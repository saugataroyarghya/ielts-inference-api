"""Resident native PyTorch models with tensor batching across sessions.

Serialization and heads come from the pinned upstream package. Chunk aggregation
preserves its feature recipe; scripts.verify_speaking checks numerical parity.
"""

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from ielts_scoring.contracts import student_turns
from ielts_scoring.rubric_questions import QUESTIONS

from ielts_api.artifacts import verify_speaking_snapshot as verify_snapshot
from ielts_api.device import inference_device
from ielts_api.schemas import MAX_SECONDS, PreparedSession


def warmup_session():
    wave = (0.1 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)).astype(np.float32)
    stream = io.BytesIO()
    sf.write(stream, wave, 16000, format="FLAC", subtype="PCM_16")
    return PreparedSession(
        audio_flac=base64.b64encode(stream.getvalue()).decode(),
        responses=[
            {
                "question": "Where do you live?",
                "part": 1,
                "text": "I live in a busy city.",
                "start": 0.0,
                "end": 1.0,
                "audio_start": 0.0,
                "audio_end": 1.0,
            }
        ],
    )


def decode(session: PreparedSession):
    wave, sr = sf.read(
        io.BytesIO(base64.b64decode(session.audio_flac, validate=True)), dtype="float32"
    )
    if (
        sr != 16000
        or wave.ndim != 1
        or not 160 <= len(wave) <= MAX_SECONDS * sr
        or not np.isfinite(wave).all()
    ):
        raise ValueError("Prepared audio must be bounded, finite, mono 16 kHz")
    responses = []
    for response in session.responses:
        begin, end = round(response.audio_start * sr), round(response.audio_end * sr)
        if not 0 <= begin < end <= len(wave):
            raise ValueError("Response audio boundaries are invalid")
        responses.append(wave[begin:end])
    return wave, responses


def audio_features(wave, response_waves, rows, vectors, predictions, durations):
    """Exactly the upstream duration weighting, pauses and pitch statistics."""
    import librosa

    weights = np.asarray(durations, dtype=np.float64)
    weights /= weights.sum()
    z, p = np.asarray(vectors), np.asarray(predictions)
    zm, pm = (weights[:, None] * z).sum(0), (weights[:, None] * p).sum(0)
    zs = np.sqrt((weights[:, None] * (z - zm) ** 2).sum(0))
    ps = np.sqrt((weights[:, None] * (p - pm) ** 2).sum(0))
    energies = [
        librosa.feature.rms(y=clip, frame_length=1024, hop_length=256)[0] for clip in response_waves
    ]
    threshold = max(1e-4, float(np.percentile(np.concatenate(energies), 90)) * 0.05)
    masks = [energy < threshold for energy in energies]
    pause_lengths, pitch_values = [], []
    for clip, mask in zip(response_waves, masks, strict=True):
        changes = np.diff(np.r_[False, mask, False].astype(int))
        lengths = (np.where(changes == -1)[0] - np.where(changes == 1)[0]) * 256 / 16000
        pause_lengths.extend(lengths[lengths >= 0.25])
        f0 = librosa.yin(clip, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=256)
        if len(f0) != len(mask):
            raise RuntimeError("Pitch and pause frame counts differ")
        pitch_values.extend(f0[~mask])
    pauses, voiced = np.asarray(pause_lengths), np.asarray(pitch_values)
    pitch_cv = float(np.std(voiced) / max(np.mean(voiced), 1)) if len(voiced) else 0.0
    seconds = len(wave) / 16000
    words = sum(len(r["text"].split()) for r in rows)
    span = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    timing = np.array(
        [
            np.log1p(seconds),
            np.log1p(words),
            words / seconds,
            seconds / max(span, seconds),
            np.concatenate(masks).mean(),
            len(pauses) * 60 / seconds,
            float(pauses.mean()) if len(pauses) else 0.0,
            pitch_cv,
        ]
    )
    vector = np.r_[zm, zs, pm, ps, timing].astype(np.float32)
    if vector.shape != (1550,) or not np.isfinite(vector).all():
        raise RuntimeError("Invalid audio feature vector")
    return vector


class AudioEncoder:
    def __init__(self, model, head, device, chunk_batch_size=2):
        self.model = model
        self.head = head
        self.device = device
        self.chunk_batch_size = chunk_batch_size

    def encode(self, sessions: list[PreparedSession]):
        import torch
        from sentence_transformers.util import batch_to_device

        decoded = [decode(session) for session in sessions]
        chunks = []
        for session_index, (_, responses) in enumerate(decoded):
            for wave in responses:
                for offset in range(0, len(wave), 20 * 16000):
                    clip = wave[offset : offset + 20 * 16000]
                    if len(clip) >= 160:
                        chunks.append((session_index, clip))
        # Similar durations reduce padding; indices restore the original session order.
        chunks.sort(key=lambda item: len(item[1]), reverse=True)
        vectors, predictions, durations = ([[] for _ in sessions] for _ in range(3))
        with tempfile.TemporaryDirectory(prefix="ielts-chunks-") as folder:
            for offset in range(0, len(chunks), self.chunk_batch_size):
                batch = chunks[offset : offset + self.chunk_batch_size]
                inputs = []
                for index, (_, clip) in enumerate(batch):
                    path = Path(folder) / f"{index}.wav"
                    sf.write(path, clip, 16000)
                    inputs.append({"audio": str(path)})
                tensors = batch_to_device(self.model.tokenize(inputs), self.device)
                with torch.inference_mode():
                    z = self.model(tensors)["sentence_embedding"].float()
                    p = self.head(z)
                z, p = z.cpu().numpy(), p.cpu().numpy()
                if len(z) != len(batch) or len(p) != len(batch):
                    raise RuntimeError("Audio encoder did not preserve batch cardinality")
                for (owner, clip), embedding, prediction in zip(batch, z, p, strict=True):
                    vectors[owner].append(embedding)
                    predictions[owner].append(prediction)
                    durations[owner].append(len(clip) / 16000)
        return np.stack(
            [
                audio_features(
                    *decoded[i],
                    [r.model_dump() for r in session.responses],
                    vectors[i],
                    predictions[i],
                    durations[i],
                )
                for i, session in enumerate(sessions)
            ]
        )


class TextEncoder:
    def __init__(self, model, tokenizer, device):
        self.text_model = model
        self.tokenizer = tokenizer
        self.device = device

    def tokenize(self, session: PreparedSession):
        from ielts_scoring.clef_text import encode_text_record

        state = {"responses": student_turns(session.model_dump())}
        return encode_text_record(self.tokenizer, state, QUESTIONS)

    def encode(self, records):
        import torch

        device = self.device
        if device == "mps" and len(records) > 1:
            return np.concatenate([self.encode([record]) for record in records])
        length = max(map(len, records))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise RuntimeError("Text tokenizer has no padding or EOS token")
        ids = torch.full((len(records), length), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, record in enumerate(records):
            ids[index, : len(record)] = torch.tensor(record, dtype=torch.long, device=device)
            mask[index, : len(record)] = 1
        with torch.inference_mode():
            hidden = self.text_model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state.float()
            mask = mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        result = pooled.cpu().numpy()
        if result.shape != (len(records), 1024) or not np.isfinite(result).all():
            raise RuntimeError("Invalid text feature vectors")
        return result


class SpeakingRuntime:
    def __init__(self, bundle: str, chunk_batch_size=2):
        import torch
        from ielts_scoring import IELTSSpeakingModel

        torch.set_num_threads(2)
        folder = Path(bundle)
        verify_snapshot(folder)
        self.device = inference_device("speaking")
        self.model = IELTSSpeakingModel.from_pretrained(
            folder, device=self.device, local_files_only=True
        ).eval()
        self.audio = AudioEncoder(
            self.model.audio_encoder,
            self.model.audio_head,
            self.device,
            1 if self.device == "mps" else chunk_batch_size,
        )
        self.text = TextEncoder(self.model.text_encoder, self.model.tokenizer, self.device)

    def predict(self, sessions: list[PreparedSession]):
        outputs = [None] * len(sessions)
        valid, records, indices = [], [], []
        for index, session in enumerate(sessions):
            try:
                decode(session)
                records.append(self.text.tokenize(session))
            except ValueError as exc:
                outputs[index] = {"error": str(exc)}
                continue
            valid.append(session)
            indices.append(index)
        if valid:
            audio = self.audio.encode(valid)
            text = self.text.encode(records)
            for index, prediction in zip(
                indices, self.model.score_features(text, audio), strict=True
            ):
                outputs[index] = {**prediction, "device": self.device}
        return outputs
