"""Resident CUDA encoders with real tensor batches and the released feature recipe.

The unchanged vendor workers are the numerical reference. CUDA parity must be
checked before deploying this refactor or changing batch sizes/dependencies.
"""

import base64
import importlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from speaking_api.artifacts import MODEL, verify_snapshot
from speaking_api.schemas import MAX_SECONDS, PreparedSession
from speaking_api.vendor.checkpoint import load_audio_checkpoint
from speaking_api.vendor.contracts import student_turns
from speaking_api.vendor.portable_heads import FeatureScorer
from speaking_api.vendor.rubric_questions import QUESTIONS


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
    def __init__(self, bundle: Path, base_path: str, chunk_batch_size=4):
        import torch
        from unsloth import FastSentenceTransformer

        config = json.loads((bundle / "audio/config.json").read_text())
        state, _ = load_audio_checkpoint(bundle)
        self.chunk_batch_size = chunk_batch_size
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.model = FastSentenceTransformer.from_pretrained(
            model_name=base_path,
            max_seq_length=2048,
            config_kwargs={"vision_config": None},
            dtype=dtype,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        # Identity is checked before the caller resolves this pinned base snapshot.
        if (config["model"], config["model_revision"]) != (
            MODEL["bases"]["audio"]["repo"],
            MODEL["bases"]["audio"]["revision"],
        ):
            raise RuntimeError("Unexpected audio base model")
        self.model = FastSentenceTransformer.get_peft_model(
            self.model,
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            finetune_audio_layers=True,
            use_gradient_checkpointing=False,
            random_state=3407,
            task_type="FEATURE_EXTRACTION",
        )
        expected = {name for name, param in self.model.named_parameters() if param.requires_grad}
        if set(state["adapters"]) != expected:
            raise RuntimeError("Audio adapter names differ from the trained checkpoint")
        loaded = self.model.load_state_dict(state["adapters"], strict=False)
        if loaded.unexpected_keys or set(loaded.missing_keys) & expected:
            raise RuntimeError("Audio adapters did not load completely")
        self.head = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 3),
            torch.nn.Sigmoid(),
        ).cuda()
        self.head.load_state_dict(state["head"], strict=True)
        self.model.eval().requires_grad_(False)
        self.head.eval().requires_grad_(False)

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
                tensors = batch_to_device(self.model.tokenize(inputs), "cuda:0")
                with torch.no_grad():
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
    def __init__(self, base_path: str):
        import torch
        from bitsandbytes.nn import Linear4bit
        from unsloth import FastDecisionModel
        from unsloth.models.decision import _clef_question

        self.model, tokenizer = FastDecisionModel.from_pretrained(
            model_name=base_path,
            max_seq_length=8192,
            load_in_4bit=True,
        )
        FastDecisionModel.freeze_backbone(self.model)
        for module in self.model.encoder.modules():
            if isinstance(module, Linear4bit):
                module.compute_dtype = torch.float32
                module.compute_type_is_set = True
        self.model.eval().requires_grad_(False)
        backbone = self.model._backbone()
        text_model = getattr(backbone, "model", None) or backbone.base_model
        self.text_model = getattr(text_model, "language_model", text_model)
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.questions = {key: _clef_question(value) for key, value in QUESTIONS.items()}

    def tokenize(self, session: PreparedSession):
        from unsloth.models.clef import encode_record

        state = {"responses": student_turns(session.model_dump())}
        record = encode_record(
            self.tokenizer, {"state": state, "questions": self.questions}, max_length=8193
        )
        if len(record.input_ids) > 8192:
            raise ValueError("Session exceeds the trained 8192-token context; shorten the input")
        return record.input_ids

    def encode(self, records, *, device="cuda"):
        import torch
        from unsloth.models._decision_fast import pad_length
        from unsloth.models.decision import _no_cudnn_attention

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
        inputs = pad_length(self.model, {"input_ids": ids, "attention_mask": mask})
        with torch.no_grad(), _no_cudnn_attention():
            hidden = self.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state.float()
            mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        result = pooled.cpu().numpy()
        if result.shape != (len(records), 1024) or not np.isfinite(result).all():
            raise RuntimeError("Invalid text feature vectors")
        return result


class SpeakingRuntime:
    def __init__(self, bundle: str, audio_path: str, text_path: str, chunk_batch_size=4):
        # Import Unsloth before torch/transformers so its supported patches take effect.
        importlib.import_module("unsloth")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("The released speaking encoders require an NVIDIA CUDA host")
        torch.set_num_threads(2)
        folder = Path(bundle)
        verify_snapshot(folder)
        self.audio = AudioEncoder(folder, audio_path, chunk_batch_size)
        self.text = TextEncoder(text_path)
        self.heads = FeatureScorer.from_bundle(folder)

    def predict(self, sessions: list[PreparedSession]):
        outputs = [None] * len(sessions)
        valid, records, indices = [], [], []
        for index, session in enumerate(sessions):
            try:
                records.append(self.text.tokenize(session))
            except ValueError as exc:
                outputs[index] = {"error": str(exc)}
                continue
            valid.append(session)
            indices.append(index)
        if valid:
            audio = self.audio.encode(valid)
            text = self.text.encode(records)
            for index, prediction in zip(indices, self.heads.predict(text, audio), strict=True):
                outputs[index] = prediction
        return outputs
