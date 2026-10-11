"""Bounded decoding and the released candidate-only timing recipe."""

import base64
import io
import math
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from speaking_api.schemas import MAX_SECONDS, PreparedSession, SessionInput


class Preparer:
    def __init__(self, asr_path: str | None = None):
        self.recognizer = None
        self.asr_lock = Lock()
        if asr_path:
            from faster_whisper import WhisperModel

            self.recognizer = WhisperModel(
                asr_path, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1
            )

    def prepare(self, paths: list[Path], metadata: SessionInput):
        if len(paths) != len(metadata.responses):
            raise ValueError("Provide exactly one audio upload per response, in the same order")
        if any(r.transcript is None for r in metadata.responses) and self.recognizer is None:
            raise ValueError("Provide transcripts, or enable the server's optional ASR model")
        blocks, waves, sources = [], [], []
        offset = 0.0
        for path, response in zip(paths, metadata.responses, strict=True):
            # Inspect headers before allocating decoded samples; read only the selected interval.
            with sf.SoundFile(path) as recording:
                sr = recording.samplerate
                if recording.format not in {"WAV", "FLAC", "OGG"}:
                    raise ValueError("Supported recordings are WAV, FLAC and OGG")
                if not 8000 <= sr <= 48000 or not 1 <= recording.channels <= 2:
                    raise ValueError("Use mono/stereo audio sampled at 8–48 kHz")
                end = response.end if response.end is not None else len(recording) / sr
                if not response.start < end <= len(recording) / sr + 1e-3:
                    raise ValueError("Invalid response audio interval")
                if offset + end - response.start > MAX_SECONDS:
                    raise ValueError(
                        f"At most {MAX_SECONDS} seconds of candidate audio per request"
                    )
                recording.seek(round(response.start * sr))
                clip = recording.read(
                    round(end * sr) - round(response.start * sr),
                    dtype="float32",
                    always_2d=True,
                ).mean(axis=1)
            if sr != 16000:
                divisor = math.gcd(sr, 16000)
                clip = resample_poly(clip, 16000 // divisor, sr // divisor).astype(np.float32)
            if len(clip) < 160 or not np.isfinite(clip).all():
                raise ValueError("Response audio must contain at least 10 ms of finite samples")
            duration = len(clip) / 16000
            if offset + duration > MAX_SECONDS:
                raise ValueError(f"At most {MAX_SECONDS} seconds of candidate audio per request")
            transcript = response.transcript
            if transcript is None:
                with self.asr_lock:
                    segments, _ = self.recognizer.transcribe(
                        clip,
                        language="en",
                        beam_size=5,
                        word_timestamps=True,
                        vad_filter=False,
                        condition_on_previous_text=False,
                    )
                    transcript = " ".join(s.text.strip() for s in segments)
                sources.append("faster_whisper_unedited")
            else:
                sources.append("caller_supplied")
            start = response.timeline_start if response.timeline_start is not None else offset
            if blocks and start < blocks[-1]["end"] - 1e-3:
                raise ValueError("Response timelines overlap or are out of order")
            blocks.append(
                dict(
                    question=response.question,
                    part=response.part,
                    text=transcript,
                    start=start,
                    end=start + duration,
                    audio_start=offset,
                    audio_end=offset + duration,
                )
            )
            offset += duration
            waves.append(clip)
        if not any(block["text"].strip() for block in blocks):
            raise ValueError("Scoring requires assessable candidate speech and a transcript")
        stream = io.BytesIO()
        # The released preparation writes FLAC with the library's default PCM_16 subtype.
        sf.write(stream, np.concatenate(waves), 16000, format="FLAC", subtype="PCM_16")
        prepared = PreparedSession(
            audio_flac=base64.b64encode(stream.getvalue()).decode(),
            responses=blocks,
        )
        return prepared, sources, offset
