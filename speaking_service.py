"""BentoML speaking API and its private, resident GPU dependency."""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import bentoml
from bentoml.exceptions import InvalidArgument
from bentoml.models import HuggingFaceModel

from ielts_api.middleware import APIGuard
from speaking_api.artifacts import MODEL
from speaking_api.schemas import MAX_UPLOAD_BYTES, PreparedSession, SessionInput, SpeakingResult

with bentoml.importing():
    from soundfile import LibsndfileError

    from speaking_api.encoders import SpeakingRuntime
    from speaking_api.preparation import Preparer

logger = logging.getLogger(__name__)

image = (
    bentoml.images.Image(
        base_image=(
            "python:3.12.12-slim-bookworm@sha256:"
            "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
        ),
        python_version="3.12",
        lock_python_packages=False,
    )
    .system_packages("ffmpeg", "libsndfile1")
    .requirements_file("requirements-gpu.txt")
    .build_include("requirements-overrides.txt")
)


@bentoml.service(
    workers=1,
    threads=1,
    resources={"gpu": 1, "cpu": "2", "memory": "8Gi"},
    traffic={"timeout": 240, "max_concurrency": 8},
)
class SpeakingEncoder:
    """Private service: one loaded copy of each encoder per GPU worker."""

    bundle = HuggingFaceModel(
        MODEL["repo_id"], revision=MODEL["revision"], include=list(MODEL["files"])
    )
    audio_base = HuggingFaceModel(
        MODEL["bases"]["audio"]["repo"], revision=MODEL["bases"]["audio"]["revision"]
    )
    text_base = HuggingFaceModel(
        MODEL["bases"]["text"]["repo"], revision=MODEL["bases"]["text"]["revision"]
    )

    def __init__(self):
        self.runtime = SpeakingRuntime(self.bundle, self.audio_base, self.text_base)
        # Run the actual pipeline before readiness, including lazy kernel compilation.
        from speaking_api.encoders import warmup_session

        self.runtime.predict([warmup_session()])

    @bentoml.api(batchable=True, max_batch_size=2, max_latency_ms=120000)
    def predict(self, sessions: list[PreparedSession]) -> list[dict[str, Any]]:
        return self.runtime.predict(sessions)


@bentoml.service(
    image=image,
    workers=1,
    threads=1,
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 300, "max_concurrency": 8, "concurrency": 8},
    envs=[
        {"name": "UV_OVERRIDE", "value": "requirements-overrides.txt"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "BENTOML_DO_NOT_TRACK", "value": "True"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
        {"name": "IELTS_ENABLE_ASR", "value": "0"},
    ],
)
class IELTSSpeakingService:
    encoder = bentoml.depends(SpeakingEncoder)
    asr_base = HuggingFaceModel(
        MODEL["bases"]["optional_asr"]["repo"],
        revision=MODEL["bases"]["optional_asr"]["revision"],
    )

    def __init__(self):
        enabled = os.environ.get("IELTS_ENABLE_ASR", "0") == "1"
        self.preparer = Preparer(self.asr_base if enabled else None)
        # Hold this slot until the thread exits, even if the client disconnects.
        self.preparation_slots = asyncio.Semaphore(2)
        self.preparation_tasks: set[asyncio.Task] = set()

    async def _prepare(self, audio, metadata):
        async with self.preparation_slots:
            return await asyncio.to_thread(self.preparer.prepare, audio, metadata)

    @bentoml.api
    async def score_speaking(
        self,
        audio: list[Path],
        metadata: SessionInput,
    ) -> SpeakingResult:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        task = asyncio.create_task(self._prepare(audio, metadata))
        self.preparation_tasks.add(task)
        task.add_done_callback(self.preparation_tasks.discard)
        try:
            try:
                prepared, sources, seconds = await asyncio.shield(task)
            except asyncio.CancelledError:
                # Uploaded temporary files must outlive any non-cancellable decoding thread.
                await task
                raise
            predictions = await self.encoder.to_async.predict([prepared])
            result = predictions[0]
            if "error" in result:
                raise ValueError(result["error"])
            return SpeakingResult(
                **result,
                request_id=request_id,
                model_id=MODEL["repo_id"],
                model_revision=MODEL["revision"],
                audio_seconds=seconds,
                transcript_sources=sources,
                original_session_timeline_available=metadata.responses[0].timeline_start
                is not None,
            )
        except (ValueError, OSError, LibsndfileError) as exc:
            raise InvalidArgument(str(exc)) from exc
        finally:
            # Do not log audio, transcripts, bearer tokens or user filenames.
            logger.info(
                "speaking request_id=%s seconds=%.3f", request_id, time.perf_counter() - started
            )


IELTSSpeakingService.add_asgi_middleware(
    APIGuard,
    required=True,
    upload=True,
    max_bytes=MAX_UPLOAD_BYTES,
    max_concurrency=8,
)
