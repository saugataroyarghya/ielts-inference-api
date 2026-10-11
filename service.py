"""One BentoML API for writing and speaking, with independent model batch queues."""

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

from ielts_api.artifacts import MODEL as WRITING_MODEL
from ielts_api.artifacts import SPEAKING_MODEL as MODEL
from ielts_api.middleware import APIGuard
from ielts_api.schemas import (
    MAX_UPLOAD_BYTES,
    EssayInput,
    PreparedSession,
    ScoreResponse,
    SessionInput,
    SpeakingResult,
)

with bentoml.importing():
    from soundfile import LibsndfileError

    from ielts_api.audio import Preparer
    from ielts_api.speaking import SpeakingRuntime
    from ielts_api.writing import Scorer

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
    .requirements_file("requirements.txt")
)


@bentoml.service(
    workers=1,
    threads=1,
    resources={"cpu": "2", "memory": "8Gi"},
    traffic={"timeout": 240, "max_concurrency": 8},
)
class SpeakingEncoder:
    """Resident speaking model with a bounded request queue."""

    bundle = HuggingFaceModel(MODEL["repo_id"], revision=MODEL["revision"])

    def __init__(self):
        self.runtime = SpeakingRuntime(self.bundle)
        # Run the actual pipeline before readiness, including lazy kernel compilation.
        from ielts_api.speaking import warmup_session

        self.runtime.predict([warmup_session()])

    @bentoml.api(batchable=True, max_batch_size=2, max_latency_ms=120000)
    def predict(self, sessions: list[PreparedSession]) -> list[dict[str, Any]]:
        return self.runtime.predict(sessions)


@bentoml.service(
    workers=1,
    threads=1,
    resources={"cpu": "2", "memory": "3Gi"},
    traffic={"timeout": 60, "max_concurrency": 8},
)
class WritingEncoder:
    model_path = HuggingFaceModel(
        WRITING_MODEL["repo_id"],
        revision=WRITING_MODEL["revision"],
        include=list(WRITING_MODEL["files"]),
    )

    def __init__(self):
        self.scorer = Scorer(self.model_path)

    @bentoml.api(batchable=True, max_batch_size=4, max_latency_ms=30000)
    def predict(self, requests: list[EssayInput]) -> list[dict[str, Any]]:
        return self.scorer.score_batch(requests)


@bentoml.service(
    image=image,
    workers=1,
    threads=1,
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 300, "max_concurrency": 8, "concurrency": 8},
    envs=[
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "BENTOML_DO_NOT_TRACK", "value": "True"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
        {"name": "IELTS_ENABLE_ASR", "value": "0"},
    ],
)
class IELTSService:
    writing = bentoml.depends(WritingEncoder)
    encoder = bentoml.depends(SpeakingEncoder)
    asr_base = HuggingFaceModel(
        MODEL["optional_asr"]["repo"],
        revision=MODEL["optional_asr"]["revision"],
    )

    def __init__(self):
        enabled = os.environ.get("IELTS_ENABLE_ASR", "0") == "1"
        self.preparer = Preparer(self.asr_base if enabled else None)
        # Hold this slot until the thread exits, even if the client disconnects.
        self.preparation_slots = asyncio.Semaphore(2)
        self.preparation_tasks: set[asyncio.Task] = set()

    @bentoml.api(input_spec=EssayInput)
    async def score(self, **payload: str) -> ScoreResponse:
        """Score a Task 2 essay. Concurrent requests can share a model batch."""
        predictions = await self.writing.to_async.predict([EssayInput(**payload)])
        result = predictions[0]
        if "error" in result:
            raise InvalidArgument(result["error"])
        return ScoreResponse.model_validate(result)

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


IELTSService.add_asgi_middleware(
    APIGuard,
    required=True,
    upload_paths=("/score_speaking",),
    max_bytes=MAX_UPLOAD_BYTES,
    max_concurrency=8,
)
