"""The BentoML entry point: uv run bentoml serve service:IELTSScoringService."""

import bentoml
from bentoml.exceptions import InvalidArgument
from bentoml.models import HuggingFaceModel

from ielts_api.artifacts import MODEL
from ielts_api.middleware import APIGuard
from ielts_api.schemas import EssayInput, ScoreResponse

with bentoml.importing():
    from ielts_api.inference import Scorer

image = bentoml.images.Image(
    base_image=(
        "python:3.12.12-slim-bookworm@sha256:"
        "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
    ),
    python_version="3.12",
    lock_python_packages=False,  # requirements.txt is exported from the committed uv.lock.
).requirements_file("requirements.txt")


@bentoml.service(
    image=image,
    workers=1,
    threads=1,
    resources={"cpu": "2", "memory": "3Gi"},
    traffic={"timeout": 30, "concurrency": 4, "max_concurrency": 4},
    envs=[
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "BENTOML_DO_NOT_TRACK", "value": "True"},
    ],
)
class IELTSScoringService:
    model_path = HuggingFaceModel(
        MODEL["repo_id"], revision=MODEL["revision"], include=list(MODEL["files"])
    )

    def __init__(self) -> None:
        self.scorer = Scorer(self.model_path)

    @bentoml.api(input_spec=EssayInput)
    def score(self, **payload: str) -> ScoreResponse:
        try:
            return self.scorer.score(**payload)
        except ValueError as exc:
            raise InvalidArgument(str(exc)) from exc


IELTSScoringService.add_asgi_middleware(APIGuard)
