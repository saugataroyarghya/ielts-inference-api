import json

import pytest
from starlette.testclient import TestClient

from ielts_api.artifacts import ROOT
from scripts.verify_http import check_prediction

pytestmark = pytest.mark.integration
REFERENCE = json.loads((ROOT / "tests/fixtures/reference.json").read_text())


@pytest.fixture(scope="module")
def client():
    import asyncio
    import os
    from types import SimpleNamespace
    from unittest.mock import patch

    from bentoml._internal.configuration.containers import BentoMLContainer
    from huggingface_hub import snapshot_download
    from prometheus_client import CollectorRegistry

    from ielts_api.artifacts import MODEL
    from ielts_api.writing import Scorer
    from service import IELTSService

    scorer = Scorer(
        snapshot_download(
            MODEL["repo_id"],
            revision=MODEL["revision"],
            allow_patterns=list(MODEL["files"]),
            local_files_only=True,
        )
    )

    async def predict(items):
        return await asyncio.to_thread(scorer.score_batch, items)

    key = "integration-test-token-123456789012345"
    metrics = BentoMLContainer.metrics_client.get()
    _ = metrics.registry
    with (
        patch.object(metrics, "_registry", CollectorRegistry()),
        patch.dict(os.environ, {"IELTS_API_KEY": key}),
        patch.object(
            IELTSService.inner,
            "writing",
            SimpleNamespace(to_async=SimpleNamespace(predict=predict)),
        ),
        TestClient(
            IELTSService.to_asgi(), headers={"Authorization": f"Bearer {key}"}
        ) as test_client,
    ):
        yield test_client


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_api_matches_independent_upstream_reference(client, case):
    response = client.post("/score", json=case["input"])
    assert response.status_code == 200, response.text
    check_prediction(response.json(), case["expected"], REFERENCE["absolute_tolerance"])


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " ", "essay": "Essay"},
        {"question": "Question", "essay": ""},
        {"question": "word " * 1100, "essay": "Essay"},
        {"question": "Why?", "essay": "Essay", "unexpected": True},
    ],
)
def test_invalid_requests_return_client_errors(client, payload):
    response = client.post("/score", json=payload)
    assert response.status_code == 400, response.text


def test_ready_and_documented_after_model_initialization(client):
    assert client.get("/readyz").status_code == 200
    assert client.get("/livez").status_code == 200
    assert "/score" in client.get("/docs.json").json()["paths"]
