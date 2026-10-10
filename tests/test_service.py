import json

import pytest
from starlette.testclient import TestClient

from ielts_api.artifacts import ROOT
from scripts.verify_http import check_prediction

pytestmark = pytest.mark.integration
REFERENCE = json.loads((ROOT / "tests/fixtures/reference.json").read_text())


@pytest.fixture(scope="module")
def client():
    from service import IELTSScoringService

    with TestClient(IELTSScoringService.to_asgi()) as test_client:
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
