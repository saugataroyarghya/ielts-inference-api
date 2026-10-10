"""Verify a running source service, built Bento, or Docker container against frozen outputs."""

import argparse
import json
import time

import httpx

from ielts_api.artifacts import MODEL, RELEASE, ROOT, sha256
from ielts_api.schemas import ScoreResponse


def check_prediction(actual: dict, expected: dict, tolerance: float) -> float:
    parsed = ScoreResponse.model_validate(actual)
    assert parsed.release.model_id == MODEL["repo_id"]
    assert parsed.release.model_revision == MODEL["revision"]
    assert parsed.release.application_version == RELEASE["application_version"]
    for key in ["track", "max_tokens", "truncated"]:
        assert actual[key] == expected[key], key
    errors = [abs(actual["scores"][k] - expected["scores"][k]) for k in expected["scores"]]
    errors.append(abs(actual["task2_mean_estimate"] - expected["task2_mean_estimate"]))
    assert max(errors) <= tolerance, f"Prediction drift {max(errors)} exceeds {tolerance}"
    return max(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--wait-seconds", type=float, default=120)
    args = parser.parse_args()
    fixture = json.loads((ROOT / "tests/fixtures/reference.json").read_text())
    assert fixture["provenance"]["model_manifest_sha256"] == sha256(ROOT / "model.lock.json")
    with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            try:
                if client.get("/readyz").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() >= deadline:
                raise SystemExit("Service did not become ready before the deadline.")
            time.sleep(1)
        assert client.get("/livez").status_code == 200
        schema = client.get("/docs.json")
        schema.raise_for_status()
        assert "/score" in schema.json()["paths"]
        largest_error = 0.0
        for case in fixture["cases"]:
            response = client.post("/score", json=case["input"])
            response.raise_for_status()
            error = check_prediction(
                response.json(), case["expected"], fixture["absolute_tolerance"]
            )
            largest_error = max(largest_error, error)
            print(f"PASS {case['name']}: max absolute error={error:.8f}")
        for payload in [
            {"question": " ", "essay": "A valid essay."},
            {"question": "Why?", "essay": ""},
            {"question": "word " * 1100, "essay": "A valid essay."},
            {"question": "Why?", "essay": "Essay", "unexpected": True},
        ]:
            assert client.post("/score", json=payload).status_code == 400
        print(
            "PASS readiness, liveness, schema, four invalid requests; "
            f"max drift={largest_error:.8f}"
        )


if __name__ == "__main__":
    main()
