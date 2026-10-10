"""Capture candidate golden outputs directly from the pinned upstream model, outside the API."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from ielts_api.artifacts import MODEL, ROOT, sha256, verify_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/reference-candidate.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an existing reference: {args.output}")
    folder = Path(
        snapshot_download(MODEL["repo_id"], revision=MODEL["revision"], local_files_only=True)
    )
    verify_snapshot(folder)
    torch.set_num_threads(2)
    # Only this explicit reference-capture tool executes the reviewed upstream module.
    # Production imports our byte-for-byte verified local copy instead.
    model = (
        AutoModel.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
        .eval()
        .to("cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    fixture = json.loads((ROOT / "tests/fixtures/reference.json").read_text())
    cases = []
    for case in fixture["cases"]:
        payload = case["input"]
        result = model.score(tokenizer, [payload["question"]], [payload["essay"]])[0]
        cases.append({"name": case["name"], "input": payload, "expected": result})
    output = {
        "provenance": {
            "model_id": MODEL["repo_id"],
            "model_revision": MODEL["revision"],
            "model_manifest_sha256": sha256(ROOT / "model.lock.json"),
            "device": "cpu",
            "precision": "float32",
            "platform": platform.platform(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["torch", "transformers", "huggingface-hub"]
            },
            "purpose": "Numerical regression fixtures from synthetic essays; not accuracy labels.",
        },
        "absolute_tolerance": 0.001,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Captured {len(cases)} upstream reference predictions: {args.output}")


if __name__ == "__main__":
    main()
