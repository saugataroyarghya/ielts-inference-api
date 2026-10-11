"""Pinned speaking checkpoint, source, and base-model identities."""

import json
from pathlib import Path

from ielts_api.artifacts import ROOT, sha256

MODEL = json.loads((ROOT / "speaking-model.lock.json").read_text())


def verify_snapshot(folder: Path) -> None:
    for filename, digest in MODEL["files"].items():
        if sha256(folder / filename) != digest:
            raise RuntimeError(f"Speaking artifact checksum mismatch: {filename}")
    for filename, digest in MODEL["vendor"].items():
        if sha256(Path(__file__).parent / "vendor" / filename) != digest:
            raise RuntimeError(f"Speaking reference source differs: {filename}")
