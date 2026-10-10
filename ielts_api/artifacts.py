"""Read the immutable model manifest and verify files before loading any weights."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((ROOT / "model.lock.json").read_text())
RELEASE = json.loads((ROOT / "release.json").read_text())


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_snapshot(folder: Path) -> None:
    for filename, expected in MODEL["files"].items():
        if sha256(folder / filename) != expected:
            raise RuntimeError(f"Model artifact checksum mismatch: {filename}")
    vendor = ROOT / "ielts_api/vendor/modeling_ielts.py"
    if sha256(vendor) != MODEL["files"]["modeling_ielts.py"]:
        raise RuntimeError("Vendored scoring code differs from the pinned upstream implementation.")
