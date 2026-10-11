"""Read the immutable model manifest and verify files before loading any weights."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((ROOT / "models.lock.json").read_text())
MODEL = MODELS["writing"]
SPEAKING_MODEL = MODELS["speaking"]
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


def verify_speaking_snapshot(folder: Path) -> None:
    # The native package checks every model file against this pinned manifest.
    if sha256(folder / "release_manifest.json") != SPEAKING_MODEL["manifest_sha256"]:
        raise RuntimeError("Speaking release manifest checksum mismatch")
