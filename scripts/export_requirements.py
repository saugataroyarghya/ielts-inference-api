"""Export the one locked runtime environment consumed by BentoML."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    exported = subprocess.check_output(
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-emit-package",
            "bentoml",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
        ],
        cwd=ROOT,
        text=True,
    )
    # uv exports index-specific versions but not their index URLs.
    content = (
        "# Generated from uv.lock; run uv run python -m scripts.export_requirements.\n"
        "--find-links https://download.pytorch.org/whl/cpu/torchcodec/\n" + exported
    )
    path = ROOT / "requirements.txt"
    if args.check:
        if not path.exists() or path.read_text() != content:
            raise SystemExit("requirements.txt is stale; run scripts.export_requirements")
        print("requirements.txt matches uv.lock")
    else:
        path.write_text(content)
        print("Exported requirements.txt")


if __name__ == "__main__":
    main()
