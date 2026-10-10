"""Build a release from a clean Git commit and record its exact Bento tag."""

import json
import subprocess

from ielts_api.artifacts import MODEL, ROOT, sha256


def main() -> None:
    def output(*command: str) -> str:
        return subprocess.check_output(command, cwd=ROOT, text=True).strip()

    if output("git", "status", "--porcelain"):
        raise SystemExit("Commit the source changes before building a release Bento.")
    subprocess.run(
        ["uv", "run", "python", "-m", "scripts.export_requirements", "--check"],
        cwd=ROOT,
        check=True,
    )
    commit = output("git", "rev-parse", "HEAD")
    expected_tag = f"ielts-inference-api:{commit[:12]}"
    existing = subprocess.run(
        ["uv", "run", "bentoml", "get", expected_tag, "-o", "json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        tag = expected_tag
        print(f"Reusing existing Bento {tag}")
    else:
        log = output("uv", "run", "bentoml", "build", "--version", commit[:12], "-o", "tag")
        tags = [
            line.removeprefix("__tag__:")
            for line in log.splitlines()
            if line.startswith("__tag__:")
        ]
        if tags != [expected_tag]:
            raise SystemExit(f"Unexpected build result: {log}")
        tag = tags[0]
    record = {
        "git_commit": commit,
        "bento_tag": tag,
        "image_tag": tag,
        "model_revision": MODEL["revision"],
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "requirements_sha256": sha256(ROOT / "requirements.txt"),
    }
    folder = ROOT / ".runtime"
    folder.mkdir(exist_ok=True)
    (folder / "release.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
