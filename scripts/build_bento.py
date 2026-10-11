"""Build a release from a clean Git commit and record its exact Bento tag."""

import json
import subprocess

from ielts_api.artifacts import MODELS, ROOT, sha256


def main() -> None:
    def output(*command: str) -> str:
        return subprocess.check_output(command, cwd=ROOT, text=True).strip()

    if output("git", "status", "--porcelain"):
        raise SystemExit("Commit the source changes before building a release Bento.")
    subprocess.run(
        ["uv", "run", "--no-sync", "python", "-m", "scripts.export_requirements", "--check"],
        cwd=ROOT,
        check=True,
    )
    commit = output("git", "rev-parse", "HEAD")
    name = "ielts-inference-api"
    entry = "service:IELTSService"
    expected_tag = f"{name}:{commit[:12]}"
    existing = subprocess.run(
        ["uv", "run", "--no-sync", "bentoml", "get", expected_tag, "-o", "json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        tag = expected_tag
        print(f"Reusing existing Bento {tag}")
    else:
        import bentoml

        bento = bentoml.bentos.build(
            entry,
            name=name,
            version=commit[:12],
            build_ctx=str(ROOT),
        )
        tag = str(bento.tag)
        if tag != expected_tag:
            raise SystemExit(f"Unexpected build result: {tag}")
    record = {
        "git_commit": commit,
        "bento_tag": tag,
        "image_tag": tag,
        "model_revisions": {name: model["revision"] for name, model in MODELS.items()},
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "requirements_sha256": sha256(ROOT / "requirements.txt"),
    }
    folder = ROOT / ".runtime"
    folder.mkdir(exist_ok=True)
    (folder / "release.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
