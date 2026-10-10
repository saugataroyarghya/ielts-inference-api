# IELTS inference API

BentoML service for
[saugataroy/ielts-task2-modernbert-base-moe](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe).
`POST /score` accepts `question` and `essay` strings and returns continuous
TR, CC, LR and GRA estimates, their mean, truncation status, and release identity.

## Run

Use Python 3.12.12 and uv:

```bash
uv sync --locked
uv run python -m scripts.fetch_model
uv run bentoml serve service:IELTSScoringService --host 127.0.0.1 --port 3007
```

The fetch downloads and verifies the pinned model snapshot. Inference uses CPU
FP32, one BentoML worker, and two PyTorch threads. Open
<http://127.0.0.1:3007/> for the API UI. Verify the running service with:

```bash
uv run python -m scripts.verify_http --url http://127.0.0.1:3007
```

## Test

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m scripts.export_requirements --check
uv run pytest -q
```

Integration tests require the fetched model. GitHub Actions runs the source
checks and tests marked `not integration`.

## Build and containerize

Commit source changes first. The build helper requires a clean Git tree and
uses the commit as the Bento version:

```bash
uv run python -m scripts.build_bento
```

Set `BENTO_TAG` to the `bento_tag` printed by the helper, then build and verify:

```bash
uv run bentoml containerize "$BENTO_TAG" -t "$BENTO_TAG" --opt platform=linux/arm64
uv run python -m scripts.smoke_container "$BENTO_TAG"
```

The smoke test runs a temporary container without network access or host mounts.
Build and verify the appropriate architecture for the target host; the command
above targets arm64. After stopping any source server using port 3007:

```bash
docker run -d --name ielts-inference-api-local \
  --cpus 2 --memory 3g -p 127.0.0.1:3007:3000 "$BENTO_TAG"
uv run python -m scripts.verify_http --url http://127.0.0.1:3007
```

Authentication is not implemented. The commands bind the service to host loopback.

## Reproducibility and model behavior

`uv.lock` pins dependencies; `requirements.txt` is the generated runtime export
used by the BentoML Image. After dependency changes, regenerate it with
`uv run python -m scripts.export_requirements`.

`model.lock.json` pins the Hub revision and artifact checksums. `release.json`
holds the application version. Build output in `.runtime/release.json` maps the
Git commit to the Bento tag. Promote tested container images by digest.

Input is limited to 1,024 combined tokens. Only the essay is truncated; a question
leaving no essay space is rejected. Blank text, extra fields, and oversized
strings return HTTP 400. Estimates can fall outside 0–9, and their mean is not an
official IELTS Writing band. Regression fixtures check numerical consistency,
not grading accuracy.

See [third-party notices](THIRD_PARTY_NOTICES.md) for model and code attribution.
