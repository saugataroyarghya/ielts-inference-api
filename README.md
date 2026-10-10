# IELTS inference API

Learn production MLOps one milestone at a time: reproducible local serving first,
then load testing, observability, and a cloud VM.

This service wraps
[saugataroy/ielts-task2-modernbert-base-moe](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe)
with BentoML. `POST /score` accepts an original Task 2 question and typed essay.
It returns continuous TR, CC, LR and GRA estimates, their arithmetic mean,
truncation status, and application/model identity. It does not generate feedback.

## Run from source

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use this directory:

```bash
uv sync --locked
uv run python -m scripts.fetch_model
uv run bentoml serve service:IELTSScoringService --host 127.0.0.1 --port 3007
```

The first fetch downloads about 600 MB of model weights and verifies all 13
manifest files. Python is pinned to 3.12.12. This milestone uses CPU FP32 with
two PyTorch threads and one BentoML worker. No GPU setup is required.

In another terminal:

```bash
curl -fsS http://127.0.0.1:3007/readyz
curl -fsS http://127.0.0.1:3007/score \
  -H 'Content-Type: application/json' --data-binary @examples/essay.json
uv run python -m scripts.verify_http --url http://127.0.0.1:3007
```

Open <http://127.0.0.1:3007/> for BentoML's API UI. Stop the server with Ctrl+C.

## Test and package

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m scripts.export_requirements --check
uv run pytest -q
```

The real-model tests need the fetched snapshot. The small GitHub Actions workflow
runs lint, formatting, dependency-export consistency, and tests marked `not integration`.
Real inference and Docker verification are explicit local release checks at this stage.

Commit your changes, then build the release. The build script refuses a dirty
working tree and uses the Git commit as the Bento version:

```bash
uv run python -m scripts.build_bento
```

Read `bento_tag` from `.runtime/release.json` and set it in your shell. Use the exact
tag printed by the build script in place of `COMMIT`:

```bash
BENTO_TAG=ielts-inference-api:COMMIT
uv run bentoml serve "$BENTO_TAG" --host 127.0.0.1 --port 3008
# Another terminal:
uv run python -m scripts.verify_http --url http://127.0.0.1:3008
```

## Build and test the local Docker image

Start Docker Desktop. This first image targets **linux/arm64**, matching this Mac.
An amd64 VM requires a separately built and verified amd64 image before promotion.

```bash
uv run bentoml containerize "$BENTO_TAG" -t "$BENTO_TAG" --opt platform=linux/arm64
uv run python -m scripts.smoke_container "$BENTO_TAG"
```

The smoke test creates a fresh container with `--network none`, no mounted host
directories, two CPUs and a 3 GiB memory limit. It calls HTTP from inside that
container, compares three frozen predictions, checks invalid input, and removes
its own test container even on failure. The weights are bundled by BentoML's
class-level `HuggingFaceModel` reference.

For interactive use, stop the source server on port 3007 and run:

```bash
docker run -d --name ielts-inference-api-local \
  --cpus 2 --memory 3g -p 127.0.0.1:3007:3000 "$BENTO_TAG"
uv run python -m scripts.verify_http --url http://127.0.0.1:3007
docker logs --tail 30 ielts-inference-api-local
# When finished:
docker stop ielts-inference-api-local
docker rm ielts-inference-api-local
```

No registry push or cloud resources are needed for this milestone. This is a local
learning service; authentication, ingress limits, operational alerts and VM recovery
are later milestones.

## What is pinned

| File | Role |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Direct dependencies, full resolved dependency graph, Python version |
| `requirements.txt` | Generated runtime export consumed by BentoML; never edit by hand |
| `model.lock.json` | Exact Hub revision and SHA-256 hashes of weights, tokenizer, code and metadata |
| `release.json` | Application version returned in each prediction |
| `service.py` | Bento service, runtime image, class-level model dependency and endpoint |
| `ielts_api/inference.py` | Checksum verification, one-time loading, warmup and prediction |
| `tests/fixtures/reference.json` | Upstream CPU FP32 predictions on synthetic essays |
| `.runtime/release.json` | Generated mapping from Git commit to Bento and image tag |

Update dependencies with `uv`, then regenerate the export with
`uv run python -m scripts.export_requirements`. The image base is pinned by digest.
Linux uses PyTorch's CPU build. BentoML injects its own pinned version into the
container requirements. Version locks preserve the inference environment; they
do not promise byte-identical image rebuilds, because BentoML also installs build
tools and OS packages. Promote an already tested image by its immutable digest.

## Model behavior and attribution

Inputs are limited to 1,024 combined tokens. Only the essay is truncated, and
`truncated` reports that fact. A question that leaves no essay space is rejected.
Blank text, extra fields, and oversized strings return HTTP 400. Scores remain
raw continuous estimates and may fall outside 0–9; the mean is not an official
IELTS Writing band. Numerical regression checks do not establish grading accuracy.

The unmodified upstream scorer is in `ielts_api/vendor/modeling_ielts.py`.
Its source and all runtime artifacts are verified before the model loads.
See [third-party notices](THIRD_PARTY_NOTICES.md) and the
[model card](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe/blob/ad150d954cc9af7fa848fedeafeef6c28c71e2c6/README.md).

Start with the [milestone 1 walkthrough](docs/01-reproducible-serving.md).
