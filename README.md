# IELTS Inference API

An HTTP API that estimates IELTS rubric scores from Writing Task 2 essays and
spoken responses. Built with BentoML and PyTorch, it exposes trained scoring
models through authenticated endpoints.

> **Licensing:** The writing model's training data includes ELLIPSE
> (CC BY-NC-SA 4.0), and commercial-use rights for its weights have not been
> cleared. The speaking scorer carries its own base-model and training-data terms.
> This repository's code has no license yet, so no reuse rights are granted.
> Read the [third-party notices](THIRD_PARTY_NOTICES.md) before deploying beyond
> personal evaluation.

## What it solves

A trained model still needs input validation, preprocessing, request handling
and a reliable way to serve predictions. This project implements that serving
layer for two different inputs: question–essay pairs and audio recordings with
question context.

The API handles audio decoding and resampling, optional transcription, model
loading, batching and response validation. Each model stays loaded between
requests. Model revisions and artifact checksums are pinned so a deployment
uses known weights and scoring code.

## What to expect

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /score` | Task 2 question and essay | Task Response, Coherence and Cohesion, Lexical Resource, Grammatical Range and Accuracy, and their mean estimate |
| `POST /score_speaking` | Recordings, questions and transcripts, or optional automatic transcription | Fluency and Coherence, Lexical Resource, Grammatical Range and Accuracy, Pronunciation, and an overall band |

Responses include the model revision. Speaking scores assess the supplied
responses and retain the model's experimental, weakly supervised status;
criterion-level human accuracy has not been verified.

## Run locally

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and FFmpeg's shared libraries
for the speaking model's audio decoder. On macOS, use `brew install ffmpeg`
if a compatible FFmpeg installation is not already available.

```bash
uv sync --locked
uv run python -m scripts.fetch_models
export IELTS_API_KEY="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run bentoml serve service:IELTSService --host 127.0.0.1
```

`IELTS_API_KEY` must be at least 32 characters; the server refuses to start
without it. The first run downloads model weights and warms the models during
startup. `IELTS_DEVICE` selects `auto` (CUDA → MPS → CPU), `cuda`, `mps` or `cpu`.
Docker on macOS cannot use MPS; Apple GPU inference requires native serving.

**Writing request**

```bash
curl http://127.0.0.1:3000/score \
  -H "Authorization: Bearer $IELTS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Should cities invest in public transport?","essay":"Reliable public transport helps people travel..."}'
```

**Speaking request**

```bash
curl http://127.0.0.1:3000/score_speaking \
  -H "Authorization: Bearer $IELTS_API_KEY" \
  -F 'audio=@answer.wav;type=audio/wav' \
  -F 'metadata={"responses":[{"question":"Where do you live?","part":1,"transcript":"I live in Dhaka."}]}'
```

Upload one WAV, FLAC or OGG file per response, in metadata order. Preserve
repetitions and disfluencies in transcripts. For automatic transcription, run
`uv run python -m scripts.fetch_models --asr`, start with `IELTS_ENABLE_ASR=1`,
and omit `transcript` where needed.

**Writing response**

```json
{
  "scores": {"TR": 6.2318, "CC": 6.0874, "LR": 6.3351, "GRA": 5.9462},
  "task2_mean_estimate": 6.150125,
  "track": "ielts",
  "max_tokens": 1024,
  "truncated": false,
  "release": {
    "application_version": "0.1.0",
    "model_id": "saugataroy/ielts-task2-modernbert-base-moe",
    "model_revision": "ad150d954cc9af7fa848fedeafeef6c28c71e2c6",
    "device": "cpu",
    "precision": "float32"
  }
}
```

`TR` is Task Response, `CC` Coherence and Cohesion, `LR` Lexical Resource and
`GRA` Grammatical Range and Accuracy. Scores are raw model estimates, neither
clipped nor rounded. `truncated` is `true` when the question and essay together
exceed `max_tokens`, so the end of the essay was cut.

**Speaking response**

```json
{
  "scores": {
    "fluency_and_coherence": 6.5,
    "lexical_resource": 6.5,
    "grammatical_range_and_accuracy": 6.0,
    "pronunciation": 6.5
  },
  "speaking_band": 6.5,
  "raw_scores": {
    "fluency_and_coherence": 6.43,
    "lexical_resource": 6.61,
    "grammatical_range_and_accuracy": 6.07,
    "pronunciation": 6.38
  },
  "score_status": "experimental_weakly_supervised",
  "text_repair_validation_gate_passed": true,
  "human_criterion_accuracy_verified": false,
  "device": "cpu",
  "assessment_scope": "provided_responses",
  "request_id": "9f2c41d7a0b34e5c8d1e6f7a2b3c4d5e",
  "model_id": "saugataroy/ielts-speaking-scorer",
  "model_revision": "2c9e4cae199ce353299145231638085a83afa5b9",
  "audio_seconds": 7.4,
  "transcript_sources": ["caller_supplied"],
  "original_session_timeline_available": false,
  "disfluency_preservation_verified": false
}
```

`raw_scores` are the model's unrounded estimates. `scores` clip those to 1–9 and
round each to the nearest half band, and `speaking_band` is the mean of `scores`
rounded to the nearest half band. `transcript_sources` has one entry per
response: `caller_supplied` or `faster_whisper_unedited`. The status and
`*_verified` fields repeat the experimental status described above in every
response. All values in both samples are illustrative.

The authenticated API schema is at `/docs.json`. `/livez` and `/readyz` are
unauthenticated health endpoints.

## Serving design and limits

BentoML routes requests to dedicated writing and speaking workers with separate
adaptive batching queues. Audio preparation runs before speaking inference.
Bearer authentication and admission limits apply before request processing;
overloaded clients receive HTTP 429 and should retry with backoff.

- Writing uses a 1,024-token context, including the question. Long essays are
  truncated using the original model's scoring logic.
- Speaking accepts up to 12 recordings, 300 seconds of selected audio and 32 MiB
  per request. Text exceeding the encoder's 8,192-token context is rejected.
- Optional `start` and `end` select recording intervals. `timeline_start` preserves
  original session timing when provided for every response.
- MPS uses smaller inference batches and allocator budgets. These do not impose
  a system-wide RAM limit; local BentoML resource declarations do not provide
  OS memory isolation.

**Current validation limit:** end-to-end native inference, numerical batch parity
and throughput verification are incomplete. Configured batch sizes are not a
measured capacity claim.

## Development and verification

CI runs these source checks on every push and pull request:

```bash
uv sync --locked
uv run pytest -q                                        # Tests without loading model weights
uv run ruff check .
uv run ruff format --check .
uv run python -m scripts.export_requirements --check    # requirements.txt matches uv.lock
```

`requirements.txt` is generated from `uv.lock` and is what the Bento image
installs. After changing dependencies, run `uv lock`, then
`uv run python -m scripts.export_requirements`, and commit `pyproject.toml`,
`uv.lock` and `requirements.txt` together.
`scripts.build_bento` refuses to build while `requirements.txt` is out of date.

Against a running API:

```bash
uv run python -m scripts.verify_http     # Writing regression
uv run python -m scripts.benchmark --task writing --requests 20 --concurrency 4
```

Full-model checks are explicit: `uv run pytest -q -m integration` and
`uv run python -m scripts.verify_speaking data/sessions.json`. Run them one at a
time, without another model server running. The speaking corpus is a JSON list
of sessions whose responses contain `audio_path`, `question`, `part` and
`transcript`. Use different recording lengths, including one longer than
20 seconds, to test chunk boundaries and batch parity.

Benchmark speaking with:

```bash
uv run python -m scripts.benchmark --task speaking \
  --corpus data/sessions.json --requests 20 --concurrency 4
```

Reports include successful requests per second, latency percentiles and response
status counts.

## Build and run a container

Build a versioned Bento from a clean commit, containerize its tag, and run it.
`scripts.build_bento` prints the tag, which is `ielts-inference-api:` followed by
the first 12 characters of the commit hash.

```bash
uv run python -m scripts.build_bento
uv run bentoml containerize ielts-inference-api:<commit>
docker run --rm -p 127.0.0.1:3000:3000 \
  -e IELTS_API_KEY="$IELTS_API_KEY" \
  ielts-inference-api:<commit>
```

The container serves the same endpoints on port 3000 and requires the same
`IELTS_API_KEY`, which it refuses to start without. Add
`-e IELTS_ENABLE_ASR=1` for automatic transcription.

- `bentoml containerize` embeds the pinned model weights in the image, fetching
  them first if the build host has not cached them. The image sets
  `HF_HUB_OFFLINE=1`, so a running container needs no network access and no
  mounted cache.
- The Linux image includes PyTorch's CUDA libraries. It runs on CPU unless the
  container is given an NVIDIA GPU (`docker run --gpus all` with the NVIDIA
  Container Toolkit installed). Set `IELTS_DEVICE=cpu` to force CPU. Docker on
  macOS always runs on CPU. CUDA execution in the container has not been verified.

## Code layout

```text
service.py          API endpoints, worker configuration and batching
ielts_api/          Model inference, audio preparation, schemas and middleware
models.lock.json    Model revisions and integrity checks
pyproject.toml      Dependencies; uv.lock pins the resolved environment
requirements.txt    Generated from uv.lock; installed by the Bento image
release.json        Application version reported in responses
scripts/            Model download, build, parity checks and load testing
tests/              API contracts, preprocessing and regression fixtures
```

Models: [Writing Task 2](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe)
and [Speaking](https://huggingface.co/saugataroy/ielts-speaking-scorer).
See [third-party notices](THIRD_PARTY_NOTICES.md) for model and code terms.
