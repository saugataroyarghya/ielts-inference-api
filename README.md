# IELTS Inference API

An HTTP API that estimates IELTS rubric scores from Writing Task 2 essays and
spoken responses. Built with BentoML and PyTorch, it exposes trained scoring
models through authenticated endpoints.

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

The first run downloads model weights and warms the models during startup.
`IELTS_DEVICE` selects `auto` (CUDA → MPS → CPU), `cuda`, `mps` or `cpu`.
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

```bash
uv run pytest -q                         # Tests without loading model weights
uv run ruff check .
uv run python -m scripts.verify_http     # Against a running API
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

Build a versioned Bento from a clean commit with
`uv run python -m scripts.build_bento`, then containerize its tag with
`uv run bentoml containerize <tag>`.

## Code layout

```text
service.py          API endpoints, worker configuration and batching
ielts_api/          Model inference, audio preparation, schemas and middleware
models.lock.json    Model revisions and integrity checks
scripts/            Model download, build, parity checks and load testing
tests/              API contracts, preprocessing and regression fixtures
```

Models: [Writing Task 2](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe)
and [Speaking](https://huggingface.co/saugataroy/ielts-speaking-scorer).
See [third-party notices](THIRD_PARTY_NOTICES.md) for model and code terms.
