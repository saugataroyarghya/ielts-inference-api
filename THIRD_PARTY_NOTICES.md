# Third-party notices

The model and custom scoring implementation come from
`saugataroy/ielts-task2-modernbert-base-moe`, revision
`ad150d954cc9af7fa848fedeafeef6c28c71e2c6`.
The vendored scorer is an unmodified copy verified against `models.lock.json`.

- [Upstream licensing notice](ielts_api/vendor/MODEL_LICENSE.md)
- [ModernBERT base-model Apache 2.0 license](ielts_api/vendor/BASE_MODEL_LICENSE)

The upstream notice does not grant a blanket commercial-use license for the
checkpoint or a separate open-source license for its custom scoring code.
It records ELLIPSE's CC BY-NC-SA 4.0 terms, the IELTS dataset uploader's declared
CC BY 4.0 license, and unresolved original Writing9 permissions. See the exact
notice for the scope of those statements. No additional license grant for
third-party artifacts is made by this repository.

The input texts in the regression fixtures were written for this project.
They are synthetic test inputs, not training data,
examiner-scored examples, or copied private evaluation essays.

Model weights, environment directories and local build artifacts are excluded from Git.

## Speaking scorer

The native `ielts-speaking-scorer` 0.2.0 package is installed from the pinned
Hugging Face release `2c9e4cae199ce353299145231638085a83afa5b9`.
`models.lock.json` pins the release manifest; `uv.lock` pins the package hash.
The batching adapter in `ielts_api/speaking.py` preserves the published audio
feature recipe and uses the package's serialization, encoders and scoring heads.

- [Scorer Apache 2.0 license](https://huggingface.co/saugataroy/ielts-speaking-scorer/blob/2c9e4cae199ce353299145231638085a83afa5b9/LICENSE)
- [Upstream notice](https://huggingface.co/saugataroy/ielts-speaking-scorer/blob/2c9e4cae199ce353299145231638085a83afa5b9/NOTICE)
- [Base model and training-data terms](https://huggingface.co/saugataroy/ielts-speaking-scorer/blob/2c9e4cae199ce353299145231638085a83afa5b9/THIRD_PARTY.md)

The speaking model retains its experimental, weakly supervised status. Its
criterion-level human accuracy has not been verified. Feature-head fixtures
use synthetic vectors, not examiner-scored recordings.
