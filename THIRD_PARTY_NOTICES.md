# Third-party notices

The model and custom scoring implementation come from
`saugataroy/ielts-task2-modernbert-base-moe`, revision
`ad150d954cc9af7fa848fedeafeef6c28c71e2c6`.
The vendored scorer is an unmodified copy verified against `model.lock.json`.

- [Upstream licensing notice](ielts_api/vendor/MODEL_LICENSE.md)
- [ModernBERT base-model Apache 2.0 license](ielts_api/vendor/BASE_MODEL_LICENSE)

The upstream notice does not grant a blanket commercial-use license for the
checkpoint or a separate open-source license for its custom scoring code.
It records ELLIPSE's CC BY-NC-SA 4.0 terms, the IELTS dataset uploader's declared
CC BY 4.0 license, and unresolved original Writing9 permissions. See the exact
notice for the scope of those statements. No additional license grant for
third-party artifacts is made by this repository.

`examples/essay.json` and the input texts in the regression fixtures were written
for this learning project. They are synthetic test inputs, not training data,
examiner-scored examples, or copied private evaluation essays.

The `.agents/skills/` files originate from BentoML's official skills installation;
their individual licensing notices remain intact. Model weights, environment
directories and local build artifacts are excluded from Git.
