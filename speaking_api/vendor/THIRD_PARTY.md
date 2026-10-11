# Provenance and license status

This research release includes project inference source and learned tensor
state. It does not include pretrained base weights, training recordings,
transcripts, rater feedback, or dataset-derived feature arrays.

The project's original source, documentation and learned scorer weights are
licensed under Apache-2.0, to the extent of the rights held by the project
contributors. See `LICENSE` and `NOTICE`. Third-party components and source
data retain their own terms; the project license does not relicense them.

The pinned model cards for `unsloth/embeddinggemma-2` and
`unsloth/Qwen3.5-0.8B` declare Apache-2.0. Their weights are downloaded from their
original repositories when used; see `base_models.json` for exact revisions.
The `mispeech/speechocean762` dataset card also declares Apache-2.0.

The saved provenance for the three Amanafif554 IELTS dataset repositories has
no declared license. Missing license metadata does not establish either a grant
or a prohibition on publishing trained weights. Those upstream permissions
remain unverified despite the project's Apache-2.0 license choice. The project
does not claim that all training-data rights have been resolved. Clarification
should address the original recordings' permissions, model training and public
distribution of learned weights.

See `training_data_sources.json` for dataset URLs/revisions and `README.md` for
the supervision recipe. Dependencies retain their own licenses. This project is
an experimental research scorer and has no claimed affiliation with IELTS.
