"""One model per worker, with verified files and exactly the upstream scoring path."""

from pathlib import Path

import torch
from transformers import AutoTokenizer

from ielts_api.artifacts import MODEL, RELEASE, verify_snapshot
from ielts_api.schemas import ReleaseIdentity, ScoreResponse
from ielts_api.vendor.modeling_ielts import IELTSScoringConfig, IELTSScoringModel


class Scorer:
    def __init__(self, model_path: str):
        folder = Path(model_path)
        verify_snapshot(folder)
        torch.set_num_threads(2)
        config = IELTSScoringConfig.from_pretrained(folder, local_files_only=True)
        self.model = (
            IELTSScoringModel.from_pretrained(folder, config=config, local_files_only=True)
            .eval()
            .to("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.identity = ReleaseIdentity(
            application_version=RELEASE["application_version"],
            model_id=MODEL["repo_id"],
            model_revision=MODEL["revision"],
        )
        # Complete one forward pass before initialization/readiness succeeds.
        self.score("Should cities improve public transport?", "Reliable buses help people travel.")

    def score(self, question: str, essay: str) -> ScoreResponse:
        question_tokens = len(self.tokenizer(question, add_special_tokens=False)["input_ids"])
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        if question_tokens + special_tokens >= self.model.config.max_length:
            raise ValueError("Question is too long to leave room for essay tokens.")
        result = self.model.score(self.tokenizer, [question], [essay])[0]
        return ScoreResponse(**result, release=self.identity)
