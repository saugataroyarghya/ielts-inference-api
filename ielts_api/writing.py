"""One model per worker, with verified files and exactly the upstream scoring path."""

from pathlib import Path

import torch
from transformers import AutoTokenizer

from ielts_api.artifacts import MODEL, RELEASE, verify_snapshot
from ielts_api.device import inference_device
from ielts_api.schemas import EssayInput, ReleaseIdentity, ScoreResponse
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
            .to(inference_device("writing"))
        )
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.identity = ReleaseIdentity(
            application_version=RELEASE["application_version"],
            model_id=MODEL["repo_id"],
            model_revision=MODEL["revision"],
            device=self.model.device.type,
        )
        # Complete one forward pass before initialization/readiness succeeds.
        self.score("Should cities improve public transport?", "Reliable buses help people travel.")

    def score(self, question: str, essay: str) -> ScoreResponse:
        result = self.score_batch([EssayInput(question=question, essay=essay)])[0]
        if "error" in result:
            raise ValueError(result["error"])
        return ScoreResponse.model_validate(result)

    def score_batch(self, requests: list[EssayInput]) -> list[dict]:
        outputs = [None] * len(requests)
        valid, indices = [], []
        special = self.tokenizer.num_special_tokens_to_add(pair=True)
        for index, request in enumerate(requests):
            tokens = len(self.tokenizer(request.question, add_special_tokens=False)["input_ids"])
            if tokens + special >= self.model.config.max_length:
                outputs[index] = {"error": "Question is too long to leave room for essay tokens."}
            else:
                indices.append(index)
                valid.append(request)
        if valid:
            # MPS shares RAM with the desktop; keep activation batches small.
            size = 1 if self.model.device.type == "mps" else len(valid)
            results = []
            for offset in range(0, len(valid), size):
                batch = valid[offset : offset + size]
                results.extend(
                    self.model.score(
                        self.tokenizer, [r.question for r in batch], [r.essay for r in batch]
                    )
                )
            for index, result in zip(indices, results, strict=True):
                outputs[index] = ScoreResponse(**result, release=self.identity).model_dump()
        return outputs
