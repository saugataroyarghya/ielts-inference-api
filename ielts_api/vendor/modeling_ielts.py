"""ModernBERT IELTS Task 2 rubric regression; compatible with Transformers AutoModel."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class IELTSScoringConfig(PretrainedConfig):
    model_type = "ielts_scoring"

    def __init__(self, encoder_config=None, head="mlp", targets=None, max_length=1024,
                 label_range=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        self.head = head
        self.targets = targets or ["TR", "CC", "LR", "GRA"]
        self.max_length = max_length
        self.label_range = label_range or [0, 9]


class PlainHead(nn.Module):
    def __init__(self, size, targets):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(size, 1376), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(1376, targets))

    def forward(self, representation):
        return self.network(representation), None


class MultiGateHead(nn.Module):
    def __init__(self, size, targets):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(size, 256), nn.GELU(), nn.Dropout(0.1),
                          nn.Linear(256, 256), nn.GELU()) for _ in range(4)
        ])
        self.gates = nn.ModuleList([nn.Linear(size, 4) for _ in range(targets)])
        self.outputs = nn.ModuleList([nn.Linear(256, 1) for _ in range(targets)])

    def forward(self, representation):
        features = torch.stack([expert(representation) for expert in self.experts], dim=1)
        weights = torch.stack([gate(representation).softmax(-1) for gate in self.gates], dim=1)
        mixed = torch.einsum("bte,bed->btd", weights, features)
        return torch.cat([layer(mixed[:, i]) for i, layer in enumerate(self.outputs)], dim=-1), weights


@dataclass
class IELTSScoringOutput(ModelOutput):
    logits: torch.Tensor = None
    gates: torch.Tensor | None = None


class IELTSScoringModel(PreTrainedModel):
    config_class = IELTSScoringConfig
    base_model_prefix = "encoder"
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        settings = dict(config.encoder_config)
        model_type = settings.pop("model_type", "modernbert")
        settings["reference_compile"] = False
        encoder_config = AutoConfig.for_model(model_type, **settings)
        self.encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
        if config.head not in {"mlp", "moe"}:
            raise ValueError("head must be mlp or moe")
        self.head = (MultiGateHead if config.head == "moe" else PlainHead)(
            encoder_config.hidden_size, len(config.targets))
        self.midpoint = sum(config.label_range) / 2
        self.post_init()

    def forward(self, input_ids, attention_mask, essay_mask, **kwargs):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = (essay_mask * attention_mask).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        scores, gates = self.head(pooled)
        return IELTSScoringOutput(logits=scores + self.midpoint, gates=gates)

    @torch.inference_mode()
    def score(self, tokenizer, questions, essays):
        """Score a batch. Returns raw rubric estimates and per-input truncation flags."""
        if isinstance(questions, str) or isinstance(essays, str):
            raise TypeError("Pass questions and essays as lists of strings.")
        if not questions or len(questions) != len(essays):
            raise ValueError("Provide equally sized nonempty question and essay lists.")
        if any(not isinstance(x, str) or not x.strip() for x in [*questions, *essays]):
            raise ValueError("Every question and essay must be a nonempty string.")
        if not tokenizer.is_fast:
            raise ValueError("Use the supplied fast tokenizer to identify essay tokens.")
        encoded = tokenizer(questions, essays, truncation="only_second",
                            max_length=self.config.max_length, padding=True,
                            pad_to_multiple_of=8, return_tensors="pt")
        mask = torch.tensor([[int(x == 1) for x in encoded.sequence_ids(i)]
                             for i in range(len(essays))], device=self.device)
        scores = self(input_ids=encoded["input_ids"].to(self.device),
                      attention_mask=encoded["attention_mask"].to(self.device),
                      essay_mask=mask).logits.float().cpu().tolist()
        q_lengths = tokenizer(questions, add_special_tokens=False)["input_ids"]
        e_lengths = tokenizer(essays, add_special_tokens=False)["input_ids"]
        special = tokenizer.num_special_tokens_to_add(pair=True)
        return [{"scores": dict(zip(self.config.targets, values)),
                 "task2_mean_estimate": sum(values) / len(values),
                 "track": "ielts", "max_tokens": self.config.max_length,
                 "truncated": len(q) + len(e) + special > self.config.max_length}
                for values, q, e in zip(scores, q_lengths, e_lengths)]
