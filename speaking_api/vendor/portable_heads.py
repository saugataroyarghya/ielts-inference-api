"""NumPy inference for the learned scoring heads, after audio/text encoding.

The frozen encoders are separate. This module neither transcribes nor embeds
audio. Portable exports are checked against their PyTorch originals on Kaggle.
"""
import json
import math
from pathlib import Path
import numpy as np

CRITERIA = ('fluency_and_coherence', 'lexical_resource',
            'grammatical_range_and_accuracy', 'pronunciation')

def _matrix(value, width, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite vectors of width {width}')
    return array

def mlp(features, weights, prefix=''):
    """LayerNorm(eps=1e-5), Linear, exact GELU, inference-dropout, Linear."""
    x = np.asarray(features, dtype=np.float32)
    width = weights[prefix+'0.weight'].shape[0]
    x = _matrix(x, width, 'head input')
    normalized = (x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)
    normalized = normalized*weights[prefix+'0.weight']+weights[prefix+'0.bias']
    hidden = normalized@weights[prefix+'1.weight'].T+weights[prefix+'1.bias']
    erf = np.fromiter((math.erf(float(v)) for v in (hidden/math.sqrt(2)).flat),
                      dtype=np.float32, count=hidden.size).reshape(hidden.shape)
    hidden = .5*hidden*(1+erf)
    return hidden@weights[prefix+'4.weight'].T+weights[prefix+'4.bias']

def expected_bands(logits):
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape[-1] != 9 or not np.isfinite(logits).all():
        raise ValueError('Expected finite logits for nine band levels')
    p = np.exp(logits-logits.max(-1,keepdims=True))
    p /= p.sum(-1,keepdims=True)
    return (p*np.arange(1,10,dtype=np.float32)).sum(-1)

class FeatureScorer:
    """Use E4 FC/PR and repaired LR/GRA heads on matching frozen features."""
    def __init__(self, text_artifacts, audio_normalization=None):
        folder = Path(text_artifacts)
        self.config = json.loads((folder/'compact_config.json').read_text())
        if 'checkpoint_file' in self.config:
            from .checkpoint import load_groups
            groups = load_groups(folder/self.config['checkpoint_file'],
                ('decision.text.', 'decision.fixed.', 'normalization.text.', 'normalization.audio.'),
                'np', self.config['checkpoint_sha256'])
            self.text_weights = groups['decision.text.']
            self.fixed_weights = groups['decision.fixed.']
            self.text_norm = groups['normalization.text.']
            self.audio_norm = groups['normalization.audio.']
        else:
            self.text_weights = dict(np.load(folder/'compact_text_head.npz', allow_pickle=False))
            self.fixed_weights = dict(np.load(folder/'fixed_fc_pr_heads.npz', allow_pickle=False))
            self.text_norm = dict(np.load(folder/'compact_text_normalization.npz', allow_pickle=False))
            self.audio_norm = dict(np.load(audio_normalization or folder/'normalization.npz', allow_pickle=False))
        self.report = json.loads((folder/'compact_report.json').read_text())
        if self.config['criterion_order'] != list(CRITERIA[1:3]):
            raise ValueError('Text head criterion order does not match the scoring contract')

    @classmethod
    def from_bundle(cls, folder):
        """Load a local downloaded research bundle; encoders run separately."""
        return cls(Path(folder)/'decision')

    def raw_scores(self, text_mean, audio_features):
        text = _matrix(text_mean, self.config['hidden_size'], 'text_mean')
        audio = _matrix(audio_features, 1550, 'audio_features')
        if len(text) != len(audio):
            raise ValueError('Audio and text must have the same number of sessions')
        audio = np.clip((audio-self.audio_norm['mean'])/self.audio_norm['std'], -10, 10)
        normalized_text = np.clip((text-self.text_norm['mean'])/self.text_norm['std'], -10, 10)
        text_scores = expected_bands(mlp(normalized_text,self.text_weights).reshape(-1,2,9))
        fc = expected_bands(mlp(np.concatenate([text,audio],axis=-1),self.fixed_weights,'fc_head.'))
        pr = expected_bands(mlp(audio,self.fixed_weights,'pr_head.'))
        return np.column_stack([fc,text_scores,pr])

    def predict(self, text_mean, audio_features):
        outputs = []
        for raw in self.raw_scores(text_mean,audio_features):
            displayed = np.floor(np.clip(raw,1,9)*2+.5)/2
            outputs.append({'scores':dict(zip(CRITERIA,map(float,displayed))),
                'speaking_band':float(math.floor(float(displayed.mean())*2+.5)/2),
                'raw_scores':dict(zip(CRITERIA,map(float,raw))),
                'score_status':'experimental_weakly_supervised',
                'text_repair_validation_gate_passed':bool(self.report['repair_gate_passed']),
                'human_criterion_accuracy_verified':False})
        return outputs
