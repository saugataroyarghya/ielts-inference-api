"""Load versioned research tensors without deserializing Python objects."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_groups(path, prefixes, framework, expected_sha256):
    from safetensors import safe_open
    if sha256_file(path) != expected_sha256:
        raise ValueError('Checkpoint SHA256 does not match the selected feature recipe')
    groups = {}
    with safe_open(str(path), framework=framework, device='cpu') as archive:
        if (archive.metadata() or {}).get('ielts_format_version') != '1':
            raise ValueError('Unsupported IELTS checkpoint format')
        for prefix in prefixes:
            groups[prefix] = {key[len(prefix):]: archive.get_tensor(key)
                              for key in archive.keys() if key.startswith(prefix)}
            if not groups[prefix]:
                raise ValueError('Missing checkpoint tensor group: ' + prefix)
    return groups


def load_audio_checkpoint(bundle):
    """Support the original trusted .pt bundle and the public safetensors export."""
    bundle = Path(bundle)
    config = json.loads((bundle / 'feature_config.json').read_text())
    path = bundle / config.get('checkpoint_file', 'audio/best.pt')
    checksum = config['checkpoint_sha256']
    if path.suffix == '.safetensors':
        groups = load_groups(path, ('audio.adapters.', 'audio.head.'), 'pt', checksum)
        state = {'adapters': groups['audio.adapters.'], 'head': groups['audio.head.']}
    else:
        import torch
        if sha256_file(path) != checksum:
            raise ValueError('Audio checkpoint does not match the selected feature recipe')
        state = torch.load(path, map_location='cpu', weights_only=True)
    return state, checksum
