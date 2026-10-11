"""Hub/local loading and audio inference for the released speaking scorer.

No model code is imported from the downloaded repository. Encoder workers use
the installed package and separately configured CUDA Python environments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .checkpoint import sha256_file


class IELTSSpeakingScorer:
    """One scoring interface over the audio encoder, text encoder and heads.

    ``from_pretrained`` downloads the small checkpoint; the encoder workers
    fetch the pinned base weights on their first audio scoring call. Configure
    their Python executables explicitly or with IELTS_AUDIO_PYTHON and
    IELTS_TEXT_PYTHON. Models are currently loaded in fresh subprocesses per call.
    """

    def __init__(self, bundle: str | Path, *, audio_python: str | Path | None = None,
                 text_python: str | Path | None = None, local_files_only: bool = False):
        self.bundle_path = Path(bundle).expanduser().resolve()
        self.audio_python = audio_python
        self.text_python = text_python
        self.local_files_only = local_files_only
        self.config = json.loads((self.bundle_path / 'scorer_config.json').read_text())
        if (self.config.get('architecture') != 'IELTSSpeakingScorer'
                or self.config.get('format_version') != 1):
            raise ValueError('Unsupported IELTS scorer architecture or format version')
        self._manifest = self._verify_bundle()
        feature = json.loads((self.bundle_path / 'feature_config.json').read_text())
        compact = json.loads((self.bundle_path / 'decision/compact_config.json').read_text())
        audio = json.loads((self.bundle_path / 'audio/config.json').read_text())
        bases = self.config['base_models']
        if (self.config['weights'] != 'model.safetensors'
                or feature.get('checkpoint_file') != 'model.safetensors'
                or compact.get('checkpoint_file') != '../model.safetensors'
                or {self.config['weights_sha256'], feature['checkpoint_sha256'],
                    compact['checkpoint_sha256']} != {sha256_file(self.bundle_path / 'model.safetensors')}
                or (audio['model'], audio['model_revision']) != (bases['audio']['repo'], bases['audio']['revision'])
                or (compact['base_model'], compact['revision']) != (bases['text']['repo'], bases['text']['revision'])):
            raise ValueError('Inconsistent checkpoint or base-model configuration')
        from .portable_heads import FeatureScorer
        self._heads = FeatureScorer.from_bundle(self.bundle_path)

    def _verify_bundle(self) -> dict[str, Any]:
        manifest = json.loads((self.bundle_path / 'release_manifest.json').read_text())
        files = manifest['files_sha256']
        required = {'model.safetensors', 'scorer_config.json', 'feature_config.json',
                    'audio/config.json', 'decision/compact_config.json', 'decision/compact_report.json'}
        if not required.issubset(files):
            raise ValueError('Release manifest is missing required model files')
        for relative, digest in files.items():
            # Hub cache snapshots legitimately symlink to sibling blob storage.
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('Invalid release path: ' + relative)
            path = self.bundle_path / relative
            if not path.is_file():
                raise ValueError('Missing or invalid release path: ' + relative)
            if sha256_file(path) != digest:
                raise ValueError('Release checksum mismatch: ' + relative)
        return manifest

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *,
                        revision: str | None = None, cache_dir: str | Path | None = None,
                        local_files_only: bool = False, token: str | bool | None = None,
                        audio_python: str | Path | None = None,
                        text_python: str | Path | None = None) -> 'IELTSSpeakingScorer':
        """Load a local release or download a Hub snapshot (pin revision in production).

        Loading checks every distributed file and initializes the small scoring
        heads on CPU. It does not install packages or start GPU workers.
        """
        local = Path(pretrained_model_name_or_path).expanduser()
        if local.is_dir():
            folder = local
        else:
            from huggingface_hub import snapshot_download
            folder = snapshot_download(str(pretrained_model_name_or_path), revision=revision,
                                       cache_dir=cache_dir, local_files_only=local_files_only,
                                       token=token)
        return cls(folder, audio_python=audio_python, text_python=text_python,
                   local_files_only=local_files_only)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Copy the verified release without duplicating pretrained base weights.

        The destination must be empty. Runtime paths and credentials are never
        serialized. The result can be loaded locally by ``from_pretrained``.
        """
        self._verify_bundle()
        destination = Path(save_directory).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise ValueError('save_directory must be empty')
        destination.mkdir(parents=True, exist_ok=True)
        for relative in [*self._manifest['files_sha256'], 'release_manifest.json']:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.bundle_path / relative, target)
        return destination

    def score_features(self, text_features: Any, audio_features: Any) -> list[dict[str, Any]]:
        """Score matching 1024-wide text and 1550-wide audio features on CPU.

        Accepts one vector or a batch; always returns a list of score objects.
        Features must be produced by the pinned encoders and feature recipe.
        """
        return self._heads.predict(text_features, audio_features)

    @staticmethod
    def _python(value: str | Path | None, variable: str, default: str) -> str:
        selected = str(value or os.environ.get(variable) or default)
        path = Path(selected).expanduser()
        # Resolving a venv's Python symlink would silently select the base
        # interpreter and discard the worker's isolated dependencies.
        resolved = str(path.absolute()) if path.is_file() else shutil.which(selected)
        if not resolved or not os.access(resolved, os.X_OK):
            raise RuntimeError(f'Set {variable} or the corresponding from_pretrained argument '
                               f'to the configured CUDA worker Python (not found: {selected})')
        return resolved

    def score(self, responses: Sequence[Mapping[str, Any]], *, session_id: str = 'practice',
              work_dir: str | Path | None = None) -> dict[str, Any]:
        """Score candidate audio with questions, parts and optional transcripts.

        ``audio_path`` is relative to the caller's working directory. Optional
        ``start``/``end`` select a candidate interval. ``timeline_start`` must
        be supplied for all responses or none. Missing transcripts invoke the
        pinned ASR model. One score set is returned across all responses.

        Intermediate files are temporary by default. A supplied work_dir retains
        them in a unique child directory, returned as ``artifacts_directory``.
        """
        if not isinstance(responses, (list, tuple)) or not responses:
            raise ValueError('responses must be a nonempty list or tuple')
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError('session_id must be a nonempty string')
        rows = []
        for index, response in enumerate(responses):
            if not isinstance(response, Mapping):
                raise ValueError(f'Response {index} must be an object')
            row = dict(response)
            if not isinstance(row.get('question'), str):
                raise ValueError(f'Response {index}: question must be a string')
            if isinstance(row.get('part'), bool) or row.get('part') not in (None, 1, 2, 3):
                raise ValueError(f'Response {index}: part must be 1, 2, 3 or null')
            if row.get('transcript') is not None and not isinstance(row['transcript'], str):
                raise ValueError(f'Response {index}: transcript must be a string or null')
            if not isinstance(row.get('audio_path'), (str, Path)):
                raise ValueError(f'Response {index}: audio_path is required')
            path = Path(row['audio_path']).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            row['audio_path'] = str(path)
            rows.append(row)
        audio_python = self._python(self.audio_python, 'IELTS_AUDIO_PYTHON', '.venv-audio/bin/python')
        text_python = self._python(self.text_python, 'IELTS_TEXT_PYTHON', '.venv-text/bin/python')
        asr_model = None
        if any(row.get('transcript') is None for row in rows):
            try:
                import faster_whisper  # noqa: F401
            except ImportError as exc:
                raise RuntimeError('Missing transcripts require the [asr] package extra') from exc
            from huggingface_hub import snapshot_download
            base = self.config['base_models']['optional_asr']
            asr_model = snapshot_download(base['repo'], revision=base['revision'],
                                          local_files_only=self.local_files_only)
        if work_dir is None:
            with tempfile.TemporaryDirectory(prefix='ielts-score-') as temporary:
                return self._score(rows, session_id, Path(temporary), audio_python, text_python, asr_model)
        parent = Path(work_dir).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='score-', dir=parent))
        result = self._score(rows, session_id, folder, audio_python, text_python, asr_model)
        result['artifacts_directory'] = str(folder)
        (folder / 'scores.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def _score(self, responses, session_id, folder, audio_python, text_python, asr_model):
        from .prepare_input import prepare
        from .application import score_prepared
        request = folder / 'request.json'
        request.write_text(json.dumps({'session_id': session_id, 'responses': responses}, ensure_ascii=False))
        prepare(request, folder / 'prepared', asr_model=asr_model)
        return score_prepared(folder / 'prepared/session.json', self.bundle_path,
                              self.bundle_path / 'decision', folder, audio_python, text_python,
                              local_files_only=self.local_files_only)
