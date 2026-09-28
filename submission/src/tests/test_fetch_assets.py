"""exercise pinned downloads, integrity checks and safe model restoration"""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from urllib.parse import parse_qs, unquote, urlparse

import pytest


spec = importlib.util.spec_from_file_location('fetch_assets', Path(__file__).resolve().parents[1] / 'fetch_assets.py')
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def test_corrupt_download_never_becomes_a_model(tmp_path, monkeypatch):
    def curl(cmd):
        Path(cmd[cmd.index('--output') + 1]).write_bytes(b'wrong')
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(fetcher.subprocess, 'run', curl)
    file = tmp_path / 'model.bin'
    with pytest.raises(ValueError, match='SHA256/size'):
        fetcher.download('https://www.kaggle.com/example', file, digest(b'right'), 5)
    assert not file.exists()
    assert not file.with_name(file.name + '.part').exists()


def test_pinned_fetch_restores_duplicate_content_once(tmp_path, monkeypatch):
    raw = b'model bytes'
    sha = digest(raw)
    info = {'sha256': sha, 'bytes': len(raw)}
    files = {'models/a.bin': info, 'models/sub/b.bin': info}
    manifest = json.dumps({'files': files}).encode()
    (tmp_path / 'reproduction_manifest.json').write_bytes(manifest)
    dataset = 'owner/fixed-models'
    remote = {'dataset': dataset, 'payload_manifest_sha256': digest(manifest),
              'files': {k: {**v, 'object': sha + '.bin'} for k, v in files.items()}}
    remote_bytes = json.dumps(remote).encode()
    cfg = {'dataset': dataset, 'version': 7, 'payload_manifest_sha256': digest(manifest),
           'artifact_manifest_sha256': digest(remote_bytes)}
    config = tmp_path / 'kaggle.json'
    config.write_text(json.dumps(cfg))
    requests = []
    def curl(cmd):
        parsed = urlparse(cmd[-1])
        assert parse_qs(parsed.query) == {'datasetVersionNumber': ['7'], 'raw': ['true']}
        name = unquote(parsed.path.rsplit('/', 1)[-1])
        requests.append(name)
        Path(cmd[cmd.index('--output') + 1]).write_bytes(remote_bytes if name == 'artifact-manifest.json' else raw)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(fetcher.subprocess, 'run', curl)
    monkeypatch.setattr(fetcher.shutil, 'which', lambda _: '/usr/bin/curl')
    result = fetcher.fetch(tmp_path, config, 2)
    assert result['verified'] and result['files'] == 2
    assert (tmp_path / 'models/a.bin').read_bytes() == raw
    assert (tmp_path / 'models/sub/b.bin').read_bytes() == raw
    assert requests.count(sha + '.bin') == 1
    fetcher.fetch(tmp_path, config, 2)
    assert requests.count(sha + '.bin') == 1


def test_manifest_paths_and_versions_are_bounded(tmp_path):
    with pytest.raises(ValueError, match='unsafe'):
        fetcher.target(tmp_path, '../outside.bin')
    with pytest.raises(ValueError, match='fixed Kaggle dataset version'):
        fetcher.url({'dataset': 'owner/models', 'version': 0}, 'file.bin')
