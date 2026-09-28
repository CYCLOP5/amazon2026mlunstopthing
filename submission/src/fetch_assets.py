"""fetch the pinned public Kaggle model assets with curl and verify every byte"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path as path
import re
import shutil
import subprocess
from urllib.parse import quote, urlencode


root = path(__file__).resolve().parents[1]


def sha(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def target(bundle, name):
    relative = path(name)
    file = bundle / relative
    if relative.is_absolute() or '..' in relative.parts or not file.resolve().is_relative_to(bundle.resolve()):
        raise ValueError('unsafe model artifact path: ' + name)
    return file


def url(cfg, name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', cfg['dataset']):
        raise ValueError('invalid Kaggle dataset identity')
    version = cfg.get('version')
    if not isinstance(version, int) or version < 1:
        raise ValueError('a fixed Kaggle dataset version is required')
    return ('https://www.kaggle.com/api/v1/datasets/download/' + cfg['dataset'] + '/' + quote(name, safe='') +
            '?' + urlencode({'datasetVersionNumber': version, 'raw': 'true'}))


def download(address, file, expected_sha, expected_bytes=None):
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.is_file():
        if (expected_bytes is None or file.stat().st_size == expected_bytes) and sha(file) == expected_sha:
            return file
        raise ValueError('existing download differs: ' + str(file))
    partial = file.with_name(file.name + '.part')
    if partial.is_file() and (expected_bytes is None or partial.stat().st_size == expected_bytes):
        if sha(partial) == expected_sha:
            os.replace(partial, file)
            return file
    if partial.is_file() and expected_bytes is not None and partial.stat().st_size >= expected_bytes:
        partial.unlink()
    command = ['curl', '--disable', '--fail', '--location', '--silent', '--show-error', '--proto', '=https',
               '--proto-redir', '=https', '--connect-timeout', '30', '--retry', '5', '--retry-delay', '2',
               '--retry-connrefused', '--continue-at', '-', '--output', str(partial), address]
    result = subprocess.run(command)
    if result.returncode == 33 and partial.exists():
        partial.unlink()
        result = subprocess.run(command)
    if result.returncode:
        raise RuntimeError('Kaggle download failed; rerun to resume: ' + address)
    if (expected_bytes is not None and partial.stat().st_size != expected_bytes) or sha(partial) != expected_sha:
        partial.unlink(missing_ok=True)
        raise ValueError('downloaded model artifact failed its SHA256/size check: ' + address)
    os.replace(partial, file)
    return file


def fetch(bundle, config=None, workers=4):
    bundle = path(bundle).resolve()
    config = path(config or root / 'configs/kaggle.json')
    cfg = json.loads(config.read_text())
    manifest_file = bundle / 'reproduction_manifest.json'
    if not manifest_file.is_file():
        manifest_file = bundle / 'manifest.json'
    if sha(manifest_file) != cfg['payload_manifest_sha256']:
        raise ValueError('the local model manifest differs from the pinned Kaggle release')
    manifest = json.loads(manifest_file.read_text())
    if not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError('download workers must be between 1 and 16')
    if shutil.which('curl') is None:
        raise RuntimeError('curl is required to fetch the public model assets')
    cache = bundle / '.model-downloads' / str(cfg['version'])
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'fetch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        remote_file = download(url(cfg, 'artifact-manifest.json'), cache / 'artifact-manifest.json', cfg['artifact_manifest_sha256'])
        remote = json.loads(remote_file.read_text())
        if remote['dataset'] != cfg['dataset'] or remote['payload_manifest_sha256'] != cfg['payload_manifest_sha256']:
            raise ValueError('the remote Kaggle manifest describes a different release')
        if set(remote['files']) != set(manifest['files']):
            raise ValueError('the remote model inventory differs')
        groups = {}
        for name, info in manifest['files'].items():
            file = target(bundle, name)
            model = remote['files'][name]
            obj = info['sha256'] + '.bin'
            if not re.fullmatch(r'[0-9a-f]{64}', info['sha256']) or model['object'] != obj:
                raise ValueError('invalid content object identity')
            if any(model[k] != info[k] for k in ('sha256', 'bytes')):
                raise ValueError('remote checkpoint metadata differs: ' + name)
            groups.setdefault(obj, {'info': info, 'files': []})['files'].append(file)
        def restore(item):
            obj, group = item
            info = group['info']
            existing = []
            missing = []
            for file in group['files']:
                if file.is_file():
                    if file.stat().st_size != info['bytes'] or sha(file) != info['sha256']:
                        raise ValueError('existing model artifact differs; move it aside before fetching: ' + str(file))
                    existing.append(file)
                else:
                    missing.append(file)
            if not missing:
                return 0
            source = existing[0] if existing else download(url(cfg, obj), cache / obj, info['sha256'], info['bytes'])
            for file in missing:
                file.parent.mkdir(parents=True, exist_ok=True)
                temporary = file.with_name(file.name + '.fetching')
                if source.parent == cache:
                    os.replace(source, temporary)
                else:
                    shutil.copyfile(source, temporary)
                if temporary.stat().st_size != info['bytes'] or sha(temporary) != info['sha256']:
                    temporary.unlink(missing_ok=True)
                    raise ValueError('restored checkpoint differs: ' + str(file))
                os.replace(temporary, file)
                source = file
            return len(missing)
        restored = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(restore, item) for item in groups.items()]
            for number, job in enumerate(as_completed(jobs), 1):
                try:
                    restored += job.result()
                except BaseException:
                    for pending in jobs:
                        pending.cancel()
                    raise
                print(f'verified content objects {number}/{len(jobs)}; restored files {restored}', flush=True)
        result = {'verified': True, 'dataset': cfg['dataset'], 'version': cfg['version'],
                  'files': len(manifest['files']), 'bytes': sum(v['bytes'] for v in manifest['files'].values()),
                  'restored_files': restored, 'payload_manifest_sha256': cfg['payload_manifest_sha256']}
        (bundle / 'model-fetch.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=path, default=root)
    parser.add_argument('--config', type=path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    fetch(args.bundle, args.config, args.workers)
