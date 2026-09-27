'run the cpu stacker using the existing prepared records and pretrained scores'
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))

TRAIN_RUNS = ['re5b20a0f26','r87960abcbe','rded60990b8','ra205a3c2aa']
TEST_RUNS = ['ref279743ff','r9330cf7cd5','r91e3174eb8','r78d8ba87b7','r5498d54451',
    'r8f3b6a7145','r50dc384b14','r6865ff447f','r825107015a','r2841c20a21','r2eca5b91ce',
    'r5c4f880466','r5c5baac438','r61a8c15623','ra4d4cb174d','rf79e87eb21',
    'r43ac4e9f6b','r05e5bad2b7','raf45f0bdf6','r50f8bd5b23']


def score_roots(root, split):

    names = TRAIN_RUNS if split == 'train' else TEST_RUNS
    found = [root/f'aml26-gpu-{name}'/'out'/split for name in names]
    found = [p for p in found if p.is_dir()]
    if not found:
        found = [root/split] if (root/split).is_dir() else [root]
    return [str(p.resolve()) for p in found]


def lexical_roots(root, split):
    if root is None:
        return []
    if (root/split).is_dir():
        found = [root/split]
    else:
        found = sorted(root.glob(f'blocking/*/{split}'))
    if len(found) != 1:
        raise ValueError(f'Expected one lexical cache for {split} under {root}; found {found}. '
                         'Pass the directory directly above its train/test folders.')
    if not list(found[0].glob('*.parquet')):
        raise ValueError(f'No lexical parquet files in {found[0]}')
    return [str(found[0].resolve())]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True, help='Prepared data directory containing train/ and test/')
    ap.add_argument('--scores', type=Path, required=True, help='Mounted amazon-ml-2026 directory, or local split folders')
    ap.add_argument('--lexical', type=Path, help='Lexical job work directory, or blocking/<hash> directory')
    ap.add_argument('--output', type=Path, default=Path('experiments'))
    ap.add_argument('--work', type=Path, help='Optional local SSD cache (separate path per variant)')
    ap.add_argument('--variant', choices=['control','v4','recall'], default='v4')
    ap.add_argument('--bundle', type=Path, help='Saved stack/bundle directory: inference only')
    ap.add_argument('--threads', type=int, default=0, help='0 uses the config default')
    ap.add_argument('--check-inputs', action='store_true', help='Inspect schemas/manifests without fitting')
    args = ap.parse_args()
    from er.stack.inputs import find_runs
    import polars as pl
    splits = ['test'] if args.bundle else ['train','test']
    overrides = {'stack.data':str(args.data.resolve()), 'paths.experiments':str(args.output.resolve())}
    for split in splits:
        for name in ('ref','s2','s3'):
            path = args.data/split/f'{name}.parquet'
            schema = pl.read_parquet_schema(path)
            need = {'rid','eid','nm','ad','co'}
            if split == 'train':
                need |= {'fold','deg'} if name == 'ref' else {'own'}
            if not need <= set(schema):
                raise ValueError(f'{path}: missing {need-set(schema)}')
        roots = score_roots(args.scores, split)
        runs = find_runs(roots)
        if not runs:
            raise ValueError(f'No scored {split} manifests under {roots}')
        configs = set()
        for path in runs:
            manifest = json.loads(Path(path,'manifest.json').read_text())
            configs.add(manifest.get('config_sha256'))
            for part in manifest['parts']:
                for key in ('name','coverage'):
                    if not Path(path,part[key]).is_file():
                        raise ValueError(f'Missing score part: {Path(path,part[key])}')
        if len(configs) != 1 or None in configs:
            raise ValueError(f'{split}: missing or mixed model configurations: {configs}')
        overrides[f'stack.{split}_roots'] = roots
        overrides[f'stack.lexical_{split}'] = lexical_roots(args.lexical, split)
        print(f'{split}: {len(runs)} scored runs, config {next(iter(configs))}', flush=True)
    if args.bundle:
        meta = json.loads((args.bundle/'bundle.json').read_text())
        if bool(meta['stack'].get('lexical_train')) != bool(args.lexical):
            raise ValueError('Inference must use the same lexical-candidate policy as training')
        overrides['stack.inference_bundle'] = str(args.bundle.resolve())
    if args.work:
        overrides['stack.work'] = str(args.work.resolve())
    if args.threads:
        overrides['resources.threads'] = args.threads
    if args.check_inputs:
        print('Input schemas and part paths checked. Full target coverage is checked during loading.')
        return
    config = {'control':'stack_v4_control.toml','v4':'stack_v4.toml','recall':'stack_v4_recall.toml'}[args.variant]
    cmd = [sys.executable, str(PROJECT/'src/run.py'), 'stack', '-c', str(PROJECT/'configs'/config)]
    if args.bundle:
        cmd += ['--name', 'stack_replay']
    for key,value in overrides.items():
        cmd += ['--set', f'{key}={json.dumps(value)}']
    subprocess.run(cmd,check=True)


if __name__ == '__main__':
    main()
