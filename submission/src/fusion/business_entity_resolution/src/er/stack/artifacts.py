'versioned feature caches and portable, non-pickle model bundles'
import hashlib
import json
from pathlib import Path


def source_digest():
    src = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    for path in sorted(src.rglob('*.py')):
        code.update(path.relative_to(src).as_posix().encode())
        code.update(path.read_bytes())
    return code.hexdigest()


def cache_signature(sc, split):
    from er.stack.inputs import find_runs
    files = list(Path(sc['data'], split).glob('*.parquet'))
    for run in find_runs(sc[f'{split}_roots']):
        files.append(Path(run, 'manifest.json'))
    for root in sc.get(f'lexical_{split}', []):
        files.extend(Path(root).glob('*.parquet'))

    settings = {k: sc.get(k) for k in ('data', f'{split}_roots', f'lexical_{split}',
        'eval_folds', 'text_groups', 'lexical_top_k', 'lexical_min_rel', 'require_complete',
        'enhanced_features', 'exact_rescue', 'exact_max_owners')}
    inputs = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(files)]
    return hashlib.sha256(json.dumps([settings, inputs, source_digest()], sort_keys=True).encode()).hexdigest()


def save_models(root, stage, models, features):
    path = Path(root, stage)
    path.mkdir(parents=True, exist_ok=True)
    names = []
    for i, model in enumerate(models):
        name = f'fold_{i}.txt'
        model.save_model(str(path / name))
        names.append(name)
    (path / 'manifest.json').write_text(json.dumps({'features': features, 'models': names}, indent=2))


def load_models(root, stage):
    import lightgbm as lgb
    path = Path(root, stage)
    info = json.loads((path / 'manifest.json').read_text())
    return [lgb.Booster(model_file=str(path / n)) for n in info['models']], info['features']
