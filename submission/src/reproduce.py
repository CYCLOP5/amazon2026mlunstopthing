"""reproduce the final submission from bundled checkpoints or raw-data inference"""

import argparse as ap
import hashlib
import json
import os
from pathlib import Path as path
import shutil
import subprocess as sp
import sys


root = path(__file__).resolve().parents[1]


def sha(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(file):
    return json.loads(path(file).read_text())


def manifest(bundle):
    file = bundle / 'reproduction_manifest.json'
    return file if file.is_file() else bundle / 'manifest.json'


def verify(bundle):
    inventory = read(manifest(bundle))
    required = {'models/learned/encoder/retriever.json', 'models/learned/encoder/model.safetensors',
        'models/learned/gate/lgb.txt', 'models/learned/stack/lgb.txt',
        'models/earlier/gate/lgb.txt', 'models/earlier/neural/model.safetensors',
        'models/expert/model/model.safetensors', 'models/expert/model/head.safetensors',
        'models/fusion/residual_model.txt', 'models/collective/models/reranker.txt',
        'models/learned/neural/neural_metadata.json', 'models/france/rules.json'}
    required.update(f'models/learned/neural/m{i}/{name}' for i in range(15)
                    for name in ('model.safetensors', 'head.safetensors', 'neural_metadata.json', 'tokenizer.json'))
    required.update(f'models/run6/round{r}_model_{i}.txt' for r in (1, 2) for i in range(5))
    required.update(f'models/graph-ranker/bundle/{kind}_{i}.txt' for kind in ('binary', 'rank') for i in range(3))
    required.update(f'models/graph-refine/bundle/fold_{i}.txt' for i in range(3))
    required.update(f'models/hybrid/bundle/hybrid_{i}.txt' for i in range(3))
    if not required <= set(inventory['files']):
        raise ValueError('incomplete trained model inventory: ' + ', '.join(sorted(required - set(inventory['files']))))
    total = 0
    for name, item in inventory['files'].items():
        relative = path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe artifact path')
        file = bundle / relative
        if not file.resolve().is_relative_to(bundle):
            raise ValueError('artifact link escapes bundle: ' + name)
        if not file.is_file() or file.stat().st_size != item['bytes'] or sha(file) != item['sha256']:
            raise ValueError('artifact differs: ' + name)
        total += item['bytes']
    for name, expected in read(root / 'configs/runtime-sources.json')['files'].items():
        if sha(root / name) != expected:
            raise ValueError('frozen runtime differs: ' + name)
    print(json.dumps({'verified': True, 'files': len(inventory['files']), 'bytes': total}), flush=True)
    return inventory


def uv(project, *args, neural=False):
    command = ['uv', 'run', '--project', str(project), '--frozen', '--no-default-groups']
    command += ['--python', '3.11.16', '--group', 'neural'] if neural else ['--python', '3.13']
    return command + ['python', *map(str, args)]


def stage(name, command, work, env, outputs):
    command = list(map(str, command))
    receipt = work / 'stages' / (name + '.json')
    identity = hashlib.sha256(json.dumps({'command': command, 'binding': env['AMAZITES_REPRO_BINDING']}, sort_keys=True).encode()).hexdigest()
    if receipt.is_file():
        saved = read(receipt)
        if saved['identity'] == identity and all(path(p).is_file() and sha(path(p)) == digest for p, digest in saved['outputs'].items()):
            print(name + ': verified resume', flush=True)
            return
        raise ValueError('stage receipt differs: ' + name)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    log = work / 'stages' / (name + '.log')
    print(name + ': running; log ' + str(log), flush=True)
    with log.open('w') as stream:
        sp.run(command, env=env, stdout=stream, stderr=sp.STDOUT, check=True)
    missing = [str(p) for p in outputs if not path(p).is_file()]
    if missing:
        raise ValueError('missing stage output: ' + ', '.join(missing))
    receipt.write_text(json.dumps({'identity': identity, 'command': command,
        'outputs': {str(p): sha(path(p)) for p in outputs}}, indent=2) + '\n')


def hf_cache(bundle, work):
    cache = work / 'hf'
    for folder in sorted((bundle / 'models/bases').iterdir()):
        source = read(folder / 'source.json')
        repo = cache / ('models--' + source['model'].replace('/', '--'))
        dest = repo / 'snapshots' / source['revision']
        for file in folder.rglob('*'):
            if not file.is_file() or file.name == 'source.json':
                continue
            target = dest / file.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(file.resolve())
            elif target.resolve() != file.resolve():
                raise ValueError('offline model cache points to a different bundle')
        (repo / 'refs').mkdir(parents=True, exist_ok=True)
        (repo / 'refs/main').write_text(source['revision'])
    for family in ('earlier', 'learned'):
        runtime = work / 'runtime' / family
        if not runtime.exists():
            shutil.copytree(root / 'src/frozen' / family, runtime,
                            ignore=shutil.ignore_patterns('.venv', '__pycache__'))
        for file in (root / 'src/frozen' / family).rglob('*'):
            relative = file.relative_to(root / 'src/frozen' / family)
            if file.is_file() and not {'.venv', '__pycache__'} & set(relative.parts):
                if not (runtime / relative).is_file() or sha(runtime / relative) != sha(file):
                    raise ValueError('working scorer source differs: ' + str(relative))
        (runtime / 'models').mkdir(exist_ok=True)
        link = runtime / 'models/hf'
        if not link.exists():
            link.symlink_to(cache.resolve(), target_is_directory=True)
    return cache


def data_stage(data, work, env):
    expected = read(root / 'configs/reproduction-data.json')
    for name, item in expected['files'].items():
        split = 'train' if name.startswith('train_') else 'test'
        if sha(data / split / name) != item['sha256']:
            raise ValueError('challenge input differs: ' + name)
    prepared = work / 'data'
    runtime = root / 'src/frozen/learned'
    command = uv(runtime, runtime / 'src/data.py', '--data', data, '--out', prepared, neural=True)
    stage('data', command, work, env, [prepared / 'meta.json'])
    if read(prepared / 'meta.json') != expected:
        raise ValueError('prepared identity metadata differs')
    return prepared


def neural_stage(family, data, bundle, work, env, gpus, threads):
    runtime = work / 'runtime' / family
    cfg = read(root / 'configs' / (family + '-scoring.json'))
    folder = work / (family + '-scores')
    models = bundle / 'models' / family
    gate = models / 'gate'
    model = models / 'neural'
    command = uv(root / 'src/frozen' / family, runtime / 'src/run.py',
        '--data', data, '--cache', work / (family + '-cache'), '--gate', gate, '--neural', model,
        '--out', folder, '--split', 'test', '--device', 'cuda', '--gpu-ids', *range(gpus),
        '--workers', gpus, '--threads', max(1, threads // gpus), '--shard-size', '200000',
        '--k-lex', cfg['k']['lexical'], '--k-dense', cfg['k']['dense'], '--k-gate', cfg['k']['gate'],
        '--encoder-batch', '512', '--neural-batch', '128', '--query-batch', '2048',
        '--neural-weight', cfg['numeric']['blend']['neural_weight'], neural=True)
    if family == 'earlier':
        command += ['--retrievers', *[r['family'] for r in cfg['retrievers']]]
    else:
        retrievers = [dict(r, checkpoint=str(models / 'encoder'), reverse_root=str(models / 'reverse')) for r in cfg['retrievers']]
        file = work / 'retrievers.json'
        file.write_text(json.dumps(retrievers, indent=2) + '\n')
        command += ['--retrievers-file', str(file), '--gate-floor', '0.001', '--neural-floor', '0.001']
    stage(family + '-neural', command, work, env, [folder / 'runs.json', folder / 'runpaths.json'])
    return folder


def cold(data, bundle, work, env, gpus, threads):
    if gpus != 4:
        raise ValueError('the recorded complete GPU chain requires --gpus 4; use predict/replay for the CPU checkpoint routes')
    fusion_project = root / 'src/fusion'
    heads = root / 'src/heads.py'
    models = bundle / 'models'
    earlier = neural_stage('earlier', data, bundle, work, env, gpus, threads)
    lex_work = work / 'lexical'
    lexical_command = uv(fusion_project, root / 'src/run.py', '-c', root / 'configs/big_machine.toml',
        '--name', 'reproduction', '--only', 'prep,blocking_test', '--set', 'paths.data=' + json.dumps(env['AMAZITES_RAW_DATA']),
        '--set', 'paths.work=' + json.dumps(str(lex_work)), '--set', 'paths.experiments=' + json.dumps(str(work / 'experiments')),
        '--set', 'resources.threads=' + str(threads))
    stage('lexical', lexical_command, work, env, [])
    lexical = list(lex_work.glob('blocking/*/test'))
    if len(lexical) != 1:
        raise ValueError('expected one complete lexical candidate pool')
    lexical = lexical[0]
    for name, parent in (('run6', None), ('graph-ranker', None), ('graph-refine', work / 'graph-ranker')):
        out = work / name
        command = uv(fusion_project, heads, name, '--data', data, '--models', models / name,
                     '--scores', earlier, '--lexical', lexical, '--out', out)
        if parent is not None:
            command += ['--parent', str(parent)]
        filename = 'test_pred.parquet' if name == 'run6' else 'test_predictions.parquet'
        stage(name, command, work, env, [out / filename, out / 'report.json'])
    selected = work / 'hybrid-prepared'
    stage('hybrid-selection', uv(fusion_project, heads, 'select-hybrid', '--data', data,
        '--parent', work / 'graph-refine', '--models', models, '--out', selected), work, env,
        [selected / 'test/features.parquet', selected / 'selection.json'])
    lexical_hybrid = work / 'hybrid-lexical'
    hybrid_script = fusion_project / 'business_entity_resolution/scripts/run_final_hybrid.py'
    stage('hybrid-lexical', uv(fusion_project, hybrid_script, 'prepare', '--data', data, '--prepared', selected,
        '--output', lexical_hybrid, '--split', 'test', '--topk', '8', '--threads', threads), work, env,
        [lexical_hybrid / '_test_SUCCESS'])
    gpu_output = work / 'hybrid-gpu'
    stage('hybrid-neural', uv(root / 'src/frozen/learned', hybrid_script, 'gpu', '--prepared', selected,
        '--lexical-test', lexical_hybrid, '--old-model', models / 'earlier/neural', '--expert', models / 'expert',
        '--output', gpu_output, '--split', 'test', '--rescore-all', '--workers', '4', '--batch', '128', neural=True),
        work, env, [gpu_output / '_SUCCESS', gpu_output / 'report.json'])
    hybrid_out = work / 'hybrid'
    stage('hybrid-head', uv(fusion_project, heads, 'hybrid', '--data', data, '--parent', work / 'graph-refine',
        '--models', models / 'hybrid', '--scores', gpu_output, '--out', hybrid_out), work, env,
        [hybrid_out / 'test_predictions.parquet', hybrid_out / 'report.json'])
    learned = neural_stage('learned', data, bundle, work, env, gpus, threads)
    runtime = work / 'runtime/learned'
    newest = work / 'newest'
    newest.mkdir(exist_ok=True)
    stage('learned-prepare', uv(root / 'src/frozen/learned', runtime / 'src/post.py', 'prepare', '--data', data,
        '--split', 'test', '--runs', learned, '--out', newest / 'test.parquet', neural=True), work, env,
        [newest / 'test.parquet', newest / 'test.json'])
    stage('learned-stack', uv(root / 'src/frozen/learned', runtime / 'src/stack2.py', 'score',
        '--scores', newest / 'test.parquet', '--model', models / 'learned/stack', '--out', newest / 'stack-test.parquet',
        '--cache', work / 'learned-cache', '--threads', threads, neural=True), work, env,
        [newest / 'stack-test.parquet', newest / 'stack-test.json'])
    prepared = work / 'fusion'
    stage('fusion-features', uv(fusion_project, heads, 'fusion', '--data', data, '--models', models, '--newest', newest,
        '--hybrid', hybrid_out, '--graph', work / 'graph-refine', '--friend', work / 'run6', '--out', prepared), work, env,
        [prepared / 'prepared-test.parquet', prepared / 'features-test.json'])
    return prepared / 'prepared-test.parquet'


def finish(data, prepared, bundle, work, out, cfg, env):
    import polars as pl
    from finish import run

    countries = pl.scan_parquet(prepared).select(pl.col('co').unique()).collect()['co'].sort().to_list()
    predictions = []
    for number, country in enumerate(countries):
        folder = work / ('collective-' + str(number))
        stage('collective-' + str(number), uv(root / 'src/fusion', root / 'src/heads.py', 'collective',
            '--data', data, '--models', bundle / 'models', '--prepared', prepared, '--out', folder, '--country', country),
            work, env, [folder / 'country-0.parquet', folder / 'prediction.json'])
        predictions.append(folder / 'country-0.parquet')
    assets = work / 'final-assets'
    assets.mkdir(exist_ok=True)
    pl.concat([pl.scan_parquet(p) for p in predictions]).sort('qid', 'tid').sink_parquet(assets / 'collective.parquet', compression='zstd')
    pl.scan_parquet(prepared).filter(pl.col('co') == 'france').select('qid', 'tid', 'newest', 'friend', 'graph').sink_parquet(assets / 'france.parquet', compression='zstd')
    files = {name: {'sha256': sha(assets / name), 'bytes': (assets / name).stat().st_size}
             for name in ('collective.parquet', 'france.parquet')}
    (assets / 'manifest.json').write_text(json.dumps({'files': files, 'kind': 'trained-model-reconstruction'}, indent=2) + '\n')
    run(path(env['AMAZITES_RAW_DATA']), assets, cfg, out, bundle / 'models/france/rules.json')


def main():
    parser = ap.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify', 'replay', 'predict', 'cold'))
    parser.add_argument('--bundle', type=path, default=root)
    parser.add_argument('--data', type=path)
    parser.add_argument('--work', type=path)
    parser.add_argument('--out', type=path)
    parser.add_argument('--config', type=path, default=root / 'configs/final.json')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=4)
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    if args.threads < 1 or args.gpus < 1:
        parser.error('worker counts must be positive')
    if args.mode == 'replay':
        if args.data is None or args.out is None:
            parser.error('--data and --out are required')
        from finish import run
        run(args.data.resolve(), root / 'assets', args.config, args.out.resolve())
        return
    hosted = root / 'configs/kaggle.json'
    inventory = json.loads(manifest(args.bundle).read_text())
    if hosted.is_file() and any(not (args.bundle / name).is_file() for name in inventory['files']):
        from fetch_assets import fetch
        fetch(args.bundle, hosted)
    verify(args.bundle)
    if args.mode == 'verify':
        return
    if args.data is None or args.out is None:
        parser.error('--data and --out are required')
    args.data, args.out = args.data.resolve(), args.out.resolve()
    if args.out.exists():
        raise FileExistsError(args.out)
    if shutil.which('uv') is None:
        raise RuntimeError('uv is required for the pinned reconstruction environments')
    work = (args.work or args.out.with_name(args.out.name + '-work')).resolve()
    work.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.relative_to(root).as_posix(): sha(p) for p in sorted((root / 'src').rglob('*.py'))
                     if not {'.venv', '__pycache__'} & set(p.relative_to(root).parts)}
    binding = hashlib.sha256((sha(manifest(args.bundle)) + sha(args.config) + json.dumps(source_hashes, sort_keys=True)).encode()).hexdigest()
    env = dict(os.environ, ER_THREADS=str(args.threads), POLARS_MAX_THREADS=str(args.threads),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               AMAZITES_REPRO_BINDING=binding, AMAZITES_RAW_DATA=str(args.data))
    env['HF_HUB_CACHE'] = str(hf_cache(args.bundle, work))
    data = data_stage(args.data, work, env)
    prepared = cold(data, args.bundle, work, env, args.gpus, args.threads) if args.mode == 'cold' else args.bundle / 'resume/fusion/test/prepared-test.parquet'
    finish(data, prepared, args.bundle, work, args.out, args.config, env)


if __name__ == '__main__':
    main()
