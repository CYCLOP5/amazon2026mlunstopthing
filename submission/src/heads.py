"""predict with the retained tree heads without refitting or reselecting policies"""

import argparse as ap
import gc
import json
import os
from pathlib import Path as path
import shutil
import sys
import tomllib


root = path(__file__).resolve().parents[1]


def setup(stage):
    if stage == 'run6':
        sys.path.insert(0, str(root / 'src'))
    else:
        project = root / 'src/fusion/business_entity_resolution'
        sys.path[:0] = [str(project), str(project / 'src'), str(project / 'scripts')]


def load(file):
    return json.loads(path(file).read_text())


def run6(data, scores, lexical, models, out):
    import lightgbm as lgb
    import polars as pl
    from er.stack.pipeline import add_groups, add_round1, build_round1, predict

    with (root / 'configs/stack.toml').open('rb') as f:
        cfg = tomllib.load(f)['stack']
    cfg.update(data=str(data), test_roots=[str(scores)], lexical_test=[str(lexical)], swap_features=True, rules=False)
    build_round1(cfg, 'test', str(out))
    report = load(models / 'report.json')
    frame = pl.read_parquet(out / 'r1_test.parquet')
    for number in (1, 2):
        key = f'round{number}'
        fitted = [lgb.Booster(model_file=str(p)) for p in sorted(models.glob(key + '_model_*.txt'))]
        if not fitted:
            raise ValueError('missing ' + key + ' checkpoints')
        features = report[key]['features']
        p = predict(fitted, frame, features)
        if number == 1:
            frame = add_round1(frame, p)
            frame = add_groups(frame, pl.read_parquet(out / 'tgt_test.parquet'), cfg['confident_p'], cfg['chunk_rows'])
        else:
            frame = frame.with_columns(pl.Series('p2', p))
    keep = [c for c in ('qid', 'tid', 'prob', 'gate_prob', 'neural_prob', 'p1', 'p2') if c in frame.columns]
    frame.select(keep).write_parquet(out / 'test_pred.parquet')
    shutil.copyfile(models / 'report.json', out / 'report.json')


def graph(data, scores, lexical, models, out, parent=None):
    import lightgbm as lgb
    import polars as pl
    from er.stack.pipeline import predict
    from graph_resolution.pipeline import log, prepare
    from graph_resolution.ranker import add_null, predict_models

    cfg = load(models / 'config.json')
    cfg.update(data=str(data), test_roots=[str(scores)], lexical_test=[str(lexical)])
    if parent is None:
        cfg.pop('graph_predictions', None)
    else:
        cfg['graph_predictions'] = str(parent)
    prepare(cfg, 'test', out, out)
    frame = pl.read_parquet(out / 'test_features.parquet')
    if parent is None:
        scored = predict_models(add_null(frame), models / 'bundle', log)
        scored = scored.filter(pl.col('is_null') == 0).drop('is_null')
    else:
        bundle = models / 'bundle'
        features = load(bundle / 'features.json')
        fitted = [lgb.Booster(model_file=str(p)) for p in sorted(bundle.glob('fold_*.txt'))]
        if not fitted:
            raise ValueError('missing graph refinement checkpoints')
        p = predict(fitted, frame, features)
        weight = load(bundle / 'decision.json')['parent_rank_weight']
        scored = frame.select('qid', 'tid', ((1 - weight) * pl.col('binary') + weight * pl.col('rank')).alias('base'))
        scored = scored.with_columns(pl.Series('head', p))
    scored.write_parquet(out / 'test_predictions.parquet')
    shutil.copyfile(models / 'report.json', out / 'report.json')


def hybrid(data, parent, scored, models, out):
    import lightgbm as lgb
    import numpy as np
    import polars as pl
    from final_hybrid.fit import build_features, union_predictions, update_probabilities
    from innovation.common import read_parent

    decision = load(models / 'bundle/decision.json')
    mode = decision['mode']
    if mode == 'parent':
        result, _ = read_parent(parent, 'test')
    else:
        frame, old, _ = build_features(data, parent, scored, out, 'test')
        features = load(models / 'bundle/features.json')[mode]
        fitted = [lgb.Booster(model_file=str(p)) for p in sorted((models / 'bundle').glob(mode + '_*.txt'))]
        if not fitted:
            raise ValueError('missing hybrid checkpoints')
        delta = np.empty(frame.height, dtype=np.float32)
        for start in range(0, frame.height, 250000):
            x = frame.slice(start, 250000).select(features).cast(pl.Float32).to_numpy()
            delta[start:start + len(x)] = np.mean([m.predict(x, raw_score=True, num_threads=int(os.environ.get('ER_THREADS', '8'))) for m in fitted], axis=0)
        result = union_predictions(old, update_probabilities(frame, delta, decision['strength']))
    result.write_parquet(out / 'test_predictions.parquet')
    shutil.copyfile(models / 'report.json', out / 'report.json')


def select_hybrid(data, parent, models, out):
    from innovation.prepare import run
    cfg = load(models / 'hybrid-preparation/selection.json')['test']
    return run(data, parent, out, cfg['max_targets'], cfg['topk'], splits=('test',))


def fusion(data, newest, hybrid, graph, friend, out):
    from latest_fusion.pipeline import prepare
    return prepare('test', data, newest, hybrid, graph, friend, out,
                   root / 'src/fusion/business_entity_resolution/configs/latest_fusion_data_meta.json')


def collective(data, prepared, models, out, countries=None):
    import lightgbm as lgb
    import polars as pl
    from er.stack.inputs import load_refs, load_targets
    from latest_fusion import innovation_graph, innovation_models
    from latest_fusion.model_innovation import score_frame
    from latest_fusion.structured_pipeline import replay

    features = load(prepared.with_name('features-test.json'))
    previous = load(models / 'fusion/report.json')
    residual = lgb.Booster(model_file=str(models / 'fusion/residual_model.txt'))
    report = load(models / 'collective/models/model_report.json')
    bundle = {'mode': 'collective_graph', 'features': report['features'],
              'models': {'lightgbm': [lgb.Booster(model_file=str(models / 'collective/models/reranker.txt'))]}}
    strength = load(models / 'collective/locked_policy.json')['selected']['strength']
    expected = load(models / 'collective/test_preparation.json')['message_passing']
    if expected['budget_pruned_sending_targets'] or expected['max_sending_owners_per_target'] != 2:
        raise ValueError('country-isolated inference requires an unpruned recorded graph')
    refs = load_refs(str(data), 'test')
    n2 = pl.scan_parquet(data / 'test/s2.parquet').select(pl.len()).collect().item()
    all_countries = refs['co'].unique().sort().to_list()
    countries = all_countries if countries is None else countries
    if not set(countries) <= set(all_countries):
        raise ValueError('unknown country')
    files = []
    diagnostics = {}
    for number, country in enumerate(countries):
        local_refs = refs.filter(pl.col('co') == country)
        local_targets = load_targets(str(data), 'test').filter(pl.col('co') == country)
        frame = pl.scan_parquet(prepared).filter(pl.col('co') == country).collect(engine='streaming')
        if frame.select('qid', 'tid').n_unique() != frame.height:
            raise ValueError('duplicate candidate pairs')
        frame, _, _ = replay(frame, residual, features, previous, 'test')
        edges, graph_report = innovation_graph.build_graph(frame, local_targets, n2)
        extra, message_report = innovation_graph.propagate(frame, edges)
        if not frame.select('qid', 'tid').equals(extra.select('qid', 'tid')):
            raise ValueError('graph representation row alignment changed')
        del edges, local_refs, local_targets
        diagnostic = {'graph': graph_report, 'message_passing': message_report}
        parts = []
        for start in range(0, frame.height, 250000):
            part = frame.slice(start, 250000).hstack(extra.slice(start, 250000).select(innovation_graph.FEATURES))
            probability = innovation_models.predict(bundle, part)
            parts.append(score_frame(part, probability, strength).select('qid', 'tid', 'p'))
        scored = pl.concat(parts)
        file = out / f'country-{number}.parquet'
        scored.write_parquet(file, compression='zstd')
        files.append(file)
        diagnostics[country] = diagnostic
        print(country, scored.height, flush=True)
        del frame, extra, scored, probability, parts, part
        gc.collect()
    if countries == all_countries:
        pl.concat([pl.scan_parquet(p) for p in files]).sort('qid', 'tid').sink_parquet(out / 'test_predictions.parquet', compression='zstd')
    (out / 'prediction.json').write_text(json.dumps({'model': 'collective_graph', 'countries': diagnostics}, indent=2) + '\n')


if __name__ == '__main__':
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('run6', 'graph-ranker', 'graph-refine', 'hybrid', 'select-hybrid', 'fusion', 'collective'))
    p.add_argument('--data', type=path, required=True)
    p.add_argument('--models', type=path, required=True)
    p.add_argument('--out', type=path, required=True)
    p.add_argument('--scores', type=path)
    p.add_argument('--lexical', type=path)
    p.add_argument('--parent', type=path)
    p.add_argument('--prepared', type=path)
    for name in ('newest', 'hybrid', 'graph', 'friend'):
        p.add_argument('--' + name, type=path)
    p.add_argument('--country', action='append')
    a = p.parse_args()
    required = {'run6': ('scores', 'lexical'), 'graph-ranker': ('scores', 'lexical'),
                'graph-refine': ('scores', 'lexical', 'parent'), 'hybrid': ('scores', 'parent'),
                'select-hybrid': ('parent',), 'fusion': ('newest', 'hybrid', 'graph', 'friend'),
                'collective': ('prepared',)}
    for name in required[a.stage]:
        if getattr(a, name) is None:
            p.error('--' + name + ' is required for ' + a.stage)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    setup(a.stage)
    if a.stage == 'run6':
        run6(a.data, a.scores, a.lexical, a.models, a.out)
    elif a.stage in ('graph-ranker', 'graph-refine'):
        graph(a.data, a.scores, a.lexical, a.models, a.out, a.parent)
    elif a.stage == 'hybrid':
        hybrid(a.data, a.parent, a.scores, a.models, a.out)
    elif a.stage == 'select-hybrid':
        select_hybrid(a.data, a.parent, a.models, a.out)
    elif a.stage == 'fusion':
        fusion(a.data, a.newest, a.hybrid, a.graph, a.friend, a.out)
    else:
        collective(a.data, a.prepared, a.models, a.out, a.country)
