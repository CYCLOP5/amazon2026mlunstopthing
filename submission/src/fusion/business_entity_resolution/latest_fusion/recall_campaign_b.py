'three fixed-budget recall scorers for the isolated campaign adapter'
import json
import os
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .directional_swaps import _records

METHODS = ('sibling_bridge', 'orphan_factor', 'source_corroboration')
COUNTRIES = ('us', 'india')
ROUNDS = 220
MIN_CLASS = 50
PARAMS = dict(objective='binary', metric='binary_logloss', learning_rate=.05,
              num_leaves=31, max_depth=6, min_data_in_leaf=60, lambda_l2=20.,
              verbosity=-1, seed=27931, deterministic=True, force_col_wise=True)
FORBIDDEN = {'qid', 'tid', 'rid', 'eid', 'y', 'own', 'deg', 'fold', 'co', 'nm', 'ad'}
SIBLING_FEATURES = ['sb_count_log', 'sb_name_jaccard_max', 'sb_name_jaccard_mean',
    'sb_name_exact', 'sb_address_exact', 'sb_joint_exact', 'sb_name_coverage_max',
    'sb_cross_source_name_max', 'sb_cross_source_address']


def _log(message):
    print(time.strftime('%H:%M:%S') + ' recall B: ' + message, flush=True)


def _feature_names(frame, names):
    names = list(names)
    if not names or len(names) != len(set(names)) or FORBIDDEN.intersection(names):
        raise ValueError('Features must be unique, nonempty, and label/ID free')
    if any(c not in frame or not frame.schema[c].is_numeric() for c in names):
        raise ValueError('Missing or nonnumeric feature')
    return names


def _matrix(frame, features):
    matrix = frame.select(features).cast(pl.Float32).to_numpy()
    if np.isinf(matrix).any():
        raise ValueError('Feature matrix contains infinity')
    return matrix


def _boundary(s2_count):
    if not isinstance(s2_count, (int, np.integer)) or s2_count < 0:
        raise ValueError('A nonnegative integer s2_count is required')
    return int(s2_count)


def sibling_features(frame, baseline, targets, s2_count, chunk_size=150_000):
    'label-free, leave-one-target-out evidence from confident accepted aliases'
    s2_count = _boundary(s2_count)
    if set(SIBLING_FEATURES).intersection(frame.columns):
        raise ValueError('Sibling feature columns already present')
    if baseline is None or targets is None:
        raise ValueError('Sibling context requires global baseline and targets')
    if baseline['tid'].n_unique() != baseline.height:
        raise ValueError('Baseline has duplicate occupied targets')
    if baseline.height and (baseline['p'].null_count() or
            not np.isfinite(baseline['p'].to_numpy()).all() or
            baseline.filter(~pl.col('p').is_between(0., 1.)).height):
        raise ValueError('Invalid incumbent score')

    seeds = (baseline.select('qid', 'tid', 'p').filter(pl.col('p') >= .99)
             .join(frame.select('qid').unique(), on='qid', how='semi')
             .sort(['qid', 'p', 'tid'], descending=[False, True, False])
             .group_by('qid', maintain_order=True).head(4))
    used_targets = pl.concat([frame.select('tid'), seeds.select('tid')]).unique()
    normalized = _records(targets.join(used_targets.rename({'tid': 'rid'}), on='rid', how='semi'),
                          'tid').select('tid', '_name', '_address', '_tokens')
    seeds = seeds.join(normalized, on='tid', how='left', validate='m:1')
    if seeds['_name'].null_count():
        raise ValueError('Incumbent target missing from target records')
    seeds = seeds.rename({'tid': '_seed_tid', 'p': '_seed_p',
                          '_name': '_seed_name', '_address': '_seed_address',
                          '_tokens': '_seed_tokens'})
    pieces = []
    for part in frame.select('qid', 'tid').iter_slices(chunk_size):
        keyed = part.with_row_index('_row').join(normalized, on='tid', how='left',
                                                  validate='m:1', maintain_order='left')
        if keyed['_name'].null_count():
            raise ValueError('Candidate target missing from target records')
        expanded = (keyed.join(seeds, on='qid', how='inner')
                    .filter(pl.col('tid') != pl.col('_seed_tid'))
                    .sort(['_row', '_seed_p', '_seed_tid'], descending=[False, True, False])
                    .group_by('_row', maintain_order=True).head(3))
        if expanded.height:
            expanded = expanded.with_columns(
                pl.col('_tokens').list.set_intersection('_seed_tokens').list.len().alias('_overlap'),
                pl.col('_tokens').list.set_union('_seed_tokens').list.len().alias('_union'),
                ((pl.col('_name') != '') & (pl.col('_name') == pl.col('_seed_name'))).alias('_name_same'),
                ((pl.col('_address') != '') & (pl.col('_address') == pl.col('_seed_address'))).alias('_addr_same'),
                ((pl.col('tid') < s2_count) != (pl.col('_seed_tid') < s2_count)).alias('_cross'))
            expanded = expanded.with_columns(
                (pl.col('_overlap') / pl.col('_union').clip(lower_bound=1)).alias('_jac'),
                (pl.col('_overlap') / pl.col('_tokens').list.len().clip(lower_bound=1)).alias('_coverage'))
            aggregate = expanded.group_by('_row').agg(
                pl.len().log1p().alias('sb_count_log'),
                pl.col('_jac').max().alias('sb_name_jaccard_max'),
                pl.col('_jac').mean().alias('sb_name_jaccard_mean'),
                pl.col('_name_same').any().alias('sb_name_exact'),
                pl.col('_addr_same').any().alias('sb_address_exact'),
                (pl.col('_name_same') & pl.col('_addr_same')).any().alias('sb_joint_exact'),
                pl.col('_coverage').max().alias('sb_name_coverage_max'),
                pl.when(pl.col('_cross')).then(pl.col('_jac')).otherwise(0.).max().alias('sb_cross_source_name_max'),
                (pl.col('_cross') & pl.col('_addr_same')).any().alias('sb_cross_source_address'))
            res = keyed.select('_row').join(aggregate, on='_row', how='left', maintain_order='left')
            pieces.append(res.select(SIBLING_FEATURES).cast(pl.Float32).fill_null(0.))
        else:
            pieces.append(pl.DataFrame({c: np.zeros(part.height, np.float32) for c in SIBLING_FEATURES}))
    extra = pl.concat(pieces) if pieces else pl.DataFrame(schema={c: pl.Float32 for c in SIBLING_FEATURES})
    return frame.hstack(extra)


def _fit_head(frame, features, labels, path, rounds, params, minimum, weights=None):
    labels = np.asarray(labels, dtype=np.int8)
    counts = {'positive': int(labels.sum()), 'negative': int(len(labels) - labels.sum())}
    report = {'rows': frame.height, 'classes': counts, 'features': list(features),
              'available': min(counts.values()) >= minimum}
    if not report['available']:
        report['reason'] = 'Insufficient fit-only class support'
        return None, report
    _log(f'fitting {path.name}: {frame.height:,} rows, {len(features)} features, {rounds} rounds')
    model = lgb.train(params, lgb.Dataset(_matrix(frame, features), label=labels,
                      weight=weights, feature_name=features), num_boost_round=rounds)
    model.save_model(str(path))
    report['model_file'] = str(path)
    return model, report


def fit(method, fit_frame, features, output_dir, **kwargs):
    'fit only supplied source rows; runtime heads and json-safe report returned'
    if method not in METHODS:
        raise ValueError(f'Unknown campaign B method: {method}')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = fit_frame.filter(pl.col('co').is_in(COUNTRIES))
    core = _feature_names(frame, features)
    if frame['y'].null_count() or frame['own'].null_count() or frame.filter(
            ~pl.col('y').is_in([0, 1]) |
            (pl.col('y') != (pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.Int8))).height:
        raise ValueError('Fit labels disagree with ownership')
    rounds = int(kwargs.get('_rounds', ROUNDS))
    minimum = int(kwargs.get('_minimum_class', MIN_CLASS))
    params = {**PARAMS, 'num_threads': int(os.environ.get('ER_THREADS', '16')),
              **kwargs.get('_model_params', {})}
    if rounds < 1 or minimum < 1:
        raise ValueError('Rounds and minimum class support must be positive')
    names, heads, reports = core, {}, {}
    report = {'method': method, 'fit_rows': frame.height, 'excluded_other_country_rows': fit_frame.height-frame.height,
              'countries': list(COUNTRIES), 'features': core, 'parameters': params,
              'rounds': rounds, 'minimum_class_count': minimum, 'heads': reports,
              'fit_scope': 'Only supplied isolated fit keys; no calibration/check labels',
              'inference_labels_read': False, 'score_interpretation': 'Uncalibrated action ranking score'}
    if method == 'sibling_bridge':
        _log(f'building sibling context for {frame.height:,} fitting rows')
        frame = sibling_features(frame, kwargs.get('baseline'), kwargs.get('targets'), kwargs.get('s2_count'))
        names = core + SIBLING_FEATURES
        heads['sibling'], reports['sibling'] = _fit_head(frame, names, frame['y'].to_numpy(),
            output/'sibling_model.txt', rounds, params, minimum)
        report['context'] = 'Up to three p>=.99 incumbent siblings; candidate target excluded; no truth or degree access'
    elif method == 'orphan_factor':

        multiplicity = frame.select(pl.len().over('tid').alias('n'))['n'].to_numpy()
        weights = 1. / multiplicity if len(multiplicity) else np.empty(0)
        owned = frame.filter(pl.col('own') >= 0)
        heads['exists'], reports['exists'] = _fit_head(frame, names, (frame['own'].to_numpy() >= 0),
            output/'exists_model.txt', rounds, params, minimum, weights=weights)
        heads['owner'], reports['owner'] = _fit_head(owned, names, owned['y'].to_numpy(),
            output/'owner_model.txt', rounds, params, minimum)
        report['factorization'] = 'P(target owned | pair features) * P(correct owner | target owned, pair features)'
        report['existence_weighting'] = 'Inverse supplied fitting candidate multiplicity per target'
    else:
        boundary = _boundary(kwargs.get('s2_count'))
        for src, condition in [('source2', pl.col('tid') < boundary), ('source3', pl.col('tid') >= boundary)]:
            source_frame = frame.filter(condition)
            heads[src], reports[src] = _fit_head(source_frame, names, source_frame['y'].to_numpy(),
                output/(src+'_model.txt'), rounds, params, minimum)
        report['fusion'] = 'Minimum score from separately fitted source2 and source3 experts'
        report['s2_count_fit'] = boundary
    report['features'] = names
    report['available'] = all(model is not None for model in heads.values())
    (output/'scorer_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return {'method': method, 'features': names, 'models': heads, 'report': report,
            'available': report['available']}


def _predict_head(model, frame, features, chunk_size=200_000):
    if model.feature_name() != list(features):
        raise ValueError('Persisted feature order differs')
    res = np.empty(frame.height, dtype=np.float32)
    for start in range(0, frame.height, chunk_size):
        part = frame.slice(start, chunk_size)
        res[start:start+part.height] = model.predict(_matrix(part, features),
            num_threads=int(os.environ.get('ER_THREADS', '16')))
    if not np.isfinite(res).all() or ((res < 0) | (res > 1)).any():
        raise ValueError('Invalid binary prediction')
    return res


def predict(bundle, frame, **kwargs):
    'predict every supplied owner; never inspect inference truth metadata'
    if not bundle['available']:
        return np.zeros(frame.height, dtype=np.float32)
    method, names, heads = bundle['method'], bundle['features'], bundle['models']
    _log(f'{method}: predicting {frame.height:,} candidate pairs')
    if method == 'sibling_bridge':
        frame = sibling_features(frame, kwargs.get('baseline'), kwargs.get('targets'), kwargs.get('s2_count'))
        return _predict_head(heads['sibling'], frame, names)
    if method == 'orphan_factor':
        return (_predict_head(heads['exists'], frame, names) *
                _predict_head(heads['owner'], frame, names)).astype(np.float32)
    if method == 'source_corroboration':
        return np.minimum(_predict_head(heads['source2'], frame, names),
                          _predict_head(heads['source3'], frame, names)).astype(np.float32)
    raise ValueError(f'Unknown campaign B method: {method}')
