'label-free request selection and isolated fold-two neural adaptation data'
import json
from pathlib import Path

import numpy as np
import polars as pl

from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import tuning_anchors
from innovation.common import log

KEYS = ['qid', 'tid']


def checked(frame, name):
    if frame.select(KEYS).n_unique() != frame.height:
        raise ValueError(f'Duplicate {name} pairs')
    return frame.with_columns(pl.col(KEYS).cast(pl.UInt32))


def load_predictions(hybrid_result, friend, graph, split):
    'full outer candidate union; absent systems inherit present-side scores'
    name = 'validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'
    h = checked(pl.read_parquet(Path(hybrid_result)/name).select(*KEYS, pl.col('p').alias('hybrid_p')), 'hybrid')
    friend_path = Path(friend)/('val_pred.parquet' if split == 'train' else 'test_pred.parquet')
    if not friend_path.exists():
        friend_path = Path(friend)/name
    f = checked(pl.read_parquet(friend_path).select(*KEYS, pl.col('p2').alias('friend_p')), 'friend')
    g = checked(pl.read_parquet(Path(graph)/name).select(*KEYS, pl.col('head').alias('graph_p')), 'graph')
    d = h.join(f, on=KEYS, how='full', coalesce=True, validate='1:1').join(
        g, on=KEYS, how='full', coalesce=True, validate='1:1')
    for c in ('hybrid_p', 'friend_p', 'graph_p'):
        bad = d[c].drop_nulls()
        if not bad.is_finite().all() or not bad.is_between(0, 1).all():
            raise ValueError(f'Invalid probability {c}')

    d = d.with_columns(pl.col(c).is_not_null().cast(pl.UInt8).alias('has_'+c)
                       for c in ('hybrid_p', 'friend_p', 'graph_p'))
    f, g = pl.col('friend_p').clip(1e-6, 1-1e-6), pl.col('graph_p').clip(1e-6, 1-1e-6)
    blend = 1/(1+(-(.75*(f/(1-f)).log()+.25*(g/(1-g)).log())).exp())
    return d.with_columns(pl.when(pl.col('friend_p').is_not_null() & pl.col('graph_p').is_not_null())
        .then(blend).otherwise(pl.coalesce('friend_p', 'graph_p', 'hybrid_p')).alias('friend_graph_p'))


def read_cache(root, split):
    root = Path(root)
    path = root/f'{split}_features.parquet' if root.is_dir() else root
    return checked(pl.read_parquet(path), 'cached features')


def attach_probabilities(frame, full):
    frame = frame.drop([c for c in ('hybrid_p', 'friend_p', 'graph_p', 'friend_graph_p',
                                   'has_hybrid_p', 'has_friend_p', 'has_graph_p') if c in frame])
    d = frame.join(full, on=KEYS, how='left', validate='1:1', maintain_order='left')
    if d['friend_graph_p'].null_count():
        raise ValueError('Cached candidates absent from full prediction union')
    return d.with_columns(pl.coalesce(c, 'friend_graph_p').alias(c)
                         for c in ('hybrid_p', 'friend_p', 'graph_p'))


def choose_targets(frame, targets, max_targets, max_pairs):
    'deterministic country-balanced uncertainty sampling; target groups atomic'
    if max_targets < 1 or max_pairs < 1:
        raise ValueError('Selection caps must be positive')
    scores = frame.select('tid', 'hybrid_p', 'friend_p', 'graph_p', 'friend_graph_p')
    stats = scores.group_by('tid').agg(pl.len().alias('pairs'),
        pl.col('friend_graph_p').top_k(2).alias('_top'),
        (pl.max_horizontal('hybrid_p', 'friend_p', 'graph_p')-
         pl.min_horizontal('hybrid_p', 'friend_p', 'graph_p')).max().alias('disagreement'))
    stats = stats.with_columns(pl.col('_top').list.get(0).alias('best'),
        pl.col('_top').list.get(1, null_on_oob=True).fill_null(0).alias('second')).drop('_top')
    stats = stats.join(targets.select(pl.col('rid').alias('tid'), 'co',
        (pl.col('ad').fill_null('').str.strip_chars() == '').alias('missing_address')), on='tid', validate='1:1')
    stats = stats.with_columns((pl.col('best')-pl.col('second')).alias('margin'))
    stats = stats.with_columns(((pl.col('best').is_between(.03, .97)) |
        (pl.col('disagreement') >= .08) | ((pl.col('margin') < .2) & (pl.col('best') > .02)) |
        (pl.col('missing_address') & (pl.col('best') > .02))).alias('eligible'),
        (1-(2*pl.col('best')-1).abs()+pl.col('disagreement')+
         .5*(1-pl.col('margin'))+.2*pl.col('missing_address').cast(pl.Float32)).alias('uncertainty'),
        pl.col('tid').hash(8043).alias('_hash'))
    pools = [stats.filter((pl.col('co') == co) & pl.col('eligible')).sort(
        ['uncertainty', '_hash', 'tid'], descending=[True, False, False]).to_dicts()
        for co in stats['co'].unique().sort()]
    chosen, used, cursor = [], 0, 0
    while len(chosen) < max_targets and any(cursor < len(p) for p in pools):
        for pool in pools:
            if cursor >= len(pool) or len(chosen) >= max_targets:
                continue
            row = pool[cursor]
            if used+row['pairs'] <= max_pairs:
                chosen.append(row['tid'])
                used += row['pairs']
        cursor += 1
    sel = stats.filter(pl.col('tid').is_in(chosen))
    report = {'cached_pairs': frame.height, 'eligible_targets': int(stats['eligible'].sum()),
        'selected_targets': len(chosen), 'selected_pairs': used, 'max_targets': max_targets,
        'max_pairs': max_pairs, 'label_free': True, 'complete_cached_target_groups': True,
        'countries': {str(co): {'eligible': sub.filter(pl.col('eligible')).height,
            'selected': sel.filter(pl.col('co') == co).height,
            'mean_uncertainty': float(sub['uncertainty'].mean())}
            for co in stats['co'].unique().sort() for sub in [stats.filter(pl.col('co') == co)]}}
    return sel.select('tid'), report


def raw_requests(keys, refs, targets):
    d = keys.join(refs.select(pl.col('rid').alias('qid'), pl.col('nm').alias('nm1'),
        pl.col('ad').alias('ad1'), pl.col('co').alias('co1')), on='qid', validate='m:1')
    d = d.join(targets.select(pl.col('rid').alias('tid'), pl.col('nm').alias('nm2'),
        pl.col('ad').alias('ad2'), pl.col('co').alias('co2')), on='tid', validate='m:1')
    if d.height != keys.height or d.filter(pl.col('co1') != pl.col('co2')).height:
        raise ValueError('Missing raw records or cross-country candidate')
    return d.with_columns(pl.col('nm1', 'ad1', 'nm2', 'ad2').fill_null(''))


def neural_partitions(frame, refs):
    'candidate and true-owner fold2 isolation, including independent decoys'
    d = frame.select(*KEYS, 'own', 'fold', 'y', 'co').join(refs.select(
        pl.col('rid').cast(pl.Int64).alias('own'), pl.col('fold').alias('_own_fold')),
        on='own', how='left', maintain_order='left')
    own_key = pl.when(pl.col('own') >= 0).then(pl.col('own').cast(pl.UInt32).cast(pl.UInt64))\
        .otherwise(pl.col('tid').cast(pl.UInt64)+(1 << 32))
    held = (own_key.hash(2903)%10) == 0
    candidate_held = (pl.col('qid').cast(pl.UInt64).hash(2903)%10) == 0
    eligible = (pl.col('fold') == 2) & ((pl.col('own') < 0) | (pl.col('_own_fold') == 2)).fill_null(False)
    masks = d.select((eligible & ~held & ~candidate_held).alias('fit'),
                     (eligible & held).alias('valid'))
    return masks['fit'].to_numpy(), masks['valid'].to_numpy()


def isolate_neural_decoys(cache, full, refs):
    "Remove every decoy target exposed to the new head's tune/audit corpus"
    anchors = refs.select(pl.col('rid').alias('qid'), 'deg', 'fold')
    tune = tuning_anchors(anchors, {'seed': 42, 'fit_fold': 0, 'tune_holdout_buckets': 5})
    evaluation_owners = pl.concat([tune.select('qid'),
        anchors.filter(pl.col('fold') == 1).select('qid')]).unique()
    evaluation_targets = full.select(KEYS).join(evaluation_owners, on='qid').select('tid').unique()
    excluded = cache.filter(pl.col('own') < 0).join(evaluation_targets, on='tid').select('tid').unique()
    safe = cache.join(excluded, on='tid', how='anti')
    return safe, {'excluded_decoy_targets_touching_tune_audit': excluded.height,
        'remaining_cached_decoy_targets': safe.filter(pl.col('own') < 0)['tid'].n_unique(),
        'decoy_isolation': 'full union tune0/audit1 target closure excluded from neural fit and internal validation',
        'limitation': 'No synthetic replacement decoys; zero retained decoys means adaptation has no decoy training coverage.'}


def balanced_sample(frame, cap):
    'labels used only within already isolated supervised training partitions'
    groups = frame.select('co', 'y').unique().sort('co', 'y').rows()
    if not groups:
        return frame
    quota = cap//len(groups)
    parts = [frame.filter((pl.col('co') == co) & (pl.col('y') == y)).with_columns(
        pl.struct(KEYS).hash(6901).alias('_order')).sort('_order', 'tid', 'qid').head(quota).drop('_order')
        for co, y in groups]
    return pl.concat(parts, how='vertical_relaxed')


def complete_owner_sample(frame, cap):
    'Validation retains complete eligible owner/decoy groups under atomic cap'
    d = frame.with_columns(pl.when(pl.col('own') >= 0).then(pl.col('own'))
        .otherwise(-pl.col('tid').cast(pl.Int64)-1).alias('_group'))
    stats = d.group_by('co', '_group').agg(pl.len().alias('_rows'),
        pl.col('y').max().alias('_positive')).with_columns(pl.col('_group').hash(6901).alias('_order'))
    pools = [stats.filter((pl.col('co') == co) & (pl.col('_positive') == y))
        .sort('_order', '_group').to_dicts()
        for co, y in stats.select('co', '_positive').unique().sort('co', '_positive').rows()]
    sel, used, cursor = [], 0, 0
    while any(cursor < len(pool) for pool in pools):
        for pool in pools:
            if cursor < len(pool):
                row = pool[cursor]
                if used+row['_rows'] <= cap:
                    sel.append(row['_group'])
                    used += row['_rows']
        cursor += 1
    return d.filter(pl.col('_group').is_in(sel)).drop('_group')


def prepare(data, hybrid_result, features_train, features_test, friend, graph, output,
            max_targets=120000, max_pairs=800000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, src in (('train', features_train), ('test', features_test)):
        refs, targets = load_refs(data, split), load_targets(data, split)
        cache = read_cache(src, split)
        full = load_predictions(hybrid_result, friend, graph, split)
        frame = attach_probabilities(cache, full)
        chosen, report[split] = choose_targets(frame, targets, max_targets, max_pairs)
        sel = frame.join(chosen, on='tid').sort('tid', 'qid')
        dest = output/split
        (dest/'requests').mkdir(parents=True, exist_ok=True)
        sel.write_parquet(dest/'features.parquet')
        requests = raw_requests(sel.select(KEYS), refs, targets)
        for i, offset in enumerate(range(0, requests.height, 25000)):
            requests.slice(offset, 25000).write_parquet(dest/'requests'/f'part_{i:05d}.parquet')
        if split == 'train':
            neural_cache, isolation = isolate_neural_decoys(cache, full, refs)
            fit, valid = neural_partitions(neural_cache, refs)
            (output/'training').mkdir(exist_ok=True)
            tr = {}
            for name, mask, cap in (('fit', fit, 80000), ('valid', valid, 3000)):
                eligible = neural_cache.filter(pl.Series(mask))
                sampled = balanced_sample(eligible, cap) if name == 'fit' else complete_owner_sample(eligible, cap)
                if not sampled.height or sampled['y'].n_unique() != 2:
                    raise ValueError(f'Neural {name} partition needs both classes')
                raw_requests(sampled.select(*KEYS, 'y'), refs, targets).write_parquet(
                    output/'training'/f'{name}.parquet')
                tr[name] = {'rows': sampled.height, 'targets': sampled['tid'].n_unique(),
                    'decoy_rows': sampled.filter(pl.col('own') < 0).height,
                    'classes': sampled.group_by('co', 'y').len().to_dicts()}
            report['neural_training'] = tr
            report['neural_decoy_isolation'] = isolation
        log(f'Large reranker {split}: {report[split]}')
    report['training_isolation'] = 'fold2 candidate and true owner; true-owner/decoy-target heldout groups; candidate holdout excluded from fit'
    (output/'selection.json').write_text(json.dumps(report, indent=2))
    (output/'_SUCCESS').write_text('complete\n')
    return report
