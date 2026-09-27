'country-neutral disagreement and ownership features on complete candidate pools'
import gc

import numpy as np
import polars as pl

SCORES = ['newest', 'gate', 'neural', 'hybrid', 'graph', 'friend']
MEMBERS = [f'np_m{i}' for i in range(15)]
LEGAL = r'\b(?:incorporated|inc|corporation|corp|limited|ltd|llc|llp|pvt|plc|sarl|sas|sasu|eurl|sa|sci|snc|gmbh)\b'


def normalized(column):
    return (pl.col(column).fill_null('').str.to_lowercase().str.normalize('NFKD')
            .str.replace_all(r'\p{M}', '').str.replace_all(r'[^\p{L}\p{N}]+', ' ')
            .str.replace_all(r'\s+', ' ').str.strip_chars())


def _records(frame, prefix, idname):
    d = frame.select(pl.col('rid').alias(idname), 'co', 'nm', 'ad').with_columns(
        normalized('nm').alias('_n'), normalized('ad').alias('_a'))
    d = d.with_columns(pl.col('_n').str.replace_all(LEGAL, ' legal ').str.replace_all(r'\s+', ' ').str.strip_chars().alias('_marked'))
    d = d.with_columns(
        pl.col('_marked').str.replace_all(r'\blegal\b', '').str.replace_all(r'\s+', ' ').str.strip_chars().alias('_core'),
        pl.col('_marked').str.contains(r'\blegal\b').alias('_legal'),
        pl.col('_marked').str.extract(r'^(.*?)\blegal\b', 1).fill_null('').alias('_before'),
        pl.col('_marked').str.extract(r'\blegal\b(.*)$', 1).fill_null('').str.replace_all(r'\blegal\b', '').alias('_after'),
        pl.col('_a').str.extract(r'(\d+)', 1).fill_null('').str.strip_chars_start('0').alias('_house'),
        pl.col('_a').str.replace_all(r'\b(?:ave|av)\b', 'avenue').str.replace_all(r'\b(?:bd|boul)\b', 'boulevard')
            .str.replace_all(r'\brd\b', 'road').str.replace_all(r'\bst\b', 'street').str.replace_all(r'\bste\b', 'sainte')
            .str.replace_all(r'\d+', '').str.replace_all(r'\s+', ' ').str.strip_chars().alias('_street'))
    for col in ('_core', '_before', '_after', '_street'):
        d = d.with_columns(pl.col(col).str.split(' ').list.eval(pl.element().filter(pl.element() != '')).alias(col+'_tokens'))
    counts = d.group_by('co', '_core').len().rename({'len': '_duplicates'})
    d = d.join(counts, on=['co', '_core'], how='left', validate='m:1').with_columns(
        pl.when(pl.col('_core') == '').then(0).otherwise(pl.col('_duplicates')).alias('_duplicates'))
    keep = ['_n', '_a', '_core', '_legal', '_house', '_duplicates', '_core_tokens', '_before_tokens', '_after_tokens', '_street_tokens']
    return d.select(idname, *[pl.col(c).alias(prefix+c) for c in keep])


def _lexical(pairs, refs, targets):
    'polars vector operations; no python loop per candidate pair'
    d = pairs.select('qid', 'tid').join(refs, on='qid', how='left', validate='m:1', maintain_order='left').join(targets, on='tid', how='left', validate='m:1', maintain_order='left')
    if d['r_n'].null_count() or d['t_n'].null_count():
        raise ValueError('Candidate IDs missing from full record tables')
    d = d.with_columns(
        pl.col('r_core_tokens').list.set_difference('t_core_tokens').alias('_missing'),
        pl.col('t_core_tokens').list.set_difference('r_core_tokens').alias('_extra'),
        pl.col('r_core_tokens').list.set_intersection('t_core_tokens').list.len().alias('_shared'),
        pl.col('r_street_tokens').list.set_intersection('t_street_tokens').list.len().alias('_addr_shared'))
    rt, tt = pl.col('r_core_tokens').list.len(), pl.col('t_core_tokens').list.len()
    ra, ta = pl.col('r_street_tokens').list.len(), pl.col('t_street_tokens').list.len()
    hn1 = pl.col('r_house').cast(pl.Float64, strict=False)
    hn2 = pl.col('t_house').cast(pl.Float64, strict=False)
    expressions = {
        'name_exact': pl.col('r_n') == pl.col('t_n'),
        'name_core_exact': (pl.col('r_core') != '') & (pl.col('r_core') == pl.col('t_core')),
        'name_ref_empty': rt == 0, 'name_target_empty': tt == 0,
        'name_jaccard': pl.col('_shared') / (rt+tt-pl.col('_shared')).clip(1, None),
        'name_ref_coverage': pl.col('_shared') / rt.clip(1, None),
        'name_target_coverage': pl.col('_shared') / tt.clip(1, None),
        'name_missing': pl.col('_missing').list.len(), 'name_extra': pl.col('_extra').list.len(),
        'name_single_substitution': (pl.col('_missing').list.len() == 1) & (pl.col('_extra').list.len() == 1),
        'extra_before_legal': pl.col('_extra').list.set_intersection('t_before_tokens').list.len(),
        'extra_after_legal': pl.col('_extra').list.set_intersection('t_after_tokens').list.len(),
        'extra_at_end': pl.col('_extra').list.contains(pl.col('t_core_tokens').list.last()),
        'ref_has_legal': pl.col('r_legal'), 'target_has_legal': pl.col('t_legal'),
        'ref_name_twins_log': pl.col('r_duplicates').log1p(),
        'target_name_twins_log': pl.col('t_duplicates').log1p(),
        'address_exact': (pl.col('r_a') != '') & (pl.col('r_a') == pl.col('t_a')),
        'ref_address_empty': pl.col('r_a') == '', 'target_address_empty': pl.col('t_a') == '',
        'street_jaccard': pl.col('_addr_shared') / (ra+ta-pl.col('_addr_shared')).clip(1, None),
        'street_ref_coverage': pl.col('_addr_shared') / ra.clip(1, None),
        'street_target_coverage': pl.col('_addr_shared') / ta.clip(1, None),
        'house_equal': (pl.col('r_house') != '') & (pl.col('r_house') == pl.col('t_house')),
        'house_conflict': (pl.col('r_house') != '') & (pl.col('t_house') != '') & (pl.col('r_house') != pl.col('t_house')),
        'house_gap_log': (hn1-hn2).abs().clip(0, 1e9).log1p().fill_null(-1),
    }
    return d.select(*[x.fill_null(False if k == 'extra_at_end' else 0).cast(pl.Float32).alias(k) for k, x in expressions.items()])


def prepare_features(frame, refs, targets, chunk_size=500_000):
    'return (numeric-feature frame with original metadata, ordered feature names)'
    missing = set(SCORES)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing probability columns: {sorted(missing)}')
    if frame.select('qid', 'tid').n_unique() != frame.height:
        raise ValueError('Duplicate candidate pairs')
    if not frame.height:
        raise ValueError('Empty feature pool')
    members = [c for c in MEMBERS if c in frame]
    names, expressions = [], []
    for c in SCORES+members:
        if frame.filter(pl.col(c).is_not_null() & (~pl.col(c).is_finite() | ~pl.col(c).is_between(0, 1))).height:
            raise ValueError(f'Invalid probabilities: {c}')
        p = pl.col(c).fill_null(1e-4).clip(1e-6, 1-1e-6)
        for name, expr in [(c+'_present', pl.col(c).is_not_null()), (c+'_logit', (p/(1-p)).log())]:
            expressions.append(expr.cast(pl.Float32).alias(name)); names.append(name)
    d = frame.with_columns(expressions)
    extras = {
        'model_spread': pl.max_horizontal([pl.col(c) for c in SCORES])-pl.min_horizontal([pl.col(c) for c in SCORES]),
        'old_models_mean': pl.mean_horizontal('hybrid', 'graph', 'friend'),
        'new_old_gap': pl.col('newest')-pl.mean_horizontal('hybrid', 'graph', 'friend'),
        'gate_neural_gap': pl.col('gate')-pl.col('neural'),
        'models_above_90': pl.sum_horizontal([(pl.col(c) >= .9).fill_null(False).cast(pl.Float32) for c in SCORES]),
        'member_spread': (pl.max_horizontal(members)-pl.min_horizontal(members)) if members else pl.lit(0.),
    }
    d = d.with_columns(*[expr.fill_null(0).cast(pl.Float32).alias(name) for name, expr in extras.items()])
    names.extend(extras)


    aggregates = [pl.len().cast(pl.Float32).log1p().alias('competitor_count_log')]
    for c in ('newest', 'gate', 'hybrid', 'graph'):
        aggregates.extend([pl.col(c).fill_null(0).max().alias(c+'_target_max'),
                           pl.when(pl.len()>1).then(pl.col(c).fill_null(0).top_k(2).min()).otherwise(0.).alias(c+'_target_second')])
    agg = d.group_by('tid').agg(aggregates)
    d = d.join(agg, on='tid', how='left', validate='m:1')
    comp_names = ['competitor_count_log']
    for c in ('newest', 'gate', 'hybrid', 'graph'):
        d = d.with_columns((pl.col(c).fill_null(0)-pl.col(c+'_target_max')).cast(pl.Float32).alias(c+'_behind_best'),
                           (pl.col(c+'_target_max')-pl.col(c+'_target_second')).cast(pl.Float32).alias(c+'_owner_margin'))
        comp_names.extend([c+'_target_max', c+'_target_second', c+'_behind_best', c+'_owner_margin'])
    names.extend(comp_names)
    rr, tt = _records(refs, 'r', 'qid'), _records(targets, 't', 'tid')
    chunks = []
    for start in range(0, d.height, chunk_size):
        chunk = _lexical(d.slice(start, chunk_size), rr, tt)
        chunks.append(chunk)
        print(f'  lexical features {min(start+chunk_size,d.height):,}/{d.height:,}', flush=True)
    lexical = pl.concat(chunks)
    names.extend(lexical.columns)
    d = pl.concat([d, lexical], how='horizontal')
    del lexical, chunks, rr, tt, agg
    gc.collect()
    d = d.with_columns(pl.col(names).cast(pl.Float32))
    if d.select(pl.any_horizontal([~pl.col(c).is_finite() for c in names]).any()).item():
        raise ValueError('Nonfinite model feature')
    return d, names
