'one-step set utility actions over a frozen, globally unique winner pool'
import math

import polars as pl

KEYS = ['qid', 'tid']
FEATURE_NAMES = [
    'action_sign', 'action_probability', 'accepted_count', 'accepted_min_p',
    'accepted_mean_p', 'accepted_max_p', 'accepted_sum_p', 'candidate_count',
    'candidate_sum_p', 'candidate_max_p', 'action_minus_accepted_min_p',
    'action_minus_accepted_mean_p', 'action_minus_accepted_max_p',
    'candidate_max_minus_accepted_min_p',
]


def _unique(frame, keys, name):
    if frame.select(keys).null_count().sum_horizontal().sum():
        raise ValueError(f'{name} has null keys')
    if frame.select(keys).unique().height != frame.height:
        raise ValueError(f'{name} contains duplicate {keys}')


def _probabilities(frame, column, name):
    if column not in frame or not frame.schema[column].is_numeric():
        raise ValueError(f'{name} requires numeric {column}')
    if frame.filter(pl.col(column).is_null() | ~pl.col(column).is_finite()
                    | (pl.col(column) < 0) | (pl.col(column) > 1)).height:
        raise ValueError(f'{name} probabilities must be nonnull finite in [0, 1]')


def _labels(frame, column, name):
    if frame.filter(pl.col(column).is_null() | ~pl.col(column).is_in([0, 1])).height:
        raise ValueError(f'{name} labels must be binary and nonnull')


def _f05(n, tp, degree):
    return pl.when(degree == 0).then((n == 0).cast(pl.Float64)).otherwise(
        1.25 * tp / (n + .25 * degree))


def build_actions(winners, base_accepted, refs, *, probability_col='p', label_col='y'):
    'Return ``(actions, feature_names)`` for at most one ADD and REMOVE/owner'
    _unique(winners, ['tid'], 'Winners')
    _unique(base_accepted, ['tid'], 'Baseline')
    _unique(winners, KEYS, 'Winners')
    _unique(base_accepted, KEYS, 'Baseline')
    _unique(refs, ['qid'], 'References')
    _probabilities(winners, probability_col, 'Winners')
    _probabilities(base_accepted, probability_col, 'Baseline')
    labeled = label_col in winners and label_col in base_accepted and 'deg' in refs
    columns = KEYS + [pl.col(probability_col).cast(pl.Float64).alias('p')]
    if labeled:
        _labels(winners, label_col, 'Winners')
        _labels(base_accepted, label_col, 'Baseline')
        columns += [pl.col(label_col).cast(pl.Float64).alias('_action_y')]
        if not refs.schema['deg'].is_numeric() or refs.filter(
                pl.col('deg').is_null() | ~pl.col('deg').is_finite()
                | (pl.col('deg') < 0) | (pl.col('deg') != pl.col('deg').floor())).height:
            raise ValueError('Reference degrees must be nonnegative finite integers')
    base = base_accepted.select(columns).join(refs.select('qid'), on='qid', how='semi')
    cands = winners.select(columns).join(base_accepted.select('tid'), on='tid', how='anti')
    cands = cands.filter(pl.col('p') > 0).join(refs.select('qid'), on='qid', how='semi')
    accepted_stats = base.group_by('qid').agg(
        pl.len().alias('accepted_count'), pl.col('p').min().alias('accepted_min_p'),
        pl.col('p').mean().alias('accepted_mean_p'), pl.col('p').max().alias('accepted_max_p'),
        pl.col('p').sum().alias('accepted_sum_p'),
        *([pl.col('_action_y').sum().alias('_tp')] if labeled else []))
    candidate_stats = cands.group_by('qid').agg(
        pl.len().alias('candidate_count'), pl.col('p').sum().alias('candidate_sum_p'),
        pl.col('p').max().alias('candidate_max_p'))
    stats = refs.select('qid', *(['deg'] if labeled else [])).join(
        accepted_stats, on='qid', how='left').join(candidate_stats, on='qid', how='left')
    numeric_stats = [c for c in stats.columns if c not in ('qid', 'deg')]
    stats = stats.with_columns(pl.col(numeric_stats).fill_null(0).cast(pl.Float64))
    adds = cands.sort(['qid', 'p', 'tid'], descending=[False, True, False]).unique(
        'qid', keep='first', maintain_order=True).with_columns(pl.lit(1, dtype=pl.Int8).alias('operation'))
    removes = base.sort(['qid', 'p', 'tid']).unique('qid', keep='first', maintain_order=True).with_columns(
        pl.lit(-1, dtype=pl.Int8).alias('operation'))
    actions = pl.concat([adds, removes], how='vertical').join(stats, on='qid', how='left')
    actions = actions.with_columns(
        pl.col('operation').cast(pl.Float64).alias('action_sign'),
        pl.col('p').alias('action_probability'),
        (pl.col('p') - pl.col('accepted_min_p')).alias('action_minus_accepted_min_p'),
        (pl.col('p') - pl.col('accepted_mean_p')).alias('action_minus_accepted_mean_p'),
        (pl.col('p') - pl.col('accepted_max_p')).alias('action_minus_accepted_max_p'),
        (pl.col('candidate_max_p') - pl.col('accepted_min_p')).alias('candidate_max_minus_accepted_min_p'))
    if labeled:
        n, tp, deg = pl.col('accepted_count'), pl.col('_tp'), pl.col('deg')
        after_n = n + pl.col('operation')
        after_tp = tp + pl.col('operation') * pl.col('_action_y')
        if stats.filter(pl.col('_tp') > pl.col('deg')).height or actions.filter(after_tp > deg).height:
            raise ValueError('True-positive counts exceed reference degree')
        actions = actions.with_columns((_f05(after_n, after_tp, deg) - _f05(n, tp, deg)).alias('utility_delta'))
        actions = actions.with_columns(pl.col('_action_y').cast(pl.Int8).alias(label_col))
    result_cols = KEYS + ['p', 'operation'] + FEATURE_NAMES + ([label_col, 'utility_delta'] if labeled else [])
    actions = actions.select(result_cols).sort(['qid', 'tid', 'operation'])
    _unique(actions, KEYS + ['operation'], 'Actions')
    return actions, list(FEATURE_NAMES)


def choose_actions(actions, predicted_delta, threshold=0):
    'choose one strictly positive score above threshold per owner, or keep'
    _unique(actions, KEYS + ['operation'], 'Actions')
    if not math.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    if isinstance(predicted_delta, str):
        scored = actions.with_columns(pl.col(predicted_delta).cast(pl.Float64).alias('predicted_delta'))
    else:
        values = pl.Series('predicted_delta', predicted_delta, dtype=pl.Float64)
        if len(values) != actions.height:
            raise ValueError('Predictions must align with actions')
        scored = actions.with_columns(values)
    if scored.filter(pl.col('predicted_delta').is_null() | ~pl.col('predicted_delta').is_finite()).height:
        raise ValueError('Predicted deltas must be nonnull and finite')
    if scored.filter(~pl.col('operation').is_in([-1, 1]) | pl.col('operation').is_null()).height:
        raise ValueError('Action operation must be +1 or -1')
    return scored.filter(pl.col('predicted_delta') > max(0., threshold)).sort(
        ['qid', 'predicted_delta', 'tid', 'operation'], descending=[False, True, False, False]
    ).unique('qid', keep='first', maintain_order=True)


def apply_actions(base_accepted, chosen, *, probability_col='p', label_col='y'):
    'Apply one action/owner without releasing incumbent targets to additions'
    _unique(base_accepted, ['tid'], 'Baseline')
    _probabilities(base_accepted, probability_col, 'Baseline')
    _unique(chosen, ['qid'], 'Chosen actions')
    _unique(chosen, KEYS + ['operation'], 'Chosen actions')
    if chosen.filter(~pl.col('operation').is_in([-1, 1]) | pl.col('operation').is_null()).height:
        raise ValueError('Action operation must be +1 or -1')
    if not chosen.height:
        return base_accepted.clone()
    drops = chosen.filter(pl.col('operation') == -1)
    adds = chosen.filter(pl.col('operation') == 1)
    _unique(adds, ['tid'], 'Additions')
    if drops.join(base_accepted.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Removal does not match an incumbent pair')
    if adds.join(base_accepted.select('tid'), on='tid', how='inner').height:
        raise ValueError('Addition target is globally occupied; reassignment is forbidden')
    if adds.height:
        _probabilities(adds, 'p', 'Additions')
    expressions = []
    for column, dtype in base_accepted.schema.items():
        src = 'p' if column == probability_col else column
        if column == label_col and column not in adds and 'y' in adds:
            src = 'y'
        expressions.append((pl.col(src) if src in adds else pl.lit(None)).cast(dtype).alias(column))
    kept = base_accepted.join(drops.select(KEYS), on=KEYS, how='anti', maintain_order='left')
    res = pl.concat([kept, adds.select(expressions)], how='vertical')
    _unique(res, ['tid'], 'Result')
    return res
