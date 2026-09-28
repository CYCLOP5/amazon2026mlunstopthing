'competitive cavity message passing over an independent target similarity graph'
import time

import numpy as np
import polars as pl

from .directional_swaps import _records

ITERATIONS = 3
DAMPING = .35
STRENGTH = .8
MAX_BLOCK = 16
MAX_DEGREE = 4
MAX_OWNERS = 2
MAX_MESSAGE_ROWS = 40_000_000
MESSAGE_FLOOR = .05
SEED_CONFIDENCE = .8
FEATURES = ['ig_prior_owner', 'ig_posterior_owner', 'ig_posterior_null',
    'ig_target_entropy', 'ig_owner_margin', 'ig_graph_degree_log',
    'ig_neighbor_count_log', 'ig_original_seed_count_log', 'ig_neighbor_mean',
    'ig_neighbor_weight', 'ig_logit_shift', 'ig_iteration_change',
    'ig_cavity_shift_max']


def _log(message):
    print(time.strftime('%H:%M:%S') + ' collective graph: ' + message, flush=True)


def build_graph(frame, targets, s2_count):
    'reciprocal top-four cross-source neighbors from bounded text blocks'
    if not isinstance(s2_count, (int, np.integer)) or s2_count < 0:
        raise ValueError('Graph requires a nonnegative integer source2 boundary')
    ids = frame.select(pl.col('tid').alias('rid')).unique()
    records = _records(targets.join(ids, on='rid', how='semi'), 'tid')
    if records.height != ids.height:
        raise ValueError('Graph candidate targets missing from raw records')
    records = records.with_columns(
        (pl.col('tid') < s2_count).alias('_s2'),
        pl.col('_address').str.extract(r'(\d+)', 1).str.strip_chars_start('0').replace('', '0').alias('_house'))
    edges = []
    block_report = {}
    for key in ('_name', '_address'):
        eligible = records.filter(pl.col(key) != '')
        counts = eligible.group_by('co', key).len()
        eligible = eligible.join(counts.filter(pl.col('len') <= MAX_BLOCK).select('co', key),
                                 on=['co', key], how='semi')
        left = eligible.filter(pl.col('_s2')).select('co', key, pl.col('tid').alias('src'))
        right = eligible.filter(~pl.col('_s2')).select('co', key, pl.col('tid').alias('dst'))
        joined = left.join(right, on=['co', key], how='inner').select('src', 'dst')
        edges.append(joined)
        block_report[key] = {'candidate_edges': joined.height,
                             'oversized_blocks_excluded': counts.filter(pl.col('len') > MAX_BLOCK).height}
    edges = pl.concat(edges).unique()
    details = records.select('tid', '_name', '_address', '_tokens', '_house')
    left_details = details.rename({'tid': 'src'})
    right_details = details.rename({'tid': 'dst', **{c: c+'_r' for c in details.columns if c != 'tid'}})
    scored = []

    for part in edges.iter_slices(500_000):
        part = part.join(left_details, on='src', how='left', validate='m:1').join(
            right_details, on='dst', how='left', validate='m:1')
        part = part.with_columns(
            (pl.col('_tokens').list.set_intersection('_tokens_r').list.len() /
             pl.col('_tokens').list.set_union('_tokens_r').list.len().clip(lower_bound=1)).alias('_jac'),
            ((pl.col('_name') != '') & (pl.col('_name') == pl.col('_name_r'))).alias('_name_eq'),
            ((pl.col('_address') != '') & (pl.col('_address') == pl.col('_address_r'))).alias('_addr_eq'),
            (pl.col('_house').is_not_null() & pl.col('_house_r').is_not_null() &
             (pl.col('_house') != pl.col('_house_r'))).fill_null(False).alias('_house_conflict'))
        part = part.filter(~pl.col('_house_conflict') &
            (pl.col('_name_eq') | (pl.col('_addr_eq') & (pl.col('_jac') >= .4))))
        scored.append(part.select('src', 'dst',
            (.55*pl.col('_jac') + .35*pl.col('_addr_eq').cast(pl.Float64) +
             .1*pl.col('_name_eq').cast(pl.Float64)).cast(pl.Float32).alias('weight')))
    edges = pl.concat(scored) if scored else edges.with_columns(pl.lit(0., pl.Float32).alias('weight'))
    directed = pl.concat([edges, edges.rename({'src': 'dst', 'dst': 'src'}).select(edges.columns)])
    directed = (directed.sort(['src', 'weight', 'dst'], descending=[False, True, False])
                .group_by('src', maintain_order=True).head(MAX_DEGREE))

    reverse = directed.select(pl.col('dst').alias('src'), pl.col('src').alias('dst'))
    directed = directed.join(reverse, on=['src', 'dst'], how='semi').sort('src', 'dst')
    report = {'target_nodes': records.height, 'directed_edges': directed.height,
              'connected_targets': directed['src'].n_unique(), 'blocks': block_report,
              'max_block_size': MAX_BLOCK, 'max_reciprocal_degree': MAX_DEGREE,
              'text_scoring_chunk_edges': 500_000,
              'edge_features': 'Country + legal-form-normalized names, strict addresses, name token Jaccard, house conflict exclusion',
              'graph_uses_labels': False, 'graph_uses_model_scores': False,
              'edge_samples': directed.head(12).to_dicts()}
    return directed, report


def _prior(frame):
    if frame.select('qid', 'tid').unique().height != frame.height:
        raise ValueError('Graph requires unique candidate pairs')
    raw = frame['baseline_p'].to_numpy()
    if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError('Graph prior scores must be finite in [0,1]')
    p = np.clip(raw.astype(np.float64), 1e-5, 1.-1e-5)
    rows = frame.select('qid', 'tid').with_columns(pl.Series('_odds', p/(1-p)))
    return rows.with_columns((1.+pl.col('_odds').sum().over('tid')).alias('_denominator')).with_columns(
        (pl.col('_odds')/pl.col('_denominator')).alias('_prior'))


def _incoming(messages):
    return (messages.with_columns(
        ((pl.col('message').clip(1e-5, 1-1e-5) /
          (1-pl.col('message').clip(1e-5, 1-1e-5))).log().clip(-3., 3.)*pl.col('weight')).alias('_signal'))
        .group_by('qid', 'dst').agg(pl.col('_signal').sum().alias('_sum'),
            pl.col('weight').sum().alias('_weight'), pl.len().alias('_count'),
            pl.col('_seed').sum().alias('_seeds'),
            (pl.col('weight')*pl.col('message')).sum().alias('_prob_sum'))
        .rename({'dst': 'tid'}))


def _shift(signal, weight, count, seeds):
    mean = signal / weight.clip(lower_bound=1e-9)

    return pl.when(count >= 2).then(
        pl.when((mean > 0) & (seeds < 2)).then(0.).otherwise(mean.clip(-2., 2.)*STRENGTH)
        ).otherwise(0.)


def _node_output(prior, incoming, previous=None):
    rows = prior.join(incoming, on=['qid', 'tid'], how='left', maintain_order='left').with_columns(
        pl.col('_sum', '_weight', '_count', '_seeds', '_prob_sum').fill_null(0))
    rows = rows.with_columns(_shift(pl.col('_sum'), pl.col('_weight'), pl.col('_count'),
                                    pl.col('_seeds')).alias('_shift'))
    rows = rows.with_columns((pl.col('_odds')*(DAMPING*pl.col('_shift')).exp()).alias('_updated_odds'))
    rows = rows.with_columns((1+pl.col('_updated_odds').sum().over('tid')).alias('_updated_denom'))
    rows = rows.with_columns((pl.col('_updated_odds')/pl.col('_updated_denom')).alias('_posterior'),
                              (1/pl.col('_updated_denom')).alias('_null'))
    if previous is not None:
        rows = rows.join(previous.select('qid', 'tid', pl.col('_posterior').alias('_previous')),
                         on=['qid', 'tid'], how='left', maintain_order='left')
    else:
        rows = rows.with_columns(pl.col('_prior').alias('_previous'))
    return rows


def propagate(frame, edges):
    'Three non-backtracking rounds, keeping null/competition mass explicit'
    prior = _prior(frame)
    allowed = (prior.filter(pl.col('_odds') >= MESSAGE_FLOOR/(1-MESSAGE_FLOOR))
               .sort(['tid', '_prior', 'qid'], descending=[False, True, False])
               .group_by('tid', maintain_order=True).head(MAX_OWNERS))
    projected = edges.group_by('src').len().join(allowed.group_by('tid').len().rename(
        {'tid': 'src', 'len': '_owners'}), on='src', how='inner')
    projected_count = int(projected.select((pl.col('len').cast(pl.UInt64)*pl.col('_owners')).sum()).item() or 0)
    owners_used = MAX_OWNERS
    pruned_sending_targets = 0
    if projected_count > MAX_MESSAGE_ROWS:


        owners_used = 1
        allowed = allowed.group_by('tid', maintain_order=True).head(1)
        per_source = (edges.group_by('src').len().join(allowed.select(pl.col('tid').alias('src')),
            on='src', how='inner').with_columns(pl.col('src').hash(seed=27943).alias('_budget_hash'))
            .sort('_budget_hash', 'src').with_columns(pl.col('len').cast(pl.UInt64).cum_sum().alias('_cumulative')))
        keep_sources = per_source.filter(pl.col('_cumulative') <= MAX_MESSAGE_ROWS).select(pl.col('src').alias('tid'))
        pruned_sending_targets = per_source.height-keep_sources.height
        allowed = allowed.join(keep_sources, on='tid', how='semi', maintain_order='left')
    allowed = allowed.with_columns((pl.col('_denominator') - pl.col('_odds').sum().over('tid')).alias('_other_odds'))
    messages = (edges.join(allowed.rename({'tid': 'src'}), on='src', how='inner')
                .with_columns(pl.col('_prior').alias('message'),
                              (pl.col('_prior') >= SEED_CONFIDENCE).cast(pl.UInt32).alias('_seed'))
                .select('src', 'dst', 'qid', 'weight', '_odds', '_other_odds', 'message', '_seed'))
    message_count = messages.height
    if message_count > MAX_MESSAGE_ROWS:
        raise AssertionError('Directed graph message budget exceeded')
    prev = None
    max_cavity = None
    rounds = []
    for iteration in range(ITERATIONS):
        _log(f'round {iteration+1}/{ITERATIONS}: {message_count:,} directed owner messages')
        incoming = _incoming(messages)
        reverse = messages.select(pl.col('dst').alias('src'), pl.col('src').alias('dst'), 'qid',
            ((pl.col('message').clip(1e-5, 1-1e-5)/(1-pl.col('message').clip(1e-5, 1-1e-5))).log()
             .clip(-3., 3.)*pl.col('weight')).alias('_reverse_signal'),
            pl.col('weight').alias('_reverse_weight'), pl.col('_seed').alias('_reverse_seed'),
            pl.lit(1, pl.UInt32).alias('_reverse_count'))
        cavity = (messages.join(incoming.rename({'tid': 'src'}), on=['qid', 'src'], how='left')
                  .join(reverse, on=['src', 'dst', 'qid'], how='left')
                  .with_columns(pl.col('_sum', '_weight', '_count', '_seeds', '_reverse_signal',
                     '_reverse_weight', '_reverse_seed', '_reverse_count').fill_null(0)))
        cavity = cavity.with_columns(_shift(
            pl.col('_sum')-pl.col('_reverse_signal'), pl.col('_weight')-pl.col('_reverse_weight'),
            pl.col('_count').cast(pl.Int64)-pl.col('_reverse_count').cast(pl.Int64),
            pl.col('_seeds').cast(pl.Int64)-pl.col('_reverse_seed').cast(pl.Int64)).alias('_cavity_shift'))
        cavity = cavity.with_columns((pl.col('_odds')*pl.col('_cavity_shift').exp()).alias('_cavity_odds'))
        cavity = cavity.with_columns((pl.col('_other_odds') + pl.col('_cavity_odds').sum().over('src', 'dst')).alias('_cavity_denom'))
        cavity = cavity.with_columns(((1-DAMPING)*pl.col('message') +
            DAMPING*pl.col('_cavity_odds')/pl.col('_cavity_denom')).alias('message'))
        max_cavity = cavity.group_by('qid', 'src').agg(pl.col('_cavity_shift').abs().max().alias('_cavity_max')).rename({'src': 'tid'})
        messages = cavity.select(messages.columns)
        current = _node_output(prior, _incoming(messages), prev)
        rounds.append({'iteration': iteration+1,
            'mean_absolute_owner_probability_change': float((current['_posterior']-current['_previous']).abs().mean() or 0.),
            'positively_reinforced_pairs': current.filter(pl.col('_shift') > 0).height})
        prev = current.select('qid', 'tid', '_posterior')
    del prior, allowed, projected, messages, prev, cavity, reverse, incoming
    if current.height:
        totals = current.group_by('tid').agg(pl.col('_posterior').sum(), pl.col('_null').first())
        normalization_error = float((totals['_posterior']+totals['_null']-1).abs().max())
        del totals
    else:
        normalization_error = 0.
    if normalization_error > 1e-6:
        raise ValueError('Graph ownership/null probabilities do not normalize')
    degree = edges.group_by('src').len().rename({'src': 'tid', 'len': '_degree'})
    current = current.join(degree, on='tid', how='left', maintain_order='left').join(
        max_cavity, on=['qid', 'tid'], how='left', maintain_order='left')
    current = current.with_columns(
        (-(pl.col('_posterior')*pl.col('_posterior').clip(lower_bound=1e-12).log()).sum().over('tid') -
         pl.col('_null')*pl.col('_null').clip(lower_bound=1e-12).log()).alias('_entropy'),
        pl.col('_posterior').max().over('tid').alias('_top'),
        pl.col('_posterior').rank(method='ordinal', descending=True).over('tid').alias('_rank'))
    current = current.with_columns(pl.when(pl.col('_rank') == 2).then(pl.col('_posterior')).otherwise(0.)
                                  .max().over('tid').alias('_second'))
    extra = current.select('qid', 'tid',
        pl.col('_prior').alias('ig_prior_owner'), pl.col('_posterior').alias('ig_posterior_owner'),
        pl.col('_null').alias('ig_posterior_null'), pl.col('_entropy').alias('ig_target_entropy'),
        (pl.col('_posterior')-pl.when(pl.col('_rank') == 1).then(pl.col('_second')).otherwise(pl.col('_top'))).alias('ig_owner_margin'),
        pl.col('_degree').fill_null(0).log1p().alias('ig_graph_degree_log'),
        pl.col('_count').log1p().alias('ig_neighbor_count_log'),
        pl.col('_seeds').log1p().alias('ig_original_seed_count_log'),
        (pl.col('_prob_sum')/pl.col('_weight').clip(lower_bound=1e-9)).alias('ig_neighbor_mean'),
        pl.col('_weight').alias('ig_neighbor_weight'), pl.col('_shift').alias('ig_logit_shift'),
        (pl.col('_posterior')-pl.col('_previous')).alias('ig_iteration_change'),
        pl.col('_cavity_max').fill_null(0.).alias('ig_cavity_shift_max'))
    extra = extra.with_columns(pl.col(FEATURES).cast(pl.Float32))
    for part in extra.select(FEATURES).iter_slices(500_000):
        if not np.isfinite(part.to_numpy()).all():
            raise ValueError('Nonfinite graph representation')
    report = {'iterations': ITERATIONS, 'damping': DAMPING, 'shift_strength': STRENGTH,
              'directed_owner_messages': message_count, 'max_sending_owners_per_target': owners_used,
              'projected_two_owner_messages': projected_count, 'message_row_budget': MAX_MESSAGE_ROWS,
              'budget_pruned_sending_targets': pruned_sending_targets,
              'score_floor_for_sending': MESSAGE_FLOOR, 'original_seed_confidence': SEED_CONFIDENCE,
              'null_normalization_max_error': normalization_error, 'rounds': rounds,
              'reinforcement': 'Positive shifts require at least two distinct ORIGINAL confident neighbor targets; cavities remove the exact reverse-edge message.',
              'abstention': 'Unit null odds in complete owner competition; unsent owner odds retained in cavity denominator.'}
    return extra, report


def transform(frame, refs, targets, baseline=None, s2_count=None):
    'build a new label-free representation over the complete candidate graph'
    if set(FEATURES).intersection(frame.columns):
        raise ValueError('Graph innovation feature columns already exist')
    _log(f'building independent cross-source graph for {frame.height:,} candidate pairs')
    edges, graph_report = build_graph(frame, targets, s2_count)
    extra, message_report = propagate(frame, edges)
    augmented = frame.join(extra, on=['qid', 'tid'], how='left', validate='1:1', maintain_order='left')
    report = {'method': 'competitive_cavity_graph', 'graph': graph_report,
              'message_passing': message_report, 'features': list(FEATURES),
              'baseline_accepted_table_used': False, 'truth_or_degree_features_used': False,
              'initial_score': 'Existing baseline_p prior, with complete target competition and explicit null',
              'learning': 'Graph outputs are representation features for a newly fitted full pair scorer; no action-only correction policy.'}
    return augmented, list(FEATURES), report
