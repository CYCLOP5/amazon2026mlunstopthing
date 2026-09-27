'label-free triadic evidence against a frozen-score alternative owner'
import polars as pl

from .directional_swaps import _records


FEATURES = ['ct_alternative_present', 'ct_single_candidate', 'ct_score_margin',
            'ct_name_q_distinct', 'ct_name_b_distinct',
            'ct_name_q_support', 'ct_name_b_support',
            'ct_name_q_coverage', 'ct_name_b_coverage', 'ct_name_coverage_margin',
            'ct_name_shared_support', 'ct_name_only_shared',
            'ct_address_q_distinct', 'ct_address_b_distinct',
            'ct_address_q_support', 'ct_address_b_support',
            'ct_address_q_conflict', 'ct_address_b_conflict',
            'ct_numeric_q_distinct', 'ct_numeric_b_distinct',
            'ct_numeric_q_support', 'ct_numeric_b_support',
            'ct_numeric_q_conflict', 'ct_numeric_b_conflict']


def _tokens(records, key, prefix):
    d = _records(records, key).with_columns(
        pl.col('_address').str.split(' ').list.eval(
            pl.element().filter(pl.element() != '')).list.unique().alias('_address_tokens'))


    d = d.with_columns(pl.col('_tokens').list.set_union('_address_tokens').list.eval(
        pl.element().filter(pl.element().str.contains(r'\p{N}'))).alias('_numeric_tokens'))
    return d.select(key, 'co', '_tokens', '_address_tokens', '_numeric_tokens').rename({
        'co': prefix + 'co', '_tokens': prefix + 'name',
        '_address_tokens': prefix + 'address', '_numeric_tokens': prefix + 'numeric'})


def build(frame, refs, targets, chunk_size=500_000):
    'return original rows plus 24 float64 features, names, and diagnostics'
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if not {'qid', 'tid', 'newest'}.issubset(frame.columns):
        raise ValueError('Candidate frame requires qid, tid and newest')
    if frame.select(pl.col('qid', 'tid').is_null().any()).row(0) != (False, False):
        raise ValueError('Missing candidate IDs')
    if frame.select('qid', 'tid').n_unique() != frame.height:
        raise ValueError('Duplicate candidate pairs')
    if not frame.schema['newest'].is_numeric() or frame['newest'].null_count():
        raise ValueError('newest must contain finite numeric scores')
    if not frame['newest'].cast(pl.Float64).is_finite().all():
        raise ValueError('newest must contain finite numeric scores')
    if set(FEATURES).intersection(frame.columns):
        raise ValueError('Contrastive feature columns already exist')
    for records in (refs, targets):
        if not {'rid', 'co', 'nm', 'ad'}.issubset(records.columns):
            raise ValueError('Record tables require rid, co, nm and ad')
        if records['rid'].null_count() or records['co'].null_count():
            raise ValueError('Missing record IDs or country')
    r, t = _tokens(refs, 'qid', '_q_'), _tokens(targets, 'tid', '_t_')


    ranked = frame.select('tid', 'qid', 'newest').group_by('tid').agg(
        pl.col('qid').sort_by(['newest', 'qid'], descending=[True, False]).head(2).alias('_owners'),
        pl.col('newest').cast(pl.Float64).sort_by(
            ['newest', 'qid'], descending=[True, False]).head(2).alias('_scores'))
    ranked = ranked.select('tid', pl.col('_owners').list.get(0).alias('_best'),
        pl.col('_owners').list.get(1, null_on_oob=True).alias('_second'),
        pl.col('_scores').list.get(0).alias('_best_score'),
        pl.col('_scores').list.get(1, null_on_oob=True).alias('_second_score'))
    b = r.rename({'qid': '_bid', **{c: c.replace('_q_', '_b_') for c in r.columns if c != 'qid'}})
    parts, alternative_rows, shared_only_rows = [], 0, 0
    for chunk in frame.select('qid', 'tid', 'newest', *(['co'] if 'co' in frame.columns else [])).iter_slices(chunk_size):
        d = chunk.join(ranked, on='tid', how='left', validate='m:1', maintain_order='left')
        is_best = pl.col('qid') == pl.col('_best')
        d = d.with_columns(
            pl.when(is_best).then('_second').otherwise('_best').alias('_bid'),
            pl.when(is_best).then('_second_score').otherwise('_best_score').alias('_bscore'))
        d = d.join(r, on='qid', how='left', validate='m:1', maintain_order='left').join(
            t, on='tid', how='left', validate='m:1', maintain_order='left').join(
            b, on='_bid', how='left', validate='m:1', maintain_order='left')
        if d['_q_co'].null_count() or d['_t_co'].null_count():
            raise ValueError('Candidate IDs missing from full record tables')
        if d.filter(pl.col('_q_co') != pl.col('_t_co')).height:
            raise ValueError('Cross-country candidate pair')
        if 'co' in chunk.columns and (d['co'].null_count() or d.filter(pl.col('co') != pl.col('_q_co')).height):
            raise ValueError('Candidate country mismatch')
        present = pl.col('_bid').is_not_null()
        d = d.with_columns(present.alias('ct_alternative_present'),
            (~present).alias('ct_single_candidate'),
            (pl.col('newest').cast(pl.Float64) - pl.col('_bscore')).fill_null(0.).alias('ct_score_margin'))
        alternative_rows += int(d['ct_alternative_present'].sum())
        d = d.with_columns([pl.col('_b_' + domain).fill_null(pl.lit([], pl.List(pl.String)))
                           for domain in ('name', 'address', 'numeric')])
        for domain in ('name', 'address', 'numeric'):
            d = d.with_columns(
                pl.col('_q_' + domain).list.set_difference('_b_' + domain).alias('_qd'),
                pl.col('_b_' + domain).list.set_difference('_q_' + domain).alias('_bd'))
            d = d.with_columns(
                pl.col('_qd').list.len().alias(f'ct_{domain}_q_distinct'),
                pl.col('_bd').list.len().alias(f'ct_{domain}_b_distinct'),
                pl.col('_qd').list.set_intersection('_t_' + domain).list.len().alias(f'ct_{domain}_q_support'),
                pl.col('_bd').list.set_intersection('_t_' + domain).list.len().alias(f'ct_{domain}_b_support'))
            if domain == 'name':
                d = d.with_columns(*[(pl.col(f'ct_name_{side}_support') /
                    pl.col(f'ct_name_{side}_distinct').clip(lower_bound=1)).alias(f'ct_name_{side}_coverage')
                    for side in ('q', 'b')],
                    pl.col('_q_name').list.set_intersection('_b_name').list.set_intersection(
                        '_t_name').list.len().alias('ct_name_shared_support'))
                d = d.with_columns((pl.col('ct_name_q_coverage') - pl.col('ct_name_b_coverage')).alias(
                    'ct_name_coverage_margin'),
                    ((pl.col('ct_name_shared_support') > 0) & (pl.col('ct_name_q_support') == 0) &
                     (pl.col('ct_name_b_support') == 0)).alias('ct_name_only_shared'))
            else:
                d = d.with_columns(*[((pl.col(f'ct_{domain}_{side}_distinct') > 0) &
                    (pl.col(f'ct_{domain}_{side}_support') == 0) &
                    (pl.col(f'ct_{domain}_{other}_support') > 0)).alias(f'ct_{domain}_{side}_conflict')
                    for side, other in [('q', 'b'), ('b', 'q')]])
        triadic = FEATURES[3:]
        d = d.with_columns([pl.when(present).then(pl.col(c)).otherwise(0).cast(pl.Float64).alias(c)
                            for c in triadic])
        shared_only_rows += int(d['ct_name_only_shared'].sum())
        parts.append(d.select(FEATURES).with_columns(pl.all().cast(pl.Float64)))
    added = pl.concat(parts) if parts else pl.DataFrame(schema={c: pl.Float64 for c in FEATURES})
    diagnostics = {'candidate_pairs': frame.height, 'targets': ranked.height,
                   'alternative_pairs': alternative_rows, 'single_candidate_pairs': frame.height - alternative_rows,
                   'shared_only_name_pairs': shared_only_rows,
                   'selection': 'Complete candidate pool; newest descending, qid ascending; exclude current qid.',
                   'interpretation': 'Triadic evidence only; no claim of improved matching accuracy.'}
    return frame.hstack(added), list(FEATURES), diagnostics
