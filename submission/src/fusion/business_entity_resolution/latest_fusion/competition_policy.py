'fixed ownership evidence families with separate calibration cutoffs'
import numpy as np
import polars as pl

from latest_fusion.features import LEGAL, normalized, _records
from latest_fusion.rescue_policy import select_additions, promotion_additions

FAMILIES = ('addressed', 'unique_name')


def _signature():
    return (normalized('nm').str.replace_all(LEGAL, ' ').str.split(' ')
            .list.eval(pl.element().filter(pl.element() != '')).list.unique()
            .list.sort().list.join(' '))


def evidence_actions(actions, pool, refs, targets, min_head_margin=.02):
    'gate already resolved global winners; never prune competing owners first'
    if actions['tid'].n_unique() != actions.height:
        raise ValueError('Actions must have globally unique target owners')
    if not 0 <= min_head_margin < 1:
        raise ValueError('Invalid minimum head ownership margin')
    if 'min_head_margin' not in actions:
        raise ValueError('Missing minimum head ownership margin')
    if refs['rid'].n_unique() != refs.height or targets['rid'].n_unique() != targets.height:
        raise ValueError('Duplicate raw record IDs')
    evidence = ['rescue_rescue_shared_idf_sum_margin', 'rescue_member_vote_fraction',
                'rescue_family_vote_fraction']
    d = actions.join(pool.select('qid', 'tid', *[c for c in evidence if c not in actions]),
                     on=['qid', 'tid'], how='left', validate='1:1', maintain_order='left')
    rr = refs.select(pl.col('rid').alias('qid'), pl.col('co').alias('_ref_co'),
                     _signature().alias('_ref_signature'))
    counts = rr.group_by('_ref_co', '_ref_signature').len().rename({'len': '_signature_count'})
    rr = rr.join(counts, on=['_ref_co', '_ref_signature'], how='left', validate='m:1')
    tt = targets.select(pl.col('rid').alias('tid'), pl.col('co').alias('_target_co'),
                        _signature().alias('_target_signature'))
    d = d.join(rr, on='qid', how='left', validate='m:1', maintain_order='left').join(
        tt, on='tid', how='left', validate='m:1', maintain_order='left')
    if d.filter(pl.col('co').is_null() | pl.col('_ref_co').is_null() | pl.col('_target_co').is_null() |
                (pl.col('co') != pl.col('_ref_co')) | (pl.col('co') != pl.col('_target_co'))).height:
        raise ValueError('Action country mismatch or missing raw record')
    d = d.join(_records(refs, '_r', 'qid'), on='qid', how='left', validate='m:1', maintain_order='left').join(
        _records(targets, '_t', 'tid'), on='tid', how='left', validate='m:1', maintain_order='left')
    ns = pl.col('_r_core_tokens').list.set_intersection('_t_core_tokens').list.len()
    nu = pl.col('_r_core_tokens').list.set_union('_t_core_tokens').list.len()
    ss = pl.col('_r_street_tokens').list.set_intersection('_t_street_tokens').list.len()
    su = pl.col('_r_street_tokens').list.set_union('_t_street_tokens').list.len()

    margin = pl.col('min_head_margin').is_finite() & (pl.col('min_head_margin') > min_head_margin)
    addressed = ((pl.col('_r_a') != '') & (pl.col('_t_a') != '') &
                 (pl.col('_r_house') != '') & (pl.col('_r_house') == pl.col('_t_house')) &
                 (ss / su.clip(1, None) >= .65) & (ns / nu.clip(1, None) >= .5))
    unique_name = ((pl.col('_t_a') == '') & (pl.col('_ref_signature') != '') &
                   (pl.col('_ref_signature') == pl.col('_target_signature')) &
                   (pl.col('_signature_count') == 1) &
                   (pl.col('rescue_rescue_shared_idf_sum_margin') > 0) &
                   (pl.col('rescue_member_vote_fraction') >= .8) &
                   (pl.col('rescue_family_vote_fraction') >= .66))
    d = d.with_columns(pl.when(margin & addressed).then(pl.lit('addressed'))
                       .when(margin & unique_name).then(pl.lit('unique_name'))
                       .otherwise(pl.lit(None, dtype=pl.String)).alias('evidence_family'))
    return d.filter(pl.col('evidence_family').is_not_null()).select(*actions.columns, 'evidence_family')


def apply_family_policy(actions, policy):
    'apply a frozen policy identically at search, check, and inference'
    if actions['tid'].n_unique() != actions.height:
        raise ValueError('Actions must have globally unique target owners')
    if not policy.get('enabled', False):
        return actions.head(0)
    keep = pl.lit(False)
    for family in FAMILIES:
        cfg = policy['families'][family]
        if cfg['enabled']:
            keep = keep | ((pl.col('evidence_family') == family) &
                           (pl.col('score') >= max(cfg['threshold'], cfg['calibration_floor'])))
    return actions.filter(keep)


def lock_family_policy(actions, baseline, calibration, search, min_actions=5):
    'use only calibration and search labels; return one locked additive policy'
    if min_actions < 1:
        raise ValueError('Minimum family action count must be positive')
    if calibration.select('qid').join(search.select('qid'), on='qid', how='inner').height:
        raise ValueError('Calibration and search owners overlap')
    if actions['tid'].n_unique() != actions.height:
        raise ValueError('Actions must have globally unique target owners')
    if set(actions['evidence_family'].unique().to_list()) - set(FAMILIES):
        raise ValueError('Unknown ownership evidence family')
    policy = {'version': 1, 'enabled': True, 'families': {}}
    report = {'families': {}}
    for family in FAMILIES:
        family_actions = actions.filter(pl.col('evidence_family') == family).join(
            baseline.select('tid'), on='tid', how='anti')
        cal = family_actions.join(calibration.select('qid'), on='qid', how='inner')
        false = cal.filter(pl.col('y') != 1)
        dtype = np.float32 if actions['score'].dtype == pl.Float32 else np.float64
        floor = float(np.nextafter(dtype(false['score'].max()), dtype(np.inf))) if false.height else 0.
        eligible = family_actions.filter(pl.col('score') >= floor)
        selected_policy, diagnostics = select_additions(eligible, baseline, search, min_actions)
        selected_policy['calibration_floor'] = floor
        selected_cal = cal.filter(pl.col('score') >= max(floor, selected_policy['threshold'])) if selected_policy['enabled'] else cal.head(0)


        selected_policy['enabled'] = (selected_policy['enabled'] and
                                      selected_cal.height >= min_actions and
                                      int((selected_cal['y'] != 1).sum()) == 0)
        policy['families'][family] = selected_policy
        report['families'][family] = {'calibration': {'actions': cal.height,
            'positive': int(cal['y'].sum()), 'false': false.height, 'minimum_score': floor,
            'selected_actions': selected_cal.height, 'selected_positive': int(selected_cal['y'].sum()),
            'selected_false': int((selected_cal['y'] != 1).sum())}, 'search': diagnostics}
    combined = apply_family_policy(actions, policy)
    report['combined_search'] = promotion_additions(combined, baseline, search, min_actions=2*min_actions)
    policy['enabled'] = any(p['enabled'] for p in policy['families'].values()) and report['combined_search']['passed']
    return policy, report
