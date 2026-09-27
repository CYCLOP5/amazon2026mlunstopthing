'fixed-grid lexical action policies. model scores rank actions, not posterior truth'
import math

import numpy as np
import polars as pl

from er.stack import decode

DROP_GRID = (.75, .9, .95, .99)
ADD_GRID = (.99, .995, .999)
KEYS = ['qid', 'tid']


def _unique(frame, keys, name):
    if frame.select(keys).unique().height != frame.height:
        raise ValueError(f'{name} contains duplicate {keys}')


def _scope(frame, anchors):
    return frame.join(anchors.select('qid'), on='qid', how='inner')


def _actions(base, drops, adds, anchors):
    _unique(anchors, ['qid'], 'Anchors')
    _unique(base, ['tid'], 'Baseline')
    _unique(drops, KEYS, 'Drops')
    _unique(adds, ['tid'], 'Additions')
    if drops.select('tid').join(adds.select('tid'), on='tid').height:
        raise ValueError('Addition and deletion targets must be disjoint')

    drop_columns = KEYS + (['own'] if 'own' in drops and 'own' not in base else [])
    removed = _scope(base, anchors).join(drops.select(drop_columns), on=KEYS, how='inner')

    added = _scope(adds, anchors).join(base.select('tid'), on='tid', how='anti')
    return removed, added


def apply_actions(base_accepted, proposed_drops, proposed_adds, anchors):
    'apply scoped actions, preserving global incumbent target ownership'
    removed, added = _actions(base_accepted, proposed_drops, proposed_adds, anchors)
    after = pl.concat([base_accepted.join(removed.select(KEYS), on=KEYS, how='anti'),
                       added.select(base_accepted.columns)], how='vertical_relaxed')
    _unique(after, ['tid'], 'Result')
    return after


def _counts(frame):
    if frame.height and frame.filter(~pl.col('y').is_in([0, 1]) | pl.col('y').is_null()).height:
        raise ValueError('Action labels must be binary')
    tp = int(frame['y'].sum() or 0)
    out = {'tp': tp, 'fp': frame.height - tp, 'actions': frame.height}
    if 'own' in frame:
        if frame.filter(pl.col('y') != (pl.col('own') == pl.col('qid')).cast(pl.Int8)).height:
            raise ValueError('Action label disagrees with owner')
        out.update(decoy=int((frame['own'] < 0).sum()),
                   other=int(((frame['own'] >= 0) & (frame['own'] != frame['qid'])).sum()))
    return out


def _wilson98(successes, n):

    if not n:
        return 0.
    z = 2.3263478740408408
    p = successes / n
    return max(0., (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n))


def evaluate_actions(base_accepted, proposed_drops, proposed_adds, anchors):
    'exact entity macro f0.5, including correctly empty zero-degree owners'
    if not anchors.height:
        raise ValueError('Evaluation requires at least one anchor')
    removed, added = _actions(base_accepted, proposed_drops, proposed_adds, anchors)
    scoped_base = _scope(base_accepted, anchors)
    after = pl.concat([scoped_base.join(removed.select(KEYS), on=KEYS, how='anti'),
                       added.select(base_accepted.columns)], how='vertical_relaxed')
    rc, ac = _counts(removed), _counts(added)
    old, new = decode.score(scoped_base, anchors), decode.score(after, anchors)
    n = rc['actions'] + ac['actions']
    report = {'before': old, 'after': new,
              'delta': {k: new[k]-old[k] for k in old},
              'TPadded': ac['tp'], 'FPadded': ac['fp'],
              'TPremoved': rc['tp'], 'FPremoved': rc['fp'],
              'added_actions': ac['actions'], 'removed_actions': rc['actions'],
              'changed': bool(n), 'action_benefit_wilson98_lower': _wilson98(ac['tp']+rc['fp'], n),
              'uncertainty_scope': 'Descriptive action benefit interval; model cuts are ranking rules, not calibrated posterior confidence.',
              'countries': {}}
    for prefix, counts in [('added', ac), ('removed', rc)]:
        for k in ('decoy', 'other'):
            if k in counts:
                report[prefix+'_'+k] = counts[k]
    if 'co' in anchors:
        for country in anchors['co'].unique().sort().to_list():
            subset = anchors.filter(pl.col('co') == country)
            report['countries'][str(country)] = {'before': decode.score(scoped_base, subset),
                                                  'after': decode.score(after, subset)}
    return report


def propose_actions(rows, base_accepted, anchors, frozen_rule):
    'use frozen thresholds; global qualifying-owner uniqueness precedes scope'
    _unique(rows, KEYS, 'Scored swaps')
    _unique(base_accepted, ['tid'], 'Baseline')
    _unique(anchors, ['qid'], 'Anchors')
    drop = rows.select(KEYS).head(0)
    add = rows.head(0)
    if not frozen_rule.get('enabled', False):
        return drop, add
    for field in ('p_alias', 'p_decoy', 'p_other'):
        if rows.height and (not np.isfinite(rows[field].to_numpy()).all() or
                            rows.filter((pl.col(field) < 0) | (pl.col(field) > 1)).height):
            raise ValueError(f'{field} must be finite in [0, 1]')
    if frozen_rule.get('drop_threshold') is not None:
        drop = rows.filter(pl.col('p_decoy') >= frozen_rule['drop_threshold']).select(
            KEYS + (['own'] if 'own' in rows else []))
        drop = _scope(drop, anchors).join(base_accepted.select(KEYS), on=KEYS, how='inner')
    if frozen_rule.get('add_threshold') is not None:
        qualifying = rows.filter(pl.col('p_alias') >= frozen_rule['add_threshold'])
        qualifying = qualifying.join(base_accepted.select('tid'), on='tid', how='anti')
        unique_targets = qualifying.group_by('tid').len().filter(pl.col('len') == 1).select('tid')
        add = _scope(qualifying.join(unique_targets, on='tid'), anchors)
    return drop, add


def _guard(evidence, min_actions):
    before, after = evidence['before'], evidence['after']
    checks = {'min_actions': evidence['added_actions'] + evidence['removed_actions'] >= min_actions,
              'zero_removed_tp': evidence['TPremoved'] == 0,
              'zero_added_fp': evidence['FPadded'] == 0,
              'macro_gain': after['macro_f05']-before['macro_f05'] >= 1e-5,
              'precision': after['pair_precision'] >= before['pair_precision']-1e-12,
              'recall': after['pair_recall'] >= before['pair_recall']-1e-12}
    for co, metrics in evidence['countries'].items():
        checks[co+'_nondecreasing'] = all(metrics['after'][k] >= metrics['before'][k]-1e-12
                                         for k in ('macro_f05', 'pair_precision', 'pair_recall'))
    evidence.update(checks=checks, passed=all(checks.values()))
    return evidence


def select_policy(rows, base_accepted, anchors, min_actions=25):
    'choose at most one rule using only supplied source calibration labels'
    if min_actions < 1:
        raise ValueError('min_actions must be positive')
    empty = {'enabled': False, 'drop_threshold': None, 'add_threshold': None}
    evaluated, best = [], {}
    for kind, grid in [('drop', DROP_GRID), ('add', ADD_GRID)]:
        options = []
        for cut in grid:
            rule = {**empty, 'enabled': True, kind+'_threshold': cut}
            drops, adds = propose_actions(rows, base_accepted, anchors, rule)
            evidence = _guard(evaluate_actions(base_accepted, drops, adds, anchors), min_actions)
            evaluated.append({'rule': rule, **evidence})
            if evidence['passed']:
                options.append((rule, evidence))
        if options:

            best[kind] = max(options, key=lambda item: item[1]['after']['macro_f05'])
    choices = list(best.values())
    if len(best) == 2:
        combined = {'enabled': True, 'drop_threshold': best['drop'][0]['drop_threshold'],
                    'add_threshold': best['add'][0]['add_threshold']}
        drops, adds = propose_actions(rows, base_accepted, anchors, combined)
        evidence = _guard(evaluate_actions(base_accepted, drops, adds, anchors), min_actions)
        evaluated.append({'rule': combined, **evidence})
        if evidence['passed']:
            choices.append((combined, evidence))
    if choices:
        rule, report = max(choices, key=lambda item: item[1]['after']['macro_f05'])
        reason = 'Fixed-grid rule selected on source calibration anchors only'
    else:
        drops, adds = propose_actions(rows, base_accepted, anchors, empty)
        rule, report = empty, _guard(evaluate_actions(base_accepted, drops, adds, anchors), min_actions)
        reason = 'Baseline retained: no fixed-grid rule passed all action and macro guards'
    return rule, {**report, 'reason': reason, 'evaluated': evaluated,
                  'selection_scope': 'Only supplied anchors; open evaluation must replay the frozen rule',
                  'min_actions': min_actions}
