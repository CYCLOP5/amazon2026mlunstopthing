'strictly additive rescue policies; search labels never enter check selection'
import math
import numpy as np
import polars as pl
from er.stack import decode
from latest_fusion.tuning import promotion


def _validate(actions):
    if actions['tid'].n_unique() != actions.height:
        raise ValueError('Actions must have globally unique target owners')
    if 'score' in actions and not np.isfinite(actions['score'].to_numpy()).all():
        raise ValueError('Action scores must be finite')


def apply_additions(baseline, actions, threshold):
    'preserve every incumbent row and reject every occupied target globally'
    _validate(actions)
    if baseline['tid'].n_unique() != baseline.height:
        raise ValueError('Baseline has duplicate target ownership')
    additions = actions.filter(pl.col('score') >= threshold).join(
        baseline.select('tid'), on='tid', how='anti')
    additions = additions.with_columns(pl.col('score').alias('p'))
    additions = additions.select(baseline.columns)
    res = pl.concat([baseline, additions], how='vertical_relaxed')
    assert res['tid'].n_unique() == res.height
    return res


def _diagnostics(additions, baseline, anchors, min_actions):
    scoped = additions.join(anchors.select('qid'), on='qid', how='inner')
    after = apply_additions(baseline, additions.with_columns(pl.lit(1.).alias('score')), 0.)
    before_metrics, after_metrics = decode.score(baseline, anchors), decode.score(after, anchors)
    tp = int(scoped['y'].sum())
    fp = scoped.height - tp
    checks = {'min_actions': scoped.height >= min_actions, 'zero_added_fp': fp == 0,
              'precision': after_metrics['pair_precision'] >= before_metrics['pair_precision']-1e-12,
              'recall': after_metrics['pair_recall'] >= before_metrics['pair_recall']-1e-12,
              'gain': after_metrics['macro_f05']-before_metrics['macro_f05'] >= 1e-5}
    countries = {}
    for country in anchors['co'].unique().to_list():
        subset = anchors.filter(pl.col('co') == country)
        old, new = decode.score(baseline, subset), decode.score(after, subset)
        country_checks = {key: new[key] >= old[key]-1e-12 for key in ('pair_precision', 'pair_recall', 'macro_f05')}
        countries[str(country)] = {'before': old, 'after': new, 'checks': country_checks}
        checks[str(country)] = all(country_checks.values())
    return {'passed': all(checks.values()), 'checks': checks, 'before': before_metrics,
            'after': after_metrics, 'countries': countries, 'added_tp': tp, 'added_fp': fp,
            'added_actions': scoped.height}


def select_additions(top_actions, baseline_accepted, search_anchors, min_actions=10):
    _validate(top_actions)

    actions = top_actions.join(baseline_accepted.select('tid'), on='tid', how='anti')
    actions = actions.join(search_anchors.select('qid', pl.col('co').alias('_anchor_co')), on='qid', how='inner')
    if actions.height and actions.filter(pl.col('co') != pl.col('_anchor_co')).height:
        raise ValueError('Action country differs from search anchor country')
    actions = actions.drop('_anchor_co')


    base = baseline_accepted.join(search_anchors.select('qid'), on='qid', how='inner')
    cuts = set([.9, .95, .975, .99, .995, .997, .999, .9995, .9999])
    if actions.height:
        false_scores = actions.filter(pl.col('y') != 1)['score'].to_numpy()
        if len(false_scores):



            dtype = np.float32 if actions['score'].dtype == pl.Float32 else np.float64
            cutoff = np.nextafter(dtype(false_scores.max()), dtype(np.inf))
            if np.isfinite(cutoff):
                cuts.add(float(cutoff))
        else:
            cuts.add(float(actions['score'].min()))
    else:
        evidence = _diagnostics(actions, base, search_anchors, min_actions)
        evidence.update(reason='No unoccupied search actions', evaluated=[])
        return {'threshold': 1.0, 'enabled': False}, evidence
    best = None
    evaluated = []
    for cut in sorted(cuts):
        sel = actions.filter(pl.col('score') >= cut)
        evidence = _diagnostics(sel, base, search_anchors, min_actions)
        evaluated.append({'threshold': cut, 'passed': evidence['passed'], 'added_actions': evidence['added_actions'], 'added_fp': evidence['added_fp']})
        if evidence['passed'] and (best is None or evidence['after']['macro_f05'] > best[1]['after']['macro_f05']):
            best = (cut, evidence)
    if best is None:
        evidence = _diagnostics(actions.head(0), base, search_anchors, min_actions)
        evidence.update(reason='No search threshold met strict additive precision and gain constraints', evaluated=evaluated)
        return {'threshold': 1.0, 'enabled': False}, evidence
    best[1].update(reason='Selected using search labels only', evaluated=evaluated)
    return {'threshold': best[0], 'enabled': True}, best[1]


def promotion_additions(candidate_additions, baseline, check_anchors, min_actions=10):
    _validate(candidate_additions)
    occupied = candidate_additions.join(baseline.select('tid'), on='tid', how='inner').height
    additions = candidate_additions.join(baseline.select('tid'), on='tid', how='anti')
    additions = additions.join(check_anchors.select('qid'), on='qid', how='inner')
    baseline = baseline.join(check_anchors.select('qid'), on='qid', how='inner')
    evidence = _diagnostics(additions, baseline, check_anchors, min_actions)
    after = apply_additions(baseline, additions.with_columns(pl.lit(1.).alias('score')), 0.)
    paired = promotion(after, baseline, check_anchors, max_precision_loss=0, max_recall_loss=0, family_size=1)
    evidence['checks'].update(no_occupied_targets=occupied == 0, paired=paired['passed'])
    evidence['passed'] = all(evidence['checks'].values())
    evidence['paired'] = paired
    n, tp = evidence['added_actions'], evidence['added_tp']
    if n:
        z = 1.959963984540054
        p = tp/n
        denominator = 1+z*z/n
        center = (p+z*z/(2*n))/denominator
        radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
        interval = [max(0., center-radius), min(1., center+radius)]
    else:
        interval = [0., 1.]
    evidence['action_precision_ci95'] = interval
    evidence['uncertainty_scope'] = 'Wilson action precision interval; observed zero errors do not guarantee hidden precision.'
    return evidence
