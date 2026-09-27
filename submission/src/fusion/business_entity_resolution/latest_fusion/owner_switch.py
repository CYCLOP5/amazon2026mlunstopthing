'scoped reassignment of accepted targets using a frozen contrastive ranker'
import math

import polars as pl

from er.stack import decode

GRID = ((.999, .99), (.995, .98), (.99, .95), (.98, .90))
METRICS = ('macro_f05', 'pair_precision', 'pair_recall')


def _keys(frame, keys, name):
    if not set(keys).issubset(frame.columns):
        raise ValueError(f'{name} requires {keys}')
    if any(frame[c].null_count() for c in keys):
        raise ValueError(f'{name} has missing IDs')
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError(f'{name} contains duplicate {keys}')


def _scores(frame, columns):
    for c in columns:
        if c not in frame or not frame.schema[c].is_numeric():
            raise ValueError(f'{c} must be finite numeric scores in [0, 1]')
        s = frame[c].cast(pl.Float64)
        if s.null_count() or not s.is_finite().all() or not s.is_between(0, 1).all():
            raise ValueError(f'{c} must be finite numeric scores in [0, 1]')


def _base(base):
    _keys(base, ['tid'], 'Baseline')
    if 'qid' not in base or base['qid'].null_count():
        raise ValueError('Baseline has missing owner IDs')
    _scores(base, ['p'])


def _anchors(anchors):
    _keys(anchors, ['qid'], 'Anchors')


def _scoped(frame, anchors):
    return frame.join(anchors.select('qid'), on='qid', how='inner', maintain_order='left')


def propose_switches(rows, base, anchors, min_alias, min_margin):
    'select strongest new alias globally, then require both endpoints in scope'
    _keys(rows, ['qid', 'tid'], 'Scored candidates')
    _base(base)
    _anchors(anchors)
    _scores(rows, ['p_alias', 'p_decoy', 'p_other'])
    if not all(isinstance(c, (int, float)) and math.isfinite(c) and 0 <= c <= 1
               for c in (min_alias, min_margin)):
        raise ValueError('Switch cutoffs must be finite in [0, 1]')
    if base.select('qid', 'tid').join(rows.select('qid', 'tid'), on=['qid', 'tid'], how='anti').height:
        raise ValueError('Missing current owner pair in scored candidates')


    best = rows.select('qid', 'tid', 'p_alias', 'p_decoy', 'p_other').sort(
        ['tid', 'p_alias', 'qid'], descending=[False, True, False]).unique(
        'tid', keep='first', maintain_order=True)
    old = base.select('qid', 'tid').join(rows.select('qid', 'tid', 'p_alias'),
        on=['qid', 'tid'], how='left', validate='1:1').rename({'qid': 'old_qid', 'p_alias': 'old_p'})
    d = best.join(old, on='tid', how='inner', validate='1:1', maintain_order='left')
    d = d.filter((pl.col('qid') != pl.col('old_qid')) & (pl.col('p_alias') >= min_alias) &
                 (pl.col('p_alias') - pl.col('old_p') >= min_margin))
    d = _scoped(d, anchors).join(anchors.select(pl.col('qid').alias('old_qid')),
        on='old_qid', how='inner', maintain_order='left')

    new_labels = [c for c in ('y', 'own') if c in rows]
    if new_labels:
        d = d.join(_scoped(rows, anchors).select('qid', 'tid', *new_labels),
                   on=['qid', 'tid'], how='left', validate='1:1', maintain_order='left')
    old_labels = [c for c in ('y', 'own') if c in base]
    if old_labels:
        d = d.join(_scoped(base, anchors).select(pl.col('qid').alias('old_qid'), 'tid',
            *[pl.col(c).alias('old_' + c) for c in old_labels]), on=['old_qid', 'tid'],
            how='left', validate='1:1', maintain_order='left')
    columns = ['old_qid', 'qid', 'tid', 'p', 'old_p', 'p_alias', 'p_decoy', 'p_other']
    columns += [c for c in ('y', 'old_y', 'own', 'old_own') if c in d]
    return d.with_columns(pl.col('p_alias').alias('p')).select(columns)


def _switches(base, switches):
    _base(base)
    _keys(switches, ['tid'], 'Switches')
    if not {'old_qid', 'qid'}.issubset(switches.columns) or any(
            switches[c].null_count() for c in ('old_qid', 'qid')):
        raise ValueError('Switches have missing owner IDs')
    _scores(switches, ['p'])
    if switches.filter(pl.col('old_qid') == pl.col('qid')).height:
        raise ValueError('Switch must change owner')
    if switches.select(pl.col('old_qid').alias('qid'), 'tid').join(
            base.select('qid', 'tid'), on=['qid', 'tid'], how='anti').height:
        raise ValueError('Switch is missing its current owner pair in baseline')


def apply_switches(base, switches):
    'Preserve baseline schema/order; replace owner, score and available labels'
    _switches(base, switches)
    if switches.height == 0:
        return base.clone()
    if 'y' in base and 'y' not in switches:
        raise ValueError('Labeled baseline requires switch y')
    changed = ['qid', 'p'] + [c for c in ('y', 'own') if c in base and c in switches]
    added = switches.select('tid', *[pl.col(c).alias('_switch_' + c) for c in changed],
                             pl.lit(True).alias('_switched'))
    d = base.join(added, on='tid', how='left', validate='1:1', maintain_order='left')
    out = d.with_columns([pl.when(pl.col('_switched').fill_null(False)).then(pl.col('_switch_' + c))
                         .otherwise(pl.col(c)).cast(base.schema[c]).alias(c) for c in changed]).select(base.columns)
    _keys(out, ['tid'], 'Result')
    if out.height != base.height or not out['tid'].equals(base['tid']):
        raise AssertionError('Switches changed the accepted target set/order')
    return out


def _labels(frame):
    if 'y' not in frame or frame['y'].null_count() or not frame['y'].is_in([0, 1]).all():
        raise ValueError('Evaluation requires binary labels')
    if 'own' in frame and (frame['own'].null_count() or frame.filter(
            pl.col('y') != (pl.col('own') == pl.col('qid')).cast(pl.Int8)).height):
        raise ValueError('Evaluation label disagrees with truth owner')


def evaluate_switches(base, switches, anchors):
    'exact entity macro f0.5, with empty zero-degree anchors included'
    _anchors(anchors)
    _switches(base, switches)
    if not anchors.height or 'deg' not in anchors or anchors['deg'].null_count() or anchors.filter(
            (pl.col('deg') < 0) | ~pl.col('deg').cast(pl.Float64).is_finite()).height:
        raise ValueError('Evaluation requires nonempty anchors with nonnegative degrees')
    if switches.select(pl.col('old_qid').alias('qid')).join(anchors.select('qid'), on='qid', how='anti').height or \
            switches.select('qid').join(anchors.select('qid'), on='qid', how='anti').height:
        raise ValueError('Both switch owners must be in scoped anchors')
    if switches.height:
        _labels(switches)
    before_rows = _scoped(base, anchors)
    after_rows = apply_switches(before_rows, switches)
    _labels(before_rows)
    _labels(after_rows)
    changed = after_rows.join(switches.select('tid'), on='tid', how='inner').select(
        'tid', pl.col('y').alias('_new_y')).join(
        before_rows.select('tid', pl.col('y').alias('_old_y')), on='tid', validate='1:1')
    counts = {name: changed.filter((pl.col('_old_y') == old) & (pl.col('_new_y') == new)).height
              for name, old, new in [('corrected', 0, 1), ('broken', 1, 0),
                                      ('wrong_to_wrong', 0, 0), ('true_to_true', 1, 1)]}
    before, after = decode.score(before_rows, anchors), decode.score(after_rows, anchors)
    countries = {}
    if 'co' in anchors:
        if anchors['co'].null_count():
            raise ValueError('Missing anchor country')
        for co in anchors['co'].unique().sort().to_list():
            subset = anchors.filter(pl.col('co') == co)
            countries[str(co)] = {'before': decode.score(before_rows, subset),
                                   'after': decode.score(after_rows, subset)}
    return {'before': before, 'after': after, 'delta': {k: after[k] - before[k] for k in before},
            'switches': switches.height, **counts, 'countries': countries}


def select_policy(rows, base, anchors):
    'select from four fixed cuts on supplied calibration anchors only'
    evaluated, passing = [], []
    for min_alias, min_margin in GRID:
        rule = {'enabled': True, 'min_alias': min_alias, 'min_margin': min_margin}
        switches = propose_switches(rows, base, anchors, min_alias, min_margin)
        evidence = evaluate_switches(base, switches, anchors)
        checks = {'five_corrected': evidence['corrected'] >= 5, 'zero_broken': evidence['broken'] == 0,
                  'overall_nondecreasing': all(evidence['after'][k] >= evidence['before'][k] - 1e-12 for k in METRICS)}
        for co, metrics in evidence['countries'].items():
            checks[co + '_nondecreasing'] = all(metrics['after'][k] >= metrics['before'][k] - 1e-12 for k in METRICS)
        trial = {'rule': rule, **evidence, 'checks': checks, 'passed': all(checks.values())}
        evaluated.append(trial)
        if trial['passed']:
            passing.append(trial)
    if passing:
        chosen = max(passing, key=lambda trial: trial['after']['macro_f05'])
        rule, report = chosen['rule'], chosen
    else:
        rule = {'enabled': False, 'min_alias': None, 'min_margin': None}
        empty = propose_switches(rows, base, anchors, 1., 1.).head(0)
        report = {**evaluate_switches(base, empty, anchors), 'passed': False}
    return rule, {**report, 'evaluated': evaluated,
                  'selection_scope': 'Supplied calibration anchors only; replay frozen cuts on other populations.',
                  'interpretation': 'Scoped action rule, not a posterior guarantee.'}
