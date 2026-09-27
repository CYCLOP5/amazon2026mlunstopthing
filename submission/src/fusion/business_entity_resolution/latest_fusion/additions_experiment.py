'frozen pretrained-head experiment for unoccupied targets only'
import hashlib
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import polars as pl

from . import contrastive_experiment as contrastive
from . import directional_experiment as directional
from . import directional_policy as actions
from . import owner_switch as validation
from .structured_protocol import WARNING
from .pipeline import log

KEYS = ['qid', 'tid']
COUNTRIES = ['us', 'india']
GRID = tuple((p, m) for p in (.98, .99, .995, .999) for m in (.05, .2, .5))
DISABLED = {'enabled': False, 'min_alias': None, 'min_margin': None}


def _anchors(refs):
    return refs.rename({'rid': 'qid'}) if 'rid' in refs else refs


def _country(refs):
    return _anchors(refs).filter(pl.col('co').is_in(COUNTRIES))


def rank_candidates(rows, baseline):
    'Rank every supplied candidate owner before any country/split scope'
    validation._keys(rows, KEYS, 'Scored candidates')
    validation._base(baseline)
    validation._scores(rows, ['p_alias', 'p_decoy', 'p_other'])
    unoccupied = rows.join(baseline.select('tid'), on='tid', how='anti')
    ordered = unoccupied.sort(['tid', 'p_alias', 'qid'], descending=[False, True, False])
    ordered = ordered.with_columns(pl.col('p_alias').shift(-1).over('tid').fill_null(0.).alias('runner_up'))
    return ordered.unique('tid', keep='first', maintain_order=True).with_columns(
        (pl.col('p_alias') - pl.col('runner_up')).alias('margin'))


def propose_additions(rows, baseline, anchors, rule, ranked=False):
    validation._anchors(anchors)
    winners = rows if ranked else rank_candidates(rows, baseline)
    if not rule.get('enabled', False):
        return winners.head(0).with_columns(pl.col('p_alias').alias('p'))
    validation._scores(pl.DataFrame({'cut': [rule['min_alias'], rule['min_margin']]}), ['cut'])
    winners = winners.filter((pl.col('p_alias') >= rule['min_alias']) &
                            (pl.col('margin') >= rule['min_margin']) & (pl.col('margin') > 0))

    winners = winners.join(baseline.select('tid'), on='tid', how='anti')
    return winners.join(anchors.select('qid'), on='qid', how='semi').with_columns(pl.col('p_alias').alias('p'))


def apply_additions(baseline, additions):
    validation._base(baseline)
    validation._keys(additions, ['tid'], 'Additions')
    if additions.select('tid').join(baseline.select('tid'), on='tid').height:
        raise ValueError('Additions must be globally unoccupied')
    missing = set(baseline.columns) - set(additions.columns)
    if missing and additions.height:
        raise ValueError(f'Additions missing baseline columns: {sorted(missing)}')
    if not additions.height:
        return baseline.clone()
    return pl.concat([baseline, additions.select(baseline.columns)], how='vertical_relaxed')


def evaluate_additions(baseline, additions, anchors):
    validation._anchors(anchors)
    if not anchors.height:
        raise ValueError('Evaluation requires nonempty anchors')
    validation._labels(baseline.join(anchors.select('qid'), on='qid', how='semi'))
    validation._labels(additions)
    return actions.evaluate_actions(baseline, baseline.select(KEYS).head(0), additions, anchors)


def _guard(evidence, minimum, check=False):
    checks = {'minimum_TPadded': evidence['TPadded'] >= minimum,
              'zero_FPadded': evidence['FPadded'] == 0,
              'overall_nondecreasing': all(evidence['after'][k] >= evidence['before'][k] - 1e-12
                                            for k in validation.METRICS)}
    for co, metrics in evidence['countries'].items():
        checks[co + '_nondecreasing'] = all(metrics['after'][k] >= metrics['before'][k] - 1e-12
                                           for k in validation.METRICS)
    if check:
        checks['macro_gain_above_1e_5'] = evidence['delta']['macro_f05'] > 1e-5
    return {**evidence, 'checks': checks, 'passed': all(checks.values()),
            'failure_reasons': [k for k, passed in checks.items() if not passed]}


def select_policy(ranked, baseline, anchors):
    trials = []
    if anchors.height:
        for p, m in GRID:
            rule = {'enabled': True, 'min_alias': p, 'min_margin': m}
            additions = propose_additions(ranked, baseline, anchors, rule, ranked=True)
            trials.append({'rule': rule, **_guard(evaluate_additions(baseline, additions, anchors), 5)})
    passing = [trial for trial in trials if trial['passed']]
    sel = max(passing, key=lambda t: t['after']['macro_f05']) if passing else None
    return (sel['rule'] if sel else dict(DISABLED)), {
        'passed': bool(sel), 'selected': sel, 'trials': trials,
        'reason': None if sel else 'No calibration rule meets fixed guards'}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_hash(frame, keys):
    values = frame.select(keys).unique().sort(keys).rows()
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def _source(path, family):
    src = Path(path) / 'experiment'
    if not (src / 'features.json').is_file():
        raise ValueError(f'No pretrained features.json found in {src}')
    return src


def _build_score(family, frame, refs, targets, models, features, s2_count):
    if family == 'contrastive':
        rows, _, support = contrastive._build(frame, refs, targets)
        scorer = contrastive._score
    else:
        rows, _, support, _ = directional.swaps.build(frame, refs, targets, s2_count)
        scorer = directional._score
        if 'co' not in rows:
            rows = rows.join(refs.select(pl.col('rid').alias('qid'), 'co'), on='qid', how='left')
    scored = {}
    for head, model in models.items():
        if model.feature_name() != features[head]:
            raise ValueError('Stored model feature order differs from stored features.json')
        prediction = scorer(model, rows, features[head])

        baseline_col = next((c for c in ('baseline_p', 'p', 'newest') if c in rows), None)
        if 'baseline_p' not in prediction:
            if baseline_col:
                prediction = prediction.join(rows.select(*KEYS, pl.col(baseline_col).alias('baseline_p')),
                    on=KEYS, how='left', validate='1:1')
            else:
                prediction = prediction.with_columns(pl.lit(None, pl.Float64).alias('baseline_p'))
        if 'baseline_raw' in rows and 'raw' not in prediction:
            prediction = prediction.join(rows.select(*KEYS, pl.col('baseline_raw').alias('raw')),
                on=KEYS, how='left', validate='1:1')
        scored[head] = prediction.with_columns(pl.col('p_alias').alias('p')).select(
            *KEYS, 'p_alias', 'p_decoy', 'p_other', 'baseline_p', 'p',
            *[c for c in ('y', 'own', 'co', 'raw') if c in prediction])
    return scored, support


def run(train_frame, refs, targets, fit_frame, cal_refs, check_refs,
        baseline_train, core_features, output_dir, fit_refs=None, s2_count=None,
        pretrained_heads=None):
    'reuse exact persisted models; no fitting or check-driven fallback occurs'
    if not pretrained_heads:
        raise ValueError('Pretrained contrastive/directional model sources are required')
    contrastive._validate_fit(train_frame, fit_frame, cal_refs, check_refs, fit_refs)
    validation._keys(train_frame, KEYS, 'Full candidates')
    validation._base(baseline_train)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cal, check = _country(cal_refs), _country(check_refs)
    report = {'promoted': False, 'heads': {}, 'sources': {}, 'selected': None,
              'check': None, 'matched_control_check': None,
              'class_order': ['alias', 'decoy', 'other_owned'],
              'no_French_labels': True, 'main_countries': COUNTRIES,
              'policy': 'Add globally unoccupied targets only; preserve every incumbent row. Global highest alias score, ties abstain, runner-up margin before scope.',
              'directional_competition': 'All swap-eligible owners in the full candidate pool; non-swap pairs are not directional competitors.',
              'selection': 'Best calibration macro F0.5 per head, then one overall best candidate. Check only that candidate and its matched control; no fallback.',
              'historical_upstream_exposure_warning': WARNING,
              'protocol': {'fit_pairs_sha256': _ids_hash(fit_frame, KEYS),
                           'calibration_refs_sha256': _ids_hash(cal_refs, ['rid' if 'rid' in cal_refs else 'qid']),
                           'check_refs_sha256': _ids_hash(check_refs, ['rid' if 'rid' in check_refs else 'qid'])}}
    runtime, rankings, cands = {}, {}, []
    for family in ('contrastive', 'directional'):
        if family not in pretrained_heads:
            report['sources'][family] = {'available': False, 'reason': 'No source supplied'}
            continue
        src = _source(pretrained_heads[family], family)
        ff = json.loads((src / 'features.json').read_text())
        models = {}
        provenance = {'directory': str(src.resolve()), 'features_sha256': _hash(src / 'features.json'),
                      'models': {}, 'retrained': False}
        for head in ('control', 'treatment'):
            path = src / (head + '_model.txt')
            if path.is_file():
                models[head] = lgb.Booster(model_file=str(path))
                provenance['models'][head] = {'path': str(path.resolve()), 'sha256': _hash(path)}
        report['sources'][family] = provenance
        log(f'additions: building {family} features and scoring exact saved heads')
        scored, support = _build_score(family, train_frame, refs, targets, models, ff, s2_count)
        runtime[family] = {'models': models, 'features': ff, 'source': src}
        provenance['support'] = support
        for head, rows in scored.items():
            key = family + '/' + head
            rankings[key] = rank_candidates(rows, baseline_train)
            rule, evidence = select_policy(rankings[key], baseline_train, cal)
            report['heads'][key] = {'rule': rule, 'calibration': evidence}
            sel = evidence['selected']
            log(f'additions: {key} calibration eligible={evidence["passed"]}; rule={rule}; '+
                (f'TPadded={sel["TPadded"]}, FPadded={sel["FPadded"]}, gain={sel["delta"]["macro_f05"]:.9f}' if sel else 'no eligible additions'))
            if evidence['passed']:
                cands.append((key, evidence['selected']['after']['macro_f05']))
    winner = max(cands, key=lambda item: item[1])[0] if cands else None
    log(f'additions: locked calibration winner={winner}; no check-driven fallback')
    model, ff, family, head, rule = None, None, None, None, dict(DISABLED)
    if winner:
        family, head = winner.split('/')
        rule = report['heads'][winner]['rule']
        report['selected'] = {'family': family, 'head': head, 'rule': rule}
        report['matched_control_is_selected'] = head == 'control'
        report['feature_attribution'] = ('Core-only control selected; no gain attributed to new family features'
            if head == 'control' else 'Treatment selected; matched control reported on the same frozen check population')
        selected_adds = propose_additions(rankings[winner], baseline_train, cal, rule, ranked=True)
        selected_adds.write_parquet(output / 'calibration_additions.parquet')
        if selected_adds.height <= 200:
            report['calibration_admitted_examples'] = selected_adds.to_dicts()
        if check.height:
            adds = propose_additions(rankings[winner], baseline_train, check, rule, ranked=True)
            evidence = _guard(evaluate_additions(baseline_train, adds, check), 25, check=True)
            control_key = family + '/control'
            control_rule = report['heads'].get(control_key, {}).get('rule', DISABLED)
            if control_key in rankings:
                control_adds = propose_additions(rankings[control_key], baseline_train, check, control_rule, ranked=True)
                control = _guard(evaluate_additions(baseline_train, control_adds, check), 25, check=True)
                evidence['checks']['not_worse_matched_control'] = evidence['after']['macro_f05'] >= control['after']['macro_f05'] - 1e-12
                report['matched_control_check'] = control
            else:
                evidence['checks']['not_worse_matched_control'] = False
            evidence['passed'] = all(evidence['checks'].values())
            evidence['failure_reasons'] = [k for k, passed in evidence['checks'].items() if not passed]
            report['check'] = evidence
            report['promoted'] = evidence['passed']
            log(f'additions: check promoted={evidence["passed"]}; TPadded={evidence["TPadded"]}, FPadded={evidence["FPadded"]}; gain={evidence["delta"]["macro_f05"]:.9f}; failures={evidence["failure_reasons"]}')
            adds.write_parquet(output / 'check_additions.parquet')
        else:
            report['check'] = {'passed': False, 'failure_reasons': ['No check references']}
        model, ff = runtime[family]['models'][head], runtime[family]['features'][head]
        shutil.copyfile(runtime[family]['source'] / (head + '_model.txt'), output / 'selected_model.txt')
        (output / 'features.json').write_text(json.dumps(ff, indent=2))
    else:
        report['failure_reason'] = 'No calibration-eligible candidate; exact baseline retained'
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / 'rule.json').write_text(json.dumps({'promoted': report['promoted'], 'selected': report['selected']}, indent=2))
    return {'report': report, 'promoted': report['promoted'], 'model': model,
            'features': ff, 'family': family, 'head': head, 'rule': rule,
            'output_dir': str(output)}


def infer(bundle, test_frame, test_refs, test_targets, baseline_test, s2_count=None, **unused):
    'replay just the selected frozen family; preserve every accepted row'
    diagnostics = {'promoted': bool(bundle['promoted']), 'additions': 0,
                   'france_policy': 'Exact baseline retained', 'accepted_targets': baseline_test.height}
    if not bundle['promoted']:
        return baseline_test.clone(), diagnostics
    scored, support = _build_score(bundle['family'], test_frame, test_refs, test_targets,
        {bundle['head']: bundle['model']}, {bundle['head']: bundle['features']}, s2_count)
    rows = scored[bundle['head']]
    additions = propose_additions(rows, baseline_test, _country(test_refs), bundle['rule'])
    accepted = apply_additions(baseline_test, additions)
    diagnostics.update(additions=additions.height, accepted_targets=accepted.height, support=support,
                       family=bundle['family'], head=bundle['head'], rule=bundle['rule'])
    return accepted, diagnostics
