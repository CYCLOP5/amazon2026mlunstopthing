'Fixed drop/add ablation of saved checkpoints, without fitting or search'
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export
from final_repair.pipeline import changes
from latest_fusion import additions_experiment as addition_policy
from latest_fusion.pipeline import log, logit, sha256, sigmoid, write_json
from latest_fusion.review_export import assert_anchor_replay, assert_metric_replay
from latest_fusion.structured_pipeline import (INCUMBENT_SHA, METHOD, _trim,
    assert_same_pool, replay)
from latest_fusion.structured_protocol import split_references, WARNING
from latest_fusion.tune_pipeline import anchor_metrics, compact, predict_residual
from latest_fusion.tuning import apply_policy, fit_population, top_pairs

STRENGTH_SHA = '0f91f8ec752bfba46a30aecc15488ee8b7a31a835de8ed808944db1f8a4e16ef'
DROP_ONLY_SHA = 'd3daa7ca0d23a6c55280047d1a442600d73c3dd9ef7ec2c7ac10d638975e58ba'
METRICS = ('macro_f05', 'pair_precision', 'pair_recall')
KEYS = ['qid', 'tid']


def intersection(baseline, strength):
    'Preserve exact baseline rows whose owner/target pair occurs in strength'
    for name, frame in [('Baseline', baseline), ('Strength', strength)]:
        if frame['tid'].n_unique() != frame.height:
            raise ValueError(name + ' assigns a target more than once')
    return baseline.join(strength.select(KEYS), on=KEYS, how='semi', maintain_order='left')


def fixed_arms(baseline, strength, additions):
    'both addition arms reuse proposals made against original occupancy b'
    if additions.select('tid').join(baseline.select('tid'), on='tid').height:
        raise ValueError('Additions overlap original baseline occupancy')
    dropped = intersection(baseline, strength)
    return {'B': baseline, 'D': dropped,
            'A': addition_policy.apply_additions(baseline, additions),
            'DA': addition_policy.apply_additions(dropped, additions)}


def _counts(frame):
    addition_policy.validation._labels(frame)
    tp = int(frame['y'].sum() or 0)
    return {'TP': tp, 'FP': frame.height - tp, 'pairs': frame.height}


def effects(after, before, anchors):
    scope = anchors.select('qid')
    added = after.join(before.select(KEYS), on=KEYS, how='anti').join(scope, on='qid', how='semi')
    removed = before.join(after.select(KEYS), on=KEYS, how='anti').join(scope, on='qid', how='semi')
    a, d = _counts(added), _counts(removed)
    res = {'TPadded': a['TP'], 'FPadded': a['FP'], 'TPremoved': d['TP'], 'FPremoved': d['FP'],
              'added_pairs': a['pairs'], 'removed_pairs': d['pairs'], 'country': {}}
    for country in sorted(anchors['co'].unique()):
        ids = anchors.filter(pl.col('co') == country).select('qid')
        ca, cd = _counts(added.join(ids, on='qid', how='semi')), _counts(removed.join(ids, on='qid', how='semi'))
        res['country'][country] = {'TPadded': ca['TP'], 'FPadded': ca['FP'],
            'TPremoved': cd['TP'], 'FPremoved': cd['FP'], 'added_pairs': ca['pairs'], 'removed_pairs': cd['pairs']}
    return res


def evaluate_arms(arms, populations):
    report = {}
    for name, anchors in populations.items():
        if not anchors.height:
            raise ValueError('Empty evaluation population: ' + name)
        report[name] = {arm: {'metrics': anchor_metrics(rows, anchors),
                              'effects_vs_B': effects(rows, arms['B'], anchors)}
                        for arm, rows in arms.items()}
        report[name]['DA']['effects_vs_D'] = effects(arms['DA'], arms['D'], anchors)
    return report


def combined_guard(evaluation):
    'one fixed export guard; never selects or modifies a threshold'
    base, dropped, combined = (evaluation[k]['metrics'] for k in ('B', 'D', 'DA'))
    checks = {'strict_macro_gain_vs_B': combined['overall']['macro_f05'] > base['overall']['macro_f05'],
              'macro_at_least_D': combined['overall']['macro_f05'] >= dropped['overall']['macro_f05'] - 1e-12}
    for region in ['overall'] + sorted(base['country']):
        old = base['overall'] if region == 'overall' else base['country'][region]
        now = combined['overall'] if region == 'overall' else combined['country'][region]
        for metric in METRICS:
            checks[region + '_' + metric + '_nondecreasing'] = now[metric] >= old[metric] - 1e-12
    return {'passed': all(checks.values()), 'checks': checks,
            'failure_reasons': [key for key, passed in checks.items() if not passed]}


def parse_sources(incumbent_report, strength_report, selected_policy, additions_report):
    if incumbent_report.get('matching_sha256') != INCUMBENT_SHA or incumbent_report.get('selected') != METHOD:
        raise ValueError('Wrong confirmed incumbent report')
    if strength_report.get('matching_sha256') != STRENGTH_SHA:
        raise ValueError('Wrong saved strength submission report')
    locked = strength_report.get('locked_candidate')
    if not isinstance(locked, dict) or locked != selected_policy or locked.get('strength') != 1.0 or locked.get('policy') != {'rule': 'expected_f', 'threshold': .8}:
        raise ValueError('Saved strength checkpoint/policy differs from frozen 1.0 expected-F floor .8')
    sel = additions_report.get('selected')
    exp = {'family': 'contrastive', 'head': 'treatment',
                'rule': {'enabled': True, 'min_alias': .99, 'min_margin': .05}}
    if sel != exp:
        raise ValueError('Frozen additions selection differs from completed experiment')
    return locked, sel


def _read(path):
    return json.loads(Path(path).read_text())


def _frozen_additions(frame, refs, targets, baseline, model, features, selected):
    scored, _ = addition_policy._build_score(selected['family'], frame, refs, targets,
        {selected['head']: model}, {selected['head']: features}, None)
    return addition_policy.propose_additions(scored[selected['head']], baseline,
        addition_policy._country(refs), selected['rule'])


def run(data, train, test, incumbent, strength, additions, output):
    started = time.monotonic()
    data, train, test, incumbent, strength, additions, output = map(Path,
        (data, train, test, incumbent, strength, additions, output))
    for src in (incumbent, strength, additions):
        if output.resolve() == src.resolve() or src.resolve() in output.resolve().parents:
            raise ValueError('Review output must be separate from every saved source')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Review output must be empty')
    inc_report, strength_report = _read(incumbent/'report.json'), _read(strength/'report.json')
    additions_report = _read(additions/'experiment/report.json')
    locked, sel = parse_sources(inc_report, strength_report, _read(strength/'selected_policy.json'), additions_report)
    if _read(additions/'experiment/rule.json')['selected'] != sel:
        raise ValueError('Saved additions rule differs from report')
    for root, exp in [(incumbent, INCUMBENT_SHA), (strength, STRENGTH_SHA)]:
        if sha256(root/'output/matching_results.tsv') != exp:
            raise ValueError('Source TSV hash differs: ' + str(root))
    ff = _read(train.with_name('features-train.json'))
    if ff != _read(test.with_name('features-test.json')):
        raise ValueError('Prepared feature schemas differ')
    frozen_features = _read(additions/'experiment/features.json')
    if not isinstance(frozen_features, list):
        raise ValueError('Selected additions features must be a frozen ordered array')
    model = lgb.Booster(model_file=str(additions/'experiment/selected_model.txt'))
    if model.feature_name() != frozen_features:
        raise ValueError('Selected additions model/features differ')
    saved_cal = _read(strength/'selected_calibration.json')
    output.mkdir(parents=True, exist_ok=True)
    report = {'experimental': True, 'automatic_recommendation': False,
              'training_performed': False, 'threshold_search_performed': False,
              'fixed_arms': {'B': 'Confirmed .98805 baseline', 'D': 'Exact pair intersection B with saved strength',
                             'A': 'B plus frozen unoccupied-target additions', 'DA': 'D plus the identical additions computed against B'},
              'strength_policy': locked, 'additions_selection': sel,
              'additions_original_promoted': additions_report['promoted'],
              'additions_ablation_authorized_despite_original_rejection': True,
              'historical_exposure_warning': WARNING, 'no_French_labels_or_policy_changes': True,
              'provenance': {}}
    for name, root, files in [('incumbent', incumbent, ['report.json', 'residual_model.txt', 'output/matching_results.tsv']),
        ('strength', strength, ['report.json', 'selected_model.txt', 'selected_policy.json', 'selected_calibration.json', 'output/matching_results.tsv']),
        ('additions', additions, ['report.json', 'experiment/report.json', 'experiment/selected_model.txt', 'experiment/features.json', 'experiment/rule.json'])]:
        report['provenance'][name] = {'path': str(root.resolve()), 'sha256': {file: sha256(root/file) for file in files}}
    log('drop review: replaying exact incumbent training assignments')
    refs, targets = load_refs(str(data), 'train'), load_targets(str(data), 'train')
    tr = pl.read_parquet(train)
    assert_same_pool(tr, incumbent/'validation_predictions.parquet')
    inc_model = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
    tr, top, base = replay(tr, inc_model, ff, inc_report, 'train')
    del top, inc_model
    _, tune_ids = fit_population(refs)
    known = refs.filter(pl.col('co').is_in(['us', 'india']))
    original_tune = known.filter(pl.col('rid').is_in(pl.Series(tune_ids).implode())).select(pl.col('rid').alias('qid'), 'deg', 'co')
    assert_metric_replay(anchor_metrics(base, original_tune)['overall'], strength_report['incumbent_tune_replay'], 'incumbent original tune')
    assert_metric_replay(anchor_metrics(base, original_tune)['overall'], inc_report['methods'][METHOD]['tune'], 'incumbent saved tune')
    log('drop review: replaying saved strength 1.0 and frozen calibration/decoder')
    strength_model = lgb.Booster(model_file=str(strength/'selected_model.txt'))
    raw = sigmoid(logit(tr['newest'].fill_null(1e-4).to_numpy()) +
                  locked['strength'] * predict_residual(strength_model, tr, ff)).astype(np.float32)
    strength_train = apply_policy(top_pairs(compact(tr, raw, saved_cal['train'])), locked['policy'])
    assert_anchor_replay(anchor_metrics(strength_train, original_tune),
        strength_report['candidate_all_previous_tune_descriptive'], 'strength original tune')
    del raw, strength_model
    tr = _trim(tr)
    gc.collect()
    log('drop review: computing frozen additions against original global occupancy B')
    added = _frozen_additions(tr, refs, targets, base, model, frozen_features, sel)
    arms = fixed_arms(base, strength_train, added)
    check_refs = split_references(refs)[2].filter(pl.col('co').is_in(['us', 'india']))
    populations = {'original_tune': original_tune,
        'structured_check': check_refs.select(pl.col('rid').alias('qid'), 'deg', 'co'),
        'fold1_all': known.filter(pl.col('fold') == 1).select(pl.col('rid').alias('qid'), 'deg', 'co')}
    report['evaluation'] = evaluate_arms(arms, populations)
    report['combined_export_guard'] = combined_guard(report['evaluation']['structured_check'])
    added.write_parquet(output/'frozen_train_additions.parquet')
    write_json(output/'selection.json', report)
    log('drop review: fixed arm evaluations complete; loading authoritative test assignments')
    del tr, refs, targets, base, strength_train, arms, added
    gc.collect()
    rt, tt = load_refs(str(data), 'test'), load_targets(str(data), 'test')
    te = pl.read_parquet(test)
    assert_same_pool(te, incumbent/'test_predictions.parquet')
    if te.height != strength_report['validation']['candidate_pairs']:
        raise ValueError('Saved strength full test candidate count differs')
    btest = pl.read_parquet(incumbent/'accepted_test.parquet').select('qid', 'tid', 'p')
    stest = pl.read_parquet(strength/'accepted_test.parquet').select('qid', 'tid', 'p')
    dtest = intersection(btest, stest)
    delta = changes(dtest, btest, rt)
    if (dtest.height != 5_831_055 or delta['removed_pairs'] != 11_161 or delta['added_pairs'] or
        delta['removed_by_country'] != {'us': 6508, 'india': 4653}):
        raise ValueError('Authoritative drop intersection differs from 11,161 known-country drops: ' + str(delta))
    protected = rt.filter(~pl.col('co').is_in(['us', 'india'])).select(pl.col('rid').alias('qid'))
    if not btest.join(protected, on='qid', how='semi').equals(dtest.join(protected, on='qid', how='semi')):
        raise ValueError('Drop intersection changed France/protected rows')
    from validate_outputs import validate
    def export_arm(name, accepted):
        log('drop review: exporting ' + name + ' with the full candidate pool')
        dst = output/name
        details = export(te.select('qid', 'tid').with_columns(pl.lit(0.).alias('p')), 'p',
            'fixed_ablation', 0., rt, tt, str(dst/'output'), accepted=accepted)
        accepted.write_parquet(dst/'accepted_test.parquet')
        return {'experimental': name == 'experimental_combined', 'export': details,
                'validation': validate(data, dst/'output'),
                'matching_sha256': sha256(dst/'output/matching_results.tsv'),
                'changes_vs_B': changes(accepted, btest, rt)}
    report['drop_only'] = export_arm('drop_only', dtest)
    if report['drop_only']['matching_sha256'] != DROP_ONLY_SHA:
        raise ValueError('Cloud drop-only TSV differs from the independently reconstructed local intersection')
    log('drop review: requested exact drop-only export complete')
    if report['combined_export_guard']['passed']:
        log('drop review: computing frozen test additions against original B occupancy')
        inc_model = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
        te, top, replayed = replay(te, inc_model, ff, inc_report, 'test')
        del top, replayed, inc_model
        added_test = _frozen_additions(_trim(te), rt, tt, btest, model, frozen_features, sel)
        combined = addition_policy.apply_additions(dtest, added_test)
        added_test.write_parquet(output/'frozen_test_additions.parquet')
        report['experimental_combined'] = export_arm('experimental_combined', combined)
        report['experimental_combined']['same_additions_against_B'] = added_test.height
    else:
        report['experimental_combined'] = {'exported': False,
            'reason': 'Fixed structured-check export guard failed',
            'failure_reasons': report['combined_export_guard']['failure_reasons']}
    report['elapsed_seconds'] = time.monotonic() - started
    report['submission_recommendation'] = 'Requested exact drop-only artifact; combined export, if present, is EXPERIMENTAL and carries no automatic recommendation.'
    write_json(output/'report.json', report)
    log('drop review complete; requested drop-only full-pool validator passed')
    return report
