'replay a locked uncertainty-only rejection into a separate experimental tsv'
import gc
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export
from final_repair.pipeline import changes
from latest_fusion.calibration import curves, histogram
from latest_fusion.pipeline import log, logit, partition, sha256, sigmoid, write_json
from latest_fusion.tune_pipeline import anchor_metrics, compact, predict_residual
from latest_fusion.tuning import apply_policy, fit_population, split_search_check, top_pairs

INCUMBENT_SHA = '1d1196df3840ced54b7b1af1f2e7649990d0c6183302f184eddcbda4da1ec00c'
METRICS = ('macro_f05', 'pair_precision', 'pair_recall', 'pairs')


def require_uncertainty_only(report):
    'no empirical precision, recall, country, or density guard may be waived'
    locked = report.get('locked_candidate', {})
    check = report.get('check', {})
    guards = check.get('checks', {})
    if (report.get('promoted') is not False or report.get('submission_changed') is not False or
            report.get('matching_sha256') != INCUMBENT_SHA):
        raise ValueError('Source must be the unchanged incumbent fallback')
    required = {'gain', 'positive_lower_bound', 'precision', 'recall',
                'india_score', 'us_score', 'overall_absolute_precision',
                'india_absolute_precision', 'us_absolute_precision'}
    if not required.issubset(guards) or {key for key, value in guards.items() if value is not True} != {'positive_lower_bound'}:
        raise ValueError('Source failed a guard other than uncertainty')
    if not math.isfinite(check.get('paired_gain', float('nan'))) or check['paired_gain'] <= 0:
        raise ValueError('Source has no positive paired gain')
    interval = check.get('paired_family_interval', [])
    standard_error = check.get('paired_standard_error', float('nan'))
    if (check.get('passed') is not False or len(interval) != 2 or
            not all(math.isfinite(value) for value in interval) or interval[0] > 0 or
            not math.isfinite(standard_error) or standard_error < 0):
        raise ValueError('Missing original uncertainty rejection evidence')
    if set(check.get('countries', {})) != {'india', 'us'}:
        raise ValueError('Incomplete original check country evidence')
    comparisons = [('overall', check)] + list(check['countries'].items())
    for country, evidence in comparisons:
        for metric in METRICS[:-1]:
            cand = evidence.get('candidate', {}).get(metric, float('nan'))
            baseline = evidence.get('incumbent', {}).get(metric, float('nan'))
            if (not math.isfinite(cand) or not math.isfinite(baseline) or
                    cand < baseline-1e-12):
                raise ValueError(f'Original {country} {metric} regressed or is nonfinite')
    density = report.get('density_replay_check', {})
    required_density = {region+'_'+metric for region in ('overall', 'india', 'us')
                        for metric in ('macro_f05', 'precision', 'recall')}
    if (density.get('passed') is not True or not required_density.issubset(density.get('checks', {})) or
            not all(value is True for value in density['checks'].values())):
        raise ValueError('Source failed density/country guards')
    required_search = {'overall_gain'} | {region+'_'+metric for region in ('overall', 'india', 'us')
                                         for metric in ('macro_f05', 'precision', 'recall')}
    if (locked.get('eligible') is not True or not required_search.issubset(locked.get('search_checks', {})) or
            not all(value is True for value in locked['search_checks'].values()) or
            not isinstance(locked.get('policy'), dict)):
        raise ValueError('Source has no eligible locked search candidate')
    strength = locked.get('strength', float('nan'))
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError('Invalid locked strength')
    model = locked.get('model', '')
    if not model or Path(model).name != model:
        raise ValueError('Invalid locked model name')
    return locked


def assert_metric_replay(actual, expected, name):
    'check saved empirical evidence without choosing a new policy'
    for key in METRICS:
        if (key not in expected or key not in actual or
                not math.isfinite(actual[key]) or not math.isfinite(expected[key]) or
                abs(actual[key]-expected[key]) > 1e-10):
            raise ValueError(f'{name} replay mismatch: {key}')


def assert_anchor_replay(actual, expected, name):
    assert_metric_replay(actual['overall'], expected['overall'], name)
    if set(actual['country']) != set(expected['country']):
        raise ValueError(f'{name} replay country mismatch')
    for country in expected['country']:
        assert_metric_replay(actual['country'][country], expected['country'][country], name+'/'+country)


def frozen_test_assignments(frame, raw, recipe, refs, incumbent_accepted, policy, known_countries):
    'original locked known-country decoder plus exact saved french assignments'
    mask = frame['co'].is_in(list(known_countries)).to_numpy()
    scored = compact(frame.filter(pl.Series(mask)), raw[mask], recipe)
    accepted = apply_policy(top_pairs(scored), policy).select('qid', 'tid', 'p')
    protected = refs.filter(~pl.col('co').is_in(list(known_countries)))['rid']
    frozen = incumbent_accepted.filter(pl.col('qid').is_in(protected.implode())).select('qid', 'tid', 'p')
    accepted = pl.concat([accepted, frozen], how='vertical_relaxed')
    if accepted['tid'].n_unique() != accepted.height:
        raise ValueError('Experimental export assigned a target twice')
    delta = changes(accepted, incumbent_accepted, refs)
    for key in ('added_by_country', 'removed_by_country'):
        if any(count for country, count in delta[key].items() if country not in known_countries):
            raise ValueError('Experimental export altered frozen country')
    return accepted, scored, delta


def run(data, train, test, incumbent, completed, metadata, output):
    data, train, test, incumbent, completed, metadata, output = map(Path,
        (data, train, test, incumbent, completed, metadata, output))
    for src in (incumbent, completed):
        if output.resolve() == src.resolve() or src.resolve() in output.resolve().parents:
            raise ValueError('Experimental output must be separate from source artifacts')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Experimental output directory is not empty')
    source_report = json.loads((completed/'report.json').read_text())
    locked = require_uncertainty_only(source_report)
    log(f'locked experimental replay: {locked["name"]}; original uncertainty rejection retained')
    prev = json.loads((incumbent/'report.json').read_text())
    if (prev['selected'] != 'residual_0.5' or prev['matching_sha256'] != INCUMBENT_SHA or
            sha256(incumbent/'output/matching_results.tsv') != INCUMBENT_SHA or
            source_report.get('incumbent_job') != 'amazites-latest-fusion-20260927-01'):
        raise ValueError('Wrong frozen incumbent artifact')
    ff = json.loads(train.with_name('features-train.json').read_text())
    if ff != json.loads(test.with_name('features-test.json').read_text()):
        raise ValueError('Prepared feature schemas differ')
    refs = load_refs(str(data), 'train')
    _, tuneids = fit_population(refs)
    search, check = split_search_check(refs, tuneids)
    tr = pl.read_parquet(train)
    if tr.height != prev['coverage']['train']['union_pairs']:
        raise ValueError('Prepared training row count changed')
    model_path = incumbent/'residual_model.txt' if locked['model'] == 'incumbent' else completed/(locked['model']+'_model.txt')
    model = lgb.Booster(model_file=str(model_path))
    incumbent_model = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
    base = logit(tr['newest'].fill_null(1e-4).to_numpy())
    inc_raw = sigmoid(base+.5*predict_residual(incumbent_model, tr, ff)).astype(np.float32)
    raw = sigmoid(base+locked['strength']*predict_residual(model, tr, ff)).astype(np.float32)
    saved_cal = prev['calibration']['methods']['residual_0.5']
    inc_policy = {'rule': 'top1_threshold', 'threshold': prev['methods']['residual_0.5']['tune']['threshold']}
    incumbent_accepted = apply_policy(top_pairs(compact(tr, inc_raw, saved_cal['train'])), inc_policy)
    assert_metric_replay(decode.score(incumbent_accepted, pl.concat([search, check])), source_report['incumbent_tune_replay'], 'incumbent tune')
    calibration_refs = refs.filter((pl.col('fold') == 0) & pl.Series(partition(refs['rid'].to_numpy()) == 1))
    ncal = dict(calibration_refs.group_by('co').len().rows())
    calmask = (tr['fold'].to_numpy() == 0) & (partition(tr['qid'].to_numpy()) == 1)
    edges = np.asarray(json.loads((metadata.parent/'latest_fusion_calibration.json').read_text())['edges'])
    co, seg, y = (tr[c].to_numpy() for c in ('co', 'seg', 'y'))
    h = histogram(raw, co, seg, y, edges, calmask)
    inc_h = histogram(inc_raw, co, seg, y, edges, calmask)
    recipe = saved_cal['train'] if locked['model'] == 'incumbent' and locked['strength'] == .5 else {'curves': curves(h, h, ncal, ncal, edges, transfer=False)}
    chosen = apply_policy(top_pairs(compact(tr, raw, recipe)), locked['policy'])
    replay_search, replay_check = anchor_metrics(chosen, search), anchor_metrics(chosen, check)
    assert_anchor_replay(replay_search, locked['search'], 'locked search')
    assert_metric_replay(replay_check['overall'], source_report['check']['candidate'], 'locked check')
    baseline_check = anchor_metrics(incumbent_accepted, check)
    assert_metric_replay(baseline_check['overall'], source_report['check']['incumbent'], 'incumbent check')
    for country, values in source_report['check']['countries'].items():
        assert_metric_replay(replay_check['country'][country], values['candidate'], 'locked check/'+country)
        assert_metric_replay(baseline_check['country'][country], values['incumbent'], 'incumbent check/'+country)
    target_mask = tr['qid'].is_in(check['qid'].implode()).to_numpy()
    ncheck = dict(check.group_by('co').len().rows())
    for name, probs, hist, policy in (('candidate', raw, h, locked['policy']), ('incumbent', inc_raw, inc_h, inc_policy)):
        live = histogram(probs, co, seg, None, edges, target_mask)
        density_recipe = {'curves': curves(hist, live, ncal, ncheck, edges, transfer=True)}
        density_accepted = apply_policy(top_pairs(compact(tr, probs, density_recipe)), policy)
        assert_anchor_replay(anchor_metrics(density_accepted, check), source_report['density_replay_check'][name], 'density '+name)
    descriptive = anchor_metrics(chosen, pl.concat([search, check]))
    log('saved search, owner-check and density metrics replayed exactly; exporting locked candidate')
    del tr, chosen, incumbent_accepted, raw, inc_raw, base, co, seg, y
    gc.collect()
    te = pl.read_parquet(test)
    if te.height != prev['coverage']['test']['union_pairs']:
        raise ValueError('Prepared test row count changed')
    rt, targets = load_refs(str(data), 'test'), load_targets(str(data), 'test')
    raw_test = sigmoid(logit(te['newest'].fill_null(1e-4).to_numpy())+
                       locked['strength']*predict_residual(model, te, ff)).astype(np.float32)
    if locked['model'] == 'incumbent' and locked['strength'] == .5:
        target_recipe = saved_cal['test']
    else:
        cot, segt = te['co'].to_numpy(), te['seg'].to_numpy()
        live = histogram(raw_test, cot, segt, None, edges, np.isin(cot, list(ncal)))
        target_recipe = {'curves': curves(h, live, ncal, dict(rt.group_by('co').len().rows()), edges, transfer=True)}
        target_recipe['curves'].update({key: value for key, value in saved_cal['test']['curves'].items() if key.rsplit('|', 1)[0] not in ncal})
    old = pl.read_parquet(incumbent/'accepted_test.parquet').select('qid', 'tid', 'p')
    accepted, scored, delta = frozen_test_assignments(te, raw_test, target_recipe, rt, old, locked['policy'], ncal)
    output.mkdir(parents=True, exist_ok=True)
    details = export(te.select('qid', 'tid').with_columns(pl.Series('p', raw_test)), 'p',
        locked['policy']['rule'], locked['policy'].get('threshold', 0), rt, targets, str(output/'output'), accepted=accepted)
    accepted.write_parquet(output/'accepted_test.parquet')
    scored.write_parquet(output/'known_country_test_predictions.parquet')
    write_json(output/'selected_calibration.json', {'train': recipe, 'test': target_recipe})
    write_json(output/'locked_policy.json', locked)
    from validate_outputs import validate
    report = {'promoted': False, 'experimental': True, 'training_performed': False,
        'source_report_sha256': sha256(completed/'report.json'), 'source_model_sha256': sha256(model_path),
        'incumbent_matching_sha256': INCUMBENT_SHA, 'locked_candidate': locked,
        'original_check': source_report['check'], 'original_density_replay_check': source_report['density_replay_check'],
        'original_failed_guards': ['positive_lower_bound'], 'search_replay': replay_search,
        'check_replay': replay_check, 'candidate_all_tune_descriptive': descriptive,
        'changes': delta, 'submission_changed': bool(delta['added_pairs'] or delta['removed_pairs']),
        'export': details, 'validation': validate(data, output/'output'),
        'matching_sha256': sha256(output/'output/matching_results.tsv'),
        'limitations': ['Experimental locked-candidate export; statistical promotion remains rejected.',
            'Historical upstream exposure and owner dependence limit held-out comparisons.',
            'No threshold selection, fitting, or French decision change occurred in this replay.',
            'No hidden-test or leaderboard gain is guaranteed or estimated.'],
        'submission_recommendation': 'Experimental artifact for explicit review; uncertainty guard was not passed.'}
    write_json(output/'report.json', report)
    log(f'experimental export validated: changed={report["submission_changed"]}; promoted=False')
    return report
