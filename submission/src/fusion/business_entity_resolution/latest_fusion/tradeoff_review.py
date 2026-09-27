'Replay one saved F0.5/precision trade-off without fitting or policy search'
import gc
import json
import math
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export
from latest_fusion.calibration import curves, histogram
from latest_fusion.pipeline import log, logit, partition, sha256, sigmoid, write_json
from latest_fusion.review_export import assert_anchor_replay, assert_metric_replay, frozen_test_assignments
from latest_fusion.tune_pipeline import anchor_metrics, compact, predict_residual
from latest_fusion.tuning import apply_policy, fit_population, promotion, split_search_check, top_pairs


def locked_trial(report, trial):
    if report.get('promoted') is not True or report.get('submission_changed') is not True:
        raise ValueError('Source must be the promoted changed comparison model')
    sel = report.get('locked_candidate', {})
    choices = [row for row in report.get('trials', []) if row.get('name') == trial]
    if len(choices) != 1 or trial == sel.get('name'):
        raise ValueError('Select one saved alternate trial, never a newly invented policy')
    cand = choices[0]
    if cand.get('model') != sel.get('model') or not cand.get('model'):
        raise ValueError('Review must reuse the selected checkpoint')
    for row in (sel, cand):
        strength = row.get('strength', float('nan'))
        if not math.isfinite(strength) or strength <= 0 or not isinstance(row.get('policy'), dict):
            raise ValueError('Invalid saved strength or policy')
    return cand


def run(data, train, test, completed, metadata, output, trial, expected_sha):
    started = time.monotonic()
    data, train, test, completed, metadata, output = map(Path, (data, train, test, completed, metadata, output))
    if output.resolve() == completed.resolve() or completed.resolve() in output.resolve().parents:
        raise ValueError('Experimental output must be separate from the source')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Experimental output directory must be empty')
    src = json.loads((completed/'report.json').read_text())
    cand = locked_trial(src, trial)
    baseline = src['locked_candidate']
    if (src['matching_sha256'] != expected_sha or
            sha256(completed/'output/matching_results.tsv') != expected_sha):
        raise ValueError('Wrong comparison submission artifact')
    ff = json.loads(train.with_name('features-train.json').read_text())
    if ff != json.loads(test.with_name('features-test.json').read_text()):
        raise ValueError('Prepared feature schema mismatch')
    saved_cal = json.loads((completed/'selected_calibration.json').read_text())
    if json.loads((completed/'selected_policy.json').read_text()) != baseline:
        raise ValueError('Saved selected policy differs from the report')
    model_path = completed/'selected_model.txt'
    model = lgb.Booster(model_file=str(model_path))
    refs = load_refs(str(data), 'train')
    _, tuneids = fit_population(refs)
    search, check = split_search_check(refs, tuneids)
    all_tune = pl.concat([search, check])
    tr = pl.read_parquet(train)
    if tr.select('qid', 'tid').n_unique() != tr.height:
        raise ValueError('Duplicate prepared candidate pairs')
    log(f'replaying saved {trial} against {baseline["name"]}; no fitting or threshold search')
    residual = predict_residual(model, tr, ff)
    base = logit(tr['newest'].fill_null(1e-4).to_numpy())
    baseline_raw = sigmoid(base+baseline['strength']*residual).astype(np.float32)
    candidate_raw = sigmoid(base+cand['strength']*residual).astype(np.float32)
    del base, residual
    calibration_refs = refs.filter((pl.col('fold') == 0) & pl.Series(partition(refs['rid'].to_numpy()) == 1))
    ncal = dict(calibration_refs.group_by('co').len().rows())
    edges = np.asarray(json.loads((metadata.parent/'latest_fusion_calibration.json').read_text())['edges'])
    mask = (tr['fold'].to_numpy() == 0) & (partition(tr['qid'].to_numpy()) == 1)
    co, seg, y = (tr[c].to_numpy() for c in ('co', 'seg', 'y'))
    baseline_hist = histogram(baseline_raw, co, seg, y, edges, mask)
    candidate_hist = histogram(candidate_raw, co, seg, y, edges, mask)
    recipe = {'curves': curves(candidate_hist, candidate_hist, ncal, ncal, edges, transfer=False)}
    before = apply_policy(top_pairs(compact(tr, baseline_raw, saved_cal['train'])), baseline['policy'])
    after = apply_policy(top_pairs(compact(tr, candidate_raw, recipe)), cand['policy'])
    assert_anchor_replay(anchor_metrics(before, search), baseline['search'], 'selected search')
    assert_anchor_replay(anchor_metrics(after, search), cand['search'], 'saved alternate search')
    baseline_check = anchor_metrics(before, check)
    assert_metric_replay(baseline_check['overall'], src['check']['candidate'], 'selected check')
    for country, entry in src['check']['countries'].items():
        assert_metric_replay(baseline_check['country'][country], entry['candidate'], 'selected check/'+country)
    assert_anchor_replay(anchor_metrics(before, all_tune), src['candidate_all_previous_tune_descriptive'], 'selected full tune')
    res = {'experimental': True, 'promoted': False, 'training_performed': False,
        'new_threshold_search': False, 'source_matching_sha256': expected_sha,
        'source_report_sha256': sha256(completed/'report.json'), 'source_model_sha256': sha256(model_path),
        'reference_candidate': baseline, 'reviewed_trial': cand,
        'original_search_rejection_checks': [name for name, passed in cand['search_checks'].items() if not passed],
        'candidate_check': anchor_metrics(after, check), 'reference_check': baseline_check,
        'candidate_all_tune_descriptive': anchor_metrics(after, all_tune),
        'reference_all_tune_descriptive': src['candidate_all_previous_tune_descriptive'],
        'original_0p98805_check_metrics': src['check']['incumbent'],
        'check_vs_current_strict': promotion(after, before, check, family_size=5, max_precision_loss=0, max_recall_loss=0),
        'limitations': ['Single saved trial replayed at user request; the common check is historically reused, not a new blind test.',
            'Original no-precision-loss rejection remains recorded. This export is an explicitly labeled precision/F0.5 trade-off.',
            'Confidence intervals are approximate and do not undo adaptive historical selection or shared-owner dependence.',
            'No French labels or French policy changes; no leaderboard gain is assumed.']}
    target_mask = tr['qid'].is_in(check['qid'].implode()).to_numpy()
    ncheck = dict(check.group_by('co').len().rows())
    density = {}
    for name, raw, h, policy in [('reference', baseline_raw, baseline_hist, baseline['policy']),
                               ('candidate', candidate_raw, candidate_hist, cand['policy'])]:
        live = histogram(raw, co, seg, None, edges, target_mask)
        transfer_recipe = {'curves': curves(h, live, ncal, ncheck, edges, transfer=True)}
        accepted = apply_policy(top_pairs(compact(tr, raw, transfer_recipe)), policy)
        density[name] = anchor_metrics(accepted, check)
    assert_anchor_replay(density['reference'], src['density_replay_check']['candidate'], 'selected density')
    res['density_replay'] = density
    log(f'fixed trial check F0.5 {res["candidate_check"]["overall"]["macro_f05"]:.9f}; reference {baseline_check["overall"]["macro_f05"]:.9f}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'selection.json', res)
    del tr, before, after, accepted, baseline_raw, candidate_raw, co, seg, y
    gc.collect()
    te = pl.read_parquet(test)
    if te.height != src['validation']['candidate_pairs']:
        raise ValueError('Prepared test candidate pool changed')
    rt, targets = load_refs(str(data), 'test'), load_targets(str(data), 'test')
    raw = sigmoid(logit(te['newest'].fill_null(1e-4).to_numpy())+
                  cand['strength']*predict_residual(model, te, ff)).astype(np.float32)
    cot, segt = te['co'].to_numpy(), te['seg'].to_numpy()
    live = histogram(raw, cot, segt, None, edges, np.isin(cot, list(ncal)))
    test_recipe = {'curves': curves(candidate_hist, live, ncal, dict(rt.group_by('co').len().rows()), edges, transfer=True)}
    previous_acc = pl.read_parquet(completed/'accepted_test.parquet').select('qid', 'tid', 'p')
    accepted, scored, delta = frozen_test_assignments(te, raw, test_recipe, rt, previous_acc, cand['policy'], ncal)
    res['export'] = export(te.select('qid', 'tid').with_columns(pl.Series('p', raw)), 'p',
        cand['policy']['rule'], cand['policy'].get('threshold', 0), rt, targets, str(output/'output'), accepted=accepted)
    accepted.write_parquet(output/'accepted_test.parquet')
    scored.write_parquet(output/'known_country_test_predictions.parquet')
    write_json(output/'selected_calibration.json', {'train': recipe, 'test': test_recipe})
    write_json(output/'reviewed_policy.json', cand)
    from validate_outputs import validate
    res.update(changes_vs_current=delta, submission_changed=bool(delta['added_pairs'] or delta['removed_pairs']),
        validation=validate(data, output/'output'), matching_sha256=sha256(output/'output/matching_results.tsv'),
        elapsed_seconds=time.monotonic()-started,
        submission_recommendation='Experimental 1.25 trade-off export for review; compare held-out and density results with the recommended 1.0 file before using a submission.')
    write_json(output/'report.json', res)
    log(f'fixed trade-off export complete; changed={res["submission_changed"]}; validator PASS')
    return res
