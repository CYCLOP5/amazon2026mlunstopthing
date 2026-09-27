'bounded, controlled experiments over the exact 0.98805 incumbent'
import gc
import importlib
import json
from pathlib import Path
import shutil
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export
from final_repair.pipeline import changes
from latest_fusion.pipeline import log, logit, sha256, sigmoid, write_json
from latest_fusion.structured_protocol import prepare_protocol
from latest_fusion.tune_pipeline import compact, predict_residual, anchor_metrics, error_slices
from latest_fusion.tuning import apply_policy, fit_population, top_pairs

INCUMBENT_SHA = '1d1196df3840ced54b7b1af1f2e7649990d0c6183302f184eddcbda4da1ec00c'
METHOD = 'residual_0.5'
CAMPAIGN_MODES = ('honest_leaf', 'crossview_consensus', 'hard_negative',
                  'sibling_bridge', 'orphan_factor', 'source_corroboration',
                  'monotone_verifier', 'country_consensus')
MODES = ('contrastive', 'set_utility', 'directional', 'additions') + CAMPAIGN_MODES
CORE = ['baseline_p', 'newest_present', 'newest_logit', 'gate_logit',
        'neural_logit', 'graph_logit', 'hybrid_logit', 'friend_logit',
        'name_core_exact', 'name_jaccard', 'name_ref_coverage', 'name_target_coverage',
        'house_equal', 'house_conflict', 'ref_address_empty', 'target_address_empty',
        'ref_name_twins_log', 'target_name_twins_log', 'competitor_count_log',
        'newest_owner_margin', 'gate_owner_margin']


def replay(frame, model, features, previous, split):
    'Preserve raw-score tie breaking and the original train/test calibration'
    residual = predict_residual(model, frame, features)
    raw = sigmoid(logit(frame['newest'].fill_null(1e-4).to_numpy()) + .5*residual).astype(np.float32)
    if split == 'test':
        unseen = ~frame['co'].is_in(['us', 'india']).to_numpy()
        raw[unseen] = frame['gate'].fill_null(0).to_numpy()[unseen]
    scored = compact(frame, raw, previous['calibration']['methods'][METHOD][split])
    top = top_pairs(scored)
    accepted = apply_policy(top, {'rule': 'top1_threshold',
                                 'threshold': previous['methods'][METHOD]['tune']['threshold']})
    return frame.with_columns(scored['p'].alias('baseline_p'), scored['raw'].alias('baseline_raw')), top, accepted


def assert_same_pairs(actual, expected, label):
    keys = ['qid', 'tid']
    if actual.height != expected.height or actual.select(keys).join(expected.select(keys), on=keys, how='anti').height:
        raise ValueError(f'{label}: pair replay differs from incumbent')
    if actual['tid'].n_unique() != actual.height:
        raise ValueError(f'{label}: multiple owners assigned to a target')


def assert_same_pool(frame, saved):
    'compare all candidate keys, including rejected pairs, with the incumbent'
    keys = ['qid', 'tid']
    current = frame.select(keys)
    old = pl.read_parquet(saved, columns=keys)
    if (current.height != old.height or current.unique().height != current.height or
            old.unique().height != old.height or current.join(old, on=keys, how='anti').height):
        raise ValueError('Full candidate pool differs from incumbent score artifact')


def _trim(frame):
    keep = ['qid', 'tid', 'co', 'newest', 'gate', 'baseline_raw', 'y', 'own', 'fold', 'seg'] + CORE
    return frame.select([c for c in dict.fromkeys(keep) if c in frame])


def _s2_count(data, split):
    return pl.scan_parquet(Path(data)/split/'s2.parquet').select(pl.len()).collect().item()


def run(data, train, test, incumbent, output, mode, expected_sha=INCUMBENT_SHA, pretrained_heads=None):
    started = time.monotonic()
    if mode not in MODES:
        raise ValueError(f'Unknown experiment: {mode}')
    data, train, test, incumbent, output = map(Path, (data, train, test, incumbent, output))
    output.mkdir(parents=True, exist_ok=True)
    prev = json.loads((incumbent/'report.json').read_text())
    source_tsv = incumbent/'output'/'matching_results.tsv'
    if prev['selected'] != METHOD or prev['matching_sha256'] != expected_sha or sha256(source_tsv) != expected_sha:
        raise ValueError('Incumbent provenance differs from the confirmed 0.98805 artifact')
    ff = json.loads(train.with_name('features-train.json').read_text())
    if ff != json.loads(test.with_name('features-test.json').read_text()):
        raise ValueError('Prepared train/test feature schema differs')
    module = importlib.import_module('latest_fusion.recall_campaign_experiment'
        if mode in CAMPAIGN_MODES else f'latest_fusion.{mode}_experiment')
    if mode == 'additions' and (not pretrained_heads or set(pretrained_heads) != {'contrastive', 'directional'}):
        raise ValueError('Additions experiment requires both completed head bundles')
    refs, targets = load_refs(str(data), 'train'), load_targets(str(data), 'train')
    tr = pl.read_parquet(train)
    if tr.height != prev['coverage']['train']['union_pairs']:
        raise ValueError('Prepared train candidate count differs')
    assert_same_pool(tr, incumbent/'validation_predictions.parquet')
    if set(CORE)-{'baseline_p'}-set(tr.columns):
        raise ValueError('Prepared train is missing core comparison features')
    model = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
    log(f'{mode}: replaying exact incumbent over {tr.height:,} train candidates')
    tr, top, base_train = replay(tr, model, ff, prev, 'train')
    _, tune_ids = fit_population(refs)
    tune = refs.filter(pl.col('rid').is_in(pl.Series(tune_ids).implode())).select(pl.col('rid').alias('qid'), 'deg', 'co')
    replay_metrics = decode.score(base_train, tune)
    if abs(replay_metrics['macro_f05'] - prev['methods'][METHOD]['tune']['macro_f05']) > 1e-10:
        raise ValueError(f'Incumbent tune replay differs: {replay_metrics}')
    log(f'incumbent replay PASS: original tune F0.5 {replay_metrics["macro_f05"]:.9f}')
    tr = _trim(tr)
    fit_refs, cal_refs, check_refs, fit_frame, protocol = prepare_protocol(refs, tr)
    if not cal_refs.height or not check_refs.height:
        raise ValueError('No usable calibration/check population after target isolation')
    write_json(output/'protocol.json', protocol)
    if mode == 'additions':
        for family, root in pretrained_heads.items():
            src = json.loads((Path(root)/'report.json').read_text())
            if src['incumbent_sha256'] != expected_sha or src['protocol'] != protocol:
                raise ValueError(f'{family}: pretrained head benchmark/protocol differs')
    check_anchors = check_refs.select(pl.col('rid').alias('qid'), 'deg', 'co')
    diagnostics = error_slices(tr, top, base_train, check_anchors)
    write_json(output/'baseline_errors.json', diagnostics)
    write_json(output/'baseline_metrics.json', {'original_tune': replay_metrics,
               'check': anchor_metrics(base_train, check_anchors)})
    log(f'{mode}: isolated {fit_frame.height:,} fitting pairs / {cal_refs.height:,} calibration owners / {check_refs.height:,} check owners')
    tr, fit_frame = _trim(tr), _trim(fit_frame)
    del top
    gc.collect()
    extra = {'pretrained_heads': pretrained_heads} if mode == 'additions' else {}
    if mode in CAMPAIGN_MODES:
        extra['method'] = mode
    bundle = module.run(tr, refs, targets, fit_frame, cal_refs, check_refs, base_train,
                        CORE, output/'experiment', fit_refs=fit_refs, s2_count=_s2_count(data, 'train'), **extra)

    selection = {'mode': mode, 'experiment': bundle['report'],
                 'incumbent_sha256': expected_sha, 'incumbent_leaderboard': .98805,
                 'historical_exposure': protocol['historical_upstream_exposure_warning']}
    write_json(output/'selection.json', selection)
    log(f'{mode}: training/check complete; loading test for frozen inference')
    del tr, fit_frame, refs, targets, fit_refs, cal_refs, check_refs, base_train
    gc.collect()
    rt, tt = load_refs(str(data), 'test'), load_targets(str(data), 'test')
    te = pl.read_parquet(test)
    if te.height != prev['coverage']['test']['union_pairs']:
        raise ValueError('Prepared test candidate count differs')
    assert_same_pool(te, incumbent/'test_predictions.parquet')
    te, top, replay_test = replay(te, model, ff, prev, 'test')
    base_test = pl.read_parquet(incumbent/'accepted_test.parquet').select('qid', 'tid', 'p')
    known = rt.filter(pl.col('co').is_in(['us', 'india'])).select(pl.col('rid').alias('qid'))
    assert_same_pairs(replay_test.join(known, on='qid', how='semi'),
                      base_test.join(known, on='qid', how='semi'), 'Known-country test')
    del top, replay_test, model
    te = _trim(te)
    accepted, inference = module.infer(bundle, te, rt, tt, base_test, s2_count=_s2_count(data, 'test'))
    if accepted['tid'].n_unique() != accepted.height:
        raise ValueError('Experiment assigned a target to more than one owner')
    if accepted.select('qid', 'tid').join(te.select('qid', 'tid'), on=['qid', 'tid'], how='anti').height:
        raise ValueError('Experiment introduced a pair outside the candidate pool')
    delta = changes(accepted, base_test, rt)
    changed = bool(delta['added_pairs'] or delta['removed_pairs'])
    if mode != 'directional' and (delta['added_by_country'].get('france', 0) or delta['removed_by_country'].get('france', 0)):
        raise ValueError('IN/US experiment changed a French decision')
    dest = output/'output'
    dest.mkdir(exist_ok=True)
    if changed:
        export(te, 'baseline_p', 'structured_frozen_actions', 0., rt, tt, str(dest), accepted=accepted)
    else:
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            shutil.copyfile(incumbent/'output'/name, dest/name)
        if sha256(dest/'matching_results.tsv') != expected_sha:
            raise ValueError('Unchanged fallback is not byte-identical')
    from validate_outputs import validate
    val = validate(data, dest)
    accepted.write_parquet(output/'accepted_test.parquet')
    report = {**selection, 'protocol': protocol, 'baseline_original_tune': replay_metrics,
              'baseline_check_errors': diagnostics, 'inference': inference, 'changes': delta,
              'changed': changed, 'validation': val,
              'matching_sha256': sha256(dest/'matching_results.tsv'),
              'elapsed_minutes': (time.monotonic()-started)/60,
              'submission_status': 'candidate_requires_review' if changed else 'exact_best_no_new_submission_needed'}
    write_json(output/'report.json', report)
    log(f'{mode} finished: changed={changed}, {accepted.height:,} matches; validator PASS')
    return report
