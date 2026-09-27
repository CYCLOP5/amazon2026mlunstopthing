'Train new representations/retrieval/ensembles and re-decode full ownership'
import gc
import importlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export
from final_repair.pipeline import changes
from . import innovation_models as learning
from .pipeline import log, logit, sigmoid, sha256, write_json
from .structured_pipeline import INCUMBENT_SHA, METHOD, replay, assert_same_pool, assert_same_pairs, _s2_count
from .structured_protocol import prepare_protocol, WARNING
from .tune_pipeline import anchor_metrics
from .tuning import apply_policy, top_pairs, select_policy, promotion, fit_population

MODES = ('reciprocal_retrieval', 'collective_graph', 'gated_ensemble')
BLENDS = (.5, 1.)
METRICS = ('macro_f05', 'pair_precision', 'pair_recall')


def anchors(refs):
    return refs.filter(pl.col('co').is_in(['us', 'india'])).select(pl.col('rid').alias('qid'), 'deg', 'co')


def preservation(candidate, baseline, owners, minimum_gain=0.):
    before, after = anchor_metrics(baseline, owners), anchor_metrics(candidate, owners)
    checks = {'gain': after['overall']['macro_f05'] >= before['overall']['macro_f05']+minimum_gain-1e-12}
    for name, old, new in [('overall', before['overall'], after['overall'])] + [
            (co, before['country'][co], after['country'][co]) for co in before['country']]:
        for metric in METRICS:
            checks[name+'_'+metric] = new[metric] >= old[metric]-1e-12
    return {'before': before, 'after': after, 'checks': checks, 'passed': all(checks.values())}


def score_frame(frame, probability, strength):
    probability = np.asarray(probability, dtype=np.float32)
    if probability.shape != (frame.height,) or not np.isfinite(probability).all() or \
            ((probability < 0) | (probability > 1)).any():
        raise ValueError('Invalid innovation model predictions')
    raw = sigmoid((1-strength)*logit(frame['baseline_p'].to_numpy())+strength*logit(probability))
    if '_innovation_new' in frame:
        novel = frame['_innovation_new'].to_numpy().astype(bool)

        raw[novel] = probability[novel]
    return frame.select('qid', 'tid', 'co', *[c for c in ('y', 'own') if c in frame]).with_columns(
        pl.Series('p', raw.astype(np.float32)), pl.Series('raw', raw.astype(np.float32)))


def select_model_policy(frame, probability, baseline, owners):
    trials, options = [], []
    for strength in BLENDS:
        scored = score_frame(frame, probability, strength)
        top = top_pairs(scored).filter(pl.col('co').is_in(['us', 'india']))
        feasible = lambda candidate, refs: preservation(candidate, baseline, refs, 1e-5)['passed']
        policy, metric = select_policy(top, owners, expected_f_floors=(.5, .75, .9), feasibility=feasible)
        accepted = apply_policy(top, policy)
        evidence = preservation(accepted, baseline, owners, 1e-5)
        trial = {'strength': strength, 'policy': policy, 'metrics': metric, 'preservation': evidence}
        trials.append(trial)
        options.append((trial, accepted))
    eligible = [entry for entry in options if entry[0]['preservation']['passed']]


    chosen, accepted = max(eligible or options, key=lambda entry: entry[0]['metrics']['macro_f05'])
    return chosen, accepted, trials


def transform(mode, frame, refs, targets, baseline, s2_count):
    if mode == 'gated_ensemble':
        return frame, [], {'kind': 'contextual ensemble on original full feature pool', 'new_pairs': 0}
    module = importlib.import_module('latest_fusion.'+(
        'innovation_retrieval' if mode == 'reciprocal_retrieval' else 'innovation_graph'))
    fn = module.expand if mode == 'reciprocal_retrieval' else module.transform
    augmented, extra, diagnostic = fn(frame, refs, targets, baseline, s2_count)
    if augmented.select('qid', 'tid').n_unique() != augmented.height:
        raise ValueError('Innovation introduced duplicate candidate pairs')
    if frame.select('qid', 'tid').join(augmented.select('qid', 'tid'), on=['qid', 'tid'], how='anti').height:
        raise ValueError('Innovation lost an original candidate pair')
    if mode == 'reciprocal_retrieval':
        old = frame.select('qid', 'tid').with_columns(pl.lit(False).alias('_innovation_new'))
        augmented = augmented.join(old, on=['qid', 'tid'], how='left', validate='1:1', maintain_order='left').with_columns(
            pl.col('_innovation_new').fill_null(True))
        novel = augmented.filter(pl.col('_innovation_new'))
        diagnostic = {**diagnostic, 'genuinely_new_pairs': novel.height,
                      'new_true_pairs': int(novel['y'].sum()) if 'y' in novel else None}
    return augmented, extra, diagnostic


def load(data, split):
    refs, targets = load_refs(str(data), split), load_targets(str(data), split)
    if split == 'train':
        truth = pl.concat([pl.read_parquet(Path(data)/split/f's{s}.parquet', columns=['rid', 'own']) for s in (2, 3)])
        targets = targets.join(truth, on='rid', how='left', validate='1:1', maintain_order='left')
    return refs, targets


def run(data, train, test, incumbent, output, mode, expected_sha=INCUMBENT_SHA):
    if mode not in MODES:
        raise ValueError('Unknown model innovation: '+str(mode))
    started = time.monotonic()
    data, train, test, incumbent, output = map(Path, (data, train, test, incumbent, output))
    if output.resolve() in {incumbent.resolve(), data.resolve(), train.parent.resolve(), test.parent.resolve()}:
        raise ValueError('Experiment output must not overwrite an input artifact')
    if (output/'report.json').exists():
        raise ValueError('Experiment output already contains a completed report')
    output.mkdir(parents=True, exist_ok=True)
    prev = json.loads((incumbent/'report.json').read_text())
    if prev['selected'] != METHOD or prev['matching_sha256'] != expected_sha or \
            sha256(incumbent/'output'/'matching_results.tsv') != expected_sha:
        raise ValueError('Incumbent differs from confirmed leaderboard benchmark')
    original_features = json.loads(train.with_name('features-train.json').read_text())
    if original_features != json.loads(test.with_name('features-test.json').read_text()):
        raise ValueError('Prepared feature schemas differ')
    residual = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
    refs, targets = load(data, 'train')
    tr = pl.read_parquet(train)
    assert_same_pool(tr, incumbent/'validation_predictions.parquet')
    log(f'{mode}: replay exact fusion benchmark over {tr.height:,} train pairs')
    tr, _, baseline_train = replay(tr, residual, original_features, prev, 'train')
    _, tuneids = fit_population(refs)
    tune = refs.filter(pl.col('rid').is_in(pl.Series(tuneids).implode())).select(pl.col('rid').alias('qid'), 'deg', 'co')
    replay_metric = decode.score(baseline_train, tune)
    if abs(replay_metric['macro_f05']-prev['methods'][METHOD]['tune']['macro_f05']) > 1e-10:
        raise ValueError('Benchmark score replay mismatch')
    old_coverage = int(tr['y'].sum())
    tr, added, preparation = transform(mode, tr, refs, targets, baseline_train, _s2_count(data, 'train'))
    ff = list(dict.fromkeys(original_features+['baseline_p']+added))
    if 'y' not in tr or tr['y'].null_count() or tr['own'].null_count():
        raise ValueError('Expanded training pool lacks real truth annotations')

    fitrefs, calrefs, checkrefs, fitting, protocol = prepare_protocol(refs, tr)
    fitting = fitting.filter(pl.col('co').is_in(['us', 'india']))
    cal, check = anchors(calrefs), anchors(checkrefs)
    if not cal.height or not check.height:
        raise ValueError('No isolated calibration/check anchors')
    write_json(output/'protocol.json', protocol)
    preparation.update(old_covered_true_pairs=old_coverage,
        new_covered_true_pairs=int(tr['y'].sum()),
        candidate_oracle_on_check=decode.score(tr.filter(pl.col('y') == 1).select('qid', 'tid', 'y'), check))
    write_json(output/'train_preparation.json', preparation)
    log(f'{mode}: {len(ff)} model features; {fitting.height:,} isolated fitting rows')
    bundle = learning.fit(mode, fitting, ff, output/'models')
    prob = learning.predict(bundle, tr)
    sel, candidate_train, trials = select_model_policy(tr, prob, baseline_train, cal)
    locked = {'mode': mode, 'selected': sel, 'calibration_trials': trials,
              'locked_before_check': True, 'candidate_actions': 'Full target ownership and abstention re-decoding'}
    write_json(output/'locked_policy.json', locked)
    check_report = promotion(candidate_train, baseline_train, check, min_gain=1e-5,
        max_precision_loss=0., max_recall_loss=0., family_size=3)
    strict = preservation(candidate_train, baseline_train, check, 1e-5)
    promoted = bool(sel['preservation']['passed'] and check_report['passed'] and strict['passed'])
    check_report['strict_country_preservation'] = strict
    check_report['calibration_eligible'] = sel['preservation']['passed']
    check_report['promoted'] = promoted
    original_tune = {'baseline': replay_metric, 'candidate': decode.score(candidate_train, tune)}
    log(f'{mode}: check F0.5 {check_report["incumbent"]["macro_f05"]:.9f} -> '
        f'{check_report["candidate"]["macro_f05"]:.9f}; promoted={promoted}')
    score_frame(tr, prob, sel['strength']).select('qid', 'tid', 'p', 'y').write_parquet(output/'validation_predictions.parquet')
    write_json(output/'training_result.json', {'mode': mode, 'promoted': promoted,
        'check': check_report, 'selection': locked, 'model': bundle['report'], 'original_tune': original_tune})
    del tr, fitting, refs, targets, fitrefs, calrefs, checkrefs, baseline_train, candidate_train, prob
    gc.collect()
    rt, tt = load(data, 'test')
    te = pl.read_parquet(test)
    assert_same_pool(te, incumbent/'test_predictions.parquet')
    te, _, replayed = replay(te, residual, original_features, prev, 'test')
    baseline_test = pl.read_parquet(incumbent/'accepted_test.parquet').select('qid', 'tid', 'p')
    known = rt.filter(pl.col('co').is_in(['us', 'india'])).select(pl.col('rid').alias('qid'))
    assert_same_pairs(replayed.join(known, on='qid', how='semi'), baseline_test.join(known, on='qid', how='semi'), 'Known-country test')
    del replayed, residual
    te, test_added, test_preparation = transform(mode, te, rt, tt, baseline_test, _s2_count(data, 'test'))
    if added != test_added:
        raise ValueError('Innovation train/test feature order changed')
    write_json(output/'test_preparation.json', test_preparation)
    prob = learning.predict(bundle, te)
    scored = score_frame(te, prob, sel['strength'])
    top = top_pairs(scored).filter(pl.col('co').is_in(['us', 'india']))
    cand = apply_policy(top, sel['policy']).select('qid', 'tid', 'p')

    frozen = baseline_test.join(known, on='qid', how='anti')
    accepted = pl.concat([cand, frozen], how='vertical_relaxed')
    if accepted['tid'].n_unique() != accepted.height:
        raise ValueError('Full decoder assigned multiple owners to a target')
    if accepted.select('qid', 'tid').join(te.select('qid', 'tid'), on=['qid', 'tid'], how='anti').height:
        raise ValueError('Model output includes a pair outside expanded pool')
    delta = changes(accepted, baseline_test, rt)
    if delta['added_by_country'].get('france', 0) or delta['removed_by_country'].get('france', 0):
        raise ValueError('Unexpected French policy change')
    scored.select('qid', 'tid', 'p').write_parquet(output/'test_predictions.parquet')
    accepted.write_parquet(output/'accepted_test.parquet')
    export(scored, 'p', 'full_model_innovation', 0., rt, tt, str(output/'output'), accepted=accepted)
    from validate_outputs import validate
    val = validate(data, output/'output')
    report = {'mode': mode, 'promoted': promoted, 'incumbent_sha256': expected_sha,
        'incumbent_leaderboard': .98805, 'check': check_report, 'original_tune': original_tune,
        'selection': locked, 'model': bundle['report'], 'protocol': protocol,
        'train_preparation': preparation, 'test_preparation': test_preparation,
        'changes': delta, 'changed': bool(delta['added_pairs'] or delta['removed_pairs']),
        'validation': val, 'matching_sha256': sha256(output/'output'/'matching_results.tsv'),
        'submission_status': 'locally_eligible_requires_review' if promoted else 'experimental_candidate_not_promoted_do_not_submit',
        'output_policy': 'Real model candidate exported even on failed promotion; best benchmark is never overwritten.',
        'historical_exposure': WARNING, 'elapsed_minutes': (time.monotonic()-started)/60}
    write_json(output/'report.json', report)
    log(f'{mode}: candidate export validated; promoted={promoted}; elapsed={report["elapsed_minutes"]:.1f} min')
    return report
