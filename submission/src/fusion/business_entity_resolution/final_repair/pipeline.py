'one final, bounded cpu experiment: exact reference blend plus identity repair'
import gc
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl

from er.io import write_id_lists
from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export, tuning_anchors
from final_repair import reference
from final_repair.fast_decode import apply as fast_apply
from final_repair.identity import propose

KEYS = ['qid', 'tid']
MIN_TUNE_GAIN = 0.000005
MAX_PRECISION_LOSS = 0.0002
EXPECTED_RERANKER_MATCHES = 5_884_628
START = time.monotonic()
STAGE = 'initializing'


def log(message):
    print(f'[{(time.monotonic()-START)/60:6.1f} min] {message}', flush=True)


def write_json(path, data):
    temp = Path(str(path)+'.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def counts(refs):
    d = dict(refs.group_by('co').len().rows())
    return {**d, '*': refs.height}


def truth(accepted, owners):
    d = accepted.drop('y', strict=False).join(owners, on='tid', how='left', validate='m:1')
    if d['own'].null_count():
        raise ValueError('Prediction refers to target absent from labels')
    return d.with_columns((pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8).alias('y')).drop('own')


def resolve_proposals(proposals):
    'no arbitrary tie breaking between business owners, even across rules'
    if not proposals.height:
        return proposals
    ambiguous = proposals.group_by('tid').agg(pl.col('qid').n_unique().alias('n')).filter(pl.col('n') > 1)
    return proposals.join(ambiguous.select('tid'), on='tid', how='anti').sort(
        ['tid', 'rank', 'rule']).unique('tid', keep='first', maintain_order=True)


def actions_for(proposals, accepted, refs):
    'only explicit proposals; protect confidently accepted competing owners'
    proposals = resolve_proposals(proposals)
    old = accepted.select('tid', pl.col('qid').alias('old_qid'), pl.col('p').alias('old_p'))
    d = proposals.join(old, on='tid', how='left', validate='1:1').join(
        refs.select(pl.col('rid').alias('qid'), 'co'), on='qid', how='left', validate='m:1')
    d = d.filter(pl.col('old_qid').is_null() | (pl.col('old_qid') != pl.col('qid')))
    d = d.with_columns(pl.when(pl.col('old_qid').is_null()).then(pl.lit('add')).otherwise(pl.lit('replace')).alias('action'))
    d = d.filter((pl.col('action') == 'add') | (
        pl.col('rule').is_in(['full_identity', 'core_identity']) & (pl.col('old_p') < .99)))
    return d.with_columns((pl.col('rule')+'|'+pl.col('action')).alias('cell'))


def apply_actions(accepted, actions):
    if not actions.height:
        return accepted.select(*KEYS, 'p')
    if actions['tid'].n_unique() != actions.height:
        raise ValueError('Multiple repairs proposed for a target')
    keep = accepted.select(*KEYS, 'p').join(actions.select('tid'), on='tid', how='anti')
    add = actions.select(*KEYS).with_columns(pl.lit(1., accepted.schema['p']).alias('p'))
    return pl.concat([keep, add], how='vertical_relaxed')


def wilson_lower(correct, n, z=2.5758293035):
    if not n:
        return 0.
    p = correct/n
    return float((p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))


def fit_repair_risk(actions, owners, refs, fit_qids):
    'action reliability, excluding held-out proposed, true, and old owners'
    d = actions.join(owners, on='tid', how='left', validate='1:1')
    fit_ids = fit_qids['qid']
    d = d.filter(pl.col('qid').is_in(fit_ids.implode()) &
        ((pl.col('own') < 0) | pl.col('own').is_in(fit_ids.cast(pl.Int64).implode())) &
        (pl.col('old_qid').is_null() | pl.col('old_qid').is_in(fit_ids.implode())))
    d = d.with_columns((pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8).alias('correct'))
    rows = d.group_by('co', 'cell', 'action').agg(pl.len().alias('n'), pl.col('correct').sum().alias('correct')).sort('co', 'cell')
    res = {}
    for row in rows.to_dicts():
        lower = wilson_lower(row['correct'], row['n'])
        minimum = .90 if row['action'] == 'add' else .98
        res[row['co']+'|'+row['cell']] = {**row, 'precision': row['correct']/row['n'],
            'wilson_99_lower': lower, 'required_lower': minimum,
            'eligible': row['n'] >= 30 and lower >= minimum}
    return res


def select_repairs(base, actions, owners, refs, fit_ids, tune):
    'fixed small rule set; optimize entity f0.5, not both pair p and r'
    risk = fit_repair_risk(actions, owners, refs, fit_ids)
    policies, trials = {}, {}
    for co in sorted(tune['co'].unique()):
        anchors = tune.filter(pl.col('co') == co).select('qid', 'deg')
        chosen, details = [], []
        current = decode.score(truth(base, owners), anchors)
        cells = actions.filter(pl.col('co') == co).select('cell', 'rank').unique().sort('rank', 'cell')['cell'].to_list()
        for cell in cells:
            evidence = risk.get(co+'|'+cell, {'eligible': False, 'n': 0})
            if not evidence['eligible']:
                details.append({'cell': cell, 'selected': False, 'reason': 'insufficient isolated action reliability', 'fit': evidence})
                continue
            candidate_actions = actions.filter((pl.col('co') == co) & pl.col('cell').is_in(chosen+[cell]))
            cand = apply_actions(base, candidate_actions)
            metric = decode.score(truth(cand, owners), anchors)
            gain = metric['macro_f05']-current['macro_f05']
            accept = gain >= MIN_TUNE_GAIN and metric['pair_precision'] >= current['pair_precision']-MAX_PRECISION_LOSS
            details.append({'cell': cell, 'selected': bool(accept), 'fit': evidence,
                'tune': metric, 'incremental_macro_gain': gain})
            log(f'REPAIR {co} {cell}: fit n={evidence["n"]}, lower={evidence["wilson_99_lower"]:.5f}; tune gain={gain:+.8f}; selected={accept}')
            if accept:
                chosen.append(cell)
                current = metric
        policies[co], trials[co] = chosen, details
    transferable = sorted(set.intersection(*(set(v) for v in policies.values()))) if len(policies) >= 2 else []
    return policies, transferable, {'fit_action_risk': risk, 'trials': trials,
        'transferable_cells': transferable, 'minimum_tune_gain': MIN_TUNE_GAIN,
        'max_pair_precision_loss_per_step': MAX_PRECISION_LOSS,
        'scope': 'new direct repair only; upstream models have historically seen both labeled countries'}


def choose_actions(actions, policies, transferable):
    parts = [actions.filter((pl.col('co') == co) & pl.col('cell').is_in(policies.get(co, transferable)))
             for co in actions['co'].unique().sort()]
    return pl.concat(parts, how='vertical_relaxed') if parts else actions.head(0)


def gate_transfer_bundle(cells, baselines, action_sets, owners, tune):
    'an intersection of good greedy steps need not be a good bundle'
    evidence = {}
    passed = bool(cells)
    for name, base in baselines.items():
        cand = apply_actions(base, action_sets[name].filter(pl.col('cell').is_in(cells)))
        old_labeled, new_labeled = truth(base, owners), truth(cand, owners)
        evidence[name] = {}
        for country in tune['co'].unique().sort():
            anchors = tune.filter(pl.col('co') == country).select('qid', 'deg')
            old, new = decode.score(old_labeled, anchors), decode.score(new_labeled, anchors)
            gain = new['macro_f05']-old['macro_f05']
            ok = gain >= MIN_TUNE_GAIN and new['pair_precision'] >= old['pair_precision']-MAX_PRECISION_LOSS
            evidence[name][country] = {'baseline': old, 'candidate': new, 'macro_gain': gain, 'passes': bool(ok)}
            passed &= bool(ok)
    return cells if passed else [], {'requested_cells': cells, 'passed': bool(passed), 'comparisons': evidence}


def segmented(pairs, refs, targets):
    return reference.segments(pairs.join(refs.select(pl.col('rid').alias('qid'), 'co'), on='qid', validate='m:1'),
        refs.select(pl.col('rid').alias('qid'), 'ad'), targets.select(pl.col('rid').alias('tid'), 'ad'))


def load_scores(friend, graph, split):
    ff = 'val_pred.parquet' if split == 'train' else 'test_pred.parquet'
    gf = 'validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'

    f = pl.scan_parquet(Path(friend)/ff).select(*KEYS, 'p2').filter(pl.col('p2') >= .001).collect()
    g = pl.scan_parquet(Path(graph)/gf).select(*KEYS, 'head').filter(pl.col('head') >= .001).collect()
    return f, g


def country_metrics(accepted, owners, refs, folds=(1,)):
    d = truth(accepted, owners)
    r = refs.filter(pl.col('fold').is_in(folds))
    return {co: decode.score(d, r.filter(pl.col('co') == co).select(pl.col('rid').alias('qid'), 'deg'))
            for co in r['co'].unique().sort()}


def changes(new, old, refs):
    a = new.select(*KEYS).join(old.select(*KEYS), on=KEYS, how='anti')
    b = old.select(*KEYS).join(new.select(*KEYS), on=KEYS, how='anti')
    return {'added_pairs': a.height, 'removed_pairs': b.height,
        'changed_target_owners': a.join(b.select('tid'), on='tid').height,
        'added_by_country': dict(a.join(refs.select(pl.col('rid').alias('qid'), 'co'), on='qid').group_by('co').len().rows()),
        'removed_by_country': dict(b.join(refs.select(pl.col('rid').alias('qid'), 'co'), on='qid').group_by('co').len().rows())}


def write_matches(accepted, refs, targets, path):
    m = accepted.select(*KEYS).join(refs.select(pl.col('rid').alias('qid'), pl.col('eid').alias('s1_id')), on='qid').join(
        targets.select(pl.col('rid').alias('tid'), pl.col('eid').alias('o_id')), on='tid').select('s1_id', 'o_id')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_id_lists(m, refs.sort('rid')['eid'], 'matched_entity_ids', str(path))


def run(data, hybrid, friend, graph, output):
    global STAGE
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(30):
            log('HEARTBEAT: '+STAGE)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        return _run(Path(data), Path(hybrid), Path(friend), Path(graph), output)
    finally:
        stop.set()


def _run(data, hybrid, friend, graph, output):
    global STAGE
    report = {'method': 'exact reference blend plus independently validated direct identity repair',
        'reported_leaderboard': {'previous_hybrid': .983, 'friend_recipe': .9834, 'last_reranker': .982196},
        'leaderboard_provenance': 'User reports; no private labels or portal submission performed.',
        'limits': ['France has no labels; country transfer is a proxy, not measured French accuracy.',
            'The historical audit and upstream models have been reused; this is not a new blind whole-pipeline estimate.',
            'Final reference calibration uses fold1 labels per the supplied recipe; its exact test policy has no independent train audit.',
            'Label-shift calibration assumes stable positive score density; that assumption can fail.',
            'Wilson action bounds are screening heuristics, not macro-score guarantees; same-owner actions can be correlated.',
            'Direct string identity can still merge distinct businesses at the same address. Repairs require empirical selection.']}
    STAGE = 'reading records and frozen score recipes'
    log(STAGE)
    ref_tr = load_refs(data, 'train')
    target_tr = load_targets(data, 'train')
    owners = pl.concat([pl.read_parquet(data/'train'/f's{s}.parquet', columns=['rid', 'own']) for s in (2, 3)])\
        .rename({'rid': 'tid'}).with_columns(pl.col('tid').cast(pl.UInt32))
    sel = reference.selected_decoder(json.loads((friend/'report.json').read_text()))
    if sel['score'] != 'p2':
        raise ValueError('Reference inputs changed: expected selected friend p2. Inspect before running.')
    if sel['threshold'] < .001:
        raise ValueError('Raw friend decoder requires rows removed by the score prefilter')
    frozen = json.loads((hybrid/'report.json').read_text())['decision']
    f, g = load_scores(friend, graph, 'train')
    blended = truth(reference.blend(f, g), owners)
    tr = segmented(blended.filter(pl.col('p') >= .001), ref_tr, target_tr)
    del g, blended
    anchors = ref_tr.select(pl.col('rid').alias('qid'), 'fold', 'co', 'deg')
    tune = tuning_anchors(anchors, {'fit_fold': 0, 'tune_holdout_buckets': 5, 'seed': 42}).join(
        anchors.select('qid', 'co'), on='qid')
    fit_refs = ref_tr.filter(pl.col('fold') == 0).join(tune.select(pl.col('qid').alias('rid')), on='rid', how='anti')
    fit_ids = fit_refs.select(pl.col('rid').alias('qid'))
    eval_refs = ref_tr.filter(pl.col('fold') == 1).vstack(ref_tr.join(tune.select(pl.col('qid').alias('rid')), on='rid'))
    STAGE = 'cross-fitted reference calibration and frozen baseline comparisons'
    log(STAGE)
    tables = reference.fit(tr.join(fit_ids, on='qid'),
        tr.join(eval_refs.select(pl.col('rid').alias('qid')), on='qid').drop('y'), counts(fit_refs), counts(eval_refs))
    base_tr = fast_apply('expected_f', reference.apply(tr, tables).select(*KEYS, 'p'), .05)
    raw_friend_tr = fast_apply(sel['rule'], f.rename({'p2': 'p'}), sel['threshold'])
    h = pl.scan_parquet(hybrid/'validation_predictions.parquet').select(*KEYS, 'p').filter(pl.col('p') >= .001).collect()
    hybrid_tr = fast_apply(frozen['rule'], h, frozen['threshold'])
    report['historical_audit_comparators'] = {'frozen_hybrid': country_metrics(hybrid_tr, owners, ref_tr),
        'raw_friend': country_metrics(raw_friend_tr, owners, ref_tr),
        'reference_calibrated_from_fit_fold0': country_metrics(base_tr, owners, ref_tr)}
    del f, h, hybrid_tr
    write_json(output/'progress.json', report)
    STAGE = 'full-corpus train identity lookup and isolated rule reliability'
    log(STAGE)
    proposal_tr, diagnostic_tr = propose(ref_tr.select('rid', 'nm', 'ad', 'co'), target_tr.select('rid', 'nm', 'ad', 'co'))
    actions = actions_for(proposal_tr, base_tr, ref_tr)
    policies, transferable, evidence = select_repairs(base_tr, actions, owners, ref_tr, fit_ids, tune)


    raw_actions = actions_for(proposal_tr, raw_friend_tr, ref_tr)
    _, raw_transferable, raw_evidence = select_repairs(raw_friend_tr, raw_actions, owners, ref_tr, fit_ids, tune)
    transferable = sorted(set(transferable) & set(raw_transferable))
    transferable, transfer_gate = gate_transfer_bundle(transferable,
        {'calibrated': base_tr, 'raw_friend': raw_friend_tr},
        {'calibrated': actions, 'raw_friend': raw_actions}, owners, tune)
    repair_tr = apply_actions(base_tr, choose_actions(actions, policies, transferable))
    report['repair'] = {'policies': policies, 'unlabeled_country_cells': transferable,
        'calibrated_evidence': evidence, 'raw_friend_evidence': raw_evidence, 'train_candidates': diagnostic_tr,
        'exact_transfer_bundle_gate': transfer_gate,
        'audit_after_frozen_selection': country_metrics(repair_tr, owners, ref_tr),
        'audit_before': country_metrics(base_tr, owners, ref_tr),
        'changes_all_training_candidates': changes(repair_tr, base_tr, ref_tr),
        'candidate_generation_uses_labels': False, 'audit_used_to_choose_rules': False}
    write_json(output/'repair_policy.json', report['repair'])
    write_json(output/'progress.json', report)


    final_hold = tr.join(ref_tr.filter(pl.col('fold') == 1).select(pl.col('rid').alias('qid')), on='qid')
    n_hold = counts(ref_tr.filter(pl.col('fold') == 1))
    del tr, proposal_tr, actions, raw_actions, raw_friend_tr, base_tr, repair_tr, target_tr, owners, ref_tr
    gc.collect()
    STAGE = 'reproducing colleague reference submission with actual test densities'
    log(STAGE)
    refs, targets = load_refs(data, 'test'), load_targets(data, 'test')
    f, g = load_scores(friend, graph, 'test')
    lab_countries = sorted(policies)
    labeled_refs = refs.filter(pl.col('co').is_in(lab_countries))
    unlabeled_refs = refs.filter(~pl.col('co').is_in(lab_countries))
    blend_live = reference.blend(f, g).join(labeled_refs.select(pl.col('rid').alias('qid')), on='qid')
    live = segmented(blend_live.filter(pl.col('p') >= .001), labeled_refs, targets)
    final_tables = reference.fit(final_hold, live, n_hold, counts(labeled_refs))
    calibrated = reference.apply(live, final_tables).select(*KEYS, 'p')
    acc_labeled = fast_apply('expected_f', calibrated, .05)
    raw_unlabeled = f.join(unlabeled_refs.select(pl.col('rid').alias('qid')), on='qid').rename({'p2': 'p'})
    acc_unlabeled = fast_apply(sel['rule'], raw_unlabeled, sel['threshold'])
    reference_test = pl.concat([acc_labeled, acc_unlabeled], how='vertical_relaxed')
    write_json(output/'calibration.json', {'tables': final_tables, 'unlabeled_decoder': sel,
        'validation_tables': tables, 'calibration_label_fold': 1})
    del f, g, blend_live, live, calibrated, raw_unlabeled, final_hold
    gc.collect()
    STAGE = 'full French/US/India direct identity retrieval and selected repair'
    log(STAGE)
    proposals, diagnostics = propose(refs.select('rid', 'nm', 'ad', 'co'), targets.select('rid', 'nm', 'ad', 'co'))
    actions = actions_for(proposals, reference_test, refs)
    selected_actions = choose_actions(actions, policies, transferable)
    final = apply_actions(reference_test, selected_actions)
    if final['tid'].n_unique() != final.height:
        raise ValueError('Export assigns a target to multiple owners')
    country_check = final.select(*KEYS).join(refs.select(pl.col('rid').alias('qid'), pl.col('co').alias('ref_co')), on='qid')\
        .join(targets.select(pl.col('rid').alias('tid'), pl.col('co').alias('target_co')), on='tid')
    if country_check.height != final.height or country_check.filter(pl.col('ref_co') != pl.col('target_co')).height:
        raise ValueError('Missing IDs or cross-country assignments')
    final.write_parquet(output/'accepted.parquet')
    selected_actions.write_parquet(output/'selected_actions.parquet')
    report['test'] = {'identity_candidates': diagnostics,
        'eligible_actions_by_country_cell': actions.group_by('co', 'cell').len().sort('co', 'cell').to_dicts(),
        'selected_actions_by_country_cell': selected_actions.group_by('co', 'cell').len().sort('co', 'cell').to_dicts(),
        'compared_with_exact_friend_reference': changes(final, reference_test, refs),
        'matches_per_country': final.join(refs.select(pl.col('rid').alias('qid'), 'co'), on='qid').group_by('co').len().sort('co').to_dicts()}
    examples = selected_actions.sort('co', 'cell', 'tid').group_by('co', 'cell', maintain_order=True).head(12)
    examples = examples.join(refs.select(pl.col('rid').alias('qid'), pl.col('nm').alias('reference_name'), pl.col('ad').alias('reference_address')), on='qid')\
        .join(targets.select(pl.col('rid').alias('tid'), pl.col('nm').alias('target_name'), pl.col('ad').alias('target_address')), on='tid')
    write_json(output/'repair_examples.json', examples.to_dicts())
    STAGE = 'comparing against original hybrid and exact last reranker policies'
    log(STAGE)
    h = pl.scan_parquet(hybrid/'test_predictions.parquet').select(*KEYS, 'p').filter(pl.col('p') >= .001).collect()
    acc_h = fast_apply(frozen['rule'], h, frozen['threshold'])
    report['test']['compared_with_frozen_hybrid'] = changes(final, acc_h, refs)
    hco = h.join(refs.select(pl.col('rid').alias('qid'), 'co'), on='qid')
    reranker = pl.concat([
        fast_apply('top1_threshold', hco.filter(pl.col('co') == 'india').select(*KEYS, 'p'), .7410188317298889),
        fast_apply('expected_f', hco.filter(pl.col('co') == 'us').select(*KEYS, 'p'), .3),
        acc_unlabeled], how='vertical_relaxed')
    report['test']['compared_with_last_reranker'] = changes(final, reranker, refs)
    report['test']['last_reranker_replayed_matches'] = reranker.height
    if reranker.height != EXPECTED_RERANKER_MATCHES:
        raise ValueError('Known reranker replay count mismatch; baseline inputs or policies differ')
    del h, hco, acc_h, reranker, country_check
    STAGE = 'exporting final TSV and exact reference comparator'
    log(STAGE)
    write_matches(reference_test, refs, targets, output/'reference'/'matching_results.tsv')
    with (output/'reference'/'matching_results.tsv').open('rb') as stream:
        ref_md5 = hashlib.file_digest(stream, 'md5').hexdigest()
    report['test']['reference_md5'] = ref_md5
    report['test']['friend_documented_md5'] = '1f4de3fc4f38916a35212c950ce5eaed'
    report['test']['reference_matches_documented_upload'] = ref_md5 == report['test']['friend_documented_md5']


    pools = [pl.scan_parquet(hybrid/'test_predictions.parquet').select(KEYS),
             pl.scan_parquet(graph/'test_predictions.parquet').select(KEYS),
             pl.scan_parquet(friend/'test_pred.parquet').select(KEYS), proposals.select(KEYS).lazy()]
    pool = pl.concat(pools, how='vertical_relaxed').unique().collect().with_columns(pl.lit(0., pl.Float32).alias('p'))
    report['export'] = export(pool, 'p', 'explicit_reference_plus_selected_identity', 0., refs, targets,
                              str(output/'output'), accepted=final)
    del pool, proposals, reference_test
    gc.collect()
    STAGE = 'validating IDs, candidate coverage, duplicates, and TSV structure'
    log(STAGE)
    from validate_outputs import validate
    report['validation'] = validate(data, output/'output')
    report['elapsed_minutes'] = (time.monotonic()-START)/60
    write_json(output/'report.json', report)
    (output/'_SUCCESS').write_text('complete\n')
    log(f'FINAL REPAIR COMPLETE: {report["export"]["matches"]:,} matches; reference delta {report["test"]["compared_with_exact_friend_reference"]}')
    return report
