import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import polars as pl

from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import tuning_anchors, export
from final_hybrid.fit import weighted_partitions, union_predictions, country_deltas
from graph_resolution.pipeline import subset_for_tuning, compare_audit
from innovation.common import log, logit

HEADS = ('xgboost_independent', 'catboost_independent', 'xgboost_residual', 'catboost_residual')
MIN_GAIN = .0001
MAX_PRECISION_LOSS = .0002
MAX_COUNTRY_LOSS = .0002


def read_features(root, split):
    root = Path(root)
    if not (root/f'_{split}_SUCCESS').is_file():
        raise RuntimeError(f'Incomplete {split} feature stage')
    frame = pl.read_parquet(root/f'{split}_features.parquet')
    if frame.select('qid', 'tid').n_unique() != frame.height:
        raise ValueError('Duplicate feature pairs')
    return frame


def anchors_for(data):
    refs = load_refs(data, 'train')
    all_anchors = refs.select(pl.col('rid').alias('qid'), 'deg', 'fold', 'co')
    tune = tuning_anchors(all_anchors, {'seed': 42, 'fit_fold': 0, 'tune_holdout_buckets': 5})
    return refs, all_anchors, tune.join(all_anchors.select('qid', 'co'), on='qid')


def worker(data, features_train, features_test, output, head, rounds=700, device='cuda'):
    from boosted_hybrid.models import fit_head, predict_raw, fit_one_head, matrix
    from boosted_hybrid.features import model_columns
    if head not in HEADS:
        raise ValueError('Unknown booster head')
    backend, mode = head.split('_', 1)
    dest = Path(output)/head
    dest.mkdir(parents=True, exist_ok=True)
    frame = read_features(features_train, 'train')
    refs, _, _ = anchors_for(data)
    raw, models, info = fit_head(frame, refs, dest/'bundle', backend=backend,
        mode=mode, device=device, rounds=rounds)
    cols = info['columns']
    frame.select('qid', 'tid').with_columns(pl.Series('raw', raw)).write_parquet(dest/'train.parquet')
    del raw



    stress = {}
    if mode == 'independent':
        fit, groups, weights, _ = weighted_partitions(frame, refs)
        for country in sorted(frame['co'].unique().to_list()):
            source = fit & (frame['co'].to_numpy() == country)
            ix = np.flatnonzero(source)
            if len(ix) < 30 or np.unique(frame['y'].to_numpy()[ix]).size != 2:
                stress[country] = {'skipped': 'insufficient source-country fit rows'}
                continue
            sampled = frame[ix]
            X = matrix(sampled, cols)
            y = sampled['y'].to_numpy()
            tr, va = groups[ix] != 0, groups[ix] == 0
            if not tr.any() or not va.any() or np.unique(y[tr]).size != 2 or np.unique(y[va]).size != 2:
                stress[country] = {'skipped': 'empty source-country CV partition'}
                continue
            suffix = '.json' if backend == 'xgboost' else '.cbm'
            path = dest/'bundle'/('transfer_'+country+suffix)
            fitted, details = fit_one_head(X, y, weights[ix], np.zeros(len(ix)), tr, va,
                backend=backend, mode=mode, device=device, rounds=rounds, path=path, columns=cols)
            del X, sampled
            raw = predict_raw([fitted], frame, cols, mode=mode, device=device)
            frame.select('qid', 'tid').with_columns(pl.Series('raw', raw)).write_parquet(
                dest/f'transfer_{country}.parquet')
            stress[country] = details
            del fitted, raw
            log(f'{head}: source-only {country} stress head scored')
    info['country_stress'] = stress
    (dest/'report.json').write_text(json.dumps(info, indent=2))
    del frame, refs
    gc.collect()
    test = read_features(features_test, 'test')
    raw = predict_raw(models, test, cols, mode=mode, device=device)
    test.select('qid', 'tid').with_columns(pl.Series('raw', raw)).write_parquet(dest/'test.parquet.tmp')
    (dest/'test.parquet.tmp').replace(dest/'test.parquet')
    (dest/'_SUCCESS').write_text('complete\n')
    log(f'{head}: training, transfer checks, and test prediction complete')


def gpu_preflight():
    'exercise the actual cuda booster apis before any challenge-data work'
    import tempfile
    from boosted_hybrid.models import fit_one_head, predict_raw
    n = 240
    X = np.stack([np.arange(n)%2, np.arange(n)%7], axis=1).astype(np.float32)
    y = (np.arange(n)%2).astype(np.float32)
    tr, va = np.arange(n)%3 != 0, np.arange(n)%3 == 0
    with tempfile.TemporaryDirectory(prefix='booster_preflight_') as work:
        for backend in ('xgboost', 'catboost'):
            for mode in ('independent', 'residual'):
                suffix = '.json' if backend == 'xgboost' else '.cbm'
                path = Path(work)/(backend+'_'+mode+suffix)
                fitted, _ = fit_one_head(X, y, np.ones(n), np.full(n, -1.2), tr, va,
                    backend=backend, mode=mode, device='cuda', rounds=3,
                    path=path, columns=['x0', 'x1'])
                frame = pl.DataFrame(X, schema=['x0', 'x1']).with_columns(pl.lit(-1.2).alias('base_lg'))
                values = predict_raw([fitted], frame, ['x0', 'x1'], mode=mode, device='cuda')
                replay = predict_raw([path], frame, ['x0', 'x1'], mode=mode, device='cuda')
                np.testing.assert_allclose(values, replay, atol=1e-5)
                if mode == 'residual':
                    shifted = predict_raw([fitted], frame.with_columns((pl.col('base_lg')+.75)),
                        ['x0', 'x1'], mode=mode, device='cuda')
                    np.testing.assert_allclose(shifted-values, .75, atol=1e-5)
                if not np.isfinite(values).all():
                    raise RuntimeError(f'{backend} GPU preflight failed')
                if backend == 'xgboost':
                    config = json.loads(fitted.save_config())
                    if not str(config['learner']['generic_param']['device']).startswith('cuda'):
                        raise RuntimeError('XGBoost silently fell back to CPU')
                log(f'{backend}/{mode}: actual CUDA fit/predict/replay preflight passed')


def gpu(data, features_train, features_test, output, rounds=700):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1]/'scripts/run_boosted_hybrid.py'

    subprocess.run([sys.executable, str(script), 'preflight'], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
    processes = []
    try:
        for rank, head in enumerate(HEADS):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), ER_THREADS='16',
                OMP_NUM_THREADS='16', POLARS_MAX_THREADS='8', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1')
            cmd = [sys.executable, str(script), 'worker', '--data', str(data),
                '--features-train', str(features_train), '--features-test', str(features_test),
                '--output', str(output), '--head', head, '--rounds', str(rounds), '--device', 'cuda']
            processes.append(subprocess.Popen(cmd, env=env))
        while processes:
            for proc in list(processes):
                status = proc.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f'Booster worker failed with status {status}')
                    processes.remove(proc)
            if processes:
                time.sleep(2)
    except BaseException:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        raise
    for head in HEADS:
        if not (root/head/'_SUCCESS').is_file():
            raise RuntimeError(f'Incomplete booster {head}')
    (root/'_SUCCESS').write_text('complete\n')


def read_raw(models, head, split, frame, transfer=None):
    root = Path(models)/head
    if not (root/'_SUCCESS').is_file():
        raise RuntimeError(f'Incomplete booster output: {head}')
    name = f'transfer_{transfer}' if transfer is not None else split
    raw = pl.read_parquet(root/(name+'.parquet'))
    keys = ['qid', 'tid']
    raw = raw.with_columns(pl.col(k).cast(frame.schema[k]) for k in keys)
    if raw.height != frame.height or raw.select(keys).n_unique() != raw.height:
        raise ValueError('Raw prediction coverage/uniqueness mismatch')
    out = frame.select(keys).join(raw, on=keys, how='left', maintain_order='left', validate='1:1')['raw']
    if out.null_count() or not out.is_finite().all():
        raise ValueError('Missing/nonfinite booster prediction')
    return out.to_numpy()


def combine(frame, baseline, raw, strength, mode):
    'blend existing scores; independent heads assess new retrieval pairs directly'
    prior = (frame['_current_p'] if '_current_p' in frame else
        frame.select('qid', 'tid').join(baseline.select('qid', 'tid', 'p'),
            on=['qid', 'tid'], how='left', maintain_order='left', validate='1:1')['p'])
    if prior.null_count() or len(raw) != frame.height or not np.isfinite(raw).all():
        raise ValueError('Invalid baseline alignment or raw prediction')
    lg = (1-strength)*logit(prior.to_numpy()) + strength*np.asarray(raw)
    if mode == 'independent':
        lg = np.where(frame['has_parent'].to_numpy() == 0, raw, lg)
    p = (1/(1+np.exp(-np.clip(lg, -20, 20)))).astype(np.float32)
    return frame.select('qid', 'tid').with_columns(pl.Series('p', p))


def constrained_curve(best, anchors, minimum_precision, minimum_recall):
    'Exact top-one threshold sweep, with precision/recall constraints'
    n = anchors.height
    truth = int(anchors['deg'].sum())
    base = int((anchors['deg'] == 0).sum())
    d = best.join(anchors.select('qid', 'deg'), on='qid').sort(['qid', 'p'], descending=[False, True])
    if not d.height:
        return None
    d = d.with_columns(pl.int_range(1, pl.len()+1).over('qid').alias('_n'),
        pl.col('y').cast(pl.Int64).cum_sum().over('qid').alias('_tp'))
    d = d.with_columns(pl.when(pl.col('deg') > 0)
        .then(1.25*pl.col('_tp')/(pl.col('_n')+.25*pl.col('deg'))).otherwise(0.).alias('_after'))
    d = d.with_columns(pl.col('_after').shift(1).over('qid').fill_null(
        pl.when(pl.col('deg') == 0).then(1.).otherwise(0.)).alias('_before'))
    g = d.group_by('p').agg((pl.col('_after')-pl.col('_before')).sum().alias('_delta'),
        pl.len().alias('_n'), pl.col('y').cast(pl.Int64).sum().alias('_tp')).sort('p', descending=True)
    g = g.with_columns(((base+pl.col('_delta').cum_sum())/n).alias('macro_f05'),
        pl.col('_n').cum_sum().alias('pairs'), pl.col('_tp').cum_sum().alias('_ctp'))
    g = g.with_columns((pl.col('_ctp')/pl.col('pairs')).alias('pair_precision'),
        (pl.col('_ctp')/max(truth, 1)).alias('pair_recall')).filter(
        (pl.col('pair_precision') >= minimum_precision) & (pl.col('pair_recall') >= minimum_recall))
    if not g.height:
        return None
    chosen = g.sort(['macro_f05', 'pair_recall'], descending=[True, True]).row(0, named=True)
    return {'rule': 'top1_threshold', 'threshold': chosen['p'],
        **{k: chosen[k] for k in ('macro_f05', 'pair_precision', 'pair_recall', 'pairs')}}


def tune_constrained(pairs, anchors, baseline_metrics):
    precision = baseline_metrics['pair_precision']-MAX_PRECISION_LOSS
    recall = baseline_metrics['pair_recall']-1e-12
    choices = []
    top = constrained_curve(decode.top1(pairs), anchors, precision, recall)
    if top is not None:
        choices.append(top)
    for floor in (.1, .3, .5):
        score = decode.score(decode.expected_f(pairs, floor), anchors)
        if score['pair_precision'] >= precision and score['pair_recall'] >= recall:
            choices.append({'rule': 'expected_f', 'threshold': floor, **score})
    return max(choices, key=lambda x: (x['macro_f05'], x['pair_recall'])) if choices else None


def error_changes(selected, previous, anchors):
    keys = ['qid', 'tid']
    chosen = selected.join(anchors.select('qid'), on='qid')
    old = previous.join(anchors.select('qid'), on='qid')
    added = chosen.join(old.select(keys), on=keys, how='anti')
    removed = old.join(chosen.select(keys), on=keys, how='anti')
    return {'recovered_true_links': int(added['y'].sum()),
        'new_false_links': int((added['y'] == 0).sum()),
        'lost_true_links': int(removed['y'].sum()),
        'removed_false_links': int((removed['y'] == 0).sum())}


def raw_variants(models, split, frame, transfer=None):
    heads = HEADS if transfer is None else tuple(h for h in HEADS if h.endswith('_independent'))
    if transfer is not None:
        heads = tuple(h for h in heads if (Path(models)/h/f'transfer_{transfer}.parquet').is_file())
    values = {head: read_raw(models, head, split, frame, transfer) for head in heads}
    modes = ('independent', 'residual') if transfer is None else ('independent',)
    for mode in modes:
        if all(backend+'_'+mode in values for backend in ('xgboost', 'catboost')):
            values['ensemble_'+mode] = (values['xgboost_'+mode]+values['catboost_'+mode])/2
    return values


def routed_accept(pred, policies, refs):
    'apply explicit per-country decoders without a fabricated french threshold'
    groups = refs.select(pl.col('rid').alias('qid'), 'co')
    parts = []
    for country in groups['co'].unique().sort():
        policy = policies[str(country)]
        sub = pred.join(groups.filter(pl.col('co') == country).select('qid'), on='qid')
        parts.append(decode.apply(policy['rule'], sub, policy['threshold']))
    return pl.concat(parts, how='vertical_relaxed')


def country_stress(models, frame, baseline, baseline_accepted, tune_co, tune_keys, trials):
    'calibrate in the source country, evaluate unchanged in the other country'
    report = {}
    countries = tune_co['co'].unique().sort().to_list()
    if len(countries) != 2:
        return {'status': 'requires exactly two labeled source countries'}, set()
    rs = {}
    small_base = baseline.join(tune_keys, on=['qid', 'tid'], how='inner')
    for source in countries:
        src = tune_co.filter(pl.col('co') == source).select('qid', 'deg')
        dst_country = next(c for c in countries if c != source)
        dst = tune_co.filter(pl.col('co') == dst_country).select('qid', 'deg')
        src_base = decode.score(baseline_accepted, src)
        dst_base = decode.score(baseline_accepted, dst)
        logits = raw_variants(models, 'train', frame, transfer=source)
        report['availability_'+str(source)] = {
            h: ('present' if h in logits else 'missing or skipped; cannot qualify France policy')
            for h in ('xgboost_independent', 'catboost_independent', 'ensemble_independent')}
        for head, raw in logits.items():
            for strength in (.5, 1.):
                key = f'{head}_{strength:g}'
                update = combine(frame, baseline, raw, strength, 'independent').join(
                    tune_keys, on=['qid', 'tid'], how='inner')
                patched = union_predictions(small_base, update, frame)
                decision = tune_constrained(patched, src, src_base)
                item = {'source_country': source, 'target_country': dst_country,
                        'target_baseline': dst_base, 'source_decision': decision}
                if decision is not None:
                    accepted = decode.apply(decision['rule'], patched, decision['threshold'])
                    metrics = decode.score(accepted, dst)
                    item.update(target_metrics=metrics,
                        target_delta=metrics['macro_f05']-dst_base['macro_f05'],
                        acceptable=(metrics['macro_f05'] >= dst_base['macro_f05']-MAX_COUNTRY_LOSS and
                            metrics['pair_precision'] >= dst_base['pair_precision']-MAX_PRECISION_LOSS and
                            metrics['pair_recall'] >= dst_base['pair_recall']-.0002))
                else:
                    item['acceptable'] = False
                rs.setdefault(key, []).append(item)
                log(f'Head transfer {key} {source}->{dst_country}: {item}')
    passing = set()
    for key, directions in rs.items():
        mean = float(np.mean([d.get('target_delta', -1.) for d in directions]))
        passed = len(directions) == 2 and all(d['acceptable'] for d in directions) and mean >= MIN_GAIN
        report[key] = {'directions': directions, 'mean_delta': mean, 'passes_proxy': passed}
        if passed and trials.get(key, {}).get('eligible', False):
            passing.add(key)
    return report, passing


def country_summary(pred, accepted, baseline_accepted, refs, targets, features):
    'Label-free country drift/acceptance diagnostics, never a French accuracy estimate'
    summary = {}
    for country in refs['co'].unique().sort():
        owners = refs.filter(pl.col('co') == country).select(pl.col('rid').alias('qid'))
        target = targets.filter(pl.col('co') == country)
        sel = accepted.join(owners, on='qid')
        prev = baseline_accepted.join(owners, on='qid')
        f = features.filter(pl.col('co') == country)
        new = sel.join(prev.select('qid', 'tid'), on=['qid', 'tid'], how='anti')
        old = prev.join(sel.select('qid', 'tid'), on=['qid', 'tid'], how='anti')
        summary[str(country)] = {'source1': owners.height, 'targets': target.height,
            'target_missing_address_fraction': float(target.select(
                (pl.col('ad').fill_null('').str.strip_chars() == '').mean()).item()),
            'baseline_matches': prev.height, 'selected_matches': sel.height,
            'new_accepted_pairs': new.height, 'removed_accepted_pairs': old.height,
            'selected_target_fraction': f['tid'].n_unique()/max(target.height, 1),
            'matches_per_source1': sel.height/max(owners.height, 1),
            'normalization_disagreement': {c: float((f[c].abs() > 10).mean())
                for c in ('raw_name_core_tset_legacy_delta', 'raw_address_fold_tset_legacy_delta') if c in f},
            'accuracy': None}
    return summary


def select(data, hybrid_result, features_train, features_test, models, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (Path(models)/'_SUCCESS').is_file():
        raise RuntimeError('Booster stage is incomplete')
    frame = read_features(features_train, 'train')
    refs, anchors, tune_co = anchors_for(data)
    tune = tune_co.select('qid', 'deg')
    audit = anchors.filter(pl.col('fold') == 1).select('qid', 'deg')
    base = pl.read_parquet(Path(hybrid_result)/'validation_predictions.parquet')
    frame = frame.join(base.select('qid', 'tid', pl.col('p').alias('_current_p')),
        on=['qid', 'tid'], how='left', maintain_order='left', validate='1:1')
    previous_report = json.loads((Path(hybrid_result)/'report.json').read_text())
    base_decision = previous_report['decision']
    base_accepted = decode.apply(base_decision['rule'], base, base_decision['threshold'])
    base_tune = decode.score(base_accepted, tune)
    tune_base = subset_for_tuning(base, tune)
    tune_keys = tune_base.select('qid', 'tid')
    if frame.select('qid', 'tid').join(base.select('qid', 'tid'), on=['qid', 'tid'], how='anti').height:
        raise ValueError('Cached hybrid feature pairs missing from full baseline pool')
    trials = {'baseline': {'head': 'baseline', 'strength': 0., 'eligible': True,
        'rule': base_decision['rule'], 'threshold': base_decision['threshold'], 'tune': base_tune}}
    logits = raw_variants(models, 'train', frame)
    for head, raw in logits.items():
        mode = head.split('_', 1)[1]
        for strength in (.5, 1.):
            key = f'{head}_{strength:g}'
            update = combine(frame, base, raw, strength, mode).join(tune_keys,
                on=['qid', 'tid'], how='inner')
            patched = union_predictions(tune_base, update, frame)
            decision = tune_constrained(patched, tune, base_tune)
            trial = {'head': head, 'strength': strength, 'mode': mode,
                     'eligible': False, 'tune': decision}
            if decision is not None:
                accepted = decode.apply(decision['rule'], patched, decision['threshold'])
                delta = country_deltas(accepted, base_accepted, tune_co)
                gain = decision['macro_f05']-base_tune['macro_f05']
                trial.update(rule=decision['rule'], threshold=decision['threshold'],
                    country_delta=delta, gain=gain,
                    eligible=gain >= MIN_GAIN and min(delta.values(), default=0.) >= -MAX_COUNTRY_LOSS)
            trials[key] = trial
            log(f'BOOSTED trial {key}: {trial}')
    winner = max((k for k, v in trials.items() if v['eligible']),
                 key=lambda k: (trials[k]['tune']['macro_f05'], trials[k]['tune']['pair_recall']))
    sel = trials[winner]
    stress, french_candidates = country_stress(models, frame, base, base_accepted,
                                               tune_co, tune_keys, trials)
    french = max(sorted(french_candidates), key=lambda k: stress[k]['mean_delta']) if french_candidates else 'baseline'
    train = base if winner == 'baseline' else union_predictions(base,
        combine(frame, base, logits[sel['head']], sel['strength'], sel['mode']), frame)
    accepted = decode.apply(sel['rule'], train, sel['threshold'])
    paired = compare_audit(accepted, base_accepted, audit)
    paired['reference'] = 'completed hybrid_0.5, identical historical audit anchors'
    paired.pop('historical_audit_reference', None)
    report = {'selected': winner, 'decision': sel, 'france_selected': french,
        'tuning': trials, 'transfer_stress': stress, 'baseline_tune': base_tune,
        'audit': decode.score(accepted, audit), 'baseline_audit': decode.score(base_accepted, audit),
        'comparison': paired, 'errors': error_changes(accepted, base_accepted, audit),
        'audit_by_country': {str(c): {'baseline': decode.score(base_accepted, a),
            'selected': decode.score(accepted, a), 'changes': error_changes(accepted, base_accepted, a)}
            for c in anchors['co'].unique().sort()
            for a in [anchors.filter((pl.col('fold') == 1) & (pl.col('co') == c)).select('qid', 'deg')]},
        'promotion_policy': {'minimum_tune_gain': MIN_GAIN, 'maximum_country_loss': MAX_COUNTRY_LOSS,
            'maximum_pair_precision_loss': MAX_PRECISION_LOSS, 'pair_recall_must_not_decrease': True,
            'france': 'eligible independent head must also pass both source-only transfer directions; otherwise baseline',
            'audit_used_for_selection': False},
        'limits': ['Audit has informed research; it is not a fresh blind whole-pipeline benchmark.',
            'Sampling weights correct owner/decoy subsampling within selected targets, not full-population calibration.',
            'France has no labels. Its policy uses head-transfer proxies conditional on frozen upstream models.',
            'Retrieval is unchanged and still bounded to previously selected targets.'],
        'training': {h: json.loads((Path(models)/h/'report.json').read_text()) for h in HEADS}}
    train.write_parquet(output/'validation_predictions.parquet')
    (output/'report.json').write_text(json.dumps(report, indent=2))
    log(f'BOOSTED selected {winner}; France policy {french}; audit {report["audit"]}; changes {report["errors"]}')
    del train, accepted, base, base_accepted, tune_base, tune_keys, frame, logits
    gc.collect()
    frame = read_features(features_test, 'test')
    base = pl.read_parquet(Path(hybrid_result)/'test_predictions.parquet')
    frame = frame.join(base.select('qid', 'tid', pl.col('p').alias('_current_p')),
        on=['qid', 'tid'], how='left', maintain_order='left', validate='1:1')
    refs, targets = load_refs(data, 'test'), load_targets(data, 'test')
    base_accepted = decode.apply(base_decision['rule'], base, base_decision['threshold'])
    policies = {str(c): trials[french if str(c).lower() == 'france' else winner]
                for c in refs['co'].unique().sort()}
    needed = {p['head'] for p in policies.values()}-{'baseline'}
    raw_cache = {}
    def get_raw(head):
        if head not in raw_cache:
            if head.startswith('ensemble_'):
                mode = head.split('_', 1)[1]
                raw_cache[head] = (get_raw('xgboost_'+mode)+get_raw('catboost_'+mode))/2
            else:
                raw_cache[head] = read_raw(models, head, 'test', frame)
        return raw_cache[head]
    updates = []
    for country, policy in policies.items():
        if policy['head'] == 'baseline':
            continue
        update = combine(frame, base, get_raw(policy['head']), policy['strength'], policy['mode'])
        keys = frame.filter(pl.col('co') == country).select('qid', 'tid')
        updates.append(update.join(keys, on=['qid', 'tid']))
    res = union_predictions(base, pl.concat(updates, how='vertical_relaxed')) if updates else base
    accepted = routed_accept(res, policies, refs)
    report['test_by_country'] = country_summary(res, accepted, base_accepted, refs, targets, frame)
    report['country_policies'] = policies
    res.write_parquet(output/'test_predictions.parquet')
    report['export'] = export(res, 'p', sel['rule'], sel['threshold'], refs, targets,
        str(output/'output'), accepted=accepted)
    report['export']['decoder'] = 'country_policies in report.json; accepted pairs supplied explicitly'
    (output/'report.json').write_text(json.dumps(report, indent=2))
    (output/'decision.json').write_text(json.dumps({'default': sel, 'countries': policies}, indent=2))
    (output/'_SUCCESS').write_text('complete\n')
    return report
