'Fit sampling-corrected residual matchers over lexical/dense candidate unions'
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from er.safe import THREADS, guard
from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.features import normalize, text_features, chunks_by_tid, name_frequency, _margin
from er.stack.pipeline import export, tuning_anchors, lgb_params
from graph_resolution.pipeline import compare_audit, subset_for_tuning
from innovation.common import read_parent, log
from innovation.train import training_partitions

NEW_PAIR_PRIOR = .01
MODES = ('surface', 'semantic', 'hybrid')


def weighted_partitions(frame, refs, cv=3):
    'correct extra negative downsampling while retaining both entity holdouts'
    fit, groups = training_partitions(frame, refs, cv)
    proportions = refs.group_by('co').agg(
        ((pl.col('fold') == 0) & ((pl.col('rid').cast(pl.UInt32).hash(1033)%5) != 0))
        .mean().alias('_fit_rate'))
    rates = frame.select('co').join(proportions, on='co', how='left', maintain_order='left')['_fit_rate'].to_numpy()
    y = frame['y'].to_numpy()
    own = frame['own'].to_numpy()
    weights = np.where(y == 1, 1., np.where(own < 0, 12.5, 1./np.clip(rates, 1e-6, 1.)))
    if not np.isfinite(weights[fit]).all():
        raise ValueError('Invalid sampling weights')
    return fit, groups, weights.astype(np.float32), proportions.to_dicts()


def attach_parent(scored, parent):
    'keep explicit missing-parent status; retrieval similarity is not probability'
    if scored.select('qid', 'tid').n_unique() != scored.height:
        raise ValueError('Duplicate hybrid scored pair')
    if parent.select('qid', 'tid').n_unique() != parent.height:
        raise ValueError('Duplicate parent pair')
    keys = ['qid', 'tid']
    scored = scored.with_columns(pl.col(k).cast(parent.schema[k]) for k in keys)
    out = scored.join(parent.select(*keys, pl.col('p').alias('parent_p')), on=keys,
                      how='left', maintain_order='left', validate='1:1')
    return out.with_columns(pl.col('parent_p').is_not_null().cast(pl.Int8).alias('has_parent'),
        pl.col('parent_p').fill_null(0)).with_columns(
            pl.when(pl.col('has_parent') == 1).then(pl.col('parent_p'))
              .otherwise(NEW_PAIR_PRIOR).clip(1e-6, 1-1e-6).alias('_prior')).with_columns(
                  (pl.col('_prior')/(1-pl.col('_prior'))).log().cast(pl.Float32).alias('base_lg'))


def feature_columns(frame, mode):
    if mode not in MODES:
        raise ValueError('Unknown feature suite')
    skip = {'qid', 'tid', 'y', 'own', 'fold', 'co', 'parent_p'}
    columns = [k for k, dtype in frame.schema.items() if k not in skip and
               not k.startswith('_') and dtype.is_numeric()]
    if mode == 'surface':
        columns = [k for k in columns if not k.startswith(('dense_', 'hit_dense', 'ce_', 'expert_', 'neural_'))
                   and k != 'rrf']
    elif mode == 'semantic':
        columns = [k for k in columns if not k.startswith(('expert_', 'neural_'))]
    return columns


def load_scored(hybrid, split):
    hybrid = Path(hybrid)
    if not (hybrid/'_SUCCESS').exists():
        raise RuntimeError('Hybrid GPU scoring is incomplete')
    manifest = hybrid/split/'fusion_report.json'
    paths = ([hybrid/split/'parts'/name for name in json.loads(manifest.read_text())['retrieval_shards']]
             if manifest.exists() else sorted((hybrid/split/'parts').glob('*.parquet')))
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('No scored hybrid parts')
    scored = pl.concat([pl.read_parquet(p) for p in paths], how='vertical_relaxed')
    expected = pl.read_parquet(hybrid/split/'candidates.parquet').select('qid', 'tid')
    if (scored.height != expected.height or scored.select('qid', 'tid').n_unique() != scored.height or
        expected.join(scored.select('qid', 'tid'), on=['qid', 'tid'], how='anti').height):
        raise ValueError('Hybrid scorer did not cover the exact candidate pool')
    for col in ('ce_base_lg', 'expert_lg'):
        if scored[col].null_count() or not scored[col].is_finite().all():
            raise ValueError('Missing/nonfinite neural scores')
    return scored


def build_features(data, parent_path, hybrid, output, split):
    scored = load_scored(hybrid, split)
    parent, parent_report = read_parent(parent_path, split)
    frame = attach_parent(scored, parent)
    del scored
    refs, targets = load_refs(data, split), load_targets(data, split)
    q = targets.join(frame.select(pl.col('tid').alias('rid')).unique(), on='rid')


    stats = parent.group_by('tid').agg(pl.col('p').top_k(2).alias('_top'),
                                     pl.len().alias('parent_candidates'))
    stats = stats.with_columns(pl.col('_top').list.get(0).alias('parent_best'),
        pl.col('_top').list.get(1, null_on_oob=True).fill_null(0).alias('parent_second')).drop('_top')
    frame = frame.join(stats, on='tid', how='left', maintain_order='left').with_columns(
        (pl.col('parent_p') - pl.col('parent_best')).alias('parent_best_gap'),
        (pl.col('expert_lg')-pl.col('ce_base_lg')).alias('neural_disagreement'))
    for col in ('ce_base_lg', 'expert_lg'):
        frame = _margin(frame, col, 'tid', col+'_margin').with_columns(
            pl.col(col).rank('min', descending=True).over('tid').cast(pl.Float32).alias(col+'_rank'))
    log(f'{split}: normalizing {refs.height:,} references and {q.height:,} selected targets')
    rr, tt = normalize(refs.select('rid', 'nm', 'ad', 'co')), normalize(q.select('rid', 'nm', 'ad', 'co'))
    keys = frame.select('qid', 'tid').sort('tid', 'qid')
    parts = []
    feature_dir = Path(output)/'feature_cache'/split
    feature_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks_by_tid(keys, 100_000)):
        part = feature_dir/f'part_{i:05d}.parquet'
        text_features(chunk, rr, tt, ['name', 'address', 'numbers']).write_parquet(part)
        parts.append(part)
        if i%25 == 0:
            log(f'{split}: constructed text features for approximately {min((i+1)*100_000, keys.height):,} pairs')
        guard('hybrid text features')
    text = pl.concat([pl.read_parquet(p) for p in parts])
    frame = frame.join(text, on=['qid', 'tid'], how='left', validate='1:1', maintain_order='left')
    ref_freq, target_freq = name_frequency(rr, tt)
    frame = frame.join(ref_freq, on='qid', how='left').join(target_freq, on='tid', how='left')
    frame = frame.join(q.select(pl.col('rid').alias('tid'), 'co'), on='tid', how='left')
    if split == 'train':
        owners = pl.concat([pl.read_parquet(Path(data)/'train'/f's{i}.parquet', columns=['rid', 'own'])
                            for i in (2, 3)]).select(pl.col('rid').cast(pl.UInt32).alias('tid'),
                                                    pl.col('own').cast(pl.Int64))
        frame = frame.join(owners, on='tid', how='left').join(
            refs.select(pl.col('rid').alias('qid'), 'fold'), on='qid', how='left').with_columns(
            (pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8).alias('y'))
    frame = frame.with_columns(pl.col(pl.Float64).cast(pl.Float32)).sort('tid', 'qid')
    if frame.height != keys.height:
        raise ValueError('Feature join changed pair count')
    frame.write_parquet(Path(output)/f'{split}_features.parquet')
    del refs, targets, q, rr, tt, text, keys
    gc.collect()
    return frame, parent, parent_report


def _sigmoid(values):
    return 1./(1.+np.exp(-np.clip(values, -30., 30.)))


def fit_residual(frame, columns, fit, groups, weights, rounds, bundle, mode):
    import lightgbm as lgb
    X = frame.select(columns).cast(pl.Float32).to_numpy()
    y = frame['y'].to_numpy().astype(np.float32)
    offset = frame['base_lg'].to_numpy().astype(np.float64)
    if not fit.any() or np.unique(y[fit]).size != 2:
        raise ValueError('Residual fit requires both classes')
    settings = {'lgbm': {'learning_rate': .035, 'num_leaves': 31, 'min_data_in_leaf': 80,
        'lambda_l2': 20., 'feature_fraction': .9, 'deterministic': True, 'force_col_wise': True,
        'boost_from_average': False, 'seed': 8043}}
    residual = np.zeros(frame.height, dtype=np.float64)
    other = np.flatnonzero(~fit)
    models, history = [], []
    for fold in range(3):
        tr, va = fit & (groups != fold), fit & (groups == fold)
        if not va.any() or np.unique(y[tr]).size != 2:
            raise ValueError('Empty owner CV class/partition')
        train = lgb.Dataset(X[tr], label=y[tr], weight=weights[tr], init_score=offset[tr], feature_name=columns)
        valid = lgb.Dataset(X[va], label=y[va], weight=weights[va], init_score=offset[va], reference=train)
        model = lgb.train(lgb_params(settings), train, num_boost_round=rounds, valid_sets=[valid],
            callbacks=[lgb.early_stopping(70, verbose=False)])

        residual[va] = model.predict(X[va], raw_score=True, num_threads=THREADS)
        for start in range(0, len(other), 500_000):
            ix = other[start:start+500_000]
            residual[ix] += model.predict(X[ix], raw_score=True, num_threads=THREADS)/3
        model.save_model(str(bundle/f'{mode}_{fold}.txt'))
        models.append(model)
        history.append({'iteration': model.best_iteration,
                        'weighted_logloss': model.best_score['valid_0']['binary_logloss']})
        log(f'{mode}: weighted residual owner CV {fold+1}/3 {history[-1]}')
        del train, valid
    return residual.astype(np.float32), models, {'cv': history, 'fit_rows': int(fit.sum()),
        'fit_positive_rate': float(y[fit].mean()),
        'weighted_fit_positive_rate': float(np.average(y[fit], weights=weights[fit])),
        'initial_score': 'parent logit for existing pairs; logit(0.01) for genuinely new pairs',
        'importance': sorted(zip(columns, np.mean([m.feature_importance('gain') for m in models], axis=0).tolist()),
                             key=lambda x: -x[1])[:40]}


def update_probabilities(frame, residual, strength):
    if len(residual) != frame.height or not np.isfinite(residual).all():
        raise ValueError('Invalid residual prediction')


    limits = np.where(frame['has_parent'].to_numpy() == 1, 4., 10.)
    delta = np.clip(residual, -limits, limits)
    p = _sigmoid(frame['base_lg'].to_numpy()+float(strength)*delta)
    return frame.select('qid', 'tid').with_columns(pl.Series('p', p.astype(np.float32)))


def union_predictions(parent, update, frame=None):
    'all parent rows survive; every newly scored pair enters an active hybrid'
    keys = ['qid', 'tid']
    updated = parent.join(update.rename({'p': '_new'}), on=keys, how='left', maintain_order='left').with_columns(
        pl.coalesce('_new', 'p').alias('p')).drop('_new')
    added = update.join(parent.select(keys), on=keys, how='anti')
    if 'y' in parent.columns:
        if frame is None:
            raise ValueError('New training candidates require labels for evaluation')
        added = added.join(frame.select(*keys, 'y'), on=keys, validate='1:1')
    return pl.concat([updated, added.select(updated.columns)], how='vertical_relaxed')


def country_deltas(accepted, baseline, anchors):
    return {str(co): decode.score(accepted, anchors.filter(pl.col('co') == co).select('qid', 'deg'))['macro_f05']-
            decode.score(baseline, anchors.filter(pl.col('co') == co).select('qid', 'deg'))['macro_f05']
            for co in anchors['co'].unique().sort()}


def retrieval_report(frame, parent, anchors):
    positives = frame.filter(pl.col('y') == 1)
    added = positives.join(parent.select('qid', 'tid'), on=['qid', 'tid'], how='anti')
    old_true = parent.filter(pl.col('y') == 1)
    combined = pl.concat([old_true.select('qid', 'tid', 'y'), added.select('qid', 'tid', 'y')], how='vertical_relaxed')
    return {'scored_pairs': frame.height, 'genuinely_new_pairs': int((frame['has_parent'] == 0).sum()),
        'new_true_pairs_all_selected': added.height,
        'new_true_pairs_audit': added.join(anchors.select('qid'), on='qid').height,
        'parent_oracle': decode.score(old_true, anchors), 'hybrid_oracle': decode.score(combined, anchors),
        'note': 'Oracle uses labels to diagnose coverage only; never used to choose candidates or a model.'}


def run(data, parent, prepared, hybrid, output, rounds=800):
    output = Path(output)
    bundle = output/'bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    tr, old, old_report = build_features(data, parent, hybrid, output, 'train')
    refs = load_refs(data, 'train')
    anchors = refs.select(pl.col('rid').alias('qid'), 'deg', 'fold', 'co')
    tune = tuning_anchors(anchors, {'seed': 42, 'fit_fold': 0, 'tune_holdout_buckets': 5})
    tune_co = tune.join(anchors.select('qid', 'co'), on='qid')
    audit = anchors.filter(pl.col('fold') == 1).select('qid', 'deg')
    fit, groups, weights, proportions = weighted_partitions(tr, refs)
    baseline_decision = old_report['selected_decoder']
    base = decode.apply(baseline_decision['rule'], old, baseline_decision['threshold'])
    baseline_tune = decode.score(base, tune)
    trials = {'parent': {'mode': 'parent', 'strength': 0., 'eligible': True,
        'rule': baseline_decision['rule'], 'threshold': baseline_decision['threshold'], 'tune': baseline_tune}}
    models, pred, training, columns = {}, {}, {}, {}


    template = union_predictions(old, tr.select('qid', 'tid').with_columns(pl.lit(0.).alias('p')), tr)
    tune_keys = subset_for_tuning(template, tune).select('qid', 'tid')
    tune_parent = old.join(tune_keys, on=['qid', 'tid'], how='inner')
    expanded_tune = tr.join(tune_keys, on=['qid', 'tid'], how='inner')
    del template
    for mode in MODES:
        cols = feature_columns(tr, mode)
        delta, fitted, info = fit_residual(tr, cols, fit, groups, weights, rounds, bundle, mode)
        columns[mode], pred[mode], models[mode], training[mode] = cols, delta, fitted, info
        for strength in (.25, .5, 1.):
            update = update_probabilities(tr, delta, strength).join(tune_keys, on=['qid', 'tid'], how='inner')
            patched = union_predictions(tune_parent, update, expanded_tune)
            decision = decode.tune(patched, tune, rules=('top1_threshold', 'expected_f'), floors=(.1, .3, .5))
            accepted = decode.apply(decision['rule'], patched, decision['threshold'])
            countries = country_deltas(accepted, base, tune_co)
            gain = decision['macro_f05'] - baseline_tune['macro_f05']
            key = f'{mode}_{strength:g}'
            trials[key] = {'mode': mode, 'strength': strength, 'rule': decision['rule'],
                'threshold': decision['threshold'], 'tune': decision, 'country_delta': countries,
                'gain': gain, 'eligible': gain >= .0001 and min(countries.values(), default=0.) >= -.0002}
            log(f'FINAL HYBRID {key}: tune F0.5={decision["macro_f05"]:.8f}; gain={gain:+.8f}; {countries}')
            del patched, accepted, update
    winner = max((k for k in trials if trials[k]['eligible']), key=lambda k: trials[k]['tune']['macro_f05'])
    decision = trials[winner]
    chosen = old if winner == 'parent' else union_predictions(old,
        update_probabilities(tr, pred[decision['mode']], decision['strength']), tr)
    accepted = decode.apply(decision['rule'], chosen, decision['threshold'])
    paired = compare_audit(accepted, base, audit)
    paired['reference'] = 'completed graph sibling parent, identical historical audit anchors'
    paired.pop('historical_audit_reference', None)
    report = {'selected': winner, 'decision': decision, 'tuning': trials, 'training': training,
        'sampling': {'country_candidate_fit_fractions': proportions, 'decoy_fraction': .08,
            'positive_weight': 1., 'owned_negative_weight': 'inverse empirical country entity fit fraction',
            'decoy_weight': 12.5, 'holdouts': 'both candidate and true owner, plus true-owner CV'},
        'promotion_policy': {'minimum_tune_gain': .0001, 'maximum_country_loss': .0002, 'audit_used': False},
        'audit': decode.score(accepted, audit), 'parent_audit': decode.score(base, audit),
        'comparison': paired,
        'retrieval': retrieval_report(tr, old, audit),
        'audit_by_country': {str(co): {'parent': decode.score(base, sub.select('qid', 'deg')),
              'selected': decode.score(accepted, sub.select('qid', 'deg'))}
            for co in anchors['co'].unique().sort()
            for sub in [anchors.filter((pl.col('fold') == 1) & (pl.col('co') == co))]},
        'limits': ['French data remain unlabeled; transfer is unverified.',
            'Historical audit has informed research and is not a fresh blind estimate.',
            'Sampling weights approximate original stratified entity inclusion probabilities.',
            'New retrieval is bounded to the frozen selected target set; identical records can remain ambiguous.']}
    (bundle/'features.json').write_text(json.dumps(columns, indent=2))
    (bundle/'decision.json').write_text(json.dumps(decision, indent=2))
    (output/'report.json').write_text(json.dumps(report, indent=2))
    chosen.write_parquet(output/'validation_predictions.parquet')
    log(f'FINAL HYBRID selected={winner}; audit={report["audit"]}; paired={report["comparison"]}')
    del tr, old, chosen, accepted, base, pred, tune_keys, tune_parent, expanded_tune
    gc.collect()
    if winner == 'parent':
        test, _ = read_parent(parent, 'test')
    else:
        te, old_test, _ = build_features(data, parent, hybrid, output, 'test')
        cols, fitted = columns[decision['mode']], models[decision['mode']]
        delta = np.empty(te.height, dtype=np.float32)
        for off in range(0, te.height, 500_000):
            X = te.slice(off, 500_000).select(cols).cast(pl.Float32).to_numpy()
            delta[off:off+len(X)] = np.mean([m.predict(X, raw_score=True, num_threads=THREADS) for m in fitted], axis=0)
        test = union_predictions(old_test, update_probabilities(te, delta, decision['strength']))
    test.write_parquet(output/'test_predictions.parquet')
    report['export'] = export(test, 'p', decision['rule'], decision['threshold'],
        load_refs(data, 'test'), load_targets(data, 'test'), str(output/'output'))
    (output/'report.json').write_text(json.dumps(report, indent=2))
    (output/'_SUCCESS').write_text('complete\n')
    return report
