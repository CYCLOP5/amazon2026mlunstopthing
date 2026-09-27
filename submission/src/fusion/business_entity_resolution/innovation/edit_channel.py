'learn the corruption process from disjoint training owners, then test a patch'
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy import sparse
from unidecode import unidecode

from innovation.common import log, read_parent, replace_scores

DIMENSION = 2**19
FIELD_FEATURES = (
    'chars1', 'chars2', 'words1', 'words2', 'common_words', 'removed_words',
    'added_words', 'common_fraction1', 'common_fraction2', 'edits',
    'edit_fraction', 'substitutions', 'insertions', 'deletions', 'digit_edits',
    'space_edits', 'first_edit_fraction', 'last_edit_fraction', 'prefix',
    'suffix', 'exact', 'missing1', 'missing2', 'unicode1', 'unicode2',
    'numbers1', 'numbers2', 'first_in_other', 'other_first_in_ref',
    'number_set_equal', 'closest_number_log_gap', 'closest_number_edit',
    'closest_number_same_length', 'closest_number_last_digit_only',
    'common_numbers', 'number_prefix_noise',
)
STRUCTURE_COLUMNS = [f'edit_{field}_{name}' for field in ('name', 'address')
                     for name in FIELD_FEATURES]
CHANNEL_COLUMNS = ['channel_name_lg', 'channel_address_lg']


def clean(value):

    value = unidecode(value or '').lower()[:768]
    value = re.sub(r'\b(?:[a-z]\.\s*){2,}',
                   lambda m: re.sub(r'[^a-z]', '', m.group()), value)
    return re.sub(r'[^a-z0-9]+', ' ', value).strip()


def _number_shape(a, b):
    aa = [x.lstrip('0') or '0' for x in re.findall(r'\d+', a)[:16]]
    bb = [x.lstrip('0') or '0' for x in re.findall(r'\d+', b)[:16]]
    common = set(aa) & set(bb)
    values = [len(aa), len(bb), int(bool(aa) and aa[0] in bb),
              int(bool(bb) and bb[0] in aa), int(bool(aa) and set(aa) == set(bb))]
    events = []
    if aa and bb:
        x = aa[0]

        y = min(bb, key=lambda z: (Levenshtein.distance(x, z),
                                   abs(int(x[-12:])-int(z[-12:]))))
        distance = Levenshtein.distance(x, y)
        gap = int(y[-12:])-int(x[-12:])
        last = int(len(x) == len(y) and x[:-1] == y[:-1] and x != y)
        values.extend([math.log1p(abs(gap)), distance, int(len(x) == len(y)), last])
        bucket = str(gap) if abs(gap) <= 10 else ('positive' if gap > 0 else 'negative')
        events.extend(['number:gap:'+bucket, 'number:edit:'+str(min(distance, 5)),
                       'number:last_only:'+str(last), 'number:equal:'+str(x == y)])
    else:
        values.extend([-1., -1., -1., -1.])
    values.extend([len(common), int(bool(aa and bb) and aa[0] in bb and aa[0] != bb[0])])
    return values, events


def field_evidence(raw_a, raw_b):
    'reusable edits, not whole business identities or a fixed similarity cutoff'
    raw_a, raw_b = raw_a or '', raw_b or ''
    a, b = clean(raw_a), clean(raw_b)
    ta, tb = a.split(), b.split()
    ca, cb = Counter(ta), Counter(tb)
    common = sum((ca & cb).values())
    left, right = sorted((ca-cb).elements()), sorted((cb-ca).elements())

    sa, sb = ' '.join(sorted(ta)), ' '.join(sorted(tb))
    operations = Levenshtein.editops(sa, sb)
    events = ['bias', 'empty:'+str(not bool(a))+':'+str(not bool(b)),
              'exact:'+str(a == b), 'bag_equal:'+str(sa == sb)]
    counts = Counter()
    positions = []
    for tag, i, j in operations:
        x = sa[i] if tag != 'insert' else '_'
        y = sb[j] if tag != 'delete' else '_'
        counts[tag] += 1
        counts['digit'] += int(x.isdigit() or y.isdigit())
        counts['space'] += int(x == ' ' or y == ' ')
        positions.append(i/max(len(sa), 1))
        events.append('char:'+tag+':'+x+':'+y)
        events.append('shape:'+tag+':'+('D' if x.isdigit() else 'A' if x.isalpha() else 'S')+
                      ':'+('D' if y.isdigit() else 'A' if y.isalpha() else 'S'))


    remaining = list(right)
    for x in left[:12]:
        if remaining:
            y = max(remaining, key=lambda z: fuzz.ratio(x, z))
            remaining.remove(y)
            events.append('word:replace:'+x[:32]+'>'+y[:32])
        else:
            events.append('word:delete:'+x[:32])
    events.extend('word:insert:'+y[:32] for y in remaining[:12])
    for count_name, count in (('left', len(left)), ('right', len(right)),
                              ('common', common), ('char', len(operations))):
        events.append(count_name+':'+str(min(count, 12)))
    prefix = 0
    for x, y in zip(sa, sb):
        if x != y:
            break
        prefix += 1
    suffix = 0
    for x, y in zip(reversed(sa), reversed(sb)):
        if x != y:
            break
        suffix += 1
    ff = [len(a), len(b), len(ta), len(tb), common, len(left), len(right),
                common/max(len(ta), 1), common/max(len(tb), 1), len(operations),
                len(operations)/max(len(sa), len(sb), 1), counts['replace'],
                counts['insert'], counts['delete'], counts['digit'], counts['space'],
                min(positions, default=-1.), max(positions, default=-1.),
                prefix/max(len(sa), len(sb), 1), suffix/max(len(sa), len(sb), 1),
                int(a == b), int(not a), int(not b),
                int(any(ord(x) > 127 for x in raw_a)), int(any(ord(x) > 127 for x in raw_b))]
    nums, numeric_events = _number_shape(a, b)
    ff.extend(nums)
    events.extend(numeric_events)

    return np.asarray(ff, dtype=np.float32), sorted(set(events))


def _build_part(task):
    from sklearn.feature_extraction import FeatureHasher
    src, dst = map(Path, task)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = pl.read_parquet(src)
    ff = np.empty((rows.height, len(STRUCTURE_COLUMNS)), dtype=np.float32)
    events_name, events_address = [], []
    for i, (n1, n2, a1, a2) in enumerate(rows.select('nm1', 'nm2', 'ad1', 'ad2').iter_rows()):
        n, ne = field_evidence(n1, n2)
        a, ae = field_evidence(a1, a2)
        ff[i] = np.concatenate((n, a))
        events_name.append(ne)
        events_address.append(ae)
    hasher = FeatureHasher(n_features=DIMENSION, input_type='string',
                           alternate_sign=False, dtype=np.float32)
    for name, events in [('name', events_name), ('address', events_address)]:
        sparse.save_npz(str(dst)+'.'+name+'.npz', hasher.transform(events), compressed=False)
    rows.select('qid', 'tid').hstack(pl.DataFrame(ff, schema=STRUCTURE_COLUMNS)).write_parquet(dst)
    return rows.height


def build_features(prepared, work, workers):
    tasks = []
    for split in ('train', 'test'):
        for part in sorted((Path(prepared)/split/'requests').glob('*.parquet')):
            tasks.append((str(part), str(Path(work)/split/part.name)))
    if not tasks:
        raise ValueError('No frozen request shards')
    if workers == 1:
        for i, count in enumerate(map(_build_part, tasks), 1):
            log(f'Edit features {i}/{len(tasks)} shards; {count:,} pairs')
    else:
        keys = ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
        prev = {k: os.environ.get(k) for k in keys}
        os.environ.update({k: '1' for k in keys})
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                for i, count in enumerate(pool.map(_build_part, tasks), 1):
                    log(f'Edit features {i}/{len(tasks)} shards; {count:,} pairs')
        finally:
            for k, value in prev.items():
                if value is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = value


def read_features(prepared, work, split):
    paths = sorted((Path(work)/split).glob('*.parquet'))
    structural = pl.concat([pl.read_parquet(p) for p in paths])
    base = pl.read_parquet(Path(prepared)/split/'features.parquet')
    frame = structural.join(base, on=['qid', 'tid'], how='left', maintain_order='left')
    if frame.height != base.height or frame['parent_p'].null_count():
        raise ValueError('Edit feature coverage differs from frozen selection')
    if frame.select('qid', 'tid').n_unique() != frame.height:
        raise ValueError('Duplicate edit feature pair')
    return frame, paths


def channel_partitions(frame, refs):
    'Channel supervision never sees tuning/audit owners OR candidates'
    d = frame.select('qid', 'tid', 'own', 'fold').join(
        refs.select(pl.col('rid').cast(pl.Int64).alias('own'), pl.col('fold').alias('_owner_fold')),
        on='own', how='left', maintain_order='left')
    allowed = ((pl.col('fold') == 2) & (pl.col('_owner_fold') == 2) & (pl.col('own') >= 0)).fill_null(False)
    val = (pl.col('own').cast(pl.Int64).hash(2917)%10) == 0
    part = d.select((allowed & ~val).alias('fit'), (allowed & val).alias('valid'))
    return part['fit'].to_numpy(), part['valid'].to_numpy()


def head_partitions(frame, refs, cv=3):
    'retain legitimate fit-fold-2 competing negatives; no heldout-owner labels'
    owners = refs.select(pl.col('rid').cast(pl.Int64).alias('own'), pl.col('fold').alias('_owner_fold'))
    d = frame.select('qid', 'tid', 'own', 'fold').join(owners, on='own', how='left', maintain_order='left')
    candidate_fit = (pl.col('fold') == 0) & ((pl.col('qid').hash(1033)%5) != 0)
    owner_fit = ((pl.col('_owner_fold') == 2) | ((pl.col('_owner_fold') == 0) &
        ((pl.col('own').clip(lower_bound=0).cast(pl.UInt32).hash(1033)%5) != 0))).fill_null(False)
    decoy = pl.col('own') < 0
    decoy_fit = decoy & ((pl.col('tid').hash(713)%100) < 8)
    group = pl.when(decoy).then(-pl.col('tid').cast(pl.Int64)-1).otherwise(pl.col('own'))
    d = d.select((candidate_fit & (owner_fit | decoy_fit)).alias('fit'),
                 (group.hash(42)%cv).alias('cv'),
                 pl.when(decoy).then(12.5).otherwise(1.).alias('weight'))
    return d['fit'].to_numpy(), d['cv'].to_numpy(), d['weight'].to_numpy().astype(np.float32)


def train_channels(frame, paths, refs, bundle, max_fit=600_000, epochs=5):
    import joblib
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import log_loss
    fit, valid = channel_partitions(frame, refs)
    rng = np.random.default_rng(7601)
    fit_ids, val_ids = np.flatnonzero(fit), np.flatnonzero(valid)
    if len(fit_ids) > max_fit:
        fit_ids = np.sort(rng.choice(fit_ids, max_fit, replace=False))
    if len(val_ids) > 100_000:
        val_ids = np.sort(rng.choice(val_ids, 100_000, replace=False))
    y = frame['y'].to_numpy().astype(np.int32)
    if min(len(fit_ids), len(val_ids)) == 0 or np.unique(y[fit_ids]).size != 2:
        raise ValueError('Edit-channel fold-2 partition lacks training/validation classes')
    models, report = {}, {}
    for name in ('name', 'address'):
        X = sparse.vstack([sparse.load_npz(str(p)+'.'+name+'.npz') for p in paths], format='csr')
        train, holdout = X[fit_ids], X[val_ids]
        model = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5,
            learning_rate='optimal', average=True, random_state=42)
        history, best_loss, best_blob = [], float('inf'), None
        import pickle
        for epoch in range(epochs):
            order = rng.permutation(len(fit_ids))
            for start in range(0, len(order), 50_000):
                ids = order[start:start+50_000]
                model.partial_fit(train[ids], y[fit_ids[ids]], classes=np.array([0, 1]))
            loss = float(log_loss(y[val_ids], model.predict_proba(holdout), labels=[0, 1]))
            history.append(loss)
            if loss < best_loss:
                best_loss, best_blob = loss, pickle.dumps(model)
            log(f'{name} channel epoch {epoch+1}/{epochs}: owner-heldout logloss {loss:.5f}')
        model = pickle.loads(best_blob)
        models[name] = model
        joblib.dump(model, Path(bundle)/f'channel_{name}.joblib')
        scores = np.clip(model.decision_function(X), -30., 30.).astype(np.float32)
        frame = frame.with_columns(pl.Series('channel_'+name+'_lg', scores))
        report[name] = {'fit_pairs': len(fit_ids), 'validation_pairs': len(val_ids),
            'fit_owners': frame['own'].gather(fit_ids).n_unique(),
            'fit_positive_rate': float(y[fit_ids].mean()), 'validation_logloss': history,
            'selected_epoch': int(np.argmin(history))+1, 'hash_dimension': DIMENSION,
            'partition': 'both candidate and true owner fold 2; internal true-owner holdout'}
        del X, train, holdout
        gc.collect()
    return frame, models, report


def fit_head(frame, columns, fit, groups, weights, settings, tag):
    import lightgbm as lgb
    from er.stack.features import as_matrix
    from er.stack.pipeline import lgb_params
    from er.safe import THREADS
    X = as_matrix(frame, columns)
    y = frame['y'].to_numpy().astype(np.float32)
    pred = np.zeros(frame.height, dtype=np.float64)
    other = np.flatnonzero(~fit)
    models, history = [], []
    for fold in range(settings['cv']):
        tr, va = fit & (groups != fold), fit & (groups == fold)
        if np.unique(y[tr]).size < 2 or not va.any():
            raise ValueError('Empty head CV class/partition')
        a = lgb.Dataset(X[tr], label=y[tr], weight=weights[tr], feature_name=columns)
        b = lgb.Dataset(X[va], label=y[va], weight=weights[va], reference=a)
        model = lgb.train(lgb_params(settings), a,
            num_boost_round=settings['fit']['num_boost_round'], valid_sets=[b],
            callbacks=[lgb.early_stopping(60, verbose=False)])
        pred[va] = model.predict(X[va], num_threads=THREADS)
        for start in range(0, len(other), 500_000):
            ids = other[start:start+500_000]
            pred[ids] += model.predict(X[ids], num_threads=THREADS)/settings['cv']
        models.append(model)
        history.append({'best_iteration': model.best_iteration,
                        'logloss': model.best_score['valid_0']['binary_logloss']})
        log(f'{tag} owner CV {fold+1}/{settings["cv"]}: {history[-1]}')
    gain = np.mean([m.feature_importance('gain') for m in models], axis=0)
    return pred.astype(np.float32), models, {'folds': history, 'fit_rows': int(fit.sum()),
        'fit_targets': frame.filter(pl.Series(fit))['tid'].n_unique(),
        'importance': sorted(zip(columns, gain.tolist()), key=lambda x: -x[1])[:35],
        'decoy_inverse_sampling_weight': 12.5}


def run(data, parent, prepared, output, workers=24, rounds=1000, work=None):
    from er.stack import decode
    from er.stack.features import feature_names
    from er.stack.inputs import load_refs, load_targets
    from er.stack.pipeline import tuning_anchors, predict, export
    from graph_resolution.pipeline import subset_for_tuning, compare_audit
    from innovation.train import mix
    output = Path(output)
    bundle = output/'bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    work = Path(work or '/tmp/amazites-edit-channel-cache')
    cfg = {'workers': workers, 'rounds': rounds, 'channel_max_fit_pairs': 600_000,
        'channel_epochs': 5, 'parent': str(parent), 'prepared': str(prepared),
        'method': 'disjoint supervised sparse name/address edit channels + selective residual patch'}
    (output/'configuration.json').write_text(json.dumps(cfg, indent=2))
    build_features(prepared, work, workers)
    tr, paths = read_features(prepared, work, 'train')
    refs = load_refs(data, 'train')
    tr, channels, channel_report = train_channels(tr, paths, refs, bundle)
    (output/'channel_report.json').write_text(json.dumps(channel_report, indent=2))
    settings = {'cv': 3, 'seed': 42, 'fit_fold': 0, 'tune_holdout_buckets': 5,
        'fit': {'num_boost_round': rounds, 'early_stopping_rounds': 60},
        'lgbm': {'learning_rate': .035, 'num_leaves': 63, 'min_data_in_leaf': 80,
                 'lambda_l2': 8., 'feature_fraction': .9, 'deterministic': True,
                 'force_col_wise': True, 'seed': 42}}
    fit, groups, weights = head_partitions(tr, refs, settings['cv'])
    columns = [c for c in feature_names(tr.columns) if c not in {'own', 'co', 'parent_p'}
               and not c.startswith('_')]
    independent = [c for c in columns if not c.startswith(('prob_', 's1_'))]
    orders = {'structure_control': [c for c in columns if c not in CHANNEL_COLUMNS],
              'edit_channel': columns, 'independent_channel': independent}
    predictions, models, training = {}, {}, {}
    for name, cols in orders.items():
        p, ms, report = fit_head(tr, cols, fit, groups, weights, settings, name)
        predictions[name], models[name], training[name] = p, ms, report
        for i, model in enumerate(ms):
            model.save_model(str(bundle/f'{name}_{i}.txt'))
    (bundle/'features.json').write_text(json.dumps(orders, indent=2))
    pred, parent_report = read_parent(parent, 'train')
    anchors = refs.select(pl.col('rid').alias('qid'), 'fold', 'deg', 'co')
    tune = tuning_anchors(anchors, settings)
    audit = anchors.filter(pl.col('fold') == 1).select('qid', 'deg')
    parent_decision = parent_report['selected_decoder']
    baseline = decode.apply(parent_decision['rule'], pred, parent_decision['threshold'])
    small = subset_for_tuning(pred, tune)
    base_tune = decode.score(baseline, tune)
    trials = {'baseline': {'kind': 'baseline', 'weight': 0., 'rule': parent_decision['rule'],
        'threshold': parent_decision['threshold'], 'tune': base_tune, 'eligible': True}}
    tune_countries = tune.join(anchors.select('qid', 'co'), on='qid') if 'co' not in tune.columns else tune


    for name, scores in predictions.items():
        for weight in (.25, .5, 1.):
            patched = replace_scores(small, mix(tr, scores, weight))
            chosen = decode.tune(patched, tune, rules=('top1_threshold', 'expected_f'), floors=(.3, .5))
            accepted = decode.apply(chosen['rule'], patched, chosen['threshold'])
            country_delta = {}
            for country in tune_countries['co'].unique().sort():
                sub = tune_countries.filter(pl.col('co') == country).select('qid', 'deg')
                country_delta[country] = (decode.score(accepted, sub)['macro_f05']-
                                          decode.score(baseline, sub)['macro_f05'])
            gain = chosen['macro_f05']-base_tune['macro_f05']
            eligible = gain >= .0001 and min(country_delta.values()) >= -.0002
            key = f'{name}_{weight}'
            trials[key] = {'kind': name, 'weight': weight, 'rule': chosen['rule'],
                'threshold': chosen['threshold'], 'tune': {'macro_f05': chosen['macro_f05']},
                'tune_country_delta': country_delta, 'eligible': eligible}
            log(f'Edit trial {key}: tune F0.5 {chosen["macro_f05"]:.7f}; eligible={eligible}; {country_delta}')
    winner = max((key for key in trials if trials[key]['eligible']),
                 key=lambda key: trials[key]['tune']['macro_f05'])
    decision = trials[winner]
    final = pred if winner == 'baseline' else replace_scores(
        pred, mix(tr, predictions[decision['kind']], decision['weight']))
    accepted = decode.apply(decision['rule'], final, decision['threshold'])
    country_metrics = {}
    for country in anchors['co'].unique().sort():
        sub = anchors.filter((pl.col('fold') == 1) & (pl.col('co') == country)).select('qid', 'deg')
        country_metrics[country] = {'baseline': decode.score(baseline, sub), 'selected': decode.score(accepted, sub)}
    report = {'selected': winner, 'decision': decision, 'tuning': trials,
        'channels': channel_report, 'training': training, 'audit': decode.score(accepted, audit),
        'baseline_audit': decode.score(baseline, audit), 'audit_by_country': country_metrics,
        'comparison': compare_audit(accepted, baseline, audit),
        'selection': json.loads((Path(prepared)/'selection.json').read_text()),
        'promotion_policy': 'tune gain >=0.0001 and no tune country loss worse than 0.0002',
        'limitations': ['Historical audit has informed previous research; not a fresh blind benchmark.',
            'French accuracy cannot be measured without labels; no projected test scores used for selection.',
            'Unchanged pool cannot recover missing true candidates or resolve information-free name collisions.',
            'Channels reuse labeled fold 2, which the upstream E5 was also trained on.',
            'Head CV on fold-2 negatives is not nested around channel fitting; tune/audit remain disjoint.',
            'Parent predictions inherit the historical fitting methodology.']}
    report['comparison']['reference'] = 'graph sibling refinement with frozen decoder'
    (output/'report.json').write_text(json.dumps(report, indent=2))
    (bundle/'decision.json').write_text(json.dumps(decision, indent=2))
    final.write_parquet(output/'validation_predictions.parquet')
    log(f'EDIT AUDIT: {report["audit"]}; paired comparison {report["comparison"]}')
    del tr, pred, final, small, baseline, accepted, predictions
    gc.collect()
    test, _ = read_parent(parent, 'test')
    if winner != 'baseline':
        te, paths = read_features(prepared, work, 'test')
        for name, model in channels.items():
            X = sparse.vstack([sparse.load_npz(str(p)+'.'+name+'.npz') for p in paths], format='csr')
            te = te.with_columns(pl.Series('channel_'+name+'_lg',
                np.clip(model.decision_function(X), -30., 30.).astype(np.float32)))
            del X
            gc.collect()
        name = decision['kind']
        p = predict(models[name], te, orders[name])
        test = replace_scores(test, mix(te, p, decision['weight']))
        del te
        gc.collect()
    test.write_parquet(output/'test_predictions.parquet')
    report['export'] = export(test, 'p', decision['rule'], decision['threshold'],
        load_refs(data, 'test'), load_targets(data, 'test'), str(output/'output'))
    (output/'report.json').write_text(json.dumps(report, indent=2))
    (output/'_SUCCESS').write_text('complete\n')
    log('Edit-channel experiment and export complete')
