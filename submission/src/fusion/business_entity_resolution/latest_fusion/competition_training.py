'target-group training that keeps wrong-owner competitors without holdout labels'
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from .pipeline import log


def _hash64(values):
    'version-independent splitmix64 for reproducible group assignments'
    x = np.asarray(values, dtype=np.uint64)
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _counts(frame):
    positive = int(frame['y'].sum())
    orphan = int((frame['own'] < 0).sum())
    res = {'pairs': frame.height, 'positive': positive,
            'true_owner_pairs': positive, 'wrong_owner_negative': frame.height-positive-orphan,
            'orphan_negative': orphan, 'targets': frame['tid'].n_unique(),
            'owned_targets': frame.filter(pl.col('own') >= 0)['tid'].n_unique()}
    if 'target_address_empty' in frame:
        res['missing_address'] = {name: frame.filter((pl.col('target_address_empty') > 0) & condition).height
            for name, condition in {'positive': pl.col('y') == 1,
                'wrong_owner_negative': (pl.col('own') >= 0) & (pl.col('y') == 0),
                'orphan_negative': pl.col('own') < 0}.items()}
    return res


def select_target_group_train(pool, refs, fitids, protectedids, *, folds=3, orphan_fraction=1/3):
    'derive labels only for fit-owned targets and sampled unowned targets'
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError('folds must be an integer >= 2')
    if not np.isfinite(orphan_fraction) or not 0 <= orphan_fraction <= 1:
        raise ValueError('orphan_fraction must be in [0, 1]')
    fitids = np.asarray(fitids, dtype=np.uint32)
    protectedids = np.asarray(protectedids, dtype=np.uint32)
    if np.intersect1d(fitids, protectedids).size:
        raise ValueError('Fit and protected owners overlap')
    if refs['rid'].n_unique() != refs.height:
        raise ValueError('Duplicate reference IDs')
    if pool.select('qid', 'tid').n_unique() != pool.height:
        raise ValueError('Duplicate candidate pairs')
    known = refs.select(pl.col('rid').cast(pl.UInt32).alias('qid'))
    if pool.select('qid').unique().join(known, on='qid', how='anti').height:
        raise ValueError('Unknown candidate reference')
    allowed = pool.filter((pl.col('own') < 0) |
                          pl.col('own').is_in(pl.Series(fitids).cast(pl.Int64).implode()))
    if allowed.group_by('tid').agg(pl.col('own').n_unique().alias('_n')).filter(pl.col('_n') != 1).height:
        raise ValueError('Inconsistent target ownership')
    if 'y' in allowed and allowed.filter(pl.col('y').is_null() |
            (pl.col('y') != (pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8))).height:
        raise ValueError('Training labels disagree with allowed target ownership')

    allowed = allowed.with_columns((pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8).alias('y'))
    legacy = allowed.filter(pl.col('qid').is_in(pl.Series(fitids).implode()))
    orphan_hash = _hash64(allowed['tid'].to_numpy().astype(np.uint64) ^ np.uint64(0xA24BAED4963EE407))
    sampled = orphan_hash.astype(np.float64)/float(2**64) < orphan_fraction
    sel = allowed.filter((pl.col('own') >= 0) | pl.Series(sampled))
    sel = sel.filter(~pl.col('qid').is_in(pl.Series(protectedids).implode()))


    groups = np.where(sel['own'].to_numpy() >= 0, sel['own'].to_numpy(),
                      -sel['tid'].to_numpy().astype(np.int64)-2).astype(np.int64)
    fold = (_hash64(groups.view(np.uint64)) % np.uint64(folds)).astype(np.uint8)
    sel = sel.with_columns(pl.Series('_group', groups), pl.Series('_fold', fold))
    summary = {**_counts(sel), 'legacy': _counts(legacy),
               'expanded_before_orphan_sampling': _counts(allowed.filter(~pl.col('qid').is_in(pl.Series(protectedids).implode()))),
               'fit_owners': len(np.unique(fitids)), 'protected_owners': len(np.unique(protectedids)),
               'protected_candidate_pairs': 0, 'protected_true_owner_pairs': 0,
               'orphan_fraction': orphan_fraction,
               'isolation': 'Target and true-owner groups withheld, including held positive owners as candidate references. Non-fit candidate references may recur as negative evidence. Protected candidate owners excluded from all fitting. Historical upstream exposure and shared target competition prevent a pristine pipeline check.',
               'group_assignment': 'SplitMix64(owner ID), or SplitMix64(-(orphan tid+2)); modulo folds'}
    summary['additional_wrong_owner_negative'] = summary['wrong_owner_negative']-summary['legacy']['wrong_owner_negative']
    return sel, summary


def fit_competition_heads(pool, refs, fitids, protectedids, features, cfg, output):
    'fit three (or configured) heads with complete held-target competition'
    if set(features) & {'qid', 'tid', 'own', 'y', 'fold', 'deg', 'co', '_group', '_fold'}:
        raise ValueError('Label or identifier entered competition feature list')
    fit, summary = select_target_group_train(pool, refs, fitids, protectedids,
        folds=cfg['folds'], orphan_fraction=cfg.get('orphan_fraction', 1/3))
    summary['folds'] = []
    if summary['positive'] < cfg['minimum_fit_positive'] or fit.height-summary['positive'] < cfg['minimum_fit_negative']:
        return [], {**summary, 'reason': 'Insufficient isolated competition training examples'}
    params = {**cfg['lgbm'], 'num_threads': int(os.environ.get('ER_THREADS', '48'))}
    models = []
    fitids = np.asarray(fitids, dtype=np.uint32)
    owner_folds = _hash64(fitids.astype(np.uint64)) % np.uint64(cfg['folds'])
    for fold in range(cfg['folds']):
        held_owners = fitids[owner_folds == fold]
        available_train = fit.filter(pl.col('_fold') != fold)
        train = available_train.filter(~pl.col('qid').is_in(pl.Series(held_owners).implode()))
        valid = fit.filter(pl.col('_fold') == fold)
        if train['y'].n_unique() < 2 or valid['y'].n_unique() < 2:
            return [], {**summary, 'reason': 'Insufficient positive and negative examples in target-group CV'}
        true_train = train.filter(pl.col('own') >= 0)['own'].unique().to_numpy()
        true_valid = valid.filter(pl.col('own') >= 0)['own'].unique().to_numpy()
        diagnostics = {'fold': fold, 'fit': _counts(train), 'valid': _counts(valid),
            'target_overlap': len(np.intersect1d(train['tid'].unique(), valid['tid'].unique())),
            'group_overlap': len(np.intersect1d(train['_group'].unique(), valid['_group'].unique())),
            'true_owner_overlap': len(np.intersect1d(true_train, true_valid)),
            'withheld_candidate_owner_pairs': available_train.height-train.height,
            'held_candidate_owner_pairs_in_train': train.filter(pl.col('qid').is_in(pl.Series(held_owners).implode())).height,
            'candidate_reference_overlap': len(np.intersect1d(train['qid'].unique(), valid['qid'].unique())),
            'nonfit_candidate_reference_overlap': len(np.setdiff1d(np.intersect1d(train['qid'].unique(), valid['qid'].unique()), fitids))}
        if any(diagnostics[k] for k in ('target_overlap', 'group_overlap', 'true_owner_overlap')):
            raise ValueError('Target-group isolation failed')
        log(f'competition head {fold+1}/{cfg["folds"]}: {train.height:,} pairs; {diagnostics["fit"]["wrong_owner_negative"]:,} wrong-owner negatives')
        dataset = lgb.Dataset(train.select(features).to_numpy(), label=train['y'].to_numpy(), feature_name=features)
        held = lgb.Dataset(valid.select(features).to_numpy(), label=valid['y'].to_numpy(), reference=dataset)
        model = lgb.train({**params, 'seed': 1801+fold}, dataset, num_boost_round=cfg['rounds'],
                          valid_sets=[held], callbacks=[lgb.early_stopping(cfg['early_stopping'], verbose=False)])
        model.save_model(str(output/f'rescue_model_{fold}.txt'))
        diagnostics.update(iteration=model.best_iteration,
                           validation_logloss=model.best_score['valid_0']['binary_logloss'])
        summary['folds'].append(diagnostics)
        models.append(model)
    return models, summary
