'hard competitors, protected labels, and target-group cv isolation'
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from latest_fusion.competition_training import select_target_group_train, fit_competition_heads, _hash64


def population():

    pool = pl.DataFrame({'qid': [1, 2, 50, 90, 1, 90, 1, 50, 1, 50],
        'tid': [10, 10, 10, 10, 20, 20, 30, 30, 31, 31],
        'own': [1, 1, 1, 1, 90, 90, -1, -1, -1, -1],
        'y': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'target_address_empty': [1.]*10})
    refs = pl.DataFrame({'rid': pl.Series([1, 2, 50, 90], dtype=pl.UInt32)})
    return pool.with_columns(pl.col('qid', 'tid').cast(pl.UInt32)), refs


def test_retains_hard_competitors_and_never_uses_protected_target_labels():
    pool, refs = population()
    sel, summary = select_target_group_train(pool, refs, [1, 2], [90], orphan_fraction=1.)
    assert sel.filter(pl.col('tid') == 10)['qid'].to_list() == [1, 2, 50]
    assert sel.filter(pl.col('tid') == 20).height == 0
    assert sel.filter(pl.col('qid') == 90).height == 0
    assert summary['legacy']['wrong_owner_negative'] == 1
    assert summary['wrong_owner_negative'] == 2
    assert summary['additional_wrong_owner_negative'] == 1
    assert summary['missing_address'] == {'positive': 1, 'wrong_owner_negative': 2, 'orphan_negative': 4}

    changed = pool.with_columns(pl.when(pl.col('tid') == 20).then(999).otherwise(pl.col('y')).alias('y'))
    again, diagnostics = select_target_group_train(changed, refs, [1, 2], [90], orphan_fraction=1.)
    assert sel.equals(again) and summary == diagnostics
    corrupt = pool.with_columns(pl.when(pl.col('tid') == 10).then(1-pl.col('y')).otherwise(pl.col('y')).alias('y'))
    with pytest.raises(ValueError, match='labels disagree'):
        select_target_group_train(corrupt, refs, [1, 2], [90])


def test_orphans_are_sampled_and_split_as_complete_reproducible_target_groups():
    tids = np.repeat(np.arange(100, 250, dtype=np.uint32), 2)
    pool = pl.DataFrame({'qid': np.tile(np.array([1, 50], dtype=np.uint32), 150),
        'tid': tids, 'own': [-1]*300, 'y': [0]*300})
    refs = pl.DataFrame({'rid': pl.Series([1, 50], dtype=pl.UInt32)})
    first, _ = select_target_group_train(pool, refs, [1], [], orphan_fraction=1/3)
    second, _ = select_target_group_train(pool.reverse(), refs, [1], [], orphan_fraction=1/3)
    assert 20 < first['tid'].n_unique() < 80
    assert first.sort('tid', 'qid').equals(second.sort('tid', 'qid'))
    assert first.group_by('tid').agg(pl.len().alias('n'), pl.col('_fold').n_unique().alias('f')).filter(
        (pl.col('n') != 2) | (pl.col('f') != 1)).height == 0
    assert first['_group'].max() < 0
    empty, _ = select_target_group_train(pool, refs, [1], [], orphan_fraction=0.)
    assert empty.height == 0


def test_fails_on_overlap_unknown_reference_and_inconsistent_ownership():
    pool, refs = population()
    with pytest.raises(ValueError, match='overlap'):
        select_target_group_train(pool, refs, [1], [1])
    with pytest.raises(ValueError, match='Unknown candidate'):
        select_target_group_train(pool, refs.filter(pl.col('rid') != 50), [1], [90])
    corrupt = pool.with_columns(pl.when((pl.col('tid') == 10) & (pl.col('qid') == 2)).then(2).otherwise(pl.col('own')).alias('own'))
    with pytest.raises(ValueError, match='Inconsistent target'):
        select_target_group_train(corrupt, refs, [1, 2], [90])


def test_real_heads_hold_out_targets_true_owners_and_candidate_positive_owners(tmp_path, monkeypatch):
    monkeypatch.setenv('ER_THREADS', '2')
    owners = np.arange(120, dtype=np.uint32)
    qid = np.stack([owners, np.roll(owners, -1), np.full(120, 500, dtype=np.uint32)], axis=1).ravel()
    pool = pl.DataFrame({'qid': qid, 'tid': np.repeat(owners+1000, 3),
        'own': np.repeat(owners.astype(np.int64), 3), 'y': np.tile([1, 0, 0], 120),
        'evidence': np.tile([1., -.6, -.9], 120), 'target_address_empty': [1.]*360})
    refs = pl.DataFrame({'rid': np.r_[owners, np.uint32(500)]})
    cfg = {'folds': 3, 'orphan_fraction': 1/3, 'minimum_fit_positive': 10,
        'minimum_fit_negative': 10, 'rounds': 12, 'early_stopping': 3,
        'lgbm': {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
                 'num_leaves': 3, 'min_data_in_leaf': 3, 'learning_rate': .2}}
    models, summary = fit_competition_heads(pool, refs, owners, [], ['evidence'], cfg, tmp_path)
    assert len(models) == 3 and summary['positive'] == 120
    assert summary['wrong_owner_negative'] == 240 and summary['legacy']['wrong_owner_negative'] == 120
    assert all((tmp_path/f'rescue_model_{fold}.txt').exists() for fold in range(3))
    owner_folds = _hash64(owners.astype(np.uint64)) % 3
    for fold, diagnostics in enumerate(summary['folds']):
        assert diagnostics['target_overlap'] == diagnostics['group_overlap'] == diagnostics['true_owner_overlap'] == 0
        assert diagnostics['held_candidate_owner_pairs_in_train'] == 0
        assert diagnostics['withheld_candidate_owner_pairs'] > 0
        assert diagnostics['nonfit_candidate_reference_overlap'] == 1

        assert diagnostics['valid']['pairs'] == 3*int((owner_folds == fold).sum())
