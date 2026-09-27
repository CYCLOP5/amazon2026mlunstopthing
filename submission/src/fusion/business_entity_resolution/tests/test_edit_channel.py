'small synthetic checks only; challenge-data training runs in azure'
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from innovation.edit_channel import (
    FIELD_FEATURES, STRUCTURE_COLUMNS, clean, field_evidence, channel_partitions,
    head_partitions, build_features, read_features, train_channels, run,
)
from innovation.prepare import run as prepare
from test_stack import make_fixture


def test_noise_prefix_keeps_true_house_number_and_conflict_evidence():
    ff, _ = field_evidence('1958 River Road', 'Door No #834 House No 01958 River Road')
    f = dict(zip(FIELD_FEATURES, ff))
    assert f['first_in_other'] == 1
    assert f['number_prefix_noise'] == 1
    assert f['closest_number_log_gap'] == 0
    conflict, events = field_evidence('1958 River Road', '1959 River Road')
    c = dict(zip(FIELD_FEATURES, conflict))
    assert c['first_in_other'] == 0
    assert c['closest_number_last_digit_only'] == 1
    assert 'number:gap:1' in events


def test_unseen_language_text_preserved_and_empty_fields_finite():
    assert clean('Lille École S.A.R.L.') == 'lille ecole sarl'
    assert clean('Compagnie Fils Groupe') == 'compagnie fils groupe'
    for a, b in [(None, None), ('कंपनी', ''), ('', '12 Rue'), ('5 River Road', 'Road River 5')]:
        ff, events = field_evidence(a, b)
        assert len(ff) == len(FIELD_FEATURES)
        assert np.isfinite(ff).all()
        assert len(events) == len(set(events))
    assert 'bag_equal:True' in field_evidence('5 River Road', 'Road River 5')[1]


def refs_and_rows():
    refs = pl.DataFrame({'rid': np.arange(300, dtype=np.uint32),
                         'fold': [0]*100+[1]*100+[2]*100})
    fit0 = refs.filter((pl.col('fold') == 0) & ((pl.col('rid').hash(1033)%5) != 0))['rid'][0]
    tune0 = refs.filter((pl.col('fold') == 0) & ((pl.col('rid').hash(1033)%5) == 0))['rid'][0]
    rows = pl.DataFrame({'qid': [fit0, fit0, fit0, fit0, 220, 220, 220, 221],
        'tid': [0, 1, 2, 3, 4, 5, 6, 7], 'own': [fit0, 230, 120, tune0, 230, 120, tune0, -1],
        'fold': [0, 0, 0, 0, 2, 2, 2, 2]}).with_columns(pl.col('qid', 'tid').cast(pl.UInt32))
    return refs, rows


def test_channels_exclude_every_heldout_owner_and_candidate():
    refs, rows = refs_and_rows()
    fit, valid = channel_partitions(rows, refs)
    assert (fit | valid).tolist() == [False, False, False, False, True, False, False, False]
    assert not (fit & valid).any()


def test_head_preserves_fit2_rivals_but_excludes_heldout_owners():
    refs, rows = refs_and_rows()
    fit, groups, weights = head_partitions(rows, refs)
    assert fit.tolist() == [True, True, False, False, False, False, False, False]
    assert groups[1] == groups[4]
    assert weights[-1] == 12.5
    assert (weights[:-1] == 1).all()


def fixture(tmp_path, n=240):
    data, roots = make_fixture(tmp_path, n)
    ref_path = data/'train/ref.parquet'
    refs = pl.read_parquet(ref_path).with_columns(
        pl.Series('fold', [0 if i%5 == 0 else 1 if i%5 == 1 else 2 for i in range(n)]))
    refs.write_parquet(ref_path)
    parent, prepared = tmp_path/'parent', tmp_path/'prepared'
    parent.mkdir()
    for split in ('train', 'test'):
        pairs = pl.read_parquet(Path(roots[split][0])/'parts/part_0.parquet').select('qid', 'tid',
            pl.col('prob').alias('base'), pl.col('prob').alias('head'),
            *(['y'] if split == 'train' else []))
        pairs.write_parquet(parent/('validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'))
    (parent/'report.json').write_text(json.dumps({'selected_head_weight': 1.,
        'selected_decoder': {'rule': 'top1_threshold', 'threshold': .5}}))
    prepare(data, parent, prepared, max_targets=10_000)
    return data, parent, prepared, refs


def test_sparse_features_preserve_pair_order_and_channel_label_isolation(tmp_path):
    data, parent, prepared, refs = fixture(tmp_path)
    work, bundle = tmp_path/'work', tmp_path/'bundle'
    bundle.mkdir()
    build_features(prepared, work, workers=2)
    frame, paths = read_features(prepared, work, 'train')
    assert set(STRUCTURE_COLUMNS) <= set(frame.columns)
    for field in ('name', 'address'):
        matrix = sparse.vstack([sparse.load_npz(str(p)+'.'+field+'.npz') for p in paths])
        assert matrix.shape[0] == frame.height
    first, models, _ = train_channels(frame, paths, refs, bundle, epochs=2)
    fit, valid = channel_partitions(frame, refs)
    changed = frame.with_columns(pl.when(pl.Series(~(fit | valid))).then(1-pl.col('y'))
                                .otherwise(pl.col('y')).alias('y'))
    second, other, _ = train_channels(changed, paths, refs, bundle, epochs=2)
    for field in ('name', 'address'):
        np.testing.assert_array_equal(models[field].coef_, other[field].coef_)
        np.testing.assert_array_equal(first['channel_'+field+'_lg'], second['channel_'+field+'_lg'])


def test_synthetic_full_experiment_exports_all_candidates(tmp_path):
    data, parent, prepared, refs = fixture(tmp_path)
    output = tmp_path/'result'
    run(data, parent, prepared, output, workers=1, rounds=5, work=tmp_path/'work')
    report = json.loads((output/'report.json').read_text())
    assert report['selected'] == 'baseline'
    assert report['comparison']['paired_delta'] == 0
    assert report['export']['candidate_pairs'] == 32*2*3
    assert (output/'bundle/channel_name.joblib').exists()
    orders = json.loads((output/'bundle/features.json').read_text())
    assert not {'qid', 'tid', 'y', 'own', 'fold', 'co'} & set(orders['edit_channel'])
    assert 'channel_name_lg' not in orders['structure_control']
    assert not any(c.startswith(('prob_', 's1_')) for c in orders['independent_channel'])
    from scripts.validate_outputs import validate
    assert validate(data, output/'output')['source1'] == 32
