from pathlib import Path
import sys

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from latest_fusion.structured_protocol import split_references, training_pool, prepare_protocol, _orphan_sample


def references():
    names = ['Étoile S.A.R.L.', 'ETOILE ltd', 'North South Ltd', 'South North Ltd']
    names += [f'Business number {i}' for i in range(120)]
    count = len(names)
    return pl.DataFrame({'rid': range(count + 1), 'eid': [f'original-{i}' for i in range(count + 1)],
        'nm': names + ['Fold zero business'], 'ad': [f'{i} road' for i in range(count + 1)],
        'co': ['france'] * (count + 1), 'fold': [1] * count + [0], 'deg': [1] * (count + 1)})


def pair_frame(rows):
    frame = pl.DataFrame(rows, schema={'qid': pl.Int64, 'tid': pl.Int64, 'own': pl.Int64}, orient='row')
    return frame.with_columns((pl.col('qid') == pl.col('own')).cast(pl.Int8).alias('y'),
                              (pl.col('tid') * .001).alias('observable_feature'))


def fixture():
    refs = references()
    fit, cal, check, _ = split_references(refs)
    a, b = fit['rid'][:2].to_list()
    c, d = cal['rid'][:2].to_list()
    e = check['rid'][0]
    legacy = refs.filter(pl.col('fold') == 0)['rid'].item()
    fraction = fit.height / refs.height
    orphan_tid = next(tid for tid in range(6000, 6100) if _orphan_sample([tid], fraction)[0])
    frame = pair_frame([
        (a, 1000, a), (e, 1000, a),
        (a, 2000, -1), (e, 2000, -1),
        (c, 3000, -1), (e, 3000, -1),
        (c, 3001, a), (a, 3001, a),
        (d, 4000, d), (e, 4001, e),
        (a, 5000, a), (b, 5000, a), (legacy, 5000, a),
        (a, orphan_tid, -1), (legacy, orphan_tid, -1),
    ])
    return refs, frame, (a, b, c, d, e, legacy, orphan_tid)


def test_same_normalized_full_name_groups_are_indivisible_and_preserve_schema():
    refs = references()
    fit, cal, check, report = split_references(refs)
    assigned = []
    for index, part in enumerate((fit, cal, check)):
        assert part.schema == refs.schema
        assert part['fold'].unique().to_list() == [1]
        assigned.extend((rid, index) for rid in part['rid'])
    assignments = dict(assigned)
    assert assignments[0] == assignments[1]
    assert len(assigned) == refs.filter(pl.col('fold') == 1).height
    assert report['partition_counts']['fit']['name_groups'] + report['partition_counts']['calibration']['name_groups'] + report['partition_counts']['check']['name_groups'] == 123
    assert report['excluded_other_fold_references'] == 1
    assert 'not a pristine' in report['historical_upstream_exposure_warning']


def test_assignment_is_stable_under_outcome_address_id_and_row_order_changes():
    refs = references()
    changed = refs.with_columns((1-pl.col('deg')).alias('deg'), pl.lit('new address').alias('ad'),
                               pl.lit(1).alias('y')).reverse()
    left = split_references(refs)
    right = split_references(changed)
    for before, after in zip(left[:3], right[:3]):
        assert before['rid'].sort().equals(after['rid'].sort())

    renumbered = split_references(refs.with_columns((pl.col('rid') + 1000).alias('rid')))
    for before, after in zip(left[:3], renumbered[:3]):
        assert before['rid'].sort().equals((after['rid']-1000).sort())


def test_target_closure_removes_owned_and_orphan_check_targets_from_every_fit_row():
    refs, frame, (_, _, _, _, _, legacy, orphan_tid) = fixture()
    fit, cal, check, _ = split_references(refs)
    fitting, report = training_pool(frame, refs, fit, cal, check)
    assert set(fitting['tid'].to_list()) == {5000, orphan_tid}
    assert fitting.filter((pl.col('qid') == legacy) & (pl.col('tid') == 5000))['y'].item() == 0
    assert fitting.filter(pl.col('tid') == 5000).height == 3
    assert fitting.schema == frame.schema
    assert fitting.equals(frame.filter(pl.col('tid').is_in([5000, orphan_tid])))
    assert report['target_overlap'] == 0
    assert report['held_candidate_qids_in_fit'] == report['held_true_owner_ids_in_fit'] == 0
    assert report['orphan_fraction'] == pytest.approx(fit.height / refs.height)
    assert report['class_counts']['fit']['positive'] == 1
    assert report['class_counts']['fit']['wrong_owner_negative'] == 2
    assert report['class_counts']['fit']['orphan_negative'] == 2


def test_calibration_check_shared_target_drops_entire_cal_owner_and_preserves_original_closure():
    refs, frame, (_, _, dropped, kept, _, _, _) = fixture()
    fit, clean_cal, check, fitting, report = prepare_protocol(refs, frame)
    assert dropped not in clean_cal['rid'].to_list()
    assert kept in clean_cal['rid'].to_list()
    cal_rows = frame.filter(pl.col('qid').is_in(clean_cal['rid'].implode()))
    assert 3001 not in cal_rows['tid'].to_list()
    assert 3001 not in fitting['tid'].to_list()
    assert report['original_calibration_references'] == report['clean_calibration_references'] + 1
    assert report['calibration_owners_dropped_for_check_target_overlap'] == 1
    assert report['calibration_check_target_overlap'] == 0
    assert report['class_counts']['calibration']['pairs'] == 1
    assert report['class_counts']['calibration']['positive'] == 1
    assert report['protected_references'] == report['original_calibration_references'] + check.height
    assert report['fit_references_without_owned_training_targets'] == fit.height-1


@pytest.mark.parametrize('kind', ['flipped', 'null', 'nonbinary', 'owner_inconsistent'])
def test_invalid_training_labels_and_target_ownership_rejected(kind):
    refs, frame, _ = fixture()
    fit, cal, check, _ = split_references(refs)
    if kind == 'flipped':
        frame = frame.with_columns((1-pl.col('y')).alias('y'))
    elif kind == 'null':
        frame = frame.with_columns(pl.lit(None, dtype=pl.Int8).alias('y'))
    elif kind == 'nonbinary':
        frame = frame.with_columns(pl.lit(2).alias('y'))
    else:
        frame = frame.with_columns(pl.when(pl.col('qid') == frame['qid'][1]).then(-1).otherwise(pl.col('own')).alias('own'))
        frame = frame.with_columns((pl.col('qid') == pl.col('own')).cast(pl.Int8).alias('y'))
    with pytest.raises(ValueError, match='labels|ownership'):
        training_pool(frame, refs, fit, cal, check)


def test_empty_fold1_or_candidate_pool_is_supported():
    refs = references().filter(pl.col('fold') == 0)
    empty = pair_frame([])
    fit, cal, check, fitting, report = prepare_protocol(refs, empty)
    assert fit.is_empty() and cal.is_empty() and check.is_empty()
    assert fitting.equals(empty) and report['orphan_fraction'] == 0


def test_unsigned_reference_ids_and_signed_true_owner_ids_are_supported():
    refs, frame, _ = fixture()
    refs = refs.with_columns(pl.col('rid').cast(pl.UInt32))
    frame = frame.with_columns(pl.col('qid', 'tid').cast(pl.UInt32))
    fit, cal, check, fitting, report = prepare_protocol(refs, frame)
    assert fit.schema == refs.schema and cal.schema == refs.schema and check.schema == refs.schema
    assert fitting.schema == frame.schema and report['target_overlap'] == 0
