from pathlib import Path
import sys

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from latest_fusion.contrastive_owners import build


def records(names, addresses=None):
    return pl.DataFrame({'rid': list(range(len(names))), 'nm': names,
                         'ad': addresses or [''] * len(names), 'co': ['france'] * len(names)})


def pairs(qids, tids=None, scores=None):
    return pl.DataFrame({'qid': qids, 'tid': tids or [0] * len(qids),
                         'newest': scores or [0.9] * len(qids)})


def test_same_name_branches_distinguish_24_and_26_lake_road():
    refs = records(['Lake Club SARL', 'Lake Club SAS'], ['24 Lake Road', '26 Lake Road'])
    targets = records(['Lake Club'], ['26 Lake Road'])
    frame = pairs([0, 1], scores=[0.95, 0.8])
    d, names, diagnostics = build(frame, refs, targets, chunk_size=1)
    assert len(names) == 24 and all(d[c].dtype == pl.Float64 for c in names)
    assert d.select(frame.columns).equals(frame)
    assert d['ct_name_only_shared'].to_list() == [1, 1]
    assert d['ct_name_shared_support'].to_list() == [2, 2]
    assert d['ct_name_q_distinct'].to_list() == [0, 0]
    for domain in ('address', 'numeric'):
        assert d[f'ct_{domain}_q_distinct'].to_list() == [1, 1]
        assert d[f'ct_{domain}_q_support'].to_list() == [0, 1]
        assert d[f'ct_{domain}_b_support'].to_list() == [1, 0]
        assert d[f'ct_{domain}_q_conflict'].to_list() == [1, 0]
        assert d[f'ct_{domain}_b_conflict'].to_list() == [0, 1]
    assert d['ct_score_margin'].to_list() == pytest.approx([0.15, -0.15])
    assert diagnostics['alternative_pairs'] == 2


def test_category_words_survive_unicode_punctuation_and_legal_normalization():
    refs = records(['Étoile Club S.A.R.L.', 'Etoile Committee LLC'])
    targets = records(['ETOILE, Committee!'])
    d, _, _ = build(pairs([0, 1]), refs, targets)
    assert d['ct_name_q_distinct'].to_list() == [1, 1]
    assert d['ct_name_q_support'].to_list() == [0, 1]
    assert d['ct_name_b_support'].to_list() == [1, 0]
    assert d['ct_name_q_coverage'].to_list() == [0, 1]
    assert d['ct_name_coverage_margin'].to_list() == [-1, 1]
    assert d['ct_name_only_shared'].to_list() == [0, 0]


def test_complete_pool_ties_and_row_order_are_deterministic():
    refs = records(['Shared Alpha', 'Shared Beta', 'Shared Gamma', 'Shared Delta'])
    targets = records(['Shared Alpha'])
    frame = pairs([3, 1, 2, 0], scores=[0.7, 0.9, 0.9, 0.9])
    d, names, _ = build(frame, refs, targets, chunk_size=1)


    assert d['ct_name_b_support'].to_list() == [1, 1, 1, 0]
    assert d['ct_score_margin'].to_list() == pytest.approx([-0.2, 0, 0, 0])
    reverse, _, _ = build(frame.reverse(), refs.reverse(), targets, chunk_size=3)
    assert reverse.select('qid', 'tid', *names).reverse().equals(d.select('qid', 'tid', *names))
    whole, _, _ = build(frame, refs, targets, chunk_size=100)
    assert whole.equals(d)


def test_lone_candidate_has_explicit_missing_alternative_without_invented_owner():
    refs = records(['Club 24', 'Committee 26'], ['24 Lake Road', '26 Lake Road'])
    targets = records(['Club 24'])
    frame = pairs([0])
    d, names, diag = build(frame, refs, targets)
    assert d['ct_alternative_present'][0] == 0
    assert d['ct_single_candidate'][0] == 1
    assert d.select([c for c in names if c != 'ct_single_candidate']).row(0) == (0.,) * 23
    assert diag['single_candidate_pairs'] == 1

    assert d.select(frame.columns).equals(frame)


def test_labels_split_and_ownership_never_affect_selection_or_features():
    refs, targets = records(['City Club', 'City Committee']), records(['City Committee'])
    frame = pairs([0, 1]).with_columns(pl.Series('y', [0, 1]), pl.Series('fold', [1, 2]),
                                       pl.Series('owner', [1, 0]))
    left, names, _ = build(frame, refs, targets)
    changed = frame.with_columns((1 - pl.col('y')).alias('y'), pl.lit(-999).alias('fold'),
                                  pl.lit(17).alias('owner'))
    right, names2, _ = build(changed, refs, targets)
    assert names == names2 and left.select(names).equals(right.select(names))
    assert right.select(changed.columns).equals(changed)
    assert not set(names) & {'qid', 'tid', 'co', 'newest', 'y', 'fold', 'owner'}


def test_address_number_separators_name_numbers_and_missing_text():
    refs = records(['Club 24', 'Club 26'], ['1/2 Lake Road', '1-2 Lake Road'])
    targets = records(['Club 26'], ['1-2 Lake Road'])
    d, _, _ = build(pairs([0, 1]), refs, targets)
    assert d['ct_numeric_q_distinct'].to_list() == [2, 2]
    assert d['ct_numeric_q_support'].to_list() == [0, 2]
    assert d['ct_numeric_b_support'].to_list() == [2, 0]
    blank, names, _ = build(pairs([0, 1]), refs, records([None], [None]))
    assert all(blank[c].is_finite().all() for c in names)
    assert blank['ct_address_q_conflict'].to_list() == [0, 0]
    assert blank['ct_numeric_q_conflict'].to_list() == [0, 0]
    assert blank['ct_name_only_shared'].to_list() == [0, 0]


def test_shared_only_name_evidence_is_observed_not_a_stopword_guess():
    refs = records(['Lake Club North', 'Lake Club South'])
    d, _, _ = build(pairs([0, 1]), refs, records(['Lake Club']))
    assert d['ct_name_shared_support'].to_list() == [2, 2]
    assert d['ct_name_only_shared'].to_list() == [1, 1]
    assert d['ct_name_q_coverage'].to_list() == [0, 0]


def test_empty_frame_and_validation():
    refs, targets = records(['Club', 'Committee']), records(['Club'])
    frame = pairs([0, 1])
    empty, names, diag = build(frame.head(0), refs, targets)
    assert empty.height == 0 and empty.columns == frame.columns + names
    assert diag['targets'] == 0
    for bad, message in [
        (pl.concat([frame, frame.head(1)]), 'Duplicate candidate'),
        (pairs([99]), 'missing'),
        (pairs([0], tids=[99]), 'missing'),
        (frame.with_columns(pl.lit(None, pl.Int64).alias('qid')), 'Missing candidate'),
        (frame.with_columns(pl.lit(float('nan')).alias('newest')), 'finite numeric'),
        (frame.with_columns(pl.lit(None, pl.Float64).alias('newest')), 'finite numeric'),
        (frame.with_columns(pl.lit('high').alias('newest')), 'finite numeric'),
    ]:
        with pytest.raises(ValueError, match=message):
            build(bad, refs, targets)
    with pytest.raises(ValueError, match='Cross-country'):
        build(frame, refs.with_columns(pl.lit('india').alias('co')), targets)
    with pytest.raises(ValueError, match='country mismatch'):
        build(frame.with_columns(pl.lit('india').alias('co')), refs, targets)
    with pytest.raises(ValueError, match='Duplicate record'):
        build(frame, pl.concat([refs, refs.head(1)]), targets)
    with pytest.raises(ValueError, match='positive integer'):
        build(frame, refs, targets, chunk_size=0)
