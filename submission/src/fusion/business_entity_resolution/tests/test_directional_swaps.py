from pathlib import Path
import sys

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from latest_fusion.directional_swaps import build


def records(names, addresses=None):
    return pl.DataFrame({'rid': list(range(len(names))), 'co': ['france']*len(names),
                         'nm': names, 'ad': addresses or ['10 rue de Paris']*len(names)})


def population():
    refs = records(['Paris Club SARL', 'Paris Club SAS', 'Paris Committee EURL'])
    targets = records(['Paris Committee', 'Paris Committee', 'Paris Club'])
    pairs = pl.DataFrame({'qid': [0, 0, 1, 2, 0], 'tid': [0, 1, 0, 2, 2],
                          'co': ['france']*5, 'gate': [.9]*5, 'y': [0, 1, 0, 1, 1]})
    return pairs, refs, targets


def test_all_edges_whole_business_exclusion_and_distinct_targets():
    pairs, refs, targets = population()
    d, names, summary, table = build(pairs, refs, targets, s2_count=2)
    assert d.select('qid', 'tid').rows() == [(0, 0), (0, 1), (1, 0), (2, 2)]
    first = d.row(0, named=True)
    assert first['ds_forward_edges'] == 1
    assert first['ds_forward_qids'] == 1
    assert first['ds_forward_targets'] == 1
    assert first['ds_forward_address_targets'] == 1
    assert first['ds_reverse_edges'] == 1
    assert first['ds_forward_s2_edges'] == 1
    assert first['ds_reverse_s3_edges'] == 1
    assert first['ds_direction_ratio'] == .5
    assert d['ds_forward_targets'][2] == 2
    assert summary['candidate_pairs'] == 5 and summary['swap_pairs'] == 4
    forward = table.filter(pl.col('_swap_from') == 'club').row(0, named=True)
    assert forward['edges'] == 3 and forward['qids'] == 2 and forward['targets'] == 2
    assert all(d[c].is_finite().all() for c in names)


def test_labels_scores_and_input_order_do_not_determine_features():
    pairs, refs, targets = population()
    left, names, _, table = build(pairs, refs, targets)
    right, other, _, table2 = build(pairs.with_columns(
        (1-pl.col('y')).alias('y'), pl.lit(0.).alias('gate'), pl.lit(-1).alias('own')), refs, targets)
    assert names == other
    assert not set(names) & {'qid', 'tid', 'co', 'y', 'own', 'gate', 'newest', 'fold'}
    assert left.select(names).equals(right.select(names)) and table.equals(table2)
    assert right['own'].to_list() == [-1]*4
    reversed_rows, _, _, _ = build(pairs.reverse(), refs, targets)
    assert reversed_rows.select('qid', 'tid').rows() == left.select('qid', 'tid').reverse().rows()


def test_accent_legal_forms_preserve_category_words_and_two_word_sets():
    refs = records(['Étoile Club S.A.R.L.', 'Paris Grand Club SAS', 'Paris Comité EURL'])
    targets = records(['Etoile committee LLC', 'Paris Petite Amicale', 'Paris Club'])
    pairs = pl.DataFrame({'qid': [0, 1, 2], 'tid': [0, 1, 2]})
    d, _, _, _ = build(pairs, refs, targets)
    assert d['_swap_from'].to_list() == ['club', 'club grand', 'comite']
    assert d['_swap_to'].to_list() == ['committee', 'amicale petite', 'club']
    assert d['ds_missing'].to_list() == [1, 2, 1]
    assert d['ds_two_word'].to_list() == [0, 1, 0]

    assert d['ds_direction_ratio'].to_list() == [.5]*3


def test_exact_alternative_owner_uses_full_refs_not_candidate_pool():
    refs = records(['Paris Club', 'Paris Committee', 'Paris Committee'],
                   ['10 rue de Paris', '10 rue de Paris', '11 rue de Paris'])
    targets = records(['Paris Committee'])
    d, _, _, _ = build(pl.DataFrame({'qid': [0], 'tid': [0]}), refs, targets)
    assert d['ds_exact_alternative_refs'][0] == 1


def test_address_separators_suffixes_and_missing_are_material():
    refs = records(['Paris Club']*5, ['1/2 rue saint Paul', '1-2 rue saint Paul',
                                    '12 bis rue saint Paul', '12 rue saint Paul', ''])
    targets = records(['Paris Committee']*5, ['1-2 rue saint Paul', '12 rue saint Paul',
                                              '12 ter rue saint Paul', '12 rue sainte Paul', ''])
    d, _, summary, _ = build(pl.DataFrame({'qid': range(5), 'tid': range(5)}), refs, targets)
    assert d['ds_address_exact'].to_list() == [0]*5
    assert d['ds_numeric_conflict'].to_list() == [0, 1, 0, 0, 0]
    assert summary['samples'][0]['_r_address'] == '1/2 rue saint paul'
    assert summary['samples'][2]['_r_address'] == '12 bis rue saint paul'


def test_only_eligible_changes_and_no_material_repetition():
    refs = records(['Paris Club Club', 'Paris Club', 'Groupe Paris Club', 'Club',
                    'Paris Club', 'Paris Club'])
    targets = records(['Paris Committee', 'Paris Club', 'Groupe Paris Committee', 'Committee',
                       'Paris Committee Committee', 'Paris One Two Three'])
    d, _, summary, _ = build(pl.DataFrame({'qid': range(6), 'tid': range(6)}), refs, targets)
    assert d.select('qid', 'tid').rows() == [(2, 2)]
    assert summary['repeated_material_token_pairs_excluded'] == 2
    assert summary['samples'][0]['_r_name'] == 'groupe paris club'


def test_duplicate_and_missing_keys_rejected():
    pairs, refs, targets = population()
    with pytest.raises(ValueError, match='Duplicate candidate'):
        build(pl.concat([pairs, pairs.head(1)]), refs, targets)
    with pytest.raises(ValueError, match='missing'):
        build(pl.DataFrame({'qid': [99], 'tid': [0]}), refs, targets)


def test_no_swaps_and_empty_input_have_finite_feature_schema():
    refs, targets = records(['Paris Club']), records(['Paris Club'])
    pairs = pl.DataFrame({'qid': [0], 'tid': [0]})
    for frame in (pairs, pairs.head(0)):
        d, names, summary, table = build(frame, refs, targets)
        assert d.height == 0 and all(c in d.columns for c in names)
        assert summary['swap_pairs'] == 0


def test_unrelated_names_do_not_supply_shared_context_support():
    refs = records(['Paris Club', 'Paris Club', 'Rome Club'])
    targets = records(['Paris Committee', 'Rome Committee'])
    pairs = pl.DataFrame({'qid': [0, 1, 2], 'tid': [0, 0, 1]})
    d, _, summary, _ = build(pairs, refs, targets)
    first = d.row(0, named=True)
    assert first['ds_forward_qids'] == 2
    assert first['ds_context_forward_qids'] == 1
    assert first['ds_context_forward_targets'] == 1
    assert d['ds_context_forward_qids'][2] == 0
    assert summary['shared_name_context_directions'] == 2
