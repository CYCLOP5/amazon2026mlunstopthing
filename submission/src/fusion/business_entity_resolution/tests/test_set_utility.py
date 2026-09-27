from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from latest_fusion.set_utility import build_actions, choose_actions, apply_actions


def pairs(qid=(), tid=(), p=(), y=None):
    schema = {'qid': pl.Int64, 'tid': pl.Int64, 'p': pl.Float64}
    data = {'qid': qid, 'tid': tid, 'p': p}
    if y is not None:
        data['y'] = y
        schema['y'] = pl.Int8
    return pl.DataFrame(data, schema=schema)


def test_exact_owner_utility_includes_empty_singletons_and_nonempty_sets():
    refs = pl.DataFrame({'qid': [0, 1, 2, 3], 'deg': [0, 0, 2, 1], 'co': ['a'] * 4})
    base = pairs([1, 2, 2, 3], [10, 20, 21, 30], [.2, .9, .1, .8], [0, 1, 0, 1])
    winners = pairs([0, 1, 2, 3], [1, 11, 22, 31], [.5, .7, .6, .9], [0, 0, 1, 0])
    actions, _ = build_actions(winners, base, refs)
    got = {(q, op): value for q, op, value in actions.select('qid', 'operation', 'utility_delta').rows()}
    assert got[(0, 1)] == -1
    assert got[(1, -1)] == 1
    assert got[(1, 1)] == 0
    assert got[(2, -1)] == pytest.approx(1.25 / 1.5 - 1.25 / 2.5)
    assert got[(2, 1)] == pytest.approx(2.5 / 3.5 - 1.25 / 2.5)
    assert got[(3, -1)] == -1
    assert got[(3, 1)] == pytest.approx(1.25 / 2.25 - 1)

    assert actions.filter(pl.col('qid') == 0)['accepted_count'].item() == 0


def test_global_occupancy_precedes_owner_scope_and_no_reassignments():
    base = pairs([99, 0], [1, 2], [.5, .2])
    winners = pairs([0, 0, 0], [1, 3, 4], [.99, .6, .4])
    refs = pl.DataFrame({'qid': [0]})
    actions, _ = build_actions(winners, base, refs)
    add = actions.filter(pl.col('operation') == 1)
    assert add['tid'].item() == 3
    assert add['candidate_count'].item() == 2
    assert add['candidate_sum_p'].item() == pytest.approx(1.)
    res = apply_actions(base, choose_actions(actions, [.1] * actions.height))
    assert res.filter(pl.col('qid') == 99).equals(base.filter(pl.col('qid') == 99))
    with pytest.raises(ValueError, match='occupied'):
        apply_actions(base, pl.DataFrame({'qid': [0], 'tid': [1], 'p': [.9], 'operation': [1]}))


def test_label_and_degree_values_do_not_enter_features_and_unlabeled_inputs_ignore_them():
    base = pairs([0], [1], [.5], [0])
    winners = pairs([0, 1], [2, 3], [.9, .8], [0, 0])
    refs = pl.DataFrame({'qid': [0, 1], 'deg': [0, 0], 'co': ['a', 'b']})
    left, ff = build_actions(winners, base, refs)
    right, names = build_actions(winners.with_columns(pl.lit(1).alias('y')),
        base.with_columns(pl.lit(1).alias('y')), refs.with_columns(pl.lit(3).alias('deg'), pl.lit('c').alias('co')))
    assert names == ff
    assert not set(ff) & {'qid', 'tid', 'deg', 'y', 'co', 'utility_delta'}
    assert left.select(ff).equals(right.select(ff))
    unlabeled, names = build_actions(winners.drop('y'), base, refs.with_columns(pl.lit(-999).alias('deg')))
    assert 'y' not in unlabeled and 'utility_delta' not in unlabeled
    assert unlabeled.select(names).equals(left.select(ff))
    assert all(unlabeled[c].is_finite().all() for c in ff)


def test_action_construction_and_selection_ties_are_deterministic():
    base = pairs([0, 0], [12, 11], [.4, .4])
    winners = pairs([0, 0, 1], [10, 9, 20], [.7, .7, 0.])
    refs = pl.DataFrame({'qid': [0, 1]})
    actions, _ = build_actions(winners, base, refs)
    reversed_actions, _ = build_actions(winners.reverse(), base.reverse(), refs.reverse())
    assert actions.equals(reversed_actions)
    assert actions.filter(pl.col('operation') == 1)['tid'].item() == 9
    assert actions.filter(pl.col('operation') == -1)['tid'].item() == 11
    chosen = choose_actions(actions, np.array([.1, .1]))
    assert chosen.height == 1 and chosen['tid'].item() == 9
    assert choose_actions(actions, [0., -1.], threshold=-1).is_empty()
    assert choose_actions(actions, [.1, .2], threshold=.2).is_empty()
    same_target = pl.DataFrame({'qid': [0, 0], 'tid': [1, 1], 'operation': [1, -1], 'score': [.2, .2]})
    assert choose_actions(same_target, 'score')['operation'].item() == -1


def test_no_actions_returns_exact_baseline_and_preserves_schema_variants():
    base = pairs([0, 1], [1, 2], [.5, .8], [1, 0]).rename({'p': 'probability', 'y': 'label'})
    base = base.with_columns(pl.Series('extra', ['original', 'kept']))
    winners = pairs([0], [3], [.9], [1]).rename({'p': 'probability', 'y': 'label'})
    refs = pl.DataFrame({'qid': [0, 1], 'deg': [2, 0]})
    actions, _ = build_actions(winners, base, refs, probability_col='probability', label_col='label')
    empty = choose_actions(actions, [0.] * actions.height)
    assert apply_actions(base, empty, probability_col='probability', label_col='label').equals(base)
    chosen = choose_actions(actions, (actions['operation'] == 1).cast(pl.Float64))
    res = apply_actions(base, chosen, probability_col='probability', label_col='label')
    assert res.schema == base.schema
    assert res.filter(pl.col('tid') == 3)['probability'].item() == .9
    assert res.filter(pl.col('tid') == 3)['label'].item() == 1
    assert res.head(base.height).equals(base)


def test_competing_additions_duplicate_winners_and_multiple_actions_rejected():
    base = pairs()
    refs = pl.DataFrame({'qid': [0, 1]})
    competing = pairs([0, 1], [1, 1], [.8, .9])
    with pytest.raises(ValueError, match='duplicate'):
        build_actions(competing, base, refs)
    chosen = competing.with_columns(pl.lit(1).alias('operation'))
    with pytest.raises(ValueError, match='duplicate'):
        apply_actions(base, chosen)
    with pytest.raises(ValueError, match='duplicate'):
        apply_actions(base, pairs([0, 0], [1, 2], [.8, .9]).with_columns(pl.lit(1).alias('operation')))


@pytest.mark.parametrize('invalid', [None, float('nan'), float('inf'), -.01, 1.01])
def test_invalid_probabilities_rejected_in_pool_baseline_and_additions(invalid):
    refs = pl.DataFrame({'qid': [0]})
    bad = pairs([0], [1], [invalid])
    with pytest.raises(ValueError, match='probabilities'):
        build_actions(bad, pairs(), refs)
    with pytest.raises(ValueError, match='probabilities'):
        build_actions(pairs(), bad, refs)
    with pytest.raises(ValueError, match='probabilities'):
        apply_actions(pairs(), bad.with_columns(pl.lit(1).alias('operation')))


def test_invalid_decisions_and_missing_removals_rejected():
    actions = pairs([0], [1], [.5]).with_columns(pl.lit(-1).alias('operation'))
    for score in [float('nan'), None, float('inf')]:
        with pytest.raises(ValueError, match='finite'):
            choose_actions(actions, [score])
    with pytest.raises(ValueError, match='align'):
        choose_actions(actions, [])
    with pytest.raises(ValueError, match='incumbent'):
        apply_actions(pairs(), actions)


def test_empty_pool_and_baseline_produce_typed_empty_actions():
    actions, ff = build_actions(pairs(y=[]), pairs(y=[]),
        pl.DataFrame({'qid': [0], 'deg': [0], 'co': ['a']}))
    assert actions.is_empty() and 'utility_delta' in actions
    assert all(actions.schema[c].is_numeric() for c in ff)
    assert apply_actions(pairs(y=[]), choose_actions(actions, [])).equals(pairs(y=[]))
