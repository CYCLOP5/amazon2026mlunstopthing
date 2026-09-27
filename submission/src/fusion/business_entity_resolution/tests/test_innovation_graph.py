from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latest_fusion import innovation_graph as graph


def edges(pairs):
    rows = [(a, b, 1.) for a, b in pairs] + [(b, a, 1.) for a, b in pairs]
    return pl.DataFrame(rows, schema=['src', 'dst', 'weight'], orient='row')


def test_two_independent_seeds_reinforce_but_reverse_edge_cannot_echo():
    frame = pl.DataFrame({'qid': [7, 7, 7], 'tid': [0, 1, 2], 'baseline_p': [.3, .98, .98]})
    out, report = graph.propagate(frame, edges([(0, 1), (0, 2)]))
    center = out.filter(pl.col('tid') == 0).row(0, named=True)
    assert center['ig_posterior_owner'] > center['ig_prior_owner']
    assert center['ig_original_seed_count_log'] == pytest.approx(np.log(3))

    assert out['ig_cavity_shift_max'].max() == 0
    assert out.filter(pl.col('tid') != 0)['ig_logit_shift'].max() == 0
    assert report['rounds'][-1]['mean_absolute_owner_probability_change'] == 0


def test_single_confident_seed_cannot_manufacture_reinforcement():
    frame = pl.DataFrame({'qid': [7]*5, 'tid': list(range(5)), 'baseline_p': [.99, .4, .4, .4, .4]})
    out, report = graph.propagate(frame, edges([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)]))
    assert out['ig_logit_shift'].max() <= 0
    assert all(r['positively_reinforced_pairs'] == 0 for r in report['rounds'])


def test_three_round_messages_add_collective_context_beyond_one_hop():
    frame = pl.DataFrame({'qid': [7]*4, 'tid': list(range(4)), 'baseline_p': [.3, .98, .98, .4]})
    out, report = graph.propagate(frame, edges([(0, 1), (0, 2), (0, 3)]))
    leaf = out.filter(pl.col('tid') == 3).row(0, named=True)


    assert leaf['ig_neighbor_mean'] > .3
    assert out['ig_cavity_shift_max'].max() > 0
    assert report['iterations'] == 3


def test_negative_neighbor_evidence_increases_explicit_decoy_mass():
    frame = pl.DataFrame({'qid': [7, 7, 7, 9], 'tid': [0, 1, 2, 0],
                          'baseline_p': [.8, .08, .08, .2]})
    out, report = graph.propagate(frame, edges([(0, 1), (0, 2)]))
    decoy = out.filter((pl.col('qid') == 7) & (pl.col('tid') == 0)).row(0, named=True)
    assert decoy['ig_logit_shift'] < 0
    assert decoy['ig_posterior_owner'] < decoy['ig_prior_owner']
    original_null = 1/(1+.8/.2+.2/.8)
    assert decoy['ig_posterior_null'] > original_null
    normalization = out.group_by('tid').agg(pl.col('ig_posterior_owner').sum(), pl.col('ig_posterior_null').first())
    np.testing.assert_allclose(normalization['ig_posterior_owner'] + normalization['ig_posterior_null'], 1., atol=1e-7)
    assert report['null_normalization_max_error'] < 1e-12


def test_omitted_owner_mass_cannot_raise_retained_messages():
    frame = pl.DataFrame({'qid': [1, 2, 3, 1], 'tid': [0, 0, 0, 1],
                          'baseline_p': [.8, .7, .6, .4]})
    out, _ = graph.propagate(frame, edges([(0, 1)]))
    expected_message = (.8/.2)/(1+.8/.2+.7/.3+.6/.4)
    neighbor = out.filter(pl.col('tid') == 1)['ig_neighbor_mean'].item()
    assert neighbor == pytest.approx(expected_message, rel=1e-6)


def test_message_budget_prunes_senders_without_losing_competition_mass(monkeypatch):
    monkeypatch.setattr(graph, 'MAX_MESSAGE_ROWS', 1)
    frame = pl.DataFrame({'qid': [1, 2, 3, 1], 'tid': [0, 0, 0, 1],
                          'baseline_p': [.8, .7, .6, .4]})
    out, report = graph.propagate(frame, edges([(0, 1)]))
    normalization = out.group_by('tid').agg(pl.col('ig_posterior_owner').sum(), pl.col('ig_posterior_null').first())
    np.testing.assert_allclose(normalization['ig_posterior_owner']+normalization['ig_posterior_null'], 1., atol=1e-7)
    assert report['max_sending_owners_per_target'] == 1
    assert report['directed_owner_messages'] <= 1
    assert report['projected_two_owner_messages'] > report['directed_owner_messages']


def records(n=8):
    return pl.DataFrame({'rid': np.arange(n, dtype=np.uint32), 'co': ['us']*n,
        'nm': ['Aster Medical Limited']*n, 'ad': ['14 River Avenue']*n,
        'own': list(range(n)), 'deg': [123]*n})


def test_independent_graph_cross_source_reciprocal_degree_and_text_constraints():
    targets = records()
    frame = pl.DataFrame({'qid': [1]*8, 'tid': np.arange(8, dtype=np.uint32), 'baseline_p': [.5]*8})
    edge, report = graph.build_graph(frame, targets, 4)
    assert edge.height == 32
    assert edge.group_by('src').len()['len'].max() <= graph.MAX_DEGREE
    assert edge.filter((pl.col('src') < 4) == (pl.col('dst') < 4)).height == 0
    assert edge.join(edge.select(pl.col('dst').alias('src'), pl.col('src').alias('dst')),
                     on=['src', 'dst'], how='anti').height == 0
    assert report['graph_uses_model_scores'] is False
    changed = targets.with_columns(pl.when(pl.col('rid') == 7).then(pl.lit('99 Other Road'))
                                   .otherwise(pl.col('ad')).alias('ad'))
    constrained, _ = graph.build_graph(frame, changed, 4)
    assert constrained.filter((pl.col('src') == 7) | (pl.col('dst') == 7)).height == 0


def test_transform_label_poisoning_and_row_order_invariance():
    targets = records()
    frame = pl.DataFrame({'qid': [1]*8+[2]*8, 'tid': list(range(8))*2,
        'baseline_p': [.9, .9, .7, .6, .3, .4, .8, .95]+[.06]*8,
        'y': [1]*8+[0]*8, 'own': [1]*16, 'deg': [9]*16})
    out, ff, report = graph.transform(frame, None, targets, None, 4)
    poisoned = frame.reverse().with_columns(pl.lit('poison').alias('y'), pl.lit(None).alias('own'),
                                           pl.lit(-999).alias('deg'))
    replay, _, _ = graph.transform(poisoned, None, targets.with_columns(pl.lit('poison').alias('own')),
                                   pl.DataFrame({'garbage': [1]}), 4)
    assert out.select('qid', 'tid').equals(frame.select('qid', 'tid'))
    assert replay.select('qid', 'tid').equals(poisoned.select('qid', 'tid'))
    np.testing.assert_allclose(out.sort('qid', 'tid').select(ff),
                               replay.sort('qid', 'tid').select(ff), atol=1e-7)
    assert np.isfinite(out.select(ff).to_numpy()).all()
    assert not report['baseline_accepted_table_used']


def test_empty_graph_preserves_prior_competition_and_finite_features():
    targets = records(18)
    frame = pl.DataFrame({'qid': [1]*18, 'tid': list(range(18)), 'baseline_p': [.4]*18})
    out, names, report = graph.transform(frame, None, targets, s2_count=9)
    assert report['graph']['directed_edges'] == 0
    np.testing.assert_allclose(out['ig_posterior_owner'], .4)
    np.testing.assert_allclose(out['ig_posterior_null'], .6)
    assert out['ig_neighbor_count_log'].max() == 0
    assert np.isfinite(out.select(names).to_numpy()).all()


def test_invalid_prior_fails_before_propagation():
    frame = pl.DataFrame({'qid': [1], 'tid': [0], 'baseline_p': [float('nan')]})
    with pytest.raises(ValueError, match='finite'):
        graph.propagate(frame, edges([(0, 1)]))
