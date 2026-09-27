'synthetic integration checks; no challenge records are fitted locally'
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boosted_hybrid.pipeline import (
    HEADS, combine, constrained_curve, routed_accept, select, read_raw, raw_variants, worker,
)
from er.stack import decode
from test_stack import make_fixture


def test_independent_new_pair_can_escape_previous_prior_cap():
    frame = pl.DataFrame({'qid': [0, 1], 'tid': [3, 4], 'has_parent': [0, 1]})
    base = frame.select('qid', 'tid').with_columns(pl.Series('p', [.01, .01]))
    raw = np.array([5., 5.])
    independent = combine(frame, base, raw, .5, 'independent')
    residual = combine(frame, base, raw, .5, 'residual')
    assert independent['p'][0] > .99
    assert independent['p'][1] == pytest.approx(residual['p'][1])
    assert residual['p'][0] < .6


def test_constrained_threshold_matches_brute_force_with_ties():
    rows = pl.DataFrame({'qid': [0, 0, 1, 2, 2], 'tid': [0, 1, 2, 3, 4],
                         'p': [.95, .6, .85, .6, .2], 'y': [1, 0, 1, 1, 0]})
    anchors = pl.DataFrame({'qid': [0, 1, 2], 'deg': [1, 1, 1]})
    choices = []
    for cut in rows['p'].unique():
        metrics = decode.score(rows.filter(pl.col('p') >= cut), anchors)
        if metrics['pair_precision'] >= .7 and metrics['pair_recall'] >= .6:
            choices.append({'threshold': cut, **metrics})
    exp = max(choices, key=lambda x: (x['macro_f05'], x['pair_recall']))
    actual = constrained_curve(rows, anchors, .7, .6)
    for key in exp:
        assert actual[key] == pytest.approx(exp[key])


def test_country_decoder_routes_france_without_changing_other_countries():
    rows = pl.DataFrame({'qid': [0, 1], 'tid': [0, 1], 'p': [.7, .7]})
    refs = pl.DataFrame({'rid': [0, 1], 'co': ['france', 'us']})
    policies = {'france': {'rule': 'top1_threshold', 'threshold': .8},
                'us': {'rule': 'top1_threshold', 'threshold': .6}}
    assert routed_accept(rows, policies, refs)['qid'].to_list() == [1]


def test_missing_transfer_models_do_not_fabricate_france_evidence(tmp_path):
    frame = pl.DataFrame({'qid': [0], 'tid': [0]})
    assert raw_variants(tmp_path, 'train', frame, transfer='us') == {}


@pytest.mark.parametrize('backend', ['xgboost', 'catboost'])
def test_actual_cpu_worker_integrates_country_models_and_test_scores(tmp_path, backend):
    from test_boosted_hybrid_models import _owner_fixture
    frame, refs = _owner_fixture()
    data, ff, out = [tmp_path/n for n in ('data', 'features', 'models')]
    (data/'train').mkdir(parents=True)
    refs = refs.with_columns(pl.col('rid').cast(pl.String).alias('eid'),
        pl.lit('Synthetic name').alias('nm'), pl.lit('1 Synthetic street').alias('ad'),
        pl.lit(1).alias('deg'))
    refs.write_parquet(data/'train/ref.parquet')
    ff.mkdir()
    for split in ('train', 'test'):
        (frame if split == 'train' else frame.drop('own', 'fold', 'y')).write_parquet(
            ff/f'{split}_features.parquet')
        (ff/f'_{split}_SUCCESS').write_text('complete\n')
    head = backend+'_independent'
    worker(data, ff, ff, out, head, rounds=4, device='cpu')
    assert (out/head/'_SUCCESS').is_file()
    assert (out/head/'transfer_us.parquet').is_file()
    got = read_raw(out, head, 'test', frame)
    assert len(got) == frame.height and np.isfinite(got).all()


def synthetic_outputs(tmp_path, oracle_stress=True):
    data, roots = make_fixture(tmp_path, n=600)
    baseline, ff, models = [tmp_path/n for n in ('hybrid', 'features', 'models')]
    baseline.mkdir()
    ff.mkdir()
    for split, n in [('train', 600), ('test', 32)]:
        path = data/split/'ref.parquet'
        refs = pl.read_parquet(path)
        country = ['us' if i < n//2 else 'india' for i in range(n)]
        if split == 'test':
            country[:8] = ['france']*8
        refs = refs.with_columns(pl.Series('co', country))
        refs.write_parquet(path)
        for src in (2, 3):
            tpath = data/split/f's{src}.parquet'
            pl.read_parquet(tpath).with_columns(pl.Series('co', country)).write_parquet(tpath)
        pairs = pl.read_parquet(Path(roots[split][0])/'parts/part_0.parquet')

        pairs = pairs.filter(pl.col('qid')//(n//2) == (pl.col('tid')%n)//(n//2))
        positive = (pl.col('qid') == pl.col('tid')%n)
        true_p = pl.when(pl.col('tid')%5 == 0).then(.05).otherwise(.9)
        p = pairs.select('qid', 'tid', pl.when(positive).then(true_p).otherwise(.02).alias('p'),
                         *(['y'] if split == 'train' else []))
        p.write_parquet(baseline/('validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'))
        frame = pairs.select('qid', 'tid').with_columns(
            pl.lit(1).alias('has_parent'), pl.lit(.2).alias('base_lg'))

        frame = frame.join(refs.select(pl.col('rid').alias('qid'), 'co',
            *(['fold'] if split == 'train' else [])), on='qid')
        if split == 'train':
            frame = frame.with_columns((pl.col('tid')%n).alias('own'), positive.cast(pl.UInt8).alias('y'))
        frame.write_parquet(ff/f'{split}_features.parquet')
        (ff/f'_{split}_SUCCESS').write_text('complete\n')
        for head in HEADS:
            dest = models/head
            dest.mkdir(parents=True, exist_ok=True)
            raw = frame.select('qid', 'tid').with_columns(
                pl.when(positive).then(5.).otherwise(-6.).alias('raw'))
            raw.write_parquet(dest/f'{split}.parquet')
            if split == 'train' and head.endswith('independent'):
                for co in ('us', 'india'):
                    stress = raw if oracle_stress else raw.with_columns((-pl.col('raw')).alias('raw'))
                    stress.write_parquet(dest/f'transfer_{co}.parquet')
            (dest/'_SUCCESS').write_text('complete\n')
            (dest/'report.json').write_text('{}')
    (baseline/'report.json').write_text(json.dumps({'decision': {'rule': 'top1_threshold', 'threshold': .5}}))
    (models/'_SUCCESS').write_text('complete\n')
    return data, baseline, ff, models


@pytest.mark.parametrize('transfer_pass', [True, False])
def test_end_to_end_selection_export_and_country_transfer_guard(tmp_path, transfer_pass):
    data, baseline, ff, models = synthetic_outputs(tmp_path, transfer_pass)
    out = tmp_path/'result'
    report = select(data, baseline, ff, ff, models, out)
    assert report['selected'] != 'baseline'
    assert report['errors']['recovered_true_links'] > 0
    assert report['errors']['new_false_links'] == 0
    assert (report['france_selected'] != 'baseline') == transfer_pass
    from scripts.validate_outputs import validate
    assert validate(data, out/'output')['source1'] == 32
    parent_keys = pl.read_parquet(baseline/'test_predictions.parquet').select('qid', 'tid')
    final_keys = pl.read_parquet(out/'test_predictions.parquet').select('qid', 'tid')
    assert final_keys.height == parent_keys.height
    assert parent_keys.join(final_keys, on=['qid', 'tid'], how='anti').height == 0
    assert all(v['accuracy'] is None for v in report['test_by_country'].values())

    path = models/HEADS[0]/'train.parquet'
    pl.read_parquet(path).head(1).write_parquet(path)
    with pytest.raises(ValueError, match='coverage'):
        read_raw(models, HEADS[0], 'train', pl.read_parquet(ff/'train_features.parquet'))
