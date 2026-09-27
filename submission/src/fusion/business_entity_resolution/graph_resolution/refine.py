'second-stage matching over a fixed expanded pool and saved oof predictions'
import gc
import json
from pathlib import Path

import polars as pl

from er.stack import decode
from er.stack.features import feature_names
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import cv_fit, predict, export, tuning_anchors
from graph_resolution.pipeline import prepare, log, subset_for_tuning, compare_audit


def reuse_predictions(raw, frozen_seeds, path):
    saved = pl.read_parquet(path).select('qid', 'tid', 'binary', 'rank')


    raw = raw.drop('y') if 'y' in raw.columns else raw
    pairs = saved.join(raw, on=['qid', 'tid'], how='left').with_columns(
        pl.col('prob').fill_null(1e-6).alias('upstream_prob'),
        pl.col('gate_prob').fill_null(1e-6).alias('upstream_gate'),
        pl.col('neural_prob').fill_null(1e-6),
        pl.col('prob').is_not_null().cast(pl.Int8).alias('has_upstream_score'),
    ).with_columns(pl.col('binary').alias('prob'), pl.col('rank').alias('gate_prob'))
    from er.stack.decode import top1
    learned = top1(saved, 'binary').filter(pl.col('binary') >= .5).select(
        'qid', pl.col('tid').alias('sid'), pl.col('binary').alias('confidence'))


    background = frozen_seeds.join(saved.select(pl.col('tid').alias('sid')).unique(), on='sid', how='anti')
    seeds = pl.concat([learned, background], how='vertical_relaxed').sort(
        ['qid', 'confidence', 'sid'], descending=[False, True, False]).with_columns(
            pl.int_range(0, pl.len()).over('qid').alias('_n')).filter(pl.col('_n') < 32).drop('_n')
    return pairs, seeds, {'reused_graph_pairs': saved.height, 'learned_sibling_seeds': learned.height,
                          'background_sibling_seeds': background.height}


def blend(pred, weight):
    return pred.select('qid', 'tid', ((1-weight)*pl.col('base')+weight*pl.col('head')).alias('p'),
                       *(['y'] if 'y' in pred.columns else []))


def run(sc, output, work):
    output, work = Path(output), Path(work)
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    bundle = output/'bundle'
    bundle.mkdir(exist_ok=True)
    parent = json.loads((Path(sc['graph_predictions'])/'report.json').read_text())
    base_weight = parent['selected_rank_weight']
    (output/'config.json').write_text(json.dumps(sc, indent=2))
    info = prepare(sc, 'train', work, output)
    gc.collect()
    tr = pl.read_parquet(work/'train_features.parquet')
    feats = [c for c in feature_names(tr.columns) if not c.startswith('_')]
    settings = {'cv': sc.get('cv', 3), 'seed': 42, 'fit_fold': 0, 'tune_holdout_buckets': 5,
        'fit': {'num_boost_round': sc.get('rounds', 1800), 'early_stopping_rounds': 80},
        'lgbm': {'learning_rate': .035, 'num_leaves': 255, 'min_data_in_leaf': 100,
                 'feature_fraction': .9, 'lambda_l2': 3., 'max_bin': 255,
                 'deterministic': True, 'force_col_wise': True, 'seed': 42}}
    fit = tr.select(((pl.col('fold') == 0) & ((pl.col('qid').hash(1033)%5) != 0)).alias('fit'))['fit'].to_numpy()
    p, models, model_info = cv_fit(tr, feats, fit, settings, 'learned graph siblings')
    for i, model in enumerate(models):
        model.save_model(str(bundle/f'fold_{i}.txt'))
    (bundle/'features.json').write_text(json.dumps(feats, indent=2))
    pred = tr.select('qid', 'tid', 'y',
        ((1-base_weight)*pl.col('binary')+base_weight*pl.col('rank')).alias('base')).with_columns(pl.Series('head', p))
    del tr
    gc.collect()
    pred.write_parquet(output/'validation_predictions.parquet')
    anchors = load_refs(sc['data'], 'train').select(pl.col('rid').alias('qid'), 'fold', 'deg')
    tune = tuning_anchors(anchors, settings)
    audit = anchors.filter(pl.col('fold') == 1).select('qid', 'deg')
    small = subset_for_tuning(pred, tune)
    trials = {}
    for weight in (0., .5, 1.):
        trials[str(weight)] = decode.tune(blend(small, weight), tune,
            rules=('top1_threshold', 'expected_f'), floors=(.1, .3, .5))
        log(f'Refinement weight {weight}: tuning {trials[str(weight)]}')
    winner = max(trials, key=lambda key: trials[key]['macro_f05'])
    decision = trials[winner]
    chosen = decode.apply(decision['rule'], blend(pred, float(winner)), decision['threshold'])

    bd = parent['selected_decoder']
    base = decode.apply(bd['rule'], blend(pred, 0), bd['threshold'])
    comparison = compare_audit(chosen, base, audit)
    comparison['reference'] = 'parent graph model with its frozen tuning-selected decoder'
    report = {'inputs_train': info, 'training': model_info, 'tuning': trials,
        'selected_head_weight': float(winner), 'selected_decoder': decision,
        'audit': decode.score(chosen, audit), 'parent_audit': decode.score(base, audit),
        'comparison': comparison, 'audit_entities': audit.height, 'tune_entities': tune.height,
        'limitation': 'Follow-up designed after observing parent audit; leaderboard/generalization improvement remains unproven.'}
    (output/'report.json').write_text(json.dumps(report, indent=2))
    (bundle/'decision.json').write_text(json.dumps({'parent_rank_weight': base_weight,
        'head_weight': float(winner), 'decoder': decision, 'settings': settings}, indent=2))
    log(f'REFINEMENT AUDIT {report["audit"]}; paired comparison {comparison}')
    del pred, small, chosen, base
    gc.collect()
    report['inputs_test'] = prepare(sc, 'test', work, output)
    if info['config_sha256'] != report['inputs_test']['config_sha256']:
        raise ValueError('Train/test upstream configurations differ')
    gc.collect()
    te = pl.read_parquet(work/'test_features.parquet')
    p = predict(models, te, feats)
    scored = te.select('qid', 'tid',
        ((1-base_weight)*pl.col('binary')+base_weight*pl.col('rank')).alias('base')).with_columns(pl.Series('head', p))
    del te
    gc.collect()
    scored.write_parquet(output/'test_predictions.parquet')
    report['export'] = export(blend(scored, float(winner)), 'p', decision['rule'], decision['threshold'],
        load_refs(sc['data'], 'test'), load_targets(sc['data'], 'test'), str(output/'output'))
    (output/'report.json').write_text(json.dumps(report, indent=2))
    log('Refinement complete; use measured reports when deciding which submission to try')
