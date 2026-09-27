'fit a selective verifier; unchanged pairs keep the completed stacker score'
import gc
import json
from pathlib import Path
import numpy as np
import polars as pl

from er.stack import decode
from er.stack.features import feature_names, as_matrix
from er.stack.inputs import load_refs,load_targets
from er.stack.pipeline import predict,tuning_anchors,export,lgb_params
from er.safe import THREADS, guard
from graph_resolution.pipeline import subset_for_tuning,compare_audit
from innovation.common import read_parent,replace_scores,log,logit


def training_partitions(frame, refs, cv=3):
    'hold out complete true-owner groups, including their competing negatives'
    owners = refs.select(pl.col('rid').cast(pl.Int64).alias('own'),
        pl.col('fold').alias('_owner_fold'))
    d = frame.select('qid','tid','own','fold').join(owners,on='own',how='left',maintain_order='left')
    candidate_fit = (pl.col('fold') == 0) & ((pl.col('qid').hash(1033)%5) != 0)
    owned_fit = ((pl.col('own') >= 0) & (pl.col('_owner_fold') == 0) &
        ((pl.col('own').clip(lower_bound=0).cast(pl.UInt32).hash(1033)%5) != 0)).fill_null(False)
    decoy_fit = (pl.col('own') < 0) & ((pl.col('tid').hash(713)%100) < 8)
    key = pl.when(pl.col('own') >= 0).then(pl.col('own')).otherwise(-pl.col('tid').cast(pl.Int64)-1)
    d = d.select((candidate_fit & (owned_fit|decoy_fit)).alias('fit'),(key.hash(42)%cv).alias('cv'))
    return d['fit'].to_numpy(),d['cv'].to_numpy()


def grouped_fit(frame, columns, fit, groups, settings, tag):
    'oof predictions grouped by true owner; other rows use the cv ensemble'
    import lightgbm as lgb
    X = as_matrix(frame,columns)
    y = frame['y'].to_numpy().astype(np.float32)
    if np.unique(y[fit]).size != 2:
        raise ValueError('Verifier fit partition requires positive and negative examples')
    k = settings['cv']
    out = np.zeros(frame.height,dtype=np.float64)
    models,iterations = [],[]
    other_ids = np.flatnonzero(~fit)
    for fold in range(k):
        train,valid = fit & (groups != fold),fit & (groups == fold)
        if not train.any() or not valid.any():
            raise ValueError('Empty true-owner CV partition')
        dtrain = lgb.Dataset(X[train],label=y[train],feature_name=columns)
        dvalid = lgb.Dataset(X[valid],label=y[valid],reference=dtrain)
        model = lgb.train(lgb_params(settings),dtrain,
            num_boost_round=settings['fit']['num_boost_round'],valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(settings['fit']['early_stopping_rounds'],verbose=False)])
        out[valid] = model.predict(X[valid],num_threads=THREADS)
        for start in range(0,len(other_ids),500_000):
            ids = other_ids[start:start+500_000]
            out[ids] += model.predict(X[ids],num_threads=THREADS)/k
        models.append(model)
        iterations.append(model.best_iteration)
        log(f'{tag}: owner CV {fold+1}/{k}, best_iteration={model.best_iteration}, '
            f'validation_logloss={model.best_score["valid_0"]["binary_logloss"]:.5f}')
        del dtrain,dvalid
        guard(tag+' grouped fit')
    gain = np.mean([m.feature_importance('gain') for m in models],axis=0)
    return out.astype(np.float32),models,{'best_iterations':iterations,
        'importance':sorted(zip(columns,gain.tolist()),key=lambda item:-item[1])[:40],
        'fit_rows':int(fit.sum()),'fit_targets':frame.filter(pl.Series(fit))['tid'].n_unique(),
        'partition':'true owner CV; candidate and true owner fit-only; decoy target CV with 8% fit sample'}


def verified_features(prepared, verified, split):
    d = pl.read_parquet(Path(prepared)/split/'features.parquet')
    ce = pl.read_parquet(str(Path(verified)/split/'*.parquet'))
    if ce.select('qid','tid').n_unique() != ce.height:
        raise ValueError('Duplicated verification pair')
    out = d.join(ce,on=['qid','tid'],how='left',maintain_order='left')
    if out.height != d.height or out['ce_full_lg'].null_count():
        raise ValueError('Missing or duplicated neural verification scores')
    return out.with_columns(
        (pl.col('ce_full_lg')-pl.col('ce_name_lg')).alias('address_contribution'),
        (pl.col('ce_full_lg')-pl.col('ce_address_lg')).alias('name_contribution'),
        (pl.col('ce_canonical_lg')-pl.col('ce_full_lg')).alias('normalization_gain'),
        pl.min_horizontal('ce_name_lg','ce_address_lg').alias('weakest_view'),
        pl.max_horizontal('ce_name_lg','ce_address_lg').alias('strongest_view'))


def mix(selected, scores, weight):
    p = 1/(1+np.exp(-np.clip((1-weight)*logit(selected['parent_p'].to_numpy())+weight*logit(scores),-30,30)))
    return selected.select('qid','tid').with_columns(pl.Series('p',p.astype(np.float32)))


def run(data,parent,prepared,verified,output,rounds=1200):
    output = Path(output)
    (output/'bundle').mkdir(parents=True,exist_ok=True)
    if not (Path(verified)/'_SUCCESS').exists():
        raise RuntimeError('GPU verification has not completed')
    tr = verified_features(prepared,verified,'train')
    skip = {'own','co','parent_p'}
    feats = [c for c in feature_names(tr.columns) if c not in skip and not c.startswith('_')]
    view_features = {'ce_name_lg','ce_address_lg','ce_canonical_lg','address_contribution',
                     'name_contribution','normalization_gain','weakest_view','strongest_view'}
    params = {'cv':3,'seed':42,'fit_fold':0,'tune_holdout_buckets':5,
        'fit':{'num_boost_round':rounds,'early_stopping_rounds':70},
        'lgbm':{'learning_rate':.04,'num_leaves':63,'min_data_in_leaf':80,'lambda_l2':5.,
                'feature_fraction':.9,'deterministic':True,'force_col_wise':True,'seed':42}}
    fit,groups = training_partitions(tr,load_refs(data,'train'),params['cv'])


    models, predictions, model_info, orders = {}, {}, {}, {}
    for kind in ('direct','multiview'):
        columns = [c for c in feats if kind=='multiview' or c not in view_features]
        p, ms, info = grouped_fit(tr,columns,fit,groups,params,kind+' selective verifier')
        models[kind],predictions[kind],model_info[kind],orders[kind] = ms,p,info,columns
        for i,m in enumerate(ms):
            m.save_model(str(output/'bundle'/f'{kind}_{i}.txt'))
    (output/'bundle/features.json').write_text(json.dumps(orders,indent=2))
    pred,parent_report = read_parent(parent,'train')
    anchors = load_refs(data,'train').select(pl.col('rid').alias('qid'),'fold','deg','co')
    tune = tuning_anchors(anchors,params)
    audit = anchors.filter(pl.col('fold')==1).select('qid','deg')
    parent_decision = parent_report['selected_decoder']
    base_accept = decode.apply(parent_decision['rule'],pred,parent_decision['threshold'])
    trials = {'baseline':{'tune':decode.score(base_accept,tune),'rule':parent_decision['rule'],
                         'threshold':parent_decision['threshold'],'kind':'baseline','weight':0.}}
    small = subset_for_tuning(pred,tune)
    for kind in predictions:
        for weight in (.25,.5,1.):
            update = mix(tr,predictions[kind],weight)
            tuned = decode.tune(replace_scores(small,update),tune,
                rules=('top1_threshold','expected_f'),floors=(.3,.5))
            key = f'{kind}_{weight}'
            trials[key] = {'tune':{'macro_f05':tuned['macro_f05']},'rule':tuned['rule'],
                          'threshold':tuned['threshold'],'kind':kind,'weight':weight}
            log(f'Verifier trial {key}: tune macro F0.5 {tuned["macro_f05"]:.7f}')
    winner = max(trials,key=lambda k:trials[k]['tune']['macro_f05'])
    decision = trials[winner]
    if winner == 'baseline':
        final = pred
    else:
        final = replace_scores(pred,mix(tr,predictions[decision['kind']],decision['weight']))
    accept = decode.apply(decision['rule'],final,decision['threshold'])
    report = {'selected':winner,'decision':decision,'tuning':trials,'training':model_info,
        'audit':decode.score(accept,audit),'baseline_audit':decode.score(base_accept,audit),
        'comparison':compare_audit(accept,base_accept,audit),
        'selection':json.loads((Path(prepared)/'selection.json').read_text()),
        'verification':json.loads((Path(verified)/'verification.json').read_text()),
        'limits':['Country slices cover US/India only; French labels are unavailable.',
                  'Four input views reuse one frozen model and are not independent models.',
                  'Audit has informed research direction; it is not a fresh blind benchmark.']}
    report['comparison']['reference'] = 'completed graph sibling refinement with frozen decoder'
    report['audit_by_country'] = {}
    for country in anchors['co'].unique().sort():
        a = anchors.filter((pl.col('fold')==1)&(pl.col('co')==country)).select('qid','deg')
        report['audit_by_country'][country] = {'baseline':decode.score(base_accept,a),'selected':decode.score(accept,a)}
    report['selected_pair_calibration'] = tr.with_columns(
        pl.Series('multiview_probability',predictions['multiview'])).filter(pl.col('fold')==1).group_by('co').agg(
            pl.len(),pl.col('y').mean().alias('true_rate'),pl.col('parent_p').mean(),
            pl.col('multiview_probability').mean()).to_dicts()
    (output/'report.json').write_text(json.dumps(report,indent=2))
    (output/'bundle/decision.json').write_text(json.dumps(decision,indent=2))
    log(f'VERIFIER AUDIT {report["audit"]}; {report["comparison"]}')
    final.write_parquet(output/'validation_predictions.parquet')
    del tr,pred,final,small,accept,base_accept,predictions
    gc.collect()
    te = verified_features(prepared,verified,'test')
    test,_ = read_parent(parent,'test')
    if winner != 'baseline':
        kind = decision['kind']
        p = predict(models[kind],te,orders[kind])
        test = replace_scores(test,mix(te,p,decision['weight']))
    test.write_parquet(output/'test_predictions.parquet')
    report['export'] = export(test,'p',decision['rule'],decision['threshold'],load_refs(data,'test'),
        load_targets(data,'test'),str(output/'output'))
    (output/'report.json').write_text(json.dumps(report,indent=2))
    log('Selective verifier complete')
