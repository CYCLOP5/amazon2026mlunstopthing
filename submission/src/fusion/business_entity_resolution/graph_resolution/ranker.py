'owner competition with an explicit abstention candidate and grouped oof fits'
from pathlib import Path
import json
import numpy as np
import polars as pl
from scipy.special import expit

from er.safe import THREADS, guard
from er.stack.features import feature_names, as_matrix

NULL_QID = 2**32-1


def training_groups(targets, refs, k=3):
    d = targets.select(pl.col('rid').alias('tid'),'own').join(
        refs.select(pl.col('rid').cast(pl.Int64).alias('own'),pl.col('fold').alias('_owner_fold')),
        on='own',how='left')
    owned_fit = ((pl.col('_owner_fold') == 0) &
                 ((pl.col('own').clip(lower_bound=0).cast(pl.UInt32).hash(1033) % 5) != 0)).fill_null(False)
    decoy_fit = (pl.col('own') < 0) & ((pl.col('tid').hash(713) % 100) < 8)
    key = pl.when(pl.col('own') >= 0).then(pl.col('own')).otherwise(-pl.col('tid').cast(pl.Int64)-1)
    return d.select('tid',(owned_fit|decoy_fit).alias('_fit'),
                    (key.hash(42)%k).cast(pl.Int8).alias('_cv'))


def add_null(df):
    "best neural candidate's observable context provides the abstention features"
    df = df.with_columns(pl.lit(0,pl.Int8).alias('is_null'))
    top = df.sort(['tid','prob','qid'],descending=[False,True,False]).unique('tid',keep='first')
    null = top.with_columns(pl.lit(NULL_QID,pl.UInt32).alias('qid'),pl.lit(1,pl.Int8).alias('is_null'))
    if 'y' in df.columns:
        ys = df.group_by('tid').agg(pl.col('y').max().alias('_any_true'))
        null = null.join(ys,on='tid').with_columns((1-pl.col('_any_true')).cast(pl.UInt8).alias('y')).drop('_any_true')
    return pl.concat([df,null],how='vertical_relaxed').sort('tid','qid')


def sizes(tids):
    starts = np.flatnonzero(np.r_[True,tids[1:] != tids[:-1]])
    return np.diff(np.r_[starts,len(tids)])


def rank_margin(df, scores):
    'log odds of this owner against all other owners and the no-match option'
    tids = df['tid'].to_numpy()
    starts = np.flatnonzero(np.r_[True,tids[1:] != tids[:-1]])
    lengths = np.diff(np.r_[starts,len(tids)])
    top = np.maximum.reduceat(scores,starts)
    exp = np.exp(scores-np.repeat(top,lengths))
    denominator = np.repeat(np.add.reduceat(exp,starts),lengths)
    p = np.clip(exp/denominator,1e-7,1-1e-7)
    return np.log(p/(1-p)).astype(np.float32)


def fit_models(df, bundle, log, rounds=1800, folds=3):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    bundle = Path(bundle)
    bundle.mkdir(parents=True,exist_ok=True)
    feats = [c for c in feature_names(df.columns) if not c.startswith('_')]
    X = as_matrix(df,feats)
    y = df['y'].to_numpy().astype(np.float32)
    fit = df['_fit'].to_numpy().astype(bool)
    group = df['_cv'].to_numpy()
    null = df['is_null'].to_numpy().astype(bool)
    tids = df['tid'].to_numpy()
    predictions = {}
    summary = {}
    common = {'verbosity':-1,'num_threads':THREADS,'learning_rate':.045,'num_leaves':127,
        'min_data_in_leaf':100,'feature_fraction':.9,'lambda_l2':3.0,'max_bin':255,
        'deterministic':True,'force_col_wise':True,'seed':73}
    for kind in ('binary','rank'):
        out = np.zeros(df.height,dtype=np.float32)
        info = []
        for fold in range(folds):
            train = fit & (group != fold)
            valid = fit & (group == fold)
            if kind == 'binary':
                train &= ~null
                valid &= ~null
                params = {**common,'objective':'binary','metric':'binary_logloss'}
                group_train = group_valid = None
            else:
                params = {**common,'objective':'lambdarank','metric':'ndcg','eval_at':[1],
                    'label_gain':[0,1],'lambdarank_truncation_level':8}
                group_train,group_valid = sizes(tids[train]),sizes(tids[valid])
            if not train.any() or not valid.any():
                raise ValueError('Empty grouped training/validation partition')
            dtrain = lgb.Dataset(X[train],label=y[train],group=group_train,feature_name=feats)
            dvalid = lgb.Dataset(X[valid],label=y[valid],group=group_valid,reference=dtrain)
            model = lgb.train(params,dtrain,num_boost_round=rounds,valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(80,verbose=False)])
            model.save_model(str(bundle/f'{kind}_{fold}.txt'))

            held = fit & (group == fold)
            out[held] = model.predict(X[held],num_threads=THREADS).astype(np.float32)
            other_ids = np.flatnonzero(~fit)
            for start in range(0,len(other_ids),500_000):
                ids = other_ids[start:start+500_000]
                out[ids] += model.predict(X[ids],num_threads=THREADS).astype(np.float32)/folds
            info.append({'iteration':model.best_iteration,'validation':dict(model.best_score['valid_0'])})
            log(f'{kind} fold {fold+1}/{folds}: {info[-1]}')
            del model,dtrain,dvalid
            guard('graph fit')
        predictions[kind] = out
        summary[kind] = info
    margin = rank_margin(df,predictions['rank'])
    cal = LogisticRegression(C=1.0,max_iter=200)

    cal.fit(margin[fit & ~null,None],y[fit & ~null])
    calibration = {'slope':float(cal.coef_[0,0]),'intercept':float(cal.intercept_[0])}
    meta = {'features':feats,'folds':folds,'calibration':calibration,'fit':summary}
    (bundle/'models.json').write_text(json.dumps(meta,indent=2))
    pred = df.select('qid','tid','y','is_null').with_columns(
        pl.Series('binary',predictions['binary']),
        pl.Series('rank',expit(calibration['slope']*margin+calibration['intercept']).astype(np.float32)))
    return pred.filter(pl.col('is_null') == 0).drop('is_null'),meta


def predict_models(df,bundle,log):
    import lightgbm as lgb
    bundle = Path(bundle)
    meta = json.loads((bundle/'models.json').read_text())
    res = {}
    for kind in ('binary','rank'):
        models = [lgb.Booster(model_file=str(bundle/f'{kind}_{i}.txt')) for i in range(meta['folds'])]
        out = np.zeros(df.height,dtype=np.float32)
        for start in range(0,df.height,500_000):
            X = as_matrix(df.slice(start,500_000),meta['features'])
            out[start:start+len(X)] = np.mean([m.predict(X,num_threads=THREADS) for m in models],axis=0)
        res[kind] = out
        log(f'{kind}: test predictions complete')
    margin = rank_margin(df,res['rank'])
    cal = meta['calibration']
    return df.select('qid','tid','is_null').with_columns(pl.Series('binary',res['binary']),
        pl.Series('rank',expit(cal['slope']*margin+cal['intercept']).astype(np.float32)))
