'audit current errors and unlabeled country drift without fitting a model'
import gc
import json
from pathlib import Path
import polars as pl
from er.stack import decode
from er.stack.inputs import load_refs,load_targets
from innovation.common import read_parent,log


def run(data,parent,output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    pred,report = read_parent(parent,'train')
    decision = report['selected_decoder']
    accepted = decode.apply(decision['rule'],pred,decision['threshold'])
    refs = load_refs(data,'train')
    anchors = refs.filter(pl.col('fold')==1).select(pl.col('rid').alias('qid'),'deg','co','nm','ad')
    targets = pl.concat([pl.read_parquet(Path(data)/'train'/f's{sr}.parquet').with_columns(pl.lit(sr).alias('sr')) for sr in (2,3)])
    truth = targets.filter(pl.col('own')>=0).select(pl.col('rid').cast(pl.UInt32).alias('tid'),
        pl.col('own').cast(pl.UInt32).alias('qid'),'sr',pl.col('nm').alias('target_name'),
        pl.col('ad').alias('target_address')).join(anchors,on='qid')
    best = decode.top1(pred).select('tid',pl.col('qid').alias('best_qid'),pl.col('p').alias('best_p'))
    seen = pred.select('qid','tid','p')
    truth = truth.join(seen,on=['qid','tid'],how='left').join(best,on='tid',how='left').join(
        accepted.select('qid','tid').with_columns(pl.lit(True).alias('accepted')),on=['qid','tid'],how='left')
    truth = truth.with_columns((pl.col('target_address').fill_null('').str.strip_chars()=='').alias('missing_address'),
        pl.when(pl.col('accepted').fill_null(False)).then(pl.lit('found'))
        .when(pl.col('p').is_null()).then(pl.lit('owner_absent'))
        .when(pl.col('qid')==pl.col('best_qid')).then(pl.lit('best_rejected'))
        .otherwise(pl.lit('wrong_owner_ranked_first')).alias('cause'))
    res = {'parent_audit':decode.score(accepted,anchors.select('qid','deg')),
        'causes':truth.group_by('cause').len().sort('cause').to_dicts(),
        'slices':truth.group_by('co','sr','missing_address').agg(pl.len().alias('true_links'),
            (pl.col('cause')!='found').mean().alias('miss_rate')).sort('co','sr','missing_address').to_dicts(),
        'oracle':decode.score(pred.filter(pl.col('y')==1),anchors.select('qid','deg')),
        'note':'Audit labels diagnose only. Test has no labels; country drift is not an accuracy measurement.'}
    examples = truth.filter(pl.col('cause')!='found').sort('cause','tid').group_by('cause',maintain_order=True).head(20)
    examples.write_parquet(output/'audit_error_examples.parquet')
    res['examples'] = examples.to_dicts()
    del pred,accepted,truth,seen,best,targets,refs
    gc.collect()
    res['test_drift'] = []
    pred,_ = read_parent(parent,'test')
    refs,targets = load_refs(data,'test'),load_targets(data,'test')
    best = decode.top1(pred).join(targets.select(pl.col('rid').alias('tid'),'co',
        (pl.col('ad').fill_null('').str.strip_chars()=='').alias('missing_address')),on='tid')
    accepted = decode.apply(decision['rule'],pred,decision['threshold'])
    best = best.join(accepted.select('tid').unique().with_columns(pl.lit(True).alias('accepted')),on='tid',how='left')
    res['test_drift'] = best.group_by('co','missing_address').agg(pl.len(),
        pl.col('accepted').fill_null(False).mean().alias('accepted_fraction'),
        ((pl.col('p')>.1)&(pl.col('p')<.9)).mean().alias('uncertain_fraction'),
        pl.col('p').mean().alias('mean_best_p')).sort('co','missing_address').to_dicts()
    (output/'diagnostics.json').write_text(json.dumps(res,indent=2))
    log(json.dumps({k:v for k,v in res.items() if k!='examples'},indent=2))
