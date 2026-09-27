'cloud-only frozen-score fusion with fail-closed record alignment'
import json
import hashlib
import threading
import time
import os
from pathlib import Path
import numpy as np
import polars as pl
from er.stack.inputs import load_refs, load_targets
from er.stack import decode
from er.stack.pipeline import export
from final_repair.fast_decode import apply as fast_apply
from final_repair.pipeline import changes

KEYS = ['qid', 'tid']

def log(message): print(time.strftime('%H:%M:%S')+' '+message,flush=True)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str))

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def partition(ids):
    return ((ids.astype(np.uint64)*2654435761) >> 13) % 3

def aligned(data, olddata, split):
    'reject stale id layouts, changed records, labels, or reference folds'
    for name in ('ref', 's2', 's3'):
        a = pl.read_parquet(Path(data)/split/(name+'.parquet'))
        b = pl.read_parquet(Path(olddata)/split/(name+'.parquet'))
        cols = ['rid', 'eid', 'nm', 'ad', 'co']
        if split == 'train':
            cols += ['fold', 'deg'] if name == 'ref' else ['own']
        if not a.select(cols).sort('rid').equals(b.select(cols).sort('rid'), null_equal=True):
            raise ValueError(f'Record alignment failed: {split}/{name}; refusing score joins')

def checked(frame, scores):
    frame = frame.with_columns(pl.col(KEYS).cast(pl.UInt32), pl.col(scores).cast(pl.Float32))
    if frame.select(KEYS).n_unique() != frame.height:
        raise ValueError('Duplicate candidate pair')
    for c in scores:
        if frame.filter(pl.col(c).is_not_null() & (~pl.col(c).is_finite() | ~pl.col(c).is_between(0, 1))).height:
            raise ValueError(f'Invalid probability {c}')
    return frame

def prepare(split, data, newest, hybrid, graph, friend, output, expected_meta, olddata=None):
    from latest_fusion.features import prepare_features
    started=time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    log(f'preparing {split}: metadata guards and score union')
    if olddata is not None: aligned(data, olddata, split)
    meta_path=Path(data)/'meta.json'
    if json.loads(meta_path.read_text())!=json.loads(Path(expected_meta).read_text()):
        raise ValueError('Prepared data metadata changed; refusing old score ID joins')
    refs, targets = load_refs(str(data), split), load_targets(str(data), split)
    newest = Path(newest)
    src = newest/'scores'/f'stack-{split}.parquet'
    if not src.exists(): src = newest/f'stack-{split}.parquet'
    names = ['newest', 'gate', 'neural']+[f'np_m{i}' for i in range(15)]
    scoremeta=json.loads(src.with_suffix('.json').read_text())
    if scoremeta['data_meta_sha256']!=hashlib.sha256(meta_path.read_bytes()).hexdigest():
        raise ValueError('Latest score data metadata hash mismatch')
    if scoremeta['score_sha256']!=sha256(src):raise ValueError('Latest score content hash mismatch')
    source_frame=pl.read_parquet(src)
    if split=='train' and source_frame.filter(pl.col('y')!=(pl.col('qid').cast(pl.Int64)==pl.col('own')).cast(pl.UInt8)).height:
        raise ValueError('Latest labels disagree with target owners')
    refcheck=source_frame.select('qid','co',*(['fold'] if split=='train' else [])).unique().join(
        refs.select(pl.col('rid').alias('qid'),pl.col('co').alias('_co'),*([pl.col('fold').alias('_fold')] if split=='train' else [])),on='qid',how='left')
    invalid=pl.col('_co').is_null()|(pl.col('co')!=pl.col('_co'))
    if split=='train': invalid|=pl.col('fold')!=pl.col('_fold')
    if refcheck.filter(invalid).height: raise ValueError('Latest reference country/fold alignment failed')
    if split=='train':
        actual_owners=pl.concat([pl.read_parquet(Path(data)/'train'/f's{s}.parquet',columns=['rid','own']) for s in (2,3)]).rename({'rid':'tid','own':'_own'})
        if source_frame.select('tid','own').unique().join(actual_owners,on='tid',how='left').filter(
                pl.col('_own').is_null()|(pl.col('own')!=pl.col('_own'))).height: raise ValueError('Latest target ownership alignment failed')
    d = source_frame.select(*KEYS, pl.col('stack_prob').alias('newest'),
        pl.col('gate_prob').alias('gate'), pl.col('neural_prob').alias('neural'), *names[3:], 'seg')
    del source_frame
    d = checked(d, names)
    coverage = {'newest_pairs': d.height}
    for root, col, dest, filename in [(hybrid,'p','hybrid','validation_predictions.parquet' if split=='train' else 'test_predictions.parquet'),
            (graph,'head','graph','validation_predictions.parquet' if split=='train' else 'test_predictions.parquet'),
            (friend,'p2','friend','val_pred.parquet' if split=='train' else 'test_pred.parquet')]:
        old = checked(pl.scan_parquet(Path(root)/filename).select(*KEYS,pl.col(col).alias(dest))
            .filter(pl.col(dest)>=.001).collect(), [dest])
        coverage[dest+'_pairs'] = old.height
        d = d.join(old,on=KEYS,how='full',coalesce=True,validate='1:1')
    if d.join(refs.select(pl.col('rid').alias('qid')),on='qid',how='anti').height or d.join(
            targets.select(pl.col('rid').alias('tid')),on='tid',how='anti').height:
        raise ValueError('Candidate IDs outside record universe')
    if split == 'train':
        own = pl.concat([pl.read_parquet(Path(data)/'train'/f's{s}.parquet',columns=['rid','own']) for s in (2,3)]).rename({'rid':'tid'})
        d = d.join(own,on='tid',validate='m:1').with_columns((pl.col('qid').cast(pl.Int64)==pl.col('own')).cast(pl.UInt8).alias('y'))
        coverage['true_pairs'] = int((own['own']>=0).sum())
        coverage['covered_true_pairs'] = int(d['y'].sum())
        coverage['missing_true_pairs'] = coverage['true_pairs']-coverage['covered_true_pairs']
    coverage['union_pairs'] = d.height
    if split=='train':
        coverage['country']=d.join(refs.select(pl.col('rid').alias('qid'),'co'),on='qid').group_by('co').agg(
            pl.len().alias('candidate_pairs'),pl.col('y').sum().alias('covered_true_pairs')).to_dicts()
    coverage['missing_scores'] = {c:d[c].null_count() for c in names+['hybrid','graph','friend']}
    d=d.join(refs.select(pl.col('rid').alias('qid'),'co',*(['fold'] if split=='train' else [])),on='qid',validate='m:1')
    d, ff = prepare_features(d,refs,targets)
    missing=d.filter(pl.col('seg').is_null()).select(*KEYS)
    if missing.height:
        number=lambda col:pl.col(col).fill_null('').str.extract_all(r'\d+').list.eval(pl.element().str.strip_chars_start('0').replace('','0'))
        relation=missing.join(refs.select(pl.col('rid').alias('qid'),pl.col('ad').fill_null('').str.extract(r'(\d+)',1).str.strip_chars_start('0').replace('','0').fill_null('').alias('_house')),on='qid').join(
            targets.select(pl.col('rid').alias('tid'),number('ad').alias('_nums')),on='tid')
        relation=relation.select(*KEYS,pl.when((pl.col('_house')!='')&pl.col('_nums').list.contains(pl.col('_house'))).then(0)
            .when((pl.col('_house')!='')&(pl.col('_nums').list.len()>0)).then(1).otherwise(2).cast(pl.UInt8).alias('_seg'))
        d=d.join(relation,on=KEYS,how='left',validate='1:1').with_columns(pl.col('seg').fill_null(pl.col('_seg'))).drop('_seg')
    prohibited = set(KEYS+['y','own','fold','eid','deg']) & set(ff)
    if prohibited: raise ValueError(f'Leaking features: {prohibited}')
    d.write_parquet(output/f'prepared-{split}.parquet')
    write_json(output/f'features-{split}.json',ff)
    write_json(output/f'coverage-{split}.json',coverage)
    log(f'prepared {split}: {d.height:,} candidate pairs, {len(ff)} features in {time.monotonic()-started:.1f}s')
    return coverage

def logit(p):
    p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))

def select_decoder(frame,method,anchors):
    from latest_fusion.calibration import decide
    from latest_fusion import decoder as latestdecoder
    ix=latestdecoder.winners(frame['qid'].to_numpy(),frame['tid'].to_numpy(),frame[method].to_numpy(),frame['raw_'+method].to_numpy())
    top=frame[ix].select(*KEYS,pl.col(method).alias('p'),'y')
    options=[{'rule':'top1_threshold',**decode.curve(top,anchors)}]
    for floor in (.05,.2,.4):
        options.append({'rule':'expected_f','threshold':floor,**decode.score(decide(frame,method,'expected_f',floor),anchors)})
    return max(options,key=lambda x:x['macro_f05'])

def fit(data, train, test, metadata, output, calibration):
    from latest_fusion.calibration import calibrate_methods, decide
    import lightgbm as lgb
    started=time.monotonic()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tr=pl.read_parquet(train);te=pl.read_parquet(test)
    ff=json.loads(Path(train).with_name('features-train.json').read_text())
    if ff!=json.loads(Path(test).with_name('features-test.json').read_text()): raise ValueError('Feature schema mismatch')
    refs=load_refs(str(data),'train'); rt=load_refs(str(data),'test'); tg=load_targets(str(data),'test')
    q=refs['rid'].to_numpy(); fold=refs['fold'].to_numpy()
    eligible=(fold==0)&(partition(q)==2)
    parity=(((q.astype(np.uint64)*2654435761+9001)>>13)%2)
    fitids=q[eligible&(parity==0)]; tuneids=q[eligible&(parity==1)]
    if np.intersect1d(fitids,tuneids).size: raise ValueError('Fit/tune overlap')

    fitmask=tr['qid'].is_in(fitids).to_numpy() & ((tr['own']<0).to_numpy()|tr['own'].is_in(fitids).to_numpy())
    fitframe=tr.filter(pl.Series(fitmask))
    log(f'fitting residual: {fitframe.height:,} pairs, {len(fitids):,} owners')
    offset=logit(fitframe['newest'].fill_null(1e-4).to_numpy())
    meta=json.loads(Path(metadata).read_text());params=dict(meta['params']);params['num_threads']=int(os.environ.get('ER_THREADS','48'))
    model=lgb.train(params,lgb.Dataset(fitframe.select(ff).to_numpy(),label=fitframe['y'].to_numpy(),init_score=offset),num_boost_round=182)
    model.save_model(str(output/'residual_model.txt'))
    log(f'residual fit complete; {fitframe.height:,} pairs, {int(fitframe["y"].sum()):,} positive')
    tune=refs.filter(pl.col('rid').is_in(tuneids)).select(pl.col('rid').alias('qid'),'deg')
    audit=refs.filter(pl.col('fold')==1).select(pl.col('rid').alias('qid'),'deg')
    def score_methods(frame):
        raw=frame['newest'].fill_null(1e-4).to_numpy(); base=logit(raw)
        residual=np.empty(frame.height,dtype=np.float32)
        for start in range(0,frame.height,1_000_000):
            residual[start:start+1_000_000]=model.predict(frame.slice(start,1_000_000).select(ff).to_numpy(),raw_score=True)
        vals={'baseline':raw}
        vals.update({f'residual_{s}':sigmoid(base+s*residual) for s in (.25,.5,1.)})
        for name in ('hybrid','graph','friend'):
            vals[f'blend_{name}']=.75*raw+.25*frame[name].fill_null(1e-4).to_numpy()
        vals['combo']=.65*raw+.2*frame['hybrid'].fill_null(1e-4).to_numpy()+.15*frame['graph'].fill_null(1e-4).to_numpy()
        return frame.with_columns([pl.Series(k,v.astype(np.float32)) for k,v in vals.items()]),list(vals)
    tr,methods=score_methods(tr)
    te,_=score_methods(te)
    log('calibrating all methods on isolated partition1')
    tr,te,calibration_report=calibrate_methods(tr,te,refs,rt,methods,calibration)
    decisions={}
    for method in methods:
        log('selecting decoder: '+method)
        decision=({'rule':'expected_f','threshold':.05,**decode.score(decide(tr,method,baseline_only=True),tune)}
            if method=='baseline' else select_decoder(tr,method,tune))
        accepted=decide(tr,method,decision['rule'],decision['threshold'],baseline_only=method=='baseline')
        decisions[method]={'tune':decision,'audit_historically_reused':decode.score(accepted,audit),
            'tune_country':{str(co):decode.score(accepted,tune.join(refs.select(pl.col('rid').alias('qid'),'co'),on='qid').filter(pl.col('co')==co)) for co in refs['co'].unique()},
            'audit_country_historically_reused':{str(co):decode.score(accepted,audit.join(refs.select(pl.col('rid').alias('qid'),'co'),on='qid').filter(pl.col('co')==co)) for co in refs['co'].unique()}}
    base_tune=decisions['baseline']['tune']
    allowed=['baseline']
    for method in methods[1:]:
        entry=decisions[method]
        checks={'overall_gain':entry['tune']['macro_f05']>=base_tune['macro_f05']+5e-6,
            'overall_precision':entry['tune']['pair_precision']>=base_tune['pair_precision']-.0001}
        for co,metric in entry['tune_country'].items():
            base_country=decisions['baseline']['tune_country'][co]
            checks[co+'_score']=metric['macro_f05']>=base_country['macro_f05']
            checks[co+'_precision']=metric['pair_precision']>=base_country['pair_precision']-.0001
        entry['promotion_checks']=checks
        if all(checks.values()):allowed.append(method)
    sel=max(allowed,key=lambda m:decisions[m]['tune']['macro_f05'])
    log(f'selected {sel}: tune {decisions[sel]["tune"]["macro_f05"]:.9f}; baseline {base_tune["macro_f05"]:.9f}')
    from latest_fusion.transfer import run_transfer
    log('checking country transfer in both directions for France')
    french,transfer_report=run_transfer(tr,te,refs,rt,ff,fitids,tuneids,params,output/'transfer')
    log(f'country transfer selected: {transfer_report["promoted"]}')
    france_ids=rt.filter(pl.col('co').cast(pl.String).str.to_lowercase().is_in(['fr','france']))['rid']
    decision=decisions[sel]['tune'];baseline=decisions['baseline']['tune']
    selected_pairs=te.select(*KEYS,pl.col(sel).alias('p'))
    accepted=decide(te,sel,decision['rule'],decision['threshold'],baseline_only=sel=='baseline').filter(~pl.col('qid').is_in(france_ids.implode()))
    baseacc=decide(te,'baseline',baseline_only=True)
    expected_baseline=meta.get('lineage',{}).get('baseline_matching_pairs')
    if expected_baseline is not None and baseacc.height!=expected_baseline:
        raise ValueError(f'Latest baseline replay mismatch: {baseacc.height} != {expected_baseline}')
    french=baseacc.filter(pl.col('qid').is_in(france_ids.implode())) if french is None else french
    accepted=pl.concat([accepted,french],how='diagonal_relaxed')
    if accepted['tid'].n_unique()!=accepted.height:
        raise ValueError('Country policies assigned a target to multiple owners')
    report={'fit_entities':len(fitids),'tune_entities':len(tuneids),'fit_pairs':fitframe.height,
        'fit_positive':int(fitframe['y'].sum()),'methods':decisions,'selected':sel,
        'limitations':['Upstream models and fold1 audit historically reused; not a blind pipeline estimate.',
            'French accuracy is unmeasured; transfer validation excludes target-country labels only at the new head and calibrator.'],
        'france_transfer':transfer_report,'baseline_matches':baseacc.height,
        'changes':changes(accepted,baseacc,rt),'calibration':calibration_report,
        'coverage':{split:json.loads(Path(path).with_name(f'coverage-{split}.json').read_text())
            for split,path in [('train',train),('test',test)]},
        'baseline_policy':'Exact provided newest calibration, source candidate pool only, expected F0.5 floor .05 and exact64 decoder.',
        'partition_protocol':{'upstream_fit':0,'calibration':1,'residual_fit_and_tune':2,'modulus':3,'owner_hash_multiplier':2654435761,'shift':13,'fit_tune_seed':9001}}
    report['export']=export(selected_pairs,'p',decision['rule'],decision['threshold'],rt,tg,str(output/'output'),accepted=accepted)
    from validate_outputs import validate
    report['validation']=validate(Path(data),output/'output')
    accepted.write_parquet(output/'accepted_test.parquet')
    baseacc.write_parquet(output/'baseline_accepted_test.parquet')
    selected_pairs.write_parquet(output/'test_predictions.parquet')
    tr.select(*KEYS,pl.col(sel).alias('p'),'y').write_parquet(output/'validation_predictions.parquet')
    report['prediction_note']='Score parquets are the selected known-country head; accepted_test includes any separately selected France transfer policy.'
    report['matching_sha256']=sha256(output/'output/matching_results.tsv')
    report['elapsed_seconds']=time.monotonic()-started
    write_json(output/'report.json',report);write_json(output/'decisions.json',decisions)
    log(f'completed fusion: {report["export"]["matches"]:,} matches, validator PASS in {report["elapsed_seconds"]:.1f}s')
    return report
