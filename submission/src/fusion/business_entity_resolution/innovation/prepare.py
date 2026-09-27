'select difficult targets without labels and prepare frozen-e5 verification'
import gc
import json
from pathlib import Path
import polars as pl

from er.stack.inputs import load_refs, load_targets
from er.stack.features import normalize, text_features, chunks_by_tid, score_features
from innovation.common import read_parent, log


def select_targets(pred, targets, max_targets=500_000, topk=8):
    stats = pred.group_by('tid').agg(pl.col('p').max().alias('best'),
        pl.col('p').top_k(2).alias('_p'))
    stats = stats.with_columns(pl.col('_p').list.get(1, null_on_oob=True).fill_null(0).alias('second')).drop('_p')
    props = targets.select(pl.col('rid').alias('tid'),
        (pl.col('ad').fill_null('').str.strip_chars() == '').alias('missing_address'), 'co')
    wanted = stats.join(props, on='tid').filter(pl.col('missing_address') |
        ((pl.col('best') < .995) & (pl.col('best') > .001)) | (pl.col('second') > .05))
    available = wanted.height

    wanted = wanted.with_columns(pl.col('tid').hash(8501).alias('_hash')).sort(
        ['missing_address', '_hash'], descending=[True, False]).head(max_targets).drop('_hash')
    sel = pred.join(wanted.select('tid'), on='tid').sort(
        ['tid', 'p', 'qid'], descending=[False, True, False]).with_columns(
            pl.int_range(0, pl.len()).over('tid').alias('_rank')).filter(pl.col('_rank') < topk).drop('_rank')
    return sel, {'eligible_targets': available, 'selected_targets': wanted.height,
        'selected_pairs': sel.height, 'max_targets': max_targets, 'topk': topk,
        'selected_by_country': wanted.group_by('co').len().sort('co').to_dicts(),
        'selected_missing_address': int(wanted['missing_address'].sum())}


def run(data, parent, output, max_targets=500_000, topk=8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split in ('train', 'test'):
        dest = output/split
        (dest/'requests').mkdir(parents=True, exist_ok=True)
        refs, targets = load_refs(data, split), load_targets(data, split)
        pred, parent_report = read_parent(parent, split)
        chosen, report = select_targets(pred, targets, max_targets, topk)
        log(f'{split} verification selection: {report}')

        sf = score_features(pred.rename({'p': 'prob'}).drop('y', strict=False), int((targets['sr']==2).sum()))
        sf = sf.join(chosen.select('qid', 'tid'), on=['qid', 'tid'])
        del pred, chosen
        gc.collect()
        rr = normalize(refs.select('rid', 'nm', 'ad', 'co'))
        tt = normalize(targets.join(sf.select(pl.col('tid').alias('rid')).unique(), on='rid')
                       .select('rid', 'nm', 'ad', 'co'))
        parts = [text_features(chunk, rr, tt, ['name','address','numbers'])
                 for chunk in chunks_by_tid(sf.select('qid','tid').sort('tid','qid'), 100_000)]
        feats = sf.join(pl.concat(parts), on=['qid', 'tid']).rename({'prob': 'parent_p'})
        del sf, parts
        freq = rr.group_by('co','name_core').len(name='name_frequency')
        rf = rr.join(freq, on=['co','name_core']).select(pl.col('rid').alias('qid'),'name_frequency')
        feats = feats.join(rf, on='qid').join(targets.select(pl.col('rid').alias('tid'),'co'), on='tid')
        if split == 'train':
            own = pl.concat([pl.read_parquet(Path(data)/split/f's{i}.parquet', columns=['rid','own']) for i in (2,3)])
            own = own.select(pl.col('rid').cast(pl.UInt32).alias('tid'),pl.col('own').cast(pl.Int64))
            feats = feats.join(own,on='tid').with_columns((pl.col('qid').cast(pl.Int64)==pl.col('own')).cast(pl.UInt8).alias('y'))
            feats = feats.join(refs.select(pl.col('rid').alias('qid'),'fold'),on='qid')
            report['selected_true_pairs'] = int(feats['y'].sum())
            report['selected_labels_by_country'] = feats.group_by('co').agg(pl.len(),pl.col('y').sum()).to_dicts()
        feats = feats.sort('tid','qid')
        feats.write_parquet(dest/'features.parquet')
        left = refs.select(pl.col('rid').alias('qid'),pl.col('nm').alias('nm1'),pl.col('ad').alias('ad1'),pl.col('co').alias('co1'))
        right = targets.select(pl.col('rid').alias('tid'),pl.col('nm').alias('nm2'),pl.col('ad').alias('ad2'),pl.col('co').alias('co2'))
        canon1 = rr.select(pl.col('rid').alias('qid'),pl.col('name_full').alias('cn1'),pl.col('addr_can').alias('ca1'))
        canon2 = tt.select(pl.col('rid').alias('tid'),pl.col('name_full').alias('cn2'),pl.col('addr_can').alias('ca2'))
        requests = feats.select('qid','tid').join(left,on='qid').join(right,on='tid').join(canon1,on='qid').join(canon2,on='tid').sort('tid','qid')
        for i, off in enumerate(range(0, requests.height, 50_000)):
            requests.slice(off, 50_000).write_parquet(dest/'requests'/f'part_{i:05d}.parquet')
        report['request_parts'] = (requests.height+49_999)//50_000
        reports[split] = report
        (output/'selection.json').write_text(json.dumps(reports,indent=2))
        log(f'{split} prepared {requests.height:,} pairs for four independent E5 views')
        del refs,targets,rr,tt,feats,requests,rf,left,right,canon1,canon2
        gc.collect()
    (output/'parent_decision.json').write_text(json.dumps(parent_report['selected_decoder'],indent=2))
    (output/'_SUCCESS').write_text('complete\n')
