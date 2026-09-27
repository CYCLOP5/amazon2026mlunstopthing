'sparse joint owner allocation using fit-only, soft source count priors'
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export, tuning_anchors


def learn_prior(refs, targets, maximum=24, smoothing=1.0):
    'include zero-count anchors; exclude every tuning and audit owner'
    fit = refs.filter((pl.col('fold') == 0) & ((pl.col('rid').hash(1033) % 5) != 0))
    if not fit.height:
        raise ValueError('No fit owners for count priors')
    counts = targets.filter(pl.col('own') >= 0).group_by('own', 'sr').len()
    histograms = []
    for src in (2, 3):
        d = fit.select(pl.col('rid').cast(pl.Int64).alias('own')).join(
            counts.filter(pl.col('sr') == src).select('own', 'len'), on='own', how='left')
        n = d['len'].fill_null(0).to_numpy().astype(int)
        histogram = np.bincount(np.minimum(n, maximum), minlength=maximum + 1).astype(float)
        histogram += smoothing
        histogram /= histogram.sum()


        histogram[-1] *= .5
        histograms.append(histogram.tolist())
    return {'probabilities': histograms, 'maximum': maximum, 'tail_ratio': .5,
            'fit_entities': fit.height, 'smoothing': smoothing,
            'fit_rule': 'fold == 0 and hash(rid,1033)%5 != 0'}


def log_count(prior, count, source):
    maximum = prior['maximum']
    table = prior['probabilities'][source]
    return float(np.log(table[min(count, maximum)]) + max(0, count-maximum)*np.log(prior['tail_ratio']))


def prepare_problem(pairs, accepted, max_targets=300_000, max_candidates=32, floor=.03, max_component=512):
    'Only uncertain groups become Python/NumPy objects; retain the full pool outside'
    summary = pairs.group_by('tid').agg(pl.col('p').top_k(2).alias('_top'), pl.len().alias('_n'))
    summary = summary.with_columns(pl.col('_top').list.get(0).alias('_p'),
        pl.col('_top').list.get(1, null_on_oob=True).fill_null(0).alias('_second'))
    eligible = summary.filter((pl.col('_n') <= max_candidates) & (pl.col('_p') >= floor)
        & ~((pl.col('_p') >= .995) & ((pl.col('_p')-pl.col('_second')) >= .2)))
    eligible = eligible.with_columns((pl.col('_p')-.5).abs().alias('_uncertainty')).sort('_uncertainty','tid').head(max_targets)
    active = pairs.join(eligible.select('tid'), on='tid').filter(pl.col('p') >= floor)

    incumbent = accepted.join(eligible.select('tid'),on='tid').select('qid','tid','p')
    active = pl.concat([active.select('qid','tid','p'),incumbent]).unique(['qid','tid']).sort('tid','qid')


    if active.height:
        tids = active['tid'].to_numpy()
        owners,owner_index = np.unique(active['qid'].to_numpy(),return_inverse=True)
        parent = np.arange(len(owners))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        starts = np.flatnonzero(np.r_[True,tids[1:] != tids[:-1]])
        ends = np.r_[starts[1:],len(tids)]
        for s,e in zip(starts,ends):
            r = root(owner_index[s])
            for j in range(s+1,e):
                other = root(owner_index[j])
                if other != r:
                    parent[other] = r
        roots = np.asarray([root(owner_index[s]) for s in starts])
        component_sizes = np.bincount(roots,minlength=len(owners))
        allowed = tids[starts][component_sizes[roots] <= max_component]
        active = active.filter(pl.col('tid').is_in(pl.Series(allowed).implode()))
    frozen = accepted.join(active.select('tid').unique(),on='tid',how='anti')
    return active, frozen, {'eligible_targets': eligible.height, 'active_pairs': active.height,
        'active_targets':active['tid'].n_unique(),'frozen_matches': frozen.height,
        'max_targets':max_targets,'max_candidates':max_candidates,'floor':floor,'max_component':max_component}


def allocate(pairs, accepted, prior, n_s2, strength, problem=None, passes=8):
    'coordinate ascent on categorical unary odds plus soft count log likelihood'
    if strength == 0:
        return accepted, {'moves':0,'passes':0,'fallback':True}
    active,frozen,info = problem or prepare_problem(pairs,accepted)
    if not active.height:
        return accepted, {**info,'moves':0,'passes':0}
    aq = active['qid'].to_numpy()
    tids = active['tid'].to_numpy()
    owners = np.unique(np.r_[aq, accepted['qid'].to_numpy()])
    ids = np.searchsorted(owners,aq)
    count = np.zeros((len(owners),2),dtype=np.int64)
    oldq = accepted['qid'].to_numpy()
    oldt = accepted['tid'].to_numpy()
    np.add.at(count,(np.searchsorted(owners,oldq),(oldt >= n_s2).astype(int)),1)


    bound = int(count.max()) + (info.get('max_component',512)) + 2
    increments = np.asarray([[log_count(prior,n+1,source)-log_count(prior,n,source)
        for n in range(bound)] for source in (0,1)])
    starts = np.flatnonzero(np.r_[True,tids[1:] != tids[:-1]])
    ends = np.r_[starts[1:],len(tids)]
    group_tids = tids[starts]
    original = active.join(accepted.select('qid','tid').with_columns(pl.lit(True).alias('_accepted')),
                           on=['qid','tid'],how='left',maintain_order='left')['_accepted'].fill_null(False).to_numpy()
    chosen = np.full(len(starts),-1,dtype=int)
    for i,(s,e) in enumerate(zip(starts,ends)):
        hits = np.flatnonzero(original[s:e])
        if len(hits) > 1:
            raise ValueError('Joint allocation requires at most one accepted owner per target')
        if len(hits):
            chosen[i] = s+hits[0]
    p = np.clip(active['p'].to_numpy().astype(float),1e-6,1-1e-6)
    unary = np.log(p/(1-p))


    order = np.argsort(np.maximum.reduceat(unary,starts),kind='stable')
    total_moves = 0
    for iteration in range(passes):
        moves = 0
        for i in order:
            s,e = starts[i],ends[i]
            src = int(group_tids[i] >= n_s2)
            old = chosen[i]
            if old >= 0:
                count[ids[old],src] -= 1

            utilities = unary[s:e].copy()
            for j in range(s,e):
                n = count[ids[j],src]
                utilities[j-s] += strength*increments[src,n]
            best = int(np.argmax(utilities))
            new = s+best if utilities[best] > 0 else -1
            old_utility = utilities[old-s] if old >= 0 else 0.
            new_utility = utilities[new-s] if new >= 0 else 0.
            if new_utility <= old_utility + 1e-10:
                new = old
            if new >= 0:
                count[ids[new],src] += 1
            if new != old:
                moves += 1
                chosen[i] = new
        total_moves += moves
        if not moves:
            break
    sel = active[np.asarray(chosen[chosen >= 0],dtype=np.uint32).tolist()]
    keys = pl.concat([frozen.select('qid','tid'),sel.select('qid','tid')])
    res = pairs.join(keys,on=['qid','tid'],how='inner')
    return res,{**info,'moves':total_moves,'passes':iteration+1}


def run(data, parent, output):
    from graph_resolution.pipeline import compare_audit
    data,parent,output = Path(data),Path(parent),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    parent_report = json.loads((parent/'report.json').read_text())
    weight = parent_report['selected_head_weight']
    decision = parent_report['selected_decoder']
    if decision['rule'] not in ('top1_threshold','expected_f'):
        raise ValueError('Parent decoder must enforce one owner per target')
    def read(split):
        filename = 'validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'
        d = pl.read_parquet(parent/filename)
        return d.select('qid','tid',((1-weight)*pl.col('base')+weight*pl.col('head')).alias('p'),
                        *(['y'] if 'y' in d.columns else []))
    refs = load_refs(data,'train')
    targets = load_targets(data,'train')

    truth = pl.concat([pl.read_parquet(data/'train'/f's{sr}.parquet',columns=['rid','own']).with_columns(
        pl.lit(sr).alias('sr')) for sr in (2,3)])
    prior = learn_prior(refs,truth)
    (output/'prior.json').write_text(json.dumps(prior,indent=2))
    anchors = refs.select(pl.col('rid').alias('qid'),'fold','deg')
    tune = tuning_anchors(anchors,{'fit_fold':0,'seed':42,'tune_holdout_buckets':5})
    audit = anchors.filter(pl.col('fold') == 1).select('qid','deg')
    pairs = read('train')
    base = decode.apply(decision['rule'],pairs,decision['threshold'])
    problem = prepare_problem(pairs,base)
    trials = {}
    best_strength = 0.
    best_score = decode.score(base,tune)['macro_f05']
    winner = base
    n_s2 = targets.filter(pl.col('sr') == 2).height
    for strength in (0., .05, .1, .25, .5, 1.):
        res,info = allocate(pairs,base,prior,n_s2,strength,problem)
        metrics = decode.score(res,tune)
        trials[str(strength)] = {'tune':metrics,'allocation':info}
        if metrics['macro_f05'] > best_score + 1e-12:
            best_strength,best_score,winner = strength,metrics['macro_f05'],res
    comparison = compare_audit(winner,base,audit)
    comparison['reference'] = 'frozen graph-refinement parent and decoder'
    report = {'selected_strength':best_strength,'tuning':trials,'audit':decode.score(winner,audit),
        'parent_audit':decode.score(base,audit),'comparison':comparison,
        'parent_head_weight':weight,'parent_decoder':decision,'prior':prior,
        'limitations':['Coordinate ascent is a local optimum, not exact macro F0.5 optimization.',
            'Source count marginals assume transferable generation counts; no hard capacity is imposed.',
            'Validation saved pool covers only evaluation-reachable targets, so background count context is incomplete.',
            'Designed after earlier audit inspection; hidden-test improvement remains unproven.']}
    winner.write_parquet(output/'validation_accepted.parquet')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    del pairs,base,problem,winner,targets,truth
    gc.collect()
    pairs = read('test')
    targets = load_targets(data,'test')
    base = decode.apply(decision['rule'],pairs,decision['threshold'])
    accepted,info = allocate(pairs,base,prior,targets.filter(pl.col('sr') == 2).height,best_strength)
    accepted.write_parquet(output/'test_accepted.parquet')
    report['test_allocation'] = info
    report['export'] = export(pairs,'p',decision['rule'],decision['threshold'],load_refs(data,'test'),
        targets,str(output/'output'),accepted=accepted)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data','parent','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    run(args.data,args.parent,args.output)
