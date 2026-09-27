'varun-compatible density calibration and exact expected-f decoding'
import json

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from latest_fusion import decoder


def partition(ids):
    return ((np.asarray(ids, dtype=np.uint64)*np.uint64(2654435761)) >> np.uint64(13)) % 3


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1-1e-6)
    return np.log(p)-np.log1p(-p)


def adjust(p, country, segment, recipe):
    p = np.asarray(p, dtype=float)
    out = p.copy()
    for co in np.unique(country):
        for seg in range(3):
            take = (country == co) & (segment == seg)
            if not take.any():
                continue
            curve = recipe['curves'].get(f'{co}|{seg}')
            if curve is None:
                raise ValueError(f'Missing country/house calibration cell {co}|{seg}')
            out[take] = np.interp(logit(p[take]), curve['centres'], curve['posterior'])
    out = np.where(p < .02, np.minimum(p, out), out)
    return np.clip(out, 0, 1).astype(np.float32)


def histogram(p, co, seg, y, edges, use):
    bins = np.searchsorted(edges, logit(p), side='right')
    res = {}
    for country in np.unique(co[use]):
        for segment in range(3):
            take = use & (co == country) & (seg == segment)
            count = np.bincount(bins[take], minlength=len(edges)+1).astype(float)
            positive = np.bincount(bins[take], weights=y[take], minlength=len(edges)+1) if y is not None else np.zeros_like(count)
            res[f'{country}|{segment}'] = [count, positive]
    return res


def curves(train, target, ntrain, ntarget, edges, minimum=50, transfer=True):
    'sparse-bin pooling and weighted isotonic fit, as in the source run'
    centres = np.r_[edges[0]-1, (edges[:-1]+edges[1:])/2, edges[-1]+1]
    top = np.r_[-np.inf, edges] >= logit([.99])[0]
    res = {}
    for country in sorted(ntarget):
        if country not in ntrain:
            continue
        nh, nt = ntrain[country], ntarget[country]
        if not nh or not nt:
            raise ValueError('Empty calibration denominator')
        count = [train[f'{country}|{s}'][0] for s in range(3)]
        positive = [train[f'{country}|{s}'][1] for s in range(3)]
        test = [target.get(f'{country}|{s}', [np.zeros(len(edges)+1), None])[0] for s in range(3)]
        top_pos, top_test = sum(x[top].sum() for x in positive), sum(x[top].sum() for x in test)
        scale = (top_test/nt)/(top_pos/nh) if top_pos > 0 and top_test > 0 and min(top_pos,top_test) >= minimum else 1.
        if not transfer:
            scale = 1.
        for segment in range(3):
            n, y, m = count[segment], positive[segment], test[segment]
            base = np.divide(y, n, out=np.zeros_like(n), where=n > 0)
            posterior = (scale*y/nh*nt+minimum*base)/(m+minimum)
            active = (n > 0) | (m > 0)
            if active.any():
                values = IsotonicRegression(out_of_bounds='clip').fit(
                    centres[active], np.clip(posterior[active],0,1), sample_weight=(m+minimum)[active]).predict(centres)
            else:
                values = 1/(1+np.exp(-centres))
            res[f'{country}|{segment}'] = {'centres': centres.tolist(), 'posterior': values.tolist(),
                'positive_scale': float(scale), 'train_pairs': int(n.sum()), 'test_pairs': int(m.sum()), 'pooled': False}
    return res


def calibrate_methods(train, test, refs, refs_test, methods, recipe_path):
    recipe = json.loads(open(recipe_path).read())
    if recipe.get('known_score') != 'stack_prob' or recipe.get('unseen_gate') is not True:
        raise ValueError('Unexpected latest reference routing')
    if recipe.get('partition') != {'modulus':3, 'remainder':1}:
        raise ValueError('Unexpected reference calibration split')
    def metadata(frame, r):
        wanted = ['co'] + (['fold'] if 'fold' in r else [])
        frame = frame.drop(wanted, strict=False).join(r.select(pl.col('rid').alias('qid'), *wanted),on='qid',how='left',validate='m:1')
        if frame['co'].null_count():
            raise ValueError('Unmapped calibration country')


        return frame
    train, test = metadata(train,refs), metadata(test,refs_test)
    if train['seg'].null_count() or test['seg'].null_count():
        raise ValueError('Missing house relation for union-only candidates')
    co, seg = train['co'].to_numpy(), train['seg'].to_numpy()
    cot, segt = test['co'].to_numpy(), test['seg'].to_numpy()
    calibration_ids = refs.filter((pl.col('fold') == 0) & pl.Series(partition(refs['rid']) == 1))
    ntrain = dict(calibration_ids.group_by('co').len().rows())
    ntarget = dict(refs_test.group_by('co').len().rows())
    fit = (train['fold'].to_numpy() == 0) & (partition(train['qid']) == 1)
    y = train['y'].to_numpy()
    edges = np.asarray(recipe['edges'])
    report = {'fit_scope': 'fold0 partition1 labels only; complete candidate competition',
              'france': 'frozen latest gate scores and supplied calibration; no French labels', 'methods': {}}
    for method in methods:
        raw = train[method].to_numpy().astype(float)
        raw_test = test[method].to_numpy().astype(float)


        unknown = ~np.isin(cot, recipe['countries'])
        raw_test[unknown] = test['gate'].fill_null(0).to_numpy()[unknown]
        valid = train['newest_present'].to_numpy() > 0 if method == 'baseline' else np.ones(train.height, bool)
        valid_test = test['newest_present'].to_numpy() > 0 if method == 'baseline' else np.ones(test.height, bool)
        h = histogram(raw,co,seg,y,edges,fit & valid)
        empirical = {'curves':curves(h,h,ntrain,ntrain,edges,transfer=False)}
        empirical_score = adjust(raw,co,seg,empirical)
        live = histogram(raw_test,cot,segt,None,edges,valid_test & ~unknown)
        if method == 'baseline':
            target_recipe = recipe
        else:
            target_recipe = {'curves':curves(h,live,ntrain,ntarget,edges,transfer=True)}
            target_recipe['curves'].update({k:v for k,v in recipe['curves'].items() if k.rsplit('|',1)[0] not in ntrain})
        target_score = adjust(raw_test,cot,segt,target_recipe)
        train = train.with_columns(pl.Series('raw_'+method,raw.astype(np.float32)),pl.Series(method,empirical_score))
        test = test.with_columns(pl.Series('raw_'+method,raw_test.astype(np.float32)),pl.Series(method,target_score))
        report['methods'][method] = {'train':empirical, 'test':target_recipe, 'fit_pairs':int((fit & valid).sum())}
    return train, test, report


def owner_top(frame, method, baseline_only=False):
    d = frame.filter(pl.col('newest_present') > 0) if baseline_only else frame
    raw = d['raw_'+method].to_numpy()
    order = decoder.winners(d['qid'].to_numpy(),d['tid'].to_numpy(),d[method].to_numpy(),raw)
    return d[order].select('qid','tid',pl.col(method).alias('p'),*(['y'] if 'y' in d else []))


def decide(frame, method, rule='expected_f', threshold=.05, baseline_only=False):
    d = frame.filter(pl.col('newest_present') > 0) if baseline_only else frame
    if rule == 'top1_threshold':
        return owner_top(d,method).filter(pl.col('p') >= threshold)
    if rule != 'expected_f':
        raise ValueError(f'Unsupported decoder {rule}')
    keep, _ = decoder.choose(d['qid'].to_numpy(),d['tid'].to_numpy(),d[method].to_numpy(),
        raw=d['raw_'+method].to_numpy(),floor=float(threshold),exact=64,threads=8)
    return d.filter(pl.Series(keep)).select('qid','tid',pl.col(method).alias('p'),*(['y'] if 'y' in d else []))
