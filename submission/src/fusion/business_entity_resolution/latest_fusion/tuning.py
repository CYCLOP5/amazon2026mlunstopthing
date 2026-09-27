'bounded search utilities. check owners remain historically exposed upstream'
import hashlib
import os
from statistics import NormalDist
import numpy as np
import polars as pl
from er.stack import decode
from latest_fusion import decoder
from latest_fusion.pipeline import partition


def fit_population(refs):
    q = refs['rid'].to_numpy()
    eligible = (refs['fold'].to_numpy() == 0) & (partition(q) == 2)
    parity = ((q.astype(np.uint64)*2654435761+9001) >> 13) % 2
    return q[eligible & (parity == 0)], q[eligible & (parity == 1)]


def split_search_check(refs, tuneids):
    anchors = refs.filter(pl.col('rid').is_in(pl.Series(tuneids).implode())).select(pl.col('rid').alias('qid'), 'deg', 'co')
    search = np.array([int.from_bytes(hashlib.sha256(f'fusion-check-v1:{int(q)}'.encode()).digest()[:8], 'big') % 100 < 60 for q in anchors['qid']], dtype=bool)
    return anchors.filter(pl.Series(search)), anchors.filter(pl.Series(~search))


def top_pairs(frame):
    ix = decoder.winners(frame['qid'].to_numpy(), frame['tid'].to_numpy(), frame['p'].to_numpy(), frame['raw'].to_numpy())
    return frame[ix]


def apply_policy(frame, policy):
    kind = policy['rule']
    if kind == 'expected_f':
        threads = int(os.environ.get('ER_DECODER_THREADS', '8'))
        if threads < 1:
            raise ValueError('ER_DECODER_THREADS must be positive')
        threads = min(threads, os.cpu_count() or 1)
        keep, _ = decoder.choose(frame['qid'].to_numpy(), frame['tid'].to_numpy(), frame['p'].to_numpy(), raw=frame['raw'].to_numpy(), floor=float(policy['threshold']), exact=64, threads=threads)
        accepted = frame.filter(pl.Series(keep))
    else:
        accepted = frame if frame['tid'].n_unique() == frame.height else top_pairs(frame)
        if kind == 'top1_threshold':
            accepted = accepted.filter(pl.col('p') >= policy['threshold'])
        elif kind == 'country_threshold':
            cuts = policy['thresholds']
            unknown = set(accepted['co'].unique().to_list()) - set(cuts)
            if unknown:
                raise ValueError(f'Missing country thresholds: {unknown}')
            accepted = accepted.filter(pl.col('p') >= pl.col('co').replace_strict(cuts, return_dtype=pl.Float64))
        else:
            raise ValueError(f'Unknown policy: {kind}')
    return accepted.select('qid', 'tid', 'p', *(['y'] if 'y' in accepted else []))


def _policy_options(top, search_anchors):
    global_curve = decode.curve(top, search_anchors)
    options = [({'rule': 'top1_threshold', 'threshold': global_curve['threshold']}, global_curve)]
    cuts = {str(co): decode.curve(top, search_anchors.filter(pl.col('co') == co))['threshold'] for co in search_anchors['co'].unique()}

    country = {'rule': 'country_threshold', 'thresholds': cuts}

    accepted = top.filter(pl.col('co').is_in(list(cuts))).filter(pl.col('p') >= pl.col('co').replace_strict(cuts, default=float('inf'), return_dtype=pl.Float64))
    options.append((country, decode.score(accepted, search_anchors)))
    return options


def optimize_policy(top, search_anchors):
    return max(_policy_options(top, search_anchors), key=lambda item: item[1]['macro_f05'])


def select_policy(frame, anchors, expected_f_floors=(.05, .4, .65, .75, .8, .85), feasibility=None):


    top = frame if frame['tid'].n_unique() == frame.height else top_pairs(frame)
    search = top.join(anchors.select('qid'), on='qid', how='inner')
    options = _policy_options(search, anchors)
    for floor in expected_f_floors:
        cand = {'rule': 'expected_f', 'threshold': float(floor)}
        options.append((cand, decode.score(apply_policy(search, cand), anchors)))

    if feasibility is None:
        return max(options, key=lambda item: item[1]['macro_f05'])
    for cut in (.5, .6, .65, .7, .75, .8, .85, .9, .95):
        cand = {'rule': 'top1_threshold', 'threshold': cut}
        options.append((cand, decode.score(apply_policy(search, cand), anchors)))
    feasible = [item for item in options if feasibility(apply_policy(search, item[0]), anchors)]

    return max(feasible or options, key=lambda item: item[1]['macro_f05'])


def _owner_scores(accepted, anchors):
    counts = accepted.join(anchors.select('qid'), on='qid', how='inner').group_by('qid').agg(pl.len().alias('n'), pl.col('y').sum().alias('tp'))
    z = anchors.join(counts, on='qid', how='left').with_columns(pl.col('n', 'tp').fill_null(0)).sort('qid')
    n, tp, deg = (z[c].to_numpy().astype(float) for c in ('n', 'tp', 'deg'))
    return np.where(deg == 0, (n == 0).astype(float), 1.25*tp/np.maximum(n+.25*deg, 1e-9))


def promotion(candidate_accepted, incumbent_accepted, check_anchors, min_gain=1e-5, max_precision_loss=.00015, max_recall_loss=.0001, family_size=1, confidence_alpha=.05):
    if isinstance(family_size, bool) or not isinstance(family_size, int) or family_size < 1:
        raise ValueError('family_size must be a positive integer')
    if not 0 < confidence_alpha < 1:
        raise ValueError('confidence_alpha must be between zero and one')
    if not check_anchors.height:
        raise ValueError('No check owners')
    if check_anchors['qid'].n_unique() != check_anchors.height:
        raise ValueError('Duplicate check owners')
    delta = _owner_scores(candidate_accepted, check_anchors) - _owner_scores(incumbent_accepted, check_anchors)
    gain = float(delta.mean())

    se = float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta) > 1 else float('inf')
    interval = [gain-1.959963984540054*se, gain+1.959963984540054*se]
    critical = NormalDist().inv_cdf(1-confidence_alpha/(2*family_size))
    family_interval = [gain-critical*se, gain+critical*se]
    cand = decode.score(candidate_accepted, check_anchors)
    incumbent = decode.score(incumbent_accepted, check_anchors)
    checks = {'gain': gain > min_gain, 'positive_lower_bound': family_interval[0] > 0,
              'precision': cand['pair_precision'] >= incumbent['pair_precision']-max_precision_loss,
              'recall': cand['pair_recall'] >= incumbent['pair_recall']-max_recall_loss}
    countries = {}
    for co in check_anchors['co'].unique():
        anchors = check_anchors.filter(pl.col('co') == co)
        new, old = decode.score(candidate_accepted, anchors), decode.score(incumbent_accepted, anchors)
        countries[str(co)] = {'candidate': new, 'incumbent': old}
        checks[str(co)+'_score'] = new['macro_f05'] >= old['macro_f05']
    return {'passed': all(checks.values()), 'checks': checks, 'candidate': cand, 'incumbent': incumbent, 'countries': countries, 'paired_gain': gain, 'paired_ci95': interval, 'paired_family_interval': family_interval, 'paired_standard_error': se, 'family_size': family_size, 'confidence_level': 1-confidence_alpha/family_size, 'check_owners': len(delta), 'scope': 'Independent of this bounded search; historically exposed to upstream models. Approximate Bonferroni-adjusted normal interval assumes independent owner differences; shared target competition may violate that assumption. paired_ci95 is retained as an unadjusted diagnostic.'}
