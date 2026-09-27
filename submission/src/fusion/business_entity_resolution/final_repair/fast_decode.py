'batched expected-f decoder preserving the historical stack decision rule'
import numpy as np
import polars as pl
from er.decision import MAX_GROUP, best_k
from er.stack import decode


def _batch_best_k(probabilities):
    'poisson-binomial expectations for equally sized groups, with strict ties'
    batch, n = probabilities.shape
    pre = np.zeros((n+1, batch, n+1), dtype=np.float64)
    suf = np.zeros_like(pre)
    pre[0, :, 0] = suf[n, :, 0] = 1.
    for i in range(n):
        p = probabilities[:, i, None]
        pre[i+1, :, :i+1] = pre[i, :, :i+1]*(1-p)
        pre[i+1, :, 1:i+2] += pre[i, :, :i+1]*p
    for i in range(n-1, -1, -1):
        p = probabilities[:, i, None]
        length = n-i
        suf[i, :, :length] = suf[i+1, :, :length]*(1-p)
        suf[i, :, 1:length+1] += suf[i+1, :, :length]*p
    optimum = suf[0, :, 0].copy()
    chosen = np.zeros(batch, dtype=np.int64)
    ambiguous = np.zeros(batch, dtype=bool)
    for k in range(1, n+1):
        a = np.arange(k+1)[:, None]
        m = np.arange(n-k+1)[None, :]
        f = 1.25*a/(.25*(a+m)+k)
        expectation = np.einsum('bi,bj,ij->b', pre[k, :, :k+1], suf[k, :, :n-k+1], f, optimize=False)


        ambiguous |= np.abs(expectation-optimum) <= 1e-13
        improve = expectation > optimum
        optimum[improve] = expectation[improve]
        chosen[improve] = k
    for i in np.flatnonzero(ambiguous):
        chosen[i] = best_k(probabilities[i])
    return chosen


def expected_f(pairs: pl.DataFrame, floor: float) -> pl.DataFrame:
    best = decode.top1(pairs).filter(pl.col('p') >= floor).sort(
        ['qid', 'p'], descending=[False, True])
    if not best.height:
        return best
    owner = best['qid'].to_numpy()
    probs = best['p'].to_numpy().astype(np.float64)
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    sizes = np.diff(np.r_[starts, len(owner)])
    clipped = np.minimum(sizes, MAX_GROUP)
    keep = np.zeros(best.height, dtype=bool)
    single = starts[sizes == 1]
    keep[single] = probs[single] > .5
    for n in np.unique(clipped[sizes > 1]):
        owners = starts[(sizes > 1) & (clipped == n)]
        offsets = np.arange(n)
        for offset in range(0, len(owners), 8192):
            index = owners[offset:offset+8192, None]+offsets
            chosen = _batch_best_k(probs[index])
            keep[index] = offsets < chosen[:, None]
    return best.filter(pl.Series(keep))


def apply(rule: str, pairs: pl.DataFrame, threshold: float) -> pl.DataFrame:
    if rule == 'expected_f':
        return expected_f(pairs, threshold)
    return decode.apply(rule, pairs, threshold)
