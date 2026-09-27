

'source1 set decisions from calibrated independent pair probabilities'
import argparse as ap
import concurrent.futures as cf
import itertools as it

import numpy as np


def expected(p):
    'exact expected f0.5 for every prefix, including the empty prediction'
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be a finite matrix in [0,1]")
    b, n = p.shape
    pmf = np.zeros((b, n + 1, n + 1))
    pmf[:, 0, 0] = 1
    for k in range(1, n + 1):
        z = p[:, k - 1, None]
        pmf[:, k, :k] += pmf[:, k - 1, :k] * (1 - z)
        pmf[:, k, 1:k + 1] += pmf[:, k - 1, :k] * z
    score = np.zeros((b, n + 1))
    score[:, 0] = pmf[:, n, 0]
    inv = np.broadcast_to(1 / np.arange(1, 5 * n + 1), (b, 5 * n)).copy()
    for k in range(n, 0, -1):
        hits = np.arange(1, k + 1)
        score[:, k] = 5 * np.sum(pmf[:, k, 1:k + 1] * hits * inv[:, 4 * k + hits - 1], axis=1)
        z = p[:, k - 1, None]
        inv = (1 - z) * inv[:, :-1] + z * inv[:, 1:]
    return score


def winners(qid, tid, prob, raw=None):
    raw = prob if raw is None else raw
    order = np.lexsort((qid, -raw, -prob, tid))
    return order[np.r_[True, tid[order][1:] != tid[order][:-1]]] if len(order) else order


def choose(qid, tid, prob, raw=None, floor=.05, exact=64, threads=1):
    qid, tid, prob = (np.asarray(x) for x in (qid, tid, prob))
    raw = prob if raw is None else np.asarray(raw)
    if (qid.ndim != 1 or any(x.shape != qid.shape for x in (tid, prob, raw)) or
            qid.dtype.kind not in "iu" or tid.dtype.kind not in "iu" or
            not np.isfinite(prob).all() or not np.isfinite(raw).all() or
            ((prob < 0) | (prob > 1)).any() or not 0 <= floor <= 1 or
            not isinstance(exact, int) or not 1 <= exact <= 256 or threads < 1):
        raise ValueError("invalid decoder inputs")
    if not len(qid):
        return np.zeros(0, bool), {"groups": 0, "approximate_groups": 0}
    order = winners(qid, tid, prob, raw)
    order = order[np.lexsort((tid[order], -raw[order], -prob[order], qid[order]))]
    starts = np.r_[0, np.flatnonzero(qid[order][1:] != qid[order][:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    sizes = ends - starts
    keep = np.zeros(len(qid), bool)
    counts = np.zeros(len(starts), np.int32)
    ps = prob[order]

    def batch(ids):
        width = int(sizes[ids].max())
        cols = np.arange(width)
        valid = cols[None, :] < sizes[ids, None]
        rows = np.minimum(starts[ids, None] + cols, len(ps) - 1)
        p = np.where(valid, ps[rows], 0)
        gain = expected(p)
        gain[:, 1:] = np.where(valid & (p >= floor), gain[:, 1:], -1)
        return ids, gain.argmax(axis=1)

    tasks = []
    for upper in (4, 8, 16, 32, 64, 128, 256):
        lo = 0 if upper == 4 else upper // 2
        ids = np.flatnonzero((sizes > lo) & (sizes <= min(upper, exact)))
        step = max(8, min(512, 2_000_000 // (upper + 1) ** 2))
        tasks.extend(ids[i:i + step] for i in range(0, len(ids), step))
    with cf.ThreadPoolExecutor(max_workers=threads) as pool:
        for ids, k in pool.map(batch, tasks):
            counts[ids] = k
    large = np.flatnonzero(sizes > exact)
    for i in large:
        p = ps[starts[i]:ends[i]].astype(np.float64)
        # ponytail: large groups use a ratio of expectations; exact dp is quadratic
        gain = 1.25 * p.cumsum() / (.25 * p.sum() + np.arange(1, len(p) + 1))
        gain[p < floor] = -1
        empty = np.prod(1 - p)
        k = int(gain.argmax())
        counts[i] = k + 1 if gain[k] > empty else 0
    rank = np.arange(len(order)) - np.repeat(starts, sizes)
    keep[order] = rank < np.repeat(counts, sizes)
    return keep, {"groups": len(starts), "approximate_groups": len(large), "exact_limit": exact}


def check():
    rng = np.random.default_rng(7)
    for n in range(1, 7):
        for p in (rng.random(n), np.zeros(n), np.ones(n), np.array([0., 1., .5, .2, .9, .01])[:n]):
            p = np.sort(p)[::-1]
            want = np.zeros(n + 1)
            for values in it.product((0, 1), repeat=n):
                y = np.asarray(values)
                mass = np.prod(np.where(y, p, 1 - p))
                for k in range(n + 1):
                    f = float(k == 0) if not y.sum() else 5 * y[:k].sum() / (4 * k + y.sum())
                    want[k] += mass * f
            np.testing.assert_allclose(expected(p[None])[0], want, atol=1e-12)
    np.testing.assert_allclose(expected(np.array([[.4], [.6]])), [[.6, .4], [.4, .6]])
    keep, stats = choose(np.array([0, 1, 2, 2]), np.array([0, 0, 1, 2]), np.array([.7, .7, .99, .6]),
                         raw=np.array([.8, .9, .99, .6]), threads=2)
    assert keep.tolist() == [False, True, True, False] and stats["approximate_groups"] == 0
    keep, _ = choose(np.array([0]), np.array([0]), np.array([.49]))
    assert not keep[0]
    for exact in (3, 6, 10):
        q = np.repeat(np.arange(9), np.arange(1, 10))
        t = np.arange(len(q))
        p = np.ones(len(q))
        keep, _ = choose(q, t, p, exact=exact, threads=2)
        assert keep.all()
    print("decoder checks passed")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
