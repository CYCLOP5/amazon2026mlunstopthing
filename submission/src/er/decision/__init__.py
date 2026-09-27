'decision rules: pair probabilities -> accepted (s1_id, o_id) matches'
import numpy as np
import polars as pl

from er.registry import Registry

DECISIONS = Registry("decision rule")


@DECISIONS.register("argmax_threshold")
def argmax_threshold(pairs: pl.DataFrame, thr: float) -> pl.DataFrame:
    'Each Source-2/3 record goes to its highest-probability Source-1 candidate if p >= thr'
    cand = pairs.filter(pl.col("p") >= thr)
    best = cand.filter(pl.col("p") == pl.col("p").max().over("o_id"))
    return best.sort("s1_id").unique("o_id", keep="first").select("s1_id", "o_id", "p")


@DECISIONS.register("threshold")
def threshold_only(pairs: pl.DataFrame, thr: float) -> pl.DataFrame:
    'accept every pair with p >= thr (no one-owner constraint) - a baseline'
    return pairs.filter(pl.col("p") >= thr).select("s1_id", "o_id", "p")



MAX_GROUP = 25


def best_k(q: np.ndarray, beta: float = 0.5) -> int:
    'q: match probabilities sorted descending. returns k maximising the expected per-entity'
    n, b2 = len(q), beta * beta
    pre = [np.ones(1)]
    for p in q:
        pre.append(np.convolve(pre[-1], (1.0 - p, p)))
    suf = [None] * (n + 1)
    suf[n] = np.ones(1)
    for i in range(n - 1, -1, -1):
        suf[i] = np.convolve(suf[i + 1], (1.0 - q[i], q[i]))
    best_e, best = suf[0][0], 0
    for k in range(1, n + 1):
        a = np.arange(k + 1)[:, None]
        m = np.arange(n - k + 1)[None, :]
        f = (1 + b2) * a / (b2 * (a + m) + k)
        e = float((pre[k][:, None] * suf[k][None, :] * f).sum())
        if e > best_e:
            best_e, best = e, k
    return best


@DECISIONS.register("expected_f")
def expected_f(pairs: pl.DataFrame, thr: float) -> pl.DataFrame:
    'per-entity optimal decision for the macro f0.5 metric'
    cand = pairs.filter(pl.col("p") >= thr)
    best = (cand.filter(pl.col("p") == pl.col("p").max().over("o_id"))
                .sort("s1_id").unique("o_id", keep="first"))
    groups = (best.sort(["s1_id", "p"], descending=[False, True])
                  .group_by("s1_id", maintain_order=True).agg("o_id", "p"))
    s1_out, o_out, p_out = [], [], []
    for s1, oids, ps in groups.iter_rows():
        q = np.asarray(ps[:MAX_GROUP], dtype=np.float64)
        k = best_k(q)
        s1_out += [s1] * k
        o_out += oids[:k]
        p_out += ps[:k]
    return pl.DataFrame({"s1_id": s1_out, "o_id": o_out, "p": p_out},
                        schema={"s1_id": pl.String, "o_id": pl.String, "p": pl.Float32})
