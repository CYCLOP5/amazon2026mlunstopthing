import argparse as ap
import os
import re
import time

import numpy as np
import polars as pl
from rapidfuzz import distance as di
from rapidfuzz import fuzz as fz
from rapidfuzz import process as rp

try:
    from data import norm
except ImportError:
    from src.data import norm


ff = [
    "nue", "aue", "nae", "aae", "ned", "nfw", "nts", "ntt", "aed", "afw", "ats", "att",
    "xed", "xat", "nqi", "nri", "nja", "aqi", "ari", "aja", "nrl", "nql", "ndl", "arl",
    "aql", "adl", "amb", "adb", "adj", "ade", "add", "adn", "nfr", "afr", "ns", "ads",
    "en", "ea", "sr", "nrk", "ark", "nsg", "asg", "cn",
]

df = ("ds_e5", "ds_qwen3", "ds_bge_m3", "ds_e5_large", "ds_e5_small")

rr = re.compile(r"\d+")


def _need(d, cs):
    z = set(d.columns)
    if z.issuperset(cs):
        return
    raise ValueError(f"missing columns {sorted(set(cs) - z)}")


def _ids(s, nm):
    a = s.to_numpy()
    if a.dtype.kind not in "iu":
        raise ValueError(f"{nm} must be integer")
    return a.astype(np.int64, copy=False)


def _pos(a, b, nm):
    if not len(a):
        if len(b):
            raise ValueError(f"unknown {nm}")
        return np.empty(0, dtype=np.intp)
    i = np.searchsorted(a, b)
    ok = i < len(a)
    ok[ok] &= a[i[ok]] == b[ok]
    if not ok.all():
        raise ValueError(f"unknown {nm}")
    return i.astype(np.intp, copy=False)


def _txt(d, c, i):
    return d[c].gather(pl.Series(i, dtype=pl.UInt32)).fill_null("").to_list()


def _sc(a, b, fn, th, scale=100):
    return rp.cpdist(a, b, scorer=fn, workers=th, dtype=np.float32) / scale


def _tm(a, b):
    z = {}

    def get(s):
        v = z.get(s)
        if v is None:
            v = frozenset(s.split())
            z[s] = v
        return v

    q = np.empty(len(a), dtype=np.float32)
    r = np.empty(len(a), dtype=np.float32)
    j = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        u, v = get(x), get(y)
        n = len(u & v)
        q[i] = n / len(v) if v else 0
        r[i] = n / len(u) if u else 0
        j[i] = n / len(u | v) if u or v else 0
    return q, r, j


def _dg(a, b):
    z = np.empty((5, len(a)), dtype=np.float32)
    w = {}

    def get(s):
        v = w.get(s)
        if v is None:
            v = set(rr.findall(s))
            w[s] = v
        return v

    for i, (x, y) in enumerate(zip(a, b)):
        u, v = get(x), get(y)
        n = len(u & v)
        both = bool(u and v)
        z[0, i] = both
        z[1, i] = n / len(u | v) if u or v else 0
        z[2, i] = bool(both and u == v)
        z[3, i] = bool(both and not n)
        z[4, i] = n
    return z


def _rk(t, s):
    n = len(t)
    if not n:
        z = np.empty(0, dtype=np.float32)
        return z, z, z
    o = np.argsort(t, kind="stable")
    u = t[o]
    st = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
    ct = np.diff(np.r_[st, n])
    mx = np.maximum.reduceat(s[o], st)
    g = np.empty(n, dtype=np.float32)
    c = np.empty(n, dtype=np.float32)
    g[o] = np.repeat(mx, ct) - s[o]
    c[o] = np.repeat(ct, ct)
    q = np.lexsort((np.arange(n), -s, t))
    w = t[q]
    st = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1]
    ct = np.diff(np.r_[st, n])
    r = np.empty(n, dtype=np.float32)
    r[q] = np.arange(n, dtype=np.float32) - np.repeat(st, ct) + 1
    return r, g, c


def _num(d, c):
    try:
        a = d[c].to_numpy().astype(np.float32, copy=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid {c}") from e
    if not np.isfinite(a).all():
        raise ValueError(f"nonfinite {c}")
    return a


def prep(ref):
    _need(ref, {"rid", "eid", "nm", "ad", "nn", "an", "co", "sr"})
    i = _ids(ref["rid"], "rid")
    if len(np.unique(i)) != len(i):
        raise ValueError("duplicate reference rid")
    d = ref.select("rid", "nm", "ad", "nn", "an").sort("rid")
    i = _ids(d["rid"], "rid")
    d = d.with_columns(
        pl.col("nm").fill_null("").map_elements(norm, return_dtype=pl.String).alias("nu"),
        pl.col("ad").fill_null("").map_elements(norm, return_dtype=pl.String).alias("au"),
        pl.col("nn").fill_null("").alias("nn"),
        pl.col("an").fill_null("").alias("an"),
    )
    n = d.group_by("nn").len().rename({"len": "nf"})
    a = d.group_by("an").len().rename({"len": "af"})
    d = d.join(n, on="nn", how="left", maintain_order="left").join(
        a, on="an", how="left", maintain_order="left")
    return {"i": i, "d": d.select("nu", "au", "nn", "an"),
            "n": np.log1p(d["nf"].to_numpy()).astype(np.float32),
            "a": np.log1p(d["af"].to_numpy()).astype(np.float32), "f": list(ff)}


def make(st, queries, pairs, threads=1, dense_features=()):
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("positive thread count required")
    if not isinstance(dense_features, (list, tuple)) or len(set(dense_features)) != len(dense_features) or any(
            not isinstance(c, str) or c not in df for c in dense_features):
        raise ValueError("invalid dense feature names")
    names = list(ff) + list(dense_features)
    _need(queries, {"rid", "eid", "nm", "ad", "nn", "an", "co", "sr"})
    _need(pairs, {"qid", "tid", "ns", "ads", "en", "ea", "sr"})
    _need(pairs, set(dense_features))
    if pairs.is_empty():
        return np.empty((0, len(names)), dtype=np.float32), names
    q = queries.select("rid", "nm", "ad", "nn", "an").sort("rid")
    qi = _ids(q["rid"], "rid")
    if len(np.unique(qi)) != len(qi):
        raise ValueError("duplicate query rid")
    ri = _pos(st["i"], _ids(pairs["qid"], "qid"), "reference qid")
    ti = _pos(qi, _ids(pairs["tid"], "tid"), "query tid")
    d = st["d"]
    nu, au, nn, an = _txt(d, "nu", ri), _txt(d, "au", ri), _txt(d, "nn", ri), _txt(d, "an", ri)
    qu = [norm(x) for x in _txt(q, "nm", ti)]
    qa = [norm(x) for x in _txt(q, "ad", ti)]
    qn, qx = _txt(q, "nn", ti), _txt(q, "an", ti)
    z = [
        (np.asarray(nu) == qu) & (np.asarray(nu) != ""),
        (np.asarray(au) == qa) & (np.asarray(au) != ""),
        (np.asarray(nn) == qn) & (np.asarray(nn) != ""),
        (np.asarray(an) == qx) & (np.asarray(an) != ""),
        _sc(nu, qu, fz.ratio, threads), _sc(nu, qu, di.JaroWinkler.normalized_similarity, threads, 1),
        _sc(nu, qu, fz.token_sort_ratio, threads), _sc(nu, qu, fz.token_set_ratio, threads),
        _sc(au, qa, fz.ratio, threads), _sc(au, qa, di.JaroWinkler.normalized_similarity, threads, 1),
        _sc(au, qa, fz.token_sort_ratio, threads), _sc(au, qa, fz.token_set_ratio, threads),
        _sc(nn, qn, fz.ratio, threads), _sc(an, qx, fz.token_set_ratio, threads),
    ]
    z.extend(_tm(nu, qu))
    z.extend(_tm(au, qa))
    z.extend([
        np.fromiter((len(x) for x in nu), dtype=np.float32, count=len(nu)),
        np.fromiter((len(x) for x in qu), dtype=np.float32, count=len(qu)),
        np.fromiter((abs(len(x) - len(y)) for x, y in zip(nu, qu)), dtype=np.float32, count=len(nu)),
        np.fromiter((len(x) for x in au), dtype=np.float32, count=len(au)),
        np.fromiter((len(x) for x in qa), dtype=np.float32, count=len(qa)),
        np.fromiter((abs(len(x) - len(y)) for x, y in zip(au, qa)), dtype=np.float32, count=len(au)),
        np.asarray([(not x) or (not y) for x, y in zip(au, qa)], dtype=np.float32),
    ])
    z.extend(_dg(an, qx))
    ns, ads, en, ea, sr = (_num(pairs, c) for c in ("ns", "ads", "en", "ea", "sr"))
    nr, ng, cn = _rk(_ids(pairs["tid"], "tid"), ns)
    ar, ag, _ = _rk(_ids(pairs["tid"], "tid"), ads)
    z.extend([st["n"][ri], st["a"][ri], ns, ads, en, ea, sr, nr, ar, ng, ag, cn])
    for c in dense_features:
        v = _num(pairs, c)
        if (v < -1).any() or (v > 1).any():
            raise ValueError(f"out of range {c}")
        z.append(v)
    x = np.column_stack(z).astype(np.float32, copy=False)
    if x.shape[1] != len(names) or not np.isfinite(x).all():
        raise ValueError("invalid feature output")
    return x, names


def check():
    r = pl.DataFrame({
        "rid": [42, 7], "eid": ["S1-b", "S1-a"], "nm": ["école", "राम"],
        "ad": ["20 Rue de l'École", "१० main road"], "nn": ["ecole", "ram"],
        "an": ["20 rue de l ecole", "10 main road"], "co": ["france", "india"], "sr": [1, 1],
        "deg": [3, 9], "uni": [0, 1], "blank": [0, 1], "fold": [0, 2],
    }, schema_overrides={"rid": pl.UInt32, "sr": pl.UInt8})
    q = pl.DataFrame({
        "rid": [99, 3], "eid": ["S2-z", "S3-y"], "nm": ["école", "राम"],
        "ad": ["20 rue de l ecole", "10 main road"], "nn": ["ecole", "ram"],
        "an": ["20 rue de l ecole", "10 main road"], "co": ["france", "india"], "sr": [2, 3],
        "own": [42, 7],
    }, schema_overrides={"rid": pl.UInt32, "sr": pl.UInt8, "own": pl.Int32})
    p = pl.DataFrame({
        "qid": [7, 42, 7], "tid": [3, 99, 99], "ns": [0.9, 0.8, 0.1], "ads": [0.4, 0.9, 0.2],
        "en": [1, 1, 0], "ea": [0, 1, 0], "sr": [3, 2, 2], "own": [7, 42, -1], "y": [1, 1, 0],
    }, schema_overrides={"qid": pl.UInt32, "tid": pl.UInt32, "sr": pl.UInt8, "own": pl.Int32, "y": pl.UInt8})
    s = prep(r)
    x, n = make(s, q, p, threads=2)
    assert s["d"]["nu"].to_list()[0] == "राम"
    assert x.shape == (3, len(n)) and x.dtype == np.float32 and np.isfinite(x).all() and n == ff
    assert x[0, n.index("nue")] == 1 and x[0, n.index("nae")] == 1
    assert x[1, n.index("nfw")] == 1
    dp = p.with_columns(pl.Series("ds_e5", [-.25, .5, .75]), pl.Series("ds_qwen3", [.1, -.2, .3]))
    dx, dn = make(s, q, dp, threads=2, dense_features=["ds_qwen3", "ds_e5"])
    assert dn == ff + ["ds_qwen3", "ds_e5"] and np.allclose(dx[0, -2:], [.1, -.25])
    for bad in (["qid"], ["tid"], ["own"], ["y"], ["co"], ["ds_e5", "ds_e5"]):
        try:
            make(s, q, dp, threads=2, dense_features=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid dense features accepted {bad}")
    for bad in (dp.drop("ds_e5"), dp.with_columns(pl.lit(float("nan")).alias("ds_e5")),
                dp.with_columns(pl.lit(1.1).alias("ds_e5"))):
        try:
            make(s, q, bad, threads=2, dense_features=["ds_e5"])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dense values accepted")
    r2 = r.with_columns(pl.lit(99).alias("deg"), pl.lit(0).alias("uni"), pl.lit(0).alias("blank"), pl.lit(1).alias("fold"))
    q2 = q.with_columns(pl.lit(-1).alias("own"))
    p2 = p.with_columns(pl.lit(-1).alias("own"), pl.lit(1).alias("y"))
    x2, n2 = make(prep(r2), q2, p2, threads=2)
    assert n2 == n and np.array_equal(x, x2)
    for c, v in (("qid", 1000), ("tid", 1000)):
        try:
            make(s, q, p.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(v).otherwise(pl.col(c)).alias(c)))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {c} accepted")
    e, ne = make(s, q, p.head(0))
    assert e.shape == (0, len(n)) and ne == n
    b = pl.concat([p] * 3334).head(10000)
    t = time.monotonic()
    y, ny = make(s, q, b, threads=min(os.cpu_count() or 1, 8))
    assert y.shape == (10000, len(n)) and ny == n and np.isfinite(y).all()
    print("checks passed", "pairs", len(y), "seconds", round(time.monotonic() - t, 3))


def main():
    a = ap.ArgumentParser()
    a.add_argument("--check", action="store_true")
    if a.parse_args().check:
        check()


if __name__ == "__main__":
    main()
