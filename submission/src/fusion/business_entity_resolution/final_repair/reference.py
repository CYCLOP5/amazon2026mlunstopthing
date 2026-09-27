'Exact colleague blend/calibration building blocks for labeled countries'
import numpy as np
import polars as pl

KEYS = ['qid', 'tid']
BLEND_MIN_P = .001


def _checked(frame, score):
    if frame.select(KEYS).n_unique() != frame.height:
        raise ValueError('Duplicate score keys')
    if frame[score].null_count() or not frame[score].is_finite().all() or not frame[score].is_between(0, 1).all():
        raise ValueError('Invalid probability')
    return frame.select(*KEYS, score, *(['y'] if 'y' in frame else []))


def blend(friend, graph, w=.25):
    'reference prefilter, outer union, missing=1e-4, then logit blend'
    if not 0 <= w <= 1:
        raise ValueError('Blend weight outside [0, 1]')
    friend = _checked(friend, 'p2').filter(pl.col('p2') >= BLEND_MIN_P)
    graph = _checked(graph, 'head').filter(pl.col('head') >= BLEND_MIN_P)
    d = friend.join(graph, on=KEYS, how='full', coalesce=True, suffix='_r', validate='1:1')
    if 'y_r' in d:
        if d.filter(pl.col('y').is_not_null() & pl.col('y_r').is_not_null() & (pl.col('y') != pl.col('y_r'))).height:
            raise ValueError('Conflicting labels')
        d = d.with_columns(pl.coalesce('y', 'y_r').alias('y')).drop('y_r')
    d = d.with_columns(pl.col('p2', 'head').fill_null(1e-4))
    def lg(c):
        x = pl.col(c).clip(1e-6, 1-1e-6)
        return (x/(1-x)).log()
    return d.with_columns((1/(1+(-((1-w)*lg('p2')+w*lg('head'))).exp())).alias('p')).drop('p2', 'head')


def selected_decoder(report):
    "return the selected method's actual score and tuned decoder; no fallback"
    try:
        method = report['methods'][report['selected']]
        score, rule, cut = method['score'], method['tune']['rule'], float(method['tune']['threshold'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Missing selected method decoder') from exc
    if not isinstance(score, str) or not score or rule not in ('top1_threshold', 'plain_threshold', 'expected_f') or not np.isfinite(cut):
        raise ValueError('Invalid selected method decoder')
    return {'score': score, 'rule': rule, 'threshold': cut}

MIN_P, BINS, TOP_P = 0.02, 25, 0.99


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


EDGES = np.linspace(_logit(np.array([MIN_P]))[0], _logit(np.array([0.999]))[0], BINS)
CENTRES = np.r_[EDGES[0] - 1.0, (EDGES[:-1] + EDGES[1:]) / 2, EDGES[-1] + 1.0]
TOP = np.flatnonzero(np.r_[-np.inf, EDGES] >= _logit(np.array([TOP_P]))[0])


def _hist(p, y, n_ent):
    idx = np.digitize(_logit(p), EDGES)
    if y is None:
        return np.bincount(idx, minlength=BINS + 1) / n_ent
    return np.bincount(idx, weights=y, minlength=BINS + 1) / n_ent


def segments(pairs: pl.DataFrame, s1_addr: pl.DataFrame, t_addr: pl.DataFrame) -> pl.DataFrame:
    "house relation: 'equal' = source-1 house number among the target's address numbers"
    nums = lambda c: pl.col(c).fill_null("").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
    s = s1_addr.select("qid", pl.col("ad").fill_null("").str.extract(r"(\d+)", 1).str.strip_chars_start("0").fill_null("").alias("hs"))
    t = t_addr.select("tid", nums("ad").alias("nums"))
    d = pairs.join(s, on="qid", how="left").join(t, on="tid", how="left")
    rel = (pl.when((pl.col("hs") != "") & pl.col("nums").list.contains(pl.col("hs"))).then(pl.lit("equal"))
             .when((pl.col("hs") != "") & (pl.col("nums").list.len() > 0)).then(pl.lit("conflict"))
             .otherwise(pl.lit("unknown")))
    return d.with_columns(rel.alias("seg")).drop("hs", "nums")


def fit(hold: pl.DataFrame, test: pl.DataFrame, n_hold: dict, n_test: dict) -> dict:
    'hold: qid, p, y, co, seg (validation pairs of the audit anchors); test: qid, p, co, seg'
    fits = {}
    for co in sorted(test["co"].unique().to_list()):
        h = hold.filter(pl.col("co") == co)
        nh = n_hold.get(co)
        if not h.height or nh is None:
            raise ValueError(f'No labeled calibration country: {co}')
        t = test.filter(pl.col("co") == co)
        nt = n_test[co]
        n1_all = _hist(h["p"].to_numpy(), h["y"].to_numpy().astype(float), nh)
        m_all = _hist(t["p"].to_numpy(), None, nt)
        a = float(m_all[TOP].sum() / max(n1_all[TOP].sum(), 1e-9))
        for seg in ("equal", "conflict", "unknown"):
            hs, ts = h.filter(pl.col("seg") == seg), t.filter(pl.col("seg") == seg)
            n1 = _hist(hs["p"].to_numpy(), hs["y"].to_numpy().astype(float), nh)
            m = _hist(ts["p"].to_numpy(), None, nt)
            post = np.clip(a * n1 / np.maximum(m, 1e-9), 0, 1)
            post[m < 1e-6] = np.nan
            post = pl.Series(post).fill_nan(None).fill_null(strategy="forward").fill_null(0.0).to_numpy().copy()
            post[0] = min(post[0], post[1])
            post = np.maximum.accumulate(post)
            fits[f"{co}|{seg}"] = {"a": a, "posterior": post.round(4).tolist()}
    return fits


def apply(test: pl.DataFrame, fits: dict) -> pl.DataFrame:
    parts = []
    for (cell,), g in test.with_columns((pl.col("co") + "|" + pl.col("seg")).alias("_cell")).group_by("_cell"):
        f = fits.get(cell)
        p = g["p"].to_numpy()
        if f is None:
            parts.append(g.drop("_cell"))
            continue
        adj = np.interp(_logit(p), CENTRES, np.asarray(f["posterior"]))
        adj = np.where(p >= MIN_P, adj, np.minimum(p, adj))
        parts.append(g.drop("_cell").with_columns(pl.Series("p", adj.astype(np.float32))))
    return pl.concat(parts)



def calibrated_predictions(hold, live, refs_hold, targets_hold, refs_live, targets_live, n_hold, n_live):
    'reference segmented labeled-country posteriors, before expected-f decoding'
    labeled = set(hold['co'].unique().to_list())
    if set(live['co'].unique().to_list()) - labeled:
        raise ValueError('Unlabeled country in calibration input')
    h = segments(hold.filter(pl.col('p') >= BLEND_MIN_P), refs_hold, targets_hold)
    t = segments(live.filter(pl.col('p') >= BLEND_MIN_P), refs_live, targets_live)
    tables = fit(h, t, n_hold, n_live)
    return apply(t, tables), tables
