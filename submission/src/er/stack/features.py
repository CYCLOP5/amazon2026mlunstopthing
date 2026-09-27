'stacker features'
import numpy as np
import polars as pl

from er.features import ATTRS, FEATURE_GROUPS
from er.normalize import normalize_records
from er.safe import guard

EPS = 1e-6
NO_OTHER = -14.0
ID_COLS = ("qid", "tid", "y")


def _logit(c: str) -> pl.Expr:
    x = pl.col(c).clip(EPS, 1 - EPS)
    return (x / (1 - x)).log()


def _margin(d: pl.DataFrame, col: str, by: str, name: str) -> pl.DataFrame:
    'col minus the best value of the OTHER rows of the same `by` group'
    top = d.group_by(by).agg(pl.col(col).top_k(2).alias("_t")).select(
        by, pl.col("_t").list.get(0).alias("_m1"),
        pl.col("_t").list.get(1, null_on_oob=True).fill_null(NO_OTHER).alias("_m2"))
    return (d.join(top, on=by, how="left", maintain_order="left")
             .with_columns(pl.when(pl.col(col) >= pl.col("_m1")).then(pl.col(col) - pl.col("_m2"))
                             .otherwise(pl.col(col) - pl.col("_m1")).alias(name))
             .drop("_m1", "_m2"))


def score_features(pairs: pl.DataFrame, n_s2: int) -> pl.DataFrame:
    'pairs: qid, tid, prob [, gate_prob, neural_prob] [, y] of the whole split'
    scores = [c for c in ("prob", "gate_prob", "neural_prob") if c in pairs.columns]
    scores += [c for c in pairs.columns if c.startswith("ext_")]
    d = pairs.with_columns([_logit(c).alias(f"{c}_lg") for c in scores] +
                           [(pl.col("tid") >= n_s2).cast(pl.Int8).alias("is_s3"),
                            pl.len().over("tid").cast(pl.Int16).alias("t_n")])
    for c in scores:
        d = d.with_columns(pl.col(f"{c}_lg").rank("min", descending=True).over("tid").cast(pl.Int16).alias(f"{c}_trank"))
        d = _margin(d, f"{c}_lg", "tid", f"{c}_tgap")
    if {"gate_prob", "neural_prob"} <= set(scores):
        d = d.with_columns((pl.col("neural_prob_lg") - pl.col("gate_prob_lg")).alias("nn_minus_gate"))
    top1 = pl.col("prob_trank") == 1
    d = d.with_columns(
        pl.len().over("qid").cast(pl.Int32).alias("s1_nclaim"),
        top1.sum().over("qid").cast(pl.Int32).alias("s1_ntop1"),
        (top1 & (pl.col("prob") >= 0.1)).sum().over("qid").cast(pl.Int32).alias("s1_ntop1_p10"),
        (top1 & (pl.col("prob") >= 0.5)).sum().over("qid").cast(pl.Int32).alias("s1_ntop1_p50"),
        (top1 & (pl.col("prob") >= 0.5)).sum().over("qid", "is_s3").cast(pl.Int32).alias("s1_ntop1_p50_src"),
        pl.col("prob_lg").rank("min", descending=True).over("qid").cast(pl.Int32).alias("s1_rank"),
        pl.col("prob_lg").rank("min", descending=True).over("qid", "is_s3").cast(pl.Int32).alias("s1_rank_src"),
        pl.col("prob").sum().over("qid").alias("s1_sum_p"),
    )
    d = _margin(d, "prob_lg", "qid", "s1_gap")
    return d.with_columns(pl.col(pl.Float64).cast(pl.Float32))



def normalize(records: pl.DataFrame, chunk: int = 1_000_000) -> pl.DataFrame:
    'records: rid, nm, ad, co -> rid + our normalised attributes (er.normalize)'
    out = []
    for off in range(0, records.height, chunk):
        r = records.slice(off, chunk)
        raw = r.select(pl.col("rid").cast(pl.String).alias("entity_id"), pl.col("nm").alias("business_name"),
                       pl.when(pl.col("ad").str.strip_chars() == "").then(None).otherwise(pl.col("ad"))
                         .alias("business_address"), pl.col("co").alias("country"))
        n = normalize_records(raw)
        rng = r.select(pl.col("ad").str.extract_groups(r"(\d+)\s*-\s*(\d+)").alias("_g")).select(
            pl.col("_g").struct.field("1").str.slice(-12).cast(pl.Int64, strict=False).alias("rng_lo"),
            pl.col("_g").struct.field("2").str.slice(-12).cast(pl.Int64, strict=False).alias("rng_hi"))
        out.append(pl.concat([r.select("rid", "co"), n.select(ATTRS), rng], how="horizontal"))
        guard("stack/normalize")
    return pl.concat(out)


def name_frequency(ref_attr: pl.DataFrame, tgt_attr: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    'how many source-1 businesses of the country share the (core) name'
    f = ref_attr.group_by("co", "name_core").agg(pl.len().cast(pl.Int32).alias("_f"))
    s1 = ref_attr.join(f, on=["co", "name_core"], how="left").select(pl.col("rid").alias("qid"),
                                                                     pl.col("_f").alias("s1_name_freq"))
    o = tgt_attr.join(f, on=["co", "name_core"], how="left").select(
        pl.col("rid").alias("tid"), pl.col("_f").fill_null(0).alias("o_name_s1freq"))
    return s1, o


def text_features(pairs: pl.DataFrame, ref_attr: pl.DataFrame, tgt_attr: pl.DataFrame, groups: list) -> pl.DataFrame:
    'pairs: qid, tid (all candidates of every record in it). returns qid, tid + text features'
    extra = ["rng_lo", "rng_hi"]
    a1 = ref_attr.select(pl.col("rid").alias("qid"), *[pl.col(a).alias(a + "_1") for a in ATTRS + extra])
    a2 = tgt_attr.select(pl.col("rid").alias("tid"), *[pl.col(a).alias(a + "_2") for a in ATTRS + extra])
    p = (pairs.select("qid", "tid").join(a1, on="qid", how="left", maintain_order="left")
              .join(a2, on="tid", how="left", maintain_order="left"))
    f = pl.concat([p.select("qid", "tid")] + [FEATURE_GROUPS.get(g)(p) for g in groups], how="horizontal")
    first = lambda c: pl.col(c).str.split(" ").list.first().str.slice(-12).cast(pl.Int64, strict=False)

    in_rng = p.select(((first("addr_nums_1") >= pl.col("rng_lo_2")) & (first("addr_nums_1") <= pl.col("rng_hi_2"))
                       | (first("addr_nums_2") >= pl.col("rng_lo_1")) & (first("addr_nums_2") <= pl.col("rng_hi_1")))
                      .fill_null(False).cast(pl.Int8).alias("hn_in_range"))
    f = f.with_columns(pl.col(pl.Boolean).cast(pl.Int8), pl.col(pl.Float64).cast(pl.Float32),
                       (p["name_core_1"] == p["name_core_2"]).cast(pl.Int8).alias("ncore_eq"),
                       in_rng["hn_in_range"])
    if "a_tset" in f.columns:
        f = _margin(f, "a_tset", "tid", "a_tset_tgap")
    if "n_tset" in f.columns:
        f = _margin(f, "n_tset", "tid", "n_tset_tgap")
    f = f.with_columns(pl.col("ncore_eq").sum().over("tid").cast(pl.Int16).alias("t_ncore_eq"))
    if "num_first_eq" in f.columns:
        f = f.with_columns(pl.col("num_first_eq").cast(pl.Int16).sum().over("tid").alias("t_hn_eq"))
    return f.with_columns(pl.col(pl.Float64).cast(pl.Float32))



SWAP_FILL = {"sw_kind": 4, "sw_one_way": -1.0, "sw_pair_share": 0.0, "sw_fan_rec": 0.0, "sw_fan_s1": 0.0,
             "sw_add_share": 0.0, "sw_drop_share": 0.0}


def _tokens(c: str) -> pl.Expr:
    return (pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
              .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
              .list.eval(pl.element().filter(pl.element() != "")).list.unique())


def swap_features(sf: pl.DataFrame, ref_attr: pl.DataFrame, tgt_attr: pl.DataFrame) -> pl.DataFrame:
    "how the record's core name differs from the source-1 core name, and how that difference"
    p = (sf.select("qid", "tid", "prob_trank")
           .join(ref_attr.select(pl.col("rid").alias("qid"), "co", pl.col("name_core").alias("_n1")), on="qid", how="left")
           .join(tgt_attr.select(pl.col("rid").alias("tid"), pl.col("name_core").alias("_n2")), on="tid", how="left")
           .with_columns(_tokens("_n1").alias("_t1"), _tokens("_n2").alias("_t2")).drop("_n1", "_n2")
           .with_columns(pl.col("_t2").list.set_difference("_t1").alias("_a"), pl.col("_t1").list.set_difference("_t2").alias("_b"))
           .drop("_t1", "_t2")
           .with_columns(pl.col("_a").list.len().alias("_na"), pl.col("_b").list.len().alias("_nb"),
                         pl.col("_a").list.first().alias("wa"), pl.col("_b").list.first().alias("wb")).drop("_a", "_b"))
    kind = (pl.when((pl.col("_na") == 0) & (pl.col("_nb") == 0)).then(0)
              .when((pl.col("_na") == 1) & (pl.col("_nb") == 1)).then(1)
              .when((pl.col("_na") == 1) & (pl.col("_nb") == 0)).then(2)
              .when((pl.col("_na") == 0) & (pl.col("_nb") == 1)).then(3).otherwise(4))
    p = p.with_columns(kind.cast(pl.Int8).alias("sw_kind")).drop("_na", "_nb")
    top = p.filter(pl.col("prob_trank") == 1)
    n_top = top.group_by("co").len("_ntop")
    sw = top.filter(pl.col("sw_kind") == 1)
    cnt = sw.group_by("co", "wa", "wb").len("_c")
    rev = cnt.rename({"wa": "wb", "wb": "wa", "_c": "_r"})
    nw = sw.group_by("co").agg(pl.col("wa").n_unique().alias("_nwa"), pl.col("wb").n_unique().alias("_nwb"))
    fa = sw.group_by("co", "wa").agg(pl.col("wb").n_unique().alias("_fa"))
    fb = sw.group_by("co", "wb").agg(pl.col("wa").n_unique().alias("_fb"))
    pair = (cnt.join(rev, on=["co", "wa", "wb"], how="left").with_columns(pl.col("_r").fill_null(0))
               .join(n_top, on="co").join(nw, on="co").join(fa, on=["co", "wa"]).join(fb, on=["co", "wb"])
               .select("co", "wa", "wb",
                       (pl.col("_c") / (pl.col("_c") + pl.col("_r"))).alias("sw_one_way"),
                       ((pl.col("_c") + pl.col("_r")) / pl.col("_ntop") * 1e4).alias("sw_pair_share"),
                       (pl.col("_fa") / pl.col("_nwb")).alias("sw_fan_rec"), (pl.col("_fb") / pl.col("_nwa")).alias("sw_fan_s1")))
    add = (top.filter(pl.col("sw_kind") == 2).group_by("co", "wa").len("_c").join(n_top, on="co")
              .select("co", "wa", (pl.col("_c") / pl.col("_ntop") * 1e4).alias("sw_add_share")))
    drop = (top.filter(pl.col("sw_kind") == 3).group_by("co", "wb").len("_c").join(n_top, on="co")
               .select("co", "wb", (pl.col("_c") / pl.col("_ntop") * 1e4).alias("sw_drop_share")))
    out = (p.join(pair, on=["co", "wa", "wb"], how="left", maintain_order="left")
            .join(add, on=["co", "wa"], how="left", maintain_order="left")
            .join(drop, on=["co", "wb"], how="left", maintain_order="left"))
    swap, added, dropped = pl.col("sw_kind") == 1, pl.col("sw_kind") == 2, pl.col("sw_kind") == 3
    out = out.with_columns(*[pl.when(swap).then(pl.col(c)) for c in ("sw_one_way", "sw_pair_share", "sw_fan_rec", "sw_fan_s1")],
                           pl.when(added).then(pl.col("sw_add_share")), pl.when(dropped).then(pl.col("sw_drop_share")))
    return (out.select("qid", "tid", *SWAP_FILL).with_columns([pl.col(c).fill_null(v) for c, v in SWAP_FILL.items()])
               .with_columns(pl.col(pl.Float64).cast(pl.Float32)))



GROUP_FILL = {"sib_n": 0, "sib_n_src": 0, "sib_best_n": -1.0, "sib_mean_n": -1.0, "sib_best_a": -1.0,
              "sib_best_na": -1.0, "sib_hn_same": 0, "sib_exact_hn": 0, "sib_exact_hn_src": 0, "sib_max_lg": NO_OTHER}


def confident_owners(r1: pl.DataFrame, min_p: float) -> pl.DataFrame:
    'records confidently owned after round 1: best candidate of the record with p1 >= min_p'
    t = r1.sort(["tid", "p1", "qid"], descending=[False, True, False]).unique("tid", keep="first", maintain_order=True)
    return t.filter(pl.col("p1") >= min_p).select(
        "qid", pl.col("tid").alias("sid"), pl.col("is_s3").alias("s_is_s3"),
        pl.col("num_first_eq").fill_null(0).cast(pl.Int8).alias("s_hn_exact"), pl.col("r1_lg").alias("s_lg"))


def group_features(pairs: pl.DataFrame, conf: pl.DataFrame, tgt_small: pl.DataFrame) -> pl.DataFrame:
    'pairs: qid, tid, is_s3 (all candidates of the records in the chunk)'
    from er.features.name import cp
    from rapidfuzz import fuzz
    s = pairs.select("qid", "tid", "is_s3").join(conf, on="qid", how="inner").filter(pl.col("sid") != pl.col("tid"))
    out = pairs.select("qid", "tid")
    if s.height:
        tt = tgt_small.select(pl.col("rid").alias("tid"), pl.col("name_core").alias("tn"), pl.col("addr_can").alias("ta"),
                              pl.col("hn1").alias("th"))
        ss = tgt_small.select(pl.col("rid").alias("sid"), pl.col("name_core").alias("sn"), pl.col("addr_can").alias("sa"),
                              pl.col("hn1").alias("sh"))
        s = s.join(tt, on="tid", how="left").join(ss, on="sid", how="left")
        s = s.with_columns(
            pl.Series("_n", cp(s["tn"].fill_null("").to_list(), s["sn"].fill_null("").to_list(), fuzz.token_set_ratio)),
            pl.Series("_a", cp(s["ta"].fill_null("").to_list(), s["sa"].fill_null("").to_list(), fuzz.token_set_ratio)))
        same_src = pl.col("s_is_s3") == pl.col("is_s3")
        g = s.group_by("qid", "tid").agg(
            pl.len().cast(pl.Int16).alias("sib_n"),
            same_src.sum().cast(pl.Int16).alias("sib_n_src"),
            pl.col("_n").max().alias("sib_best_n"), pl.col("_n").mean().alias("sib_mean_n"),
            pl.col("_a").max().alias("sib_best_a"), ((pl.col("_n") + pl.col("_a")) / 2).max().alias("sib_best_na"),
            ((pl.col("th") == pl.col("sh")) & (pl.col("th") != "")).sum().cast(pl.Int16).alias("sib_hn_same"),
            pl.col("s_hn_exact").sum().cast(pl.Int16).alias("sib_exact_hn"),
            (pl.col("s_hn_exact").cast(pl.Boolean) & same_src).sum().cast(pl.Int16).alias("sib_exact_hn_src"),
            pl.col("s_lg").max().alias("sib_max_lg"))
        out = out.join(g, on=["qid", "tid"], how="left", maintain_order="left")
    else:
        out = out.with_columns([pl.lit(None).alias(c) for c in GROUP_FILL])
    out = out.with_columns([pl.col(c).fill_null(v) for c, v in GROUP_FILL.items()])

    out = _margin(out.with_columns(pl.col("sib_n").cast(pl.Float32).alias("_sn")), "_sn", "tid", "sib_n_tgap")
    out = _margin(out, "sib_best_na", "tid", "sib_best_na_tgap").drop("_sn")
    return out.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def chunks_by_tid(pairs: pl.DataFrame, rows: int):
    "slices of a tid-sorted frame that never split one record's candidates"
    tid = pairs["tid"].to_numpy()
    start, n = 0, len(tid)
    while start < n:
        end = min(start + rows, n)
        while end < n and tid[end] == tid[end - 1]:
            end += 1
        yield pairs.slice(start, end - start)
        start = end


def feature_names(schema_names) -> list:
    skip = set(ID_COLS) | {"prob", "gate_prob", "neural_prob", "p", "p1", "p2", "r1_lg", "fold", "deg"}
    skip |= {c for c in schema_names if c.startswith("ext_") and not c.endswith(("_lg", "_trank", "_tgap"))}
    return [c for c in schema_names if c not in skip]


def as_matrix(d: pl.DataFrame, feats: list) -> np.ndarray:
    return d.select([pl.col(f).cast(pl.Float32) for f in feats]).to_numpy()
