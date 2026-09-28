"""team feature contract without owner-sampled reference aggregates"""
import argparse as ap

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler as jw

import tm_prep

backend = "teammate-v1-nos1"
attrs = ["name_core", "name_skel", "name_legal", "name_full", "name_alt", "name_compact",
         "addr_can", "addr_skel", "addr_nums", "addr_state", "name_nonascii", "addr_missing", "n_is_dom"]
ff = ["bscore", "cos_name", "cos_addr", "brank", "o_ncand", "o_bmax", "o_nrank", "o_arank", "o_nmax", "o_amax",
      "o_gap", "o_ngap", "o_agap", "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "nf_tset", "ns_ratio",
      "ns_tset", "nc_ratio", "nc_partial", "n_alt12", "n_alt21", "a_ratio", "a_tset", "a_tsort", "a_partial",
      "as_tset", "as_tsort", "n_len1", "n_len2", "n_ntok1", "n_ntok2", "n_first_eq", "legal_eq", "legal_empty1",
      "legal_empty2", "has_alt2", "nonascii2", "is_dom2", "addr_missing2", "addr_empty2", "a_len1", "a_len2",
      "state_eq", "num_n1", "num_n2", "num_inter", "num_first_eq", "num_first1_in2", "num_ovl", "num_conflict"]


def contract(m):
    if m.get("feature_backend") != backend or m.get("feature_names") != ff or m.get("dense_features", []) != []:
        raise ValueError("teammate feature contract mismatch")
    return list(ff), []


def prep(d):
    if {"rid", "nm", "ad", "co"} - set(d.columns) or d["rid"].n_unique() != len(d):
        raise ValueError("invalid teammate source records")
    src = d.select(pl.col("rid").cast(pl.UInt32).alias("entity_id"), pl.col("co").alias("country"),
                   pl.col("nm").alias("business_name"), pl.col("ad").alias("business_address"))
    return tm_prep.process(src)


def context(p):
    if {"qid", "tid", "ns", "ads"} - set(p.columns) or p.select("qid", "tid").n_unique() != len(p):
        raise ValueError("invalid teammate candidate pairs")
    if not np.isfinite(p.select("ns", "ads").to_numpy()).all():
        raise ValueError("non-finite candidate similarities")
    c = p.select(pl.col("qid").cast(pl.UInt32).alias("s1_id"), pl.col("tid").cast(pl.UInt32).alias("o_id"),
                 (pl.col("ns").cast(pl.Float32) + pl.col("ads").cast(pl.Float32)).alias("bscore"),
                 pl.col("ns").cast(pl.Float32).alias("cos_name"), pl.col("ads").cast(pl.Float32).alias("cos_addr"))
    c = c.with_row_index("ix").sort(["o_id", "bscore", "s1_id"], descending=[False, True, False]).with_columns(
        (pl.col("s1_id").cum_count().over("o_id") - 1).cast(pl.Int32).alias("brank")).sort("ix")
    c = c.with_columns(pl.len().over("o_id").alias("o_ncand"), pl.col("bscore").max().over("o_id").alias("o_bmax"),
                       pl.col("cos_name").rank("min", descending=True).over("o_id").alias("o_nrank"),
                       pl.col("cos_addr").rank("min", descending=True).over("o_id").alias("o_arank"),
                       pl.col("cos_name").max().over("o_id").alias("o_nmax"),
                       pl.col("cos_addr").max().over("o_id").alias("o_amax"))
    second = c.filter(pl.col("brank") == 1).select("o_id", pl.col("bscore").alias("o_b2"))
    c = c.join(second, on="o_id", how="left", maintain_order="left", validate="m:1").with_columns(pl.col("o_b2").fill_null(0.0))
    c = c.with_columns(pl.when(pl.col("brank") == 0).then(pl.col("bscore") - pl.col("o_b2"))
                       .otherwise(pl.col("bscore") - pl.col("o_bmax")).alias("o_gap"),
                       (pl.col("cos_name") - pl.col("o_nmax")).alias("o_ngap"),
                       (pl.col("cos_addr") - pl.col("o_amax")).alias("o_agap")).drop("o_b2")
    return c.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def strings(p, threads):
    vals = {c: p[c].to_list() for c in p.columns if p[c].dtype == pl.String}
    f = {}
    for n, a, b, scorer in [
        ("n_ratio", "name_core_1", "name_core_2", fuzz.ratio),
        ("n_tsort", "name_core_1", "name_core_2", fuzz.token_sort_ratio),
        ("n_tset", "name_core_1", "name_core_2", fuzz.token_set_ratio),
        ("n_partial", "name_core_1", "name_core_2", fuzz.partial_ratio),
        ("n_jw", "name_core_1", "name_core_2", jw.normalized_similarity),
        ("nf_tset", "name_full_1", "name_full_2", fuzz.token_set_ratio),
        ("ns_ratio", "name_skel_1", "name_skel_2", fuzz.ratio),
        ("ns_tset", "name_skel_1", "name_skel_2", fuzz.token_set_ratio),
        ("nc_ratio", "name_compact_1", "name_compact_2", fuzz.ratio),
        ("nc_partial", "name_compact_1", "name_compact_2", fuzz.partial_ratio),
        ("n_alt12", "name_core_1", "name_alt_2", fuzz.token_set_ratio),
        ("n_alt21", "name_alt_1", "name_core_2", fuzz.token_set_ratio),
        ("a_ratio", "addr_can_1", "addr_can_2", fuzz.ratio),
        ("a_tset", "addr_can_1", "addr_can_2", fuzz.token_set_ratio),
        ("a_tsort", "addr_can_1", "addr_can_2", fuzz.token_sort_ratio),
        ("a_partial", "addr_can_1", "addr_can_2", fuzz.partial_ratio),
        ("as_tset", "addr_skel_1", "addr_skel_2", fuzz.token_set_ratio),
        ("as_tsort", "addr_skel_1", "addr_skel_2", fuzz.token_sort_ratio),
    ]:
        f[n] = process.cpdist(vals[a], vals[b], scorer=scorer, workers=threads, dtype=np.float32)
    q = p.select(
        pl.col("name_core_1").str.len_chars().alias("n_len1"), pl.col("name_core_2").str.len_chars().alias("n_len2"),
        pl.col("name_core_1").str.count_matches(" ").alias("n_ntok1"), pl.col("name_core_2").str.count_matches(" ").alias("n_ntok2"),
        (pl.col("name_core_1").str.split(" ").list.first() == pl.col("name_core_2").str.split(" ").list.first()).alias("n_first_eq"),
        (pl.col("name_legal_1") == pl.col("name_legal_2")).alias("legal_eq"),
        (pl.col("name_legal_1") == "").alias("legal_empty1"), (pl.col("name_legal_2") == "").alias("legal_empty2"),
        (pl.col("name_alt_2") != "").alias("has_alt2"), pl.col("name_nonascii_2").alias("nonascii2"),
        pl.col("n_is_dom_2").alias("is_dom2"), pl.col("addr_missing_2").alias("addr_missing2"),
        (pl.col("addr_can_2") == "").alias("addr_empty2"),
        pl.col("addr_can_1").str.len_chars().alias("a_len1"), pl.col("addr_can_2").str.len_chars().alias("a_len2"),
        pl.when((pl.col("addr_state_1") == "") | (pl.col("addr_state_2") == "")).then(None)
          .otherwise(pl.col("addr_state_1") == pl.col("addr_state_2")).alias("state_eq"),
        pl.col("addr_nums_1").str.split(" ").alias("nu1"), pl.col("addr_nums_2").str.split(" ").alias("nu2"))
    q = q.with_columns(pl.col("nu1").list.eval(pl.element().filter(pl.element() != "")),
                       pl.col("nu2").list.eval(pl.element().filter(pl.element() != "")))
    q = q.with_columns(pl.col("nu1").list.len().alias("num_n1"), pl.col("nu2").list.len().alias("num_n2"),
                       pl.col("nu1").list.set_intersection("nu2").list.len().alias("num_inter"),
                       (pl.col("nu1").list.first() == pl.col("nu2").list.first()).alias("num_first_eq"),
                       pl.col("nu1").list.first().is_in(pl.col("nu2")).alias("num_first1_in2"))
    q = q.with_columns((pl.col("num_inter") / pl.max_horizontal(pl.min_horizontal("num_n1", "num_n2"), 1)).alias("num_ovl"),
                       ((pl.col("num_n1") > 0) & (pl.col("num_n2") > 0) & (pl.col("num_inter") == 0)).alias("num_conflict"))
    return pl.concat([pl.DataFrame(f), q.drop("nu1", "nu2").with_columns(pl.col(pl.Boolean).cast(pl.Int8))], how="horizontal_extend")


def make(r, q, p, threads=1, dense=()):
    if threads < 1 or dense:
        raise ValueError("invalid teammate feature options")
    if not len(p):
        return np.empty((0, len(ff)), dtype=np.float32), list(ff)
    c = context(p)
    qt = prep(q)
    ra = r.select(pl.col("entity_id").alias("s1_id"), *[pl.col(a).alias(a + "_1") for a in attrs])
    qa = qt.select(pl.col("entity_id").alias("o_id"), *[pl.col(a).alias(a + "_2") for a in attrs])
    x = np.empty((len(p), len(ff)), dtype=np.float32)
    for lo in range(0, len(c), 50000):
        ch = c.slice(lo, 50000)
        joined = (ch.select("s1_id", "o_id").join(ra, on="s1_id", how="left", maintain_order="left", validate="m:1")
                  .join(qa, on="o_id", how="left", maintain_order="left", validate="m:1"))
        if joined.null_count().sum_horizontal().sum():
            raise ValueError("teammate feature join lost a record")
        x[lo:lo + len(ch)] = pl.concat([ch, strings(joined, threads)], how="horizontal_extend").select(ff).to_numpy().astype(np.float32)
    if np.isinf(x).any():
        raise ValueError("infinite teammate feature")
    return x, list(ff)


def check():
    r = pl.DataFrame({"rid": [1, 2], "nm": ["alpha private limited", "beta inc"],
                      "ad": ["0045 main road, maharashtra", "9 road, ohio"], "co": ["india", "us"]})
    q = pl.DataFrame({"rid": [7, 8], "nm": ["अल्फा प्राइवेट लिमिटेड", "beta.com"],
                      "ad": ["45 main rd, mh", ""], "co": ["india", "us"]})
    p = pl.DataFrame({"qid": [2, 1, 2], "tid": [7, 7, 8], "ns": [0.2, 0.9, 0.8], "ads": [0.1, 0.9, 0.0]})
    st = prep(r)
    x, names = make(st, q, p)
    assert names == ff and x.shape == (3, 54) and not np.isinf(x).any()
    assert x[0, ff.index("brank")] == 1 and x[1, ff.index("brank")] == 0
    assert x[1, ff.index("num_ovl")] == 1 and x[2, ff.index("addr_empty2")] == 1
    y, _ = make(st, q.with_columns(pl.lit(999).alias("own")), p.with_columns(pl.lit(1).alias("y")))
    np.testing.assert_equal(x, y)
    reordered, _ = make(st.reverse(), q.reverse(), p.reverse())
    np.testing.assert_equal(x, reordered[::-1])
    try:
        make(st, q, p.with_columns(pl.lit(1000).alias("qid")))
        raise AssertionError("unknown reference accepted")
    except ValueError:
        pass
    assert prep(q.tail(1))["addr_can"].to_list() == [""]
    assert tm_prep.name_tok_map("4beta8")[0] == "abetab"
    print("teammate feature checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument("--check", action="store_true")
    if p.parse_args().check:
        check()
    else:
        p.error("use --check")
