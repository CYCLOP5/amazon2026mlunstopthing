'group "name": business-name similarities'
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from er.features import FEATURE_GROUPS
from er.safe import THREADS


def cp(a, b, scorer, **kw):
    'element-wise multithreaded similarity of two equal-length string lists'
    return process.cpdist(a, b, scorer=scorer, workers=THREADS, dtype=np.float32, **kw)


@FEATURE_GROUPS.register("name")
def name_group(p: pl.DataFrame) -> pl.DataFrame:
    L = {c: p[c].to_list() for c in ("name_core_1", "name_core_2", "name_full_1", "name_full_2", "name_skel_1",
                                     "name_skel_2", "name_compact_1", "name_compact_2", "name_alt_1", "name_alt_2")}
    n1, n2 = L["name_core_1"], L["name_core_2"]
    f = {
        "n_ratio": cp(n1, n2, fuzz.ratio),
        "n_tsort": cp(n1, n2, fuzz.token_sort_ratio),
        "n_tset": cp(n1, n2, fuzz.token_set_ratio),
        "n_partial": cp(n1, n2, fuzz.partial_ratio),
        "n_jw": cp(n1, n2, JaroWinkler.normalized_similarity),
        "nf_tset": cp(L["name_full_1"], L["name_full_2"], fuzz.token_set_ratio),
        "ns_ratio": cp(L["name_skel_1"], L["name_skel_2"], fuzz.ratio),
        "ns_tset": cp(L["name_skel_1"], L["name_skel_2"], fuzz.token_set_ratio),
        "nc_ratio": cp(L["name_compact_1"], L["name_compact_2"], fuzz.ratio),
        "nc_partial": cp(L["name_compact_1"], L["name_compact_2"], fuzz.partial_ratio),

        "n_alt12": cp(n1, L["name_alt_2"], fuzz.token_set_ratio),
        "n_alt21": cp(L["name_alt_1"], n2, fuzz.token_set_ratio),
    }
    t1 = pl.col("name_core_1").str.split(" ")
    t2 = pl.col("name_core_2").str.split(" ")
    q = p.select(
        pl.col("name_core_1").str.len_chars().alias("n_len1"),
        pl.col("name_core_2").str.len_chars().alias("n_len2"),
        pl.col("name_core_1").str.count_matches(" ").alias("n_ntok1"),
        pl.col("name_core_2").str.count_matches(" ").alias("n_ntok2"),
        (t1.list.first() == t2.list.first()).alias("n_first_eq"),

        t1.list.set_difference(t2).list.len().alias("n_only1"),
        t2.list.set_difference(t1).list.len().alias("n_only2"),
        (pl.col("name_legal_1") == pl.col("name_legal_2")).alias("legal_eq"),
        (pl.col("name_legal_1") == "").alias("legal_empty1"),
        (pl.col("name_legal_2") == "").alias("legal_empty2"),
        (pl.col("name_alt_2") != "").alias("has_alt2"),
        pl.col("name_nonascii_2").alias("nonascii2"),
        pl.col("n_is_dom_2").alias("is_dom2"),
    )
    return pl.concat([pl.DataFrame(f), q], how="horizontal")
