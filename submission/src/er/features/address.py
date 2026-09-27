'group "address": address similarities (the strongest signal against hard negatives)'
import polars as pl
from rapidfuzz import fuzz

from er.features import FEATURE_GROUPS
from er.features.name import cp


@FEATURE_GROUPS.register("address")
def address_group(p: pl.DataFrame) -> pl.DataFrame:
    a1, a2 = p["addr_can_1"].to_list(), p["addr_can_2"].to_list()
    s1, s2 = p["addr_skel_1"].to_list(), p["addr_skel_2"].to_list()
    f = {
        "a_ratio": cp(a1, a2, fuzz.ratio),
        "a_tset": cp(a1, a2, fuzz.token_set_ratio),
        "a_tsort": cp(a1, a2, fuzz.token_sort_ratio),
        "a_partial": cp(a1, a2, fuzz.partial_ratio),
        "as_tset": cp(s1, s2, fuzz.token_set_ratio),
        "as_tsort": cp(s1, s2, fuzz.token_sort_ratio),
    }
    t1 = pl.col("addr_can_1").str.split(" ")
    t2 = pl.col("addr_can_2").str.split(" ")
    q = p.select(
        pl.col("addr_missing_2").alias("addr_missing2"),
        (pl.col("addr_can_2") == "").alias("addr_empty2"),
        pl.col("addr_can_1").str.len_chars().alias("a_len1"),
        pl.col("addr_can_2").str.len_chars().alias("a_len2"),

        (t1.list.set_intersection(t2).list.len()
         / pl.max_horizontal(t1.list.set_union(t2).list.len(), 1)).alias("a_jacc"),
        pl.when((pl.col("addr_state_1") == "") | (pl.col("addr_state_2") == "")).then(None)
          .otherwise(pl.col("addr_state_1") == pl.col("addr_state_2")).alias("state_eq"),
    )
    return pl.concat([pl.DataFrame(f), q], how="horizontal")
