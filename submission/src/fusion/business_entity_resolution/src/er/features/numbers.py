'Group "numbers": house / plot numbers'
import numpy as np
import polars as pl
from rapidfuzz.distance import Hamming

from er.features import FEATURE_GROUPS
from er.features.name import cp


@FEATURE_GROUPS.register("numbers")
def numbers_group(p: pl.DataFrame) -> pl.DataFrame:
    nu1 = pl.col("addr_nums_1").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    nu2 = pl.col("addr_nums_2").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    q = p.select(nu1.alias("nu1"), nu2.alias("nu2"))
    q = q.with_columns(pl.col("nu1").list.first().alias("h1"), pl.col("nu2").list.first().alias("h2"))
    q = q.with_columns(
        pl.col("nu1").list.len().alias("num_n1"),
        pl.col("nu2").list.len().alias("num_n2"),
        pl.col("nu1").list.set_intersection("nu2").list.len().alias("num_inter"),
        (pl.col("h1") == pl.col("h2")).alias("num_first_eq"),
        pl.col("h1").is_in(pl.col("nu2")).alias("num_first1_in2"),

        ((pl.col("h1") != pl.col("h2")) & (pl.col("h1").str.ends_with(pl.col("h2"))
                                         | pl.col("h2").str.ends_with(pl.col("h1")))).alias("hn_suffix"),
        ((pl.col("h1") != pl.col("h2")) & (pl.col("h1").str.starts_with(pl.col("h2"))
                                         | pl.col("h2").str.starts_with(pl.col("h1")))).alias("hn_prefix"),
        (pl.col("h1").str.len_chars().cast(pl.Int16) - pl.col("h2").str.len_chars().cast(pl.Int16)).abs().alias("hn_len_diff"),

        (pl.col("h1").str.slice(-12).cast(pl.Int64, strict=False)
         - pl.col("h2").str.slice(-12).cast(pl.Int64, strict=False)).abs().cast(pl.Float64).log1p().alias("hn_log_absdiff"),
    ).with_columns(
        (pl.col("num_inter") / pl.max_horizontal(pl.min_horizontal("num_n1", "num_n2"), 1)).alias("num_ovl"),
        ((pl.col("num_n1") > 0) & (pl.col("num_n2") > 0) & (pl.col("num_inter") == 0)).alias("num_conflict"),
    )

    h1 = q["h1"].fill_null("").to_list()
    h2 = q["h2"].fill_null("").to_list()
    ham = cp(h1, h2, Hamming.distance, scorer_kwargs={"pad": True})
    same_len = (q["h1"].str.len_chars() == q["h2"].str.len_chars()).fill_null(False).to_numpy()
    q = q.with_columns(pl.Series("hn_digit_diff", np.where(same_len, ham, np.nan), dtype=pl.Float32))
    return q.drop("nu1", "nu2", "h1", "h2")
