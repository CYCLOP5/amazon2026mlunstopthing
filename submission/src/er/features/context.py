'group "context": how a pair compares with the competing candidates'
import polars as pl

from er.features import FEATURE_GROUPS

CONTEXT_COLS = ["bscore", "brank", "cos_name", "cos_addr",
                "s1_ncand", "s1_bmax", "s1_ntop1", "s1_gap",
                "o_ncand", "o_bmax", "o_gap", "o_nrank", "o_arank", "o_nmax", "o_amax", "o_ngap", "o_agap",
                "is_s3", "o_name_ties",

                "s1_name_freq", "o_name_s1freq", "view"]


def s1_aggregates(cand_glob: str) -> pl.DataFrame:
    'per-source-1 statistics over all its blocking candidates (streamed, low memory)'
    return (pl.scan_parquet(cand_glob)
              .group_by("s1_id")
              .agg(pl.len().cast(pl.Int32).alias("s1_ncand"), pl.col("bscore").max().alias("s1_bmax"),
                   (pl.col("brank") == 0).sum().cast(pl.Int32).alias("s1_ntop1"))
              .collect(engine="streaming"))


def add_context(c: pl.DataFrame, s1agg: pl.DataFrame) -> pl.DataFrame:
    'c: one blocking chunk (all candidates of each of its Source-2/3 records)'
    c = c.join(s1agg, on="s1_id", how="left", maintain_order="left")
    c = c.with_columns(
        pl.len().over("o_id").alias("o_ncand"),
        pl.col("bscore").max().over("o_id").alias("o_bmax"),
        pl.col("cos_name").rank("min", descending=True).over("o_id").alias("o_nrank"),
        pl.col("cos_addr").rank("min", descending=True).over("o_id").alias("o_arank"),
        pl.col("cos_name").max().over("o_id").alias("o_nmax"),
        pl.col("cos_addr").max().over("o_id").alias("o_amax"),
        pl.col("o_id").str.starts_with("S3-").cast(pl.Int8).alias("is_s3"),
    )

    second = c.filter(pl.col("brank") == 1).select("o_id", pl.col("bscore").alias("o_b2"))
    c = c.join(second, on="o_id", how="left", maintain_order="left").with_columns(pl.col("o_b2").fill_null(0.0))
    c = c.with_columns(
        pl.when(pl.col("brank") == 0).then(pl.col("bscore") - pl.col("o_b2"))
          .otherwise(pl.col("bscore") - pl.col("o_bmax")).alias("o_gap"),
        (pl.col("bscore") - pl.col("s1_bmax")).alias("s1_gap"),
        (pl.col("cos_name") - pl.col("o_nmax")).alias("o_ngap"),
        (pl.col("cos_addr") - pl.col("o_amax")).alias("o_agap"),

        (pl.col("cos_name") >= pl.col("o_nmax") - 1e-4).cast(pl.Int32).sum().over("o_id").alias("o_name_ties"),
    ).drop("o_b2")
    return c.with_columns(pl.col(pl.Float64).cast(pl.Float32), pl.col(pl.UInt32).cast(pl.Int32))


@FEATURE_GROUPS.register("context")
def context_group(p: pl.DataFrame) -> pl.DataFrame:
    return p.select([c for c in CONTEXT_COLS if c in p.columns])
