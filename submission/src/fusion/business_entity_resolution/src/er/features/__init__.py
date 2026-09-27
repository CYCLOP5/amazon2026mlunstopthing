'Pair features, organised in groups that the config switches on/off:'
import polars as pl

from er.registry import Registry
from er.safe import guard

FEATURE_GROUPS = Registry("feature group")


ATTRS = ["name_core", "name_skel", "name_legal", "name_full", "name_alt", "name_compact",
         "addr_can", "addr_skel", "addr_nums", "addr_state", "name_nonascii", "addr_missing", "n_is_dom"]

from er.features import address, context, name, numbers  # noqa: E402,F401  (register groups)


def build_features(cand: pl.DataFrame, s1: pl.DataFrame, oth: pl.DataFrame, groups: list,
                   sub_rows: int = 500_000) -> pl.DataFrame:
    'cand: candidate chunk with context columns. s1/oth: normalised records needed by it'
    s1a = s1.select(pl.col("entity_id").alias("s1_id"), *[pl.col(a).alias(a + "_1") for a in ATTRS])
    oa = oth.select(pl.col("entity_id").alias("o_id"), *[pl.col(a).alias(a + "_2") for a in ATTRS])
    fns = [FEATURE_GROUPS.get(g) for g in groups]
    parts = []
    for off in range(0, cand.height, sub_rows):
        p = (cand.slice(off, sub_rows)
                 .join(s1a, on="s1_id", how="left", maintain_order="left")
                 .join(oa, on="o_id", how="left", maintain_order="left"))
        parts.append(pl.concat([p.select("s1_id", "o_id")] + [fn(p) for fn in fns], how="horizontal"))
        del p
        guard("features")
    out = pl.concat(parts)
    return out.with_columns(pl.col(pl.Boolean).cast(pl.Int8), pl.col(pl.Float64).cast(pl.Float32))


def feature_columns(schema_names) -> list:
    return [c for c in schema_names if c not in ("s1_id", "o_id", "label")]
