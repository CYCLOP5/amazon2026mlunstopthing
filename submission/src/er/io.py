'reading challenge files and writing submission files'
import polars as pl


def read_tsv(path: str, **kw) -> pl.DataFrame:
    'all challenge files are tab-separated with no quoting; read everything as strings'
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0, **kw)


def load_gt_pairs(path: str) -> pl.DataFrame:
    'ground truth as (s1_id, o_id) pairs'
    gt = read_tsv(path)
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("o_id"))
              .explode("o_id").filter(pl.col("o_id") != "")
              .select(pl.col("source1_entity_id").alias("s1_id"), "o_id"))


def s1_bucket(seed: int, col: str = "s1_id") -> pl.Expr:
    'Deterministic 0..99 bucket of a Source-1 id (train/validation sampling)'
    return pl.col(col).hash(seed=seed) % 100


def write_id_lists(pairs, s1_ids: pl.Series, col: str, path: str):
    'one row per source-1 id (input order) with comma-joined sorted unique ids (empty when none)'
    g = (pairs.lazy().group_by("s1_id").agg(pl.col("o_id").unique().sort().str.join(",").alias(col))
              .collect(engine="streaming"))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
             .join(g.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left")
             .with_columns(pl.col(col).fill_null("")))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            fh.write(f"{a}\t{b}\n")
