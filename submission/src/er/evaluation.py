"metrics: the challenge's macro f-beta (per source-1 entity, singletons included)"
import polars as pl


def macro_fbeta(matches: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5) -> float:
    'matches/truth: (s1_id, o_id) pairs; s1_ids: every evaluated Source-1 id'
    b2 = beta * beta
    base = pl.DataFrame({"s1_id": s1_ids.unique()})
    tp = matches.join(truth, on=["s1_id", "o_id"], how="inner").group_by("s1_id").agg(pl.len().alias("tp"))
    npred = matches.group_by("s1_id").agg(pl.len().alias("np"))
    ntrue = truth.group_by("s1_id").agg(pl.len().alias("nt"))
    d = (base.join(tp, on="s1_id", how="left").join(npred, on="s1_id", how="left")
             .join(ntrue, on="s1_id", how="left").fill_null(0))
    prec = d["tp"] / d["np"]
    rec = d["tp"] / d["nt"]
    f = ((1 + b2) * prec * rec / (b2 * prec + rec)).fill_nan(0.0).fill_null(0.0)
    f = pl.select(pl.when((d["np"] == 0) & (d["nt"] == 0)).then(1.0).otherwise(f)).to_series()
    return float(f.mean())


def per_entity_fbeta(matches: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5) -> pl.DataFrame:
    'per-entity scores (for error analysis): s1_id, n_true, n_pred, tp, f'
    b2 = beta * beta
    base = pl.DataFrame({"s1_id": s1_ids.unique()})
    tp = matches.join(truth, on=["s1_id", "o_id"], how="inner").group_by("s1_id").agg(pl.len().alias("tp"))
    d = (base.join(tp, on="s1_id", how="left")
             .join(matches.group_by("s1_id").agg(pl.len().alias("n_pred")), on="s1_id", how="left")
             .join(truth.group_by("s1_id").agg(pl.len().alias("n_true")), on="s1_id", how="left").fill_null(0))
    return d.with_columns(
        pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
          .when((pl.col("tp") == 0)).then(0.0)
          .otherwise((1 + b2) * (pl.col("tp") / pl.col("n_pred")) * (pl.col("tp") / pl.col("n_true"))
                     / (b2 * pl.col("tp") / pl.col("n_pred") + pl.col("tp") / pl.col("n_true"))).alias("f"))
