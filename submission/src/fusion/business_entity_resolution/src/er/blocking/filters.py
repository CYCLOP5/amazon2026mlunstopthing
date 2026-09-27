'the final candidate set fed to the matching model (= candidate_pairs.tsv)'
import polars as pl


def final_candidate_filter(params: dict, columns=()) -> pl.Expr:
    'Needs columns brank, bscore, o_bmax (= max bscore of the Source-2/3 record) and,'
    main = (pl.col("brank") < params["top_k"]) & (pl.col("bscore") >= params["min_rel_score"] * pl.col("o_bmax"))
    return (main | pl.col("view")) if "view" in columns else main
