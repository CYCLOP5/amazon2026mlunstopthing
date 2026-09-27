'hard-coded rules on top of the learned decision (reported and exported separately)'
import polars as pl

from er.stack import decode

MIN_P = 0.02
RULES = ("R1_decoy_reject", "R2_sibling_rescue", "R3_range_rescue")


def _frame(df: pl.DataFrame, col: str) -> pl.DataFrame:
    need = ["qid", "tid", col, "num_n1", "num_n2", "num_first_eq", "hn_in_range", "sib_exact_hn_src",
            "sib_best_n", "sib_best_a", "addr_empty2", "n_tset"] + (["y"] if "y" in df.columns else [])
    return df.select(need).rename({col: "p"}).with_columns(pl.col(pl.Int8, pl.Int16).fill_null(0))


def accepted(df: pl.DataFrame, col: str, rule: str, thr: float, use=RULES) -> pl.DataFrame:
    d = _frame(df, col)
    acc = decode.apply(rule, d.select("qid", "tid", "p", *(["y"] if "y" in d.columns else [])), thr)
    acc = acc.select("qid", "tid").join(d, on=["qid", "tid"], how="left")
    if "R1_decoy_reject" in use:
        decoy = ((pl.col("num_n1") > 0) & (pl.col("num_n2") > 0) & (pl.col("num_first_eq") == 0)
                 & (pl.col("hn_in_range") == 0) & (pl.col("sib_exact_hn_src") >= 1))
        acc = acc.filter(~decoy)
    rescue = []
    top = decode.top1(d).filter(pl.col("p") >= MIN_P).join(acc.select("tid"), on="tid", how="anti")
    num_ok = (pl.col("num_first_eq") == 1) | (pl.col("hn_in_range") == 1) | (pl.col("num_n2") == 0)
    if "R2_sibling_rescue" in use:
        rescue.append(top.filter((pl.col("sib_best_n") >= 95) & ((pl.col("sib_best_a") >= 90) | (pl.col("addr_empty2") == 1))
                                 & num_ok))
    if "R3_range_rescue" in use:
        rescue.append(top.filter((pl.col("hn_in_range") == 1) & (pl.col("n_tset") >= 90)))
    if rescue:
        acc = pl.concat([acc] + rescue).unique(["qid", "tid"])
    return acc


def evaluate(tr: pl.DataFrame, anchors: pl.DataFrame, col: str, rule: str, thr: float, sc: dict) -> dict:
    variants = {"learned only": ()} | {f"+ {r}": (r,) for r in RULES} | {"all rules": RULES}
    out = {"tune": {}, "audit": {}}
    for name, use in variants.items():
        acc = accepted(tr, col, rule, thr, use).select("qid", "y")
        for key, fold in (("tune", sc["fit_fold"]), ("audit", sc["audit_fold"])):
            out[key][name] = decode.score(acc, anchors.filter(pl.col("fold") == fold).select("qid", "deg"))
    return out


def apply_all(te: pl.DataFrame, col: str, rule: str, thr: float) -> pl.DataFrame:
    return accepted(te, col, rule, thr, RULES).select("qid", "tid")
