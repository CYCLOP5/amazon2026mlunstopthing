'final submission used for the 0.984 leaderboard upload ("blend-safe")'
import argparse
import json
import os
import shutil
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.io import write_id_lists  # noqa: E402
from er.stack import decode  # noqa: E402
from er.stack.calibrate import apply, fit, segments  # noqa: E402

MIN_P = 0.001


def _lg(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


EXTRA = {"frame": None, "w": None}


def shift(cal: pl.DataFrame, k: float) -> pl.DataFrame:
    'multiply the odds of p by k (k < 1 = stricter; models a higher decoy share than the calibration assumed)'
    if k == 1:
        return cal
    return cal.with_columns((k * pl.col("p") / (k * pl.col("p") + 1 - pl.col("p"))).cast(pl.Float32).alias("p"))


def blend(ours: pl.DataFrame, raj: pl.DataFrame, w: float, extra: pl.DataFrame = None) -> pl.DataFrame:
    'full outer join on (qid, tid); a pair missing from one model gets 1e-4 there (not proposed)'
    d = ours.join(raj, on=["qid", "tid"], how="full", coalesce=True, suffix="_r")
    if "y_r" in d.columns:
        d = d.with_columns(pl.coalesce("y", "y_r").alias("y")).drop("y_r")
    if extra is not None:
        d = d.join(extra, on=["qid", "tid"], how="full", coalesce=True, suffix="_x")
        if "y_x" in d.columns:
            d = d.with_columns(pl.coalesce("y", "y_x").alias("y")).drop("y_x")
        d = d.with_columns(pl.col("p2").fill_null(1e-4), pl.col("head").fill_null(1e-4), pl.col("px").fill_null(1e-4))
        wo, wr, wx = EXTRA["w"]
        z = wo * _lg("p2") + wr * _lg("head") + wx * _lg("px")
        return d.with_columns((1 / (1 + (-z).exp())).alias("p")).drop("p2", "head", "px")
    d = d.with_columns(pl.col("p2").fill_null(1e-4), pl.col("head").fill_null(1e-4))
    return d.with_columns((1 / (1 + (-((1 - w) * _lg("p2") + w * _lg("head"))).exp())).alias("p")).drop("p2", "head")


def extra_frame(a, split_file, keep_y, qids=None):
    if not a.extra:
        return None
    x = pl.scan_parquet(os.path.join(a.extra, split_file)).select(
        "qid", "tid", *(["y"] if keep_y else []), pl.col(a.extra_col).alias("px")).filter(pl.col("px") >= MIN_P).collect()
    return x if qids is None else x.join(qids, on="qid")


def load_records(a):
    '-> refs {split: (qid, eid, co, ad[, fold])}, targets {split: (tid, eid, ad)}'
    refs, tgts = {}, {}
    for split in ("train", "test"):
        if a.prepared:
            cols = ["rid", "eid", "ad", "co"] + (["fold"] if split == "train" else [])
            refs[split] = pl.read_parquet(os.path.join(a.prepared, split, "ref.parquet"), columns=cols)
            tgts[split] = pl.concat([pl.read_parquet(os.path.join(a.prepared, split, f"s{s}.parquet"), columns=["rid", "eid", "ad"])
                                     for s in (2, 3)]).rename({"rid": "tid"})
        else:
            path = a.train_ref if split == "train" else a.test_ref
            cols = ["rid", "eid", "ad", "co"] + (["fold"] if split == "train" else [])
            refs[split] = pl.read_parquet(path, columns=cols)
            parts, off = [], 0
            for s in ("2", "3"):
                x = pl.read_csv(os.path.join(a.dataset, split, f"{split}_source{s}.tsv"), separator="\t", quote_char=None,
                                columns=["entity_id", "business_address"]).with_row_index("tid")
                parts.append(x.with_columns((pl.col("tid") + off).cast(pl.UInt32)).rename(
                    {"entity_id": "eid", "business_address": "ad"}))
                off += x.height
            tgts[split] = pl.concat(parts)
        refs[split] = refs[split].rename({"rid": "qid"}).with_columns(pl.col("qid").cast(pl.UInt32),
                                                                       pl.col("co").str.to_lowercase())
        tgts[split] = tgts[split].with_columns(pl.col("tid").cast(pl.UInt32))
    return refs, tgts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--raj", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--dataset")
    ap.add_argument("--train-ref")
    ap.add_argument("--test-ref")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-raj", type=float, default=0.25, help="Raj share of the logit blend (0.984 upload: 0.25)")
    ap.add_argument("--audit-fold", type=int, default=1, help="labelled Source-1 fold used as calibration holdout")
    ap.add_argument("--floor", type=float, default=0.05, help="expected-F decoder floor for calibrated scores")
    ap.add_argument("--extra", help="optional third model: stack dir with val_pred/test_pred.parquet")
    ap.add_argument("--extra-col", default="p1")
    ap.add_argument("--weights", help="w_ours,w_raj,w_extra logit weights (overrides --w-raj when --extra is given)")
    ap.add_argument("--unlabelled", choices=("own", "blend"), default="own",
                    help="countries without labels: our stacker's own rule (0.984 file) or the same rule on the blend")
    ap.add_argument("--odds", default="1", help="odds multiplier(s) on calibrated labelled-country scores before decoding; "
                    "first = main output (1 = 0.984 file), others written to <out>/odds_<k>/ (stricter test decisions)")
    ap.add_argument("--dump", action="store_true", help="also write test_calibrated.parquet (labelled countries)")
    a = ap.parse_args()
    if not a.prepared and not (a.dataset and a.train_ref and a.test_ref):
        ap.error("give --prepared, or --dataset with --train-ref and --test-ref")
    os.makedirs(a.out, exist_ok=True)
    if a.extra:
        EXTRA["w"] = tuple(float(v) for v in (a.weights or "0.4,0.2,0.4").split(","))

    refs, tgts = load_records(a)
    rtr, rte = refs["train"], refs["test"]
    hold_s1 = rtr.filter(pl.col("fold") == a.audit_fold).select("qid", "co")
    labelled = set(hold_s1["co"].unique().to_list())
    n_hold = dict(hold_s1.group_by("co").len().rows())
    n_hold["*"] = hold_s1.height
    n_test = dict(rte.group_by("co").len().rows())
    print(f"labelled countries: {sorted(labelled)}; test countries: {n_test}", flush=True)


    vo = pl.scan_parquet(os.path.join(a.ours, "val_pred.parquet")).select("qid", "tid", "y", "p2").filter(
        pl.col("p2") >= MIN_P).collect().join(hold_s1.select("qid"), on="qid")
    vr = pl.scan_parquet(os.path.join(a.raj, "validation_predictions.parquet")).select("qid", "tid", "y", "head").filter(
        pl.col("head") >= MIN_P).collect().join(hold_s1.select("qid"), on="qid")
    vb = blend(vo, vr, a.w_raj, extra_frame(a, "val_pred.parquet", True, hold_s1.select("qid"))).join(hold_s1, on="qid")
    hold = segments(vb.filter(pl.col("p") >= MIN_P), rtr.select("qid", "ad"), tgts["train"].select("tid", "ad"))
    del vo, vr, vb
    to = pl.scan_parquet(os.path.join(a.ours, "test_pred.parquet")).select("qid", "tid", "p2").filter(pl.col("p2") >= MIN_P).collect()
    tr = pl.scan_parquet(os.path.join(a.raj, "test_predictions.parquet")).select("qid", "tid", "head").filter(
        pl.col("head") >= MIN_P).collect()
    tb = blend(to, tr, a.w_raj, extra_frame(a, "test_pred.parquet", False)).join(rte.select("qid", "co"), on="qid").filter(
        pl.col("co").is_in(list(labelled)))
    del to, tr
    live = segments(tb.filter(pl.col("p") >= MIN_P), rte.select("qid", "ad"), tgts["test"].select("tid", "ad"))
    fits = fit(hold, live, n_hold, n_test)
    cal = apply(live, fits)
    if a.dump:
        cal.select("qid", "tid", "p", "co", "seg").write_parquet(os.path.join(a.out, "test_calibrated.parquet"))
    odds = [float(k) for k in a.odds.split(",")]
    acc_lab = decode.apply("expected_f", shift(cal, odds[0]).select("qid", "tid", "p"), a.floor).select("qid", "tid")
    stricter = {k: decode.apply("expected_f", shift(cal, k).select("qid", "tid", "p"), a.floor).select("qid", "tid") for k in odds[1:]}
    del tb, live, hold, cal


    rep = json.load(open(os.path.join(a.ours, "report.json")))
    sel = rep["methods"][rep["selected"]]
    col, rule, thr = sel["score"], sel["tune"]["rule"], sel["tune"]["threshold"]
    unl = rte.filter(~pl.col("co").is_in(list(labelled))).select("qid")
    if a.unlabelled == "own":
        tu = pl.read_parquet(os.path.join(a.ours, "test_pred.parquet"), columns=["qid", "tid", col]).rename({col: "p"}).join(unl, on="qid")
    else:
        to = pl.scan_parquet(os.path.join(a.ours, "test_pred.parquet")).select("qid", "tid", pl.col(col).alias("p2")).filter(
            pl.col("p2") >= MIN_P).collect().join(unl, on="qid")
        tr = pl.scan_parquet(os.path.join(a.raj, "test_predictions.parquet")).select("qid", "tid", "head").filter(
            pl.col("head") >= MIN_P).collect().join(unl, on="qid")
        tu = blend(to, tr, a.w_raj, extra_frame(a, "test_pred.parquet", False, unl))
        del to, tr
    acc_unl = decode.apply(rule, tu.select("qid", "tid", "p"), thr).select("qid", "tid")

    acc = pl.concat([acc_lab, acc_unl]).unique("tid", keep="first").sort("qid", "tid")
    acc.write_parquet(os.path.join(a.out, "accepted.parquet"))


    s1_ids = rte.sort("qid")["eid"]
    m = (acc.join(rte.select("qid", pl.col("eid").alias("s1_id")), on="qid")
            .join(tgts["test"].select("tid", pl.col("eid").alias("o_id")), on="tid").select("s1_id", "o_id"))
    assert m.height == acc.height
    write_id_lists(m, s1_ids, "matched_entity_ids", os.path.join(a.out, "matching_results.tsv"))
    c = (pl.read_csv(a.candidates, separator="\t", schema_overrides={"candidate_entity_ids": pl.Utf8})
           .with_columns(pl.col("candidate_entity_ids").fill_null("").str.split(",")).explode("candidate_entity_ids")
           .rename({"source1_entity_id": "s1_id", "candidate_entity_ids": "o_id"}))
    missing = m.join(c, on=["s1_id", "o_id"], how="anti")
    if missing.height:
        write_id_lists(pl.concat([c.filter(pl.col("o_id") != ""), missing]), s1_ids, "candidate_entity_ids",
                       os.path.join(a.out, "candidate_pairs.tsv"))
    else:
        shutil.copyfile(a.candidates, os.path.join(a.out, "candidate_pairs.tsv"))
    per = dict(acc.join(rte.select("qid", "co"), on="qid").group_by("co").len().rows())
    info = {"w_raj": a.w_raj, "weights_3": EXTRA["w"], "extra": a.extra, "floor": a.floor, "unlabelled": a.unlabelled, "ours_rule_unlabelled": [col, rule, thr], "matches": acc.height,
            "matches_per_s1": {k: round(per.get(k, 0) / n_test[k], 3) for k in n_test},
            "candidates_added": missing.height,
            "calibration_a": {k: round(v["a"], 3) for k, v in fits.items() if k.endswith("|equal")}}
    json.dump(info, open(os.path.join(a.out, "final_blend.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)
    for k, al in stricter.items():
        d = os.path.join(a.out, f"odds_{k}")
        os.makedirs(d, exist_ok=True)
        ak = pl.concat([al, acc_unl]).unique("tid", keep="first").sort("qid", "tid")
        ak.write_parquet(os.path.join(d, "accepted.parquet"))
        mk = (ak.join(rte.select("qid", pl.col("eid").alias("s1_id")), on="qid")
                .join(tgts["test"].select("tid", pl.col("eid").alias("o_id")), on="tid").select("s1_id", "o_id"))
        write_id_lists(mk, s1_ids, "matched_entity_ids", os.path.join(d, "matching_results.tsv"))
        print(json.dumps({"odds": k, "matches": ak.height, "removed_vs_base": acc.join(ak, on=["qid", "tid"], how="anti").height,
                          "added_vs_base": ak.join(acc, on=["qid", "tid"], how="anti").height}), flush=True)
    print(f"wrote {a.out}/matching_results.tsv and candidate_pairs.tsv - run utils/validate_submission.py before upload")


if __name__ == "__main__":
    main()
