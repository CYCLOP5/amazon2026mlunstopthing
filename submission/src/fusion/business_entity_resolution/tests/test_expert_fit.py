'synthetic coverage and partition checks for expert-score calibration'
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from innovation.expert_fit import (EXPERT_FEATURES, _fit_columns, add_competition_features,
                                   blend_predictions, join_expert_scores, run)
from innovation.train import training_partitions
from test_edit_channel import fixture as make_prepared_fixture


def _fixture_pairs():
    return pl.DataFrame({
        "qid": pl.Series([10, 11, 12], dtype=pl.UInt32),
        "tid": pl.Series([1, 1, 2], dtype=pl.UInt32),
        "parent_p": [0.8, 0.2, 0.4],
        "y": [1, 0, 1],
        "own": pl.Series([10, 10, 12], dtype=pl.Int64),
        "fold": [0, 0, 0],
        "co": ["us", "us", "india"],
        "prob_lg": [1.3, -1.3, -0.4],
    })


def test_expert_join_is_keyed_and_derives_target_competition_features():
    ff = _fixture_pairs()

    scores = pl.DataFrame({"qid": [12, 11, 10], "tid": [2, 1, 1],
                           "expert_lg": [-3.0, 1.0, 2.0]})
    joined = add_competition_features(join_expert_scores(ff, scores))
    assert joined["qid"].to_list() == [10, 11, 12]
    assert joined["expert_rank"].to_list() == [1.0, 2.0, 1.0]
    assert joined["expert_margin"].to_list() == [1.0, -1.0, 11.0]
    assert np.isfinite(joined.select(*EXPERT_FEATURES).to_numpy()).all()


@pytest.mark.parametrize("scores, message", [
    (pl.DataFrame({"qid": [10, 11], "tid": [1, 1], "expert_lg": [1., 2.]}), "coverage mismatch"),
    (pl.DataFrame({"qid": [10, 11, 12, 99], "tid": [1, 1, 2, 9],
                   "expert_lg": [1., 2., 3., 4.]}), "coverage mismatch"),
    (pl.DataFrame({"qid": [10, 10, 11], "tid": [1, 1, 1],
                   "expert_lg": [1., 2., 3.]}), "duplicate qid/tid"),
    (pl.DataFrame({"qid": [10, 11, 12], "tid": [1, 1, 2],
                   "expert_lg": [1., float("nan"), 3.]}), "finite"),
])
def test_expert_join_rejects_corrupt_or_incomplete_score_coverage(scores, message):
    with pytest.raises(ValueError, match=message):
        join_expert_scores(_fixture_pairs(), scores)


def test_competition_features_do_not_depend_on_labels_and_head_columns_exclude_targets():
    frame = add_competition_features(join_expert_scores(
        _fixture_pairs(), pl.DataFrame({"qid": [10, 11, 12], "tid": [1, 1, 2],
                                        "expert_lg": [2.0, 1.0, -3.0]})))
    flipped = frame.with_columns((1 - pl.col("y")).alias("y"))
    assert frame.select("qid", "tid", *EXPERT_FEATURES).equals(
        flipped.select("qid", "tid", *EXPERT_FEATURES))
    control, expert = _fit_columns(frame)
    assert set(EXPERT_FEATURES).isdisjoint(control)
    assert set(EXPERT_FEATURES) <= set(expert)
    assert not {"qid", "tid", "y", "own", "fold", "co", "parent_p"} & set(expert)


def test_training_partitions_keep_only_fit_fold_candidate_and_true_owner_groups():
    refs = pl.DataFrame({"rid": np.arange(300, dtype=np.uint32),
                         "fold": [0] * 100 + [1] * 100 + [2] * 100})
    fit_ids = refs.filter((pl.col("fold") == 0) & ((pl.col("rid").hash(1033) % 5) != 0))["rid"].to_list()
    tune_ids = refs.filter((pl.col("fold") == 0) & ((pl.col("rid").hash(1033) % 5) == 0))["rid"].to_list()
    audit_id = refs.filter(pl.col("fold") == 1)["rid"][0]
    owner, rival = fit_ids[:2]
    frame = pl.DataFrame({"qid": [owner, rival, tune_ids[0], audit_id],
        "tid": [7, 7, 7, 7], "own": [owner] * 4, "fold": [0, 0, 0, 1]}
    ).with_columns(pl.col("qid", "tid").cast(pl.UInt32))
    fit, groups = training_partitions(frame, refs)
    assert fit.tolist() == [True, True, False, False]
    assert groups[0] == groups[1] == groups[2] == groups[3]


def test_logit_blend_retains_pair_keys_and_changes_only_probability():
    frame = _fixture_pairs().with_columns(pl.Series("control_head", [0.9, 0.1, 0.5]))
    update = blend_predictions(frame, "control", .5)
    assert update.select("qid", "tid").equals(frame.select("qid", "tid"))
    assert update["p"][0] == pytest.approx(6. / 7.)
    assert update["p"][1] == pytest.approx(1. / 7.)
    odds = np.sqrt(2. / 3.)
    assert update["p"][2] == pytest.approx(odds / (1. + odds))


def test_synthetic_pipeline_fit_exports_full_parent_pool_when_parent_is_perfect(tmp_path):
    data, parent, prepared, _ = make_prepared_fixture(tmp_path, n=240)



    refs_path = data / "train/ref.parquet"
    refs = pl.read_parquet(refs_path)
    row = pl.int_range(0, pl.len())
    refs = refs.with_columns(
        pl.when(row < 80).then(0).when(row < 160).then(1).otherwise(2).alias("fold"))
    refs.write_parquet(refs_path)
    from innovation.prepare import run as prepare_pairs
    prepare_pairs(data, parent, prepared, max_targets=10_000)
    train_features = pl.read_parquet(prepared / "train/features.parquet")
    fit, groups = training_partitions(train_features, refs)
    assert set(np.unique(train_features["y"].to_numpy()[fit])) == {0, 1}
    assert np.unique(groups[fit]).size >= 3
    expert = tmp_path / "expert"
    (expert / "train").mkdir(parents=True)
    (expert / "test").mkdir(parents=True)
    for split in ("train", "test"):
        ff = pl.read_parquet(prepared / split / "features.parquet")
        p = ff["parent_p"].to_numpy().astype(float)
        logits = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))
        scores = ff.select("qid", "tid").with_columns(pl.Series("expert_lg", logits.astype(np.float32)))
        middle = scores.height // 2
        scores.slice(0, middle).write_parquet(expert / split / "part_00000.parquet")
        scores.slice(middle).write_parquet(expert / split / "part_00001.parquet")
    (expert / "expert_report.json").write_text('{"synthetic": true, "pairs": "exact"}')
    (expert / "_SUCCESS").write_text("complete\n")

    output = tmp_path / "expert_fit"
    report = run(data, parent, prepared, expert, output, rounds=8)

    assert report["selected"] == "parent"
    assert report["promotion_policy"]["selection_uses_audit"] is False
    assert not any(report["tuning"][key]["eligible"] for key in report["tuning"]
                   if key.startswith("control_"))
    assert not any(report["tuning"][key]["eligible"] for key in report["tuning"]
                   if key.startswith("expert_"))
    assert (output / "bundle/control_0.txt").exists()
    assert (output / "bundle/expert_0.txt").exists()

    from innovation.common import read_parent
    baseline_test, _ = read_parent(parent, "test")
    final_test = pl.read_parquet(output / "test_predictions.parquet")
    assert final_test.height == baseline_test.height
    assert final_test.select("qid", "tid").n_unique() == baseline_test.height
    assert final_test.select("qid", "tid", "p").sort("qid", "tid").equals(
        baseline_test.select("qid", "tid", "p").sort("qid", "tid"))
    assert report["export"]["candidate_pairs"] == baseline_test.height

    import runpy
    validate = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  "scripts/validate_outputs.py"))["validate"]
    assert validate(data, output / "output")["source1"] == 32


def test_synthetic_pipeline_promotes_expert_and_exports_full_parent_pool(tmp_path):
    data, parent, prepared, _ = make_prepared_fixture(tmp_path, n=600)
    refs_path = data / "train/ref.parquet"
    refs = pl.read_parquet(refs_path)
    row = pl.int_range(0, pl.len())
    refs = refs.with_columns(
        pl.when(row < 200).then(0).when(row < 400).then(1).otherwise(2).alias("fold"))
    refs.write_parquet(refs_path)
    from innovation.prepare import run as prepare_pairs
    prepare_pairs(data, parent, prepared, max_targets=10_000)



    from innovation.common import read_parent
    from innovation.expert_fit import _fit_columns
    raw_train = pl.read_parquet(parent / "validation_predictions.parquet")
    raw_test = pl.read_parquet(parent / "test_predictions.parquet")
    poor = pl.when(pl.col("y") == 1).then(.1).otherwise(.9).cast(pl.Float32)
    raw_train.with_columns(poor.alias("base"), poor.alias("head")).write_parquet(
        parent / "validation_predictions.parquet")
    raw_test.with_columns(pl.lit(.5).cast(pl.Float32).alias("base"),
                          pl.lit(.5).cast(pl.Float32).alias("head")).write_parquet(
        parent / "test_predictions.parquet")
    for split in ("train", "test"):
        path = prepared / split / "features.parquet"
        frame = pl.read_parquet(path).with_columns(pl.lit(.5).cast(pl.Float32).alias("parent_p"))
        generic, _ = _fit_columns(frame)
        frame = frame.with_columns([pl.lit(0.).cast(pl.Float32).alias(c) for c in generic])
        frame.write_parquet(path)

    expert = tmp_path / "expert"
    (expert / "train").mkdir(parents=True)
    (expert / "test").mkdir(parents=True)
    for split in ("train", "test"):
        ff = pl.read_parquet(prepared / split / "features.parquet")
        if split == "train":
            logits = np.where(ff["y"].to_numpy() == 1, 8., -8.)
        else:
            logits = np.where((ff["qid"].to_numpy() + ff["tid"].to_numpy()) % 2, 8., -8.)
        scores = ff.select("qid", "tid").with_columns(
            pl.Series("expert_lg", logits.astype(np.float32)))
        middle = scores.height // 2
        scores.slice(0, middle).write_parquet(expert / split / "part_00000.parquet")
        scores.slice(middle).write_parquet(expert / split / "part_00001.parquet")
    (expert / "expert_report.json").write_text('{"synthetic": true, "labels_used_for_score": "train_fixture_only"}')
    (expert / "_SUCCESS").write_text("complete\n")

    output = tmp_path / "expert_fit"
    report = run(data, parent, prepared, expert, output, rounds=30)
    assert report["selected"].startswith("expert_")
    assert report["tuning"][report["selected"]]["promotion_gain_vs_control_ceiling"] >= .0001
    assert report["tuning"][report["selected"]]["eligible"]
    baseline_test, _ = read_parent(parent, "test")
    final_test = pl.read_parquet(output / "test_predictions.parquet")
    assert final_test.height == baseline_test.height
    assert final_test.select("qid", "tid").n_unique() == baseline_test.height
    assert not final_test.sort("qid", "tid")["p"].equals(
        baseline_test.sort("qid", "tid")["p"])
    assert report["export"]["candidate_pairs"] == baseline_test.height

    import runpy
    validate = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  "scripts/validate_outputs.py"))["validate"]
    assert validate(data, output / "output")["source1"] == 32
