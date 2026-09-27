'cpu-only checks for final retrieval fusion and vector cache contracts'
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_hybrid.gpu import (EMBED_DIM, embedding_cache_meta,
    exact_topk_numpy, fuse_candidates, load_embedding_cache, save_embedding_cache,
    _model_scores)


def test_small_exact_search_is_stable_and_respects_reference_mask():
    q = np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
    r = np.asarray([[1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    out = exact_topk_numpy(q, r, np.asarray([9, 3, 5]), k=2,
                           query_ids=np.asarray([101, 102]),
                           valid_ref=np.asarray([True, False, True]))
    first = out.filter(pl.col("tid") == 101)
    assert first["qid"].to_list() == [9, 5]
    assert first["rank"].to_list() == [1, 2]
    assert out.filter(pl.col("tid") == 102)["qid"].to_list() == [5, 9]
    assert out["score"].to_list() == pytest.approx([1., 0., 1., 0.])


def test_fusion_keeps_every_prepared_pair_and_adds_rrf_top_new_pairs():
    old = pl.DataFrame({"qid": [1, 2], "tid": [10, 10]})
    lexical = pl.DataFrame({"qid": [2, 3, 4], "tid": [10, 10, 10],
                            "lex_name_score": [.9, .8, .7], "lex_address_score": [0., .7, .9],
                            "lex_name_rank": [1, 2, 3], "lex_address_rank": [0, 2, 1]})
    dn = pl.DataFrame({"qid": [3, 5], "tid": [10, 10],
                       "dense_name": [-.1, -.2], "dense_name_rank": [1, 2]})
    da = pl.DataFrame({"qid": [4, 5], "tid": [10, 10],
                       "dense_address": [.9, .8], "dense_address_rank": [1, 2]})
    res = fuse_candidates(old, lexical, dn, da, lane_topk=8, max_new=2)
    assert set(res["qid"].to_list()) == {1, 2, 3, 4}
    assert res.filter(pl.col("from_prepared") == 1)["qid"].to_list() == [1, 2]
    assert res.select("qid", "tid").n_unique() == res.height
    assert res["rrf"].is_finite().all()
    assert res.filter(pl.col("qid") == 3)["hit_dense_name"].item() == 1
    assert res.filter(pl.col("qid") == 4)["hit_dense_address"].item() == 1
    assert res.group_by("tid").len()["len"].max() == 4


def test_fusion_rejects_duplicate_pairs_and_invalid_scores():
    old = pl.DataFrame({"qid": [1], "tid": [10]})
    duplicate = pl.DataFrame({"qid": [2, 2], "tid": [10, 10],
        "lex_name_score": [.5, .4], "lex_address_score": [0., 0.]})
    empty = pl.DataFrame(schema={"qid": pl.Int64, "tid": pl.Int64,
        "dense_name": pl.Float32, "dense_name_rank": pl.UInt16})
    with pytest.raises(ValueError, match="duplicate"):
        fuse_candidates(old, duplicate, empty,
            pl.DataFrame(schema={"qid": pl.Int64, "tid": pl.Int64,
                "dense_address": pl.Float32, "dense_address_rank": pl.UInt16}))
    bad = pl.DataFrame({"qid": [2], "tid": [10], "lex_name_score": [-.1], "lex_address_score": [0.]})
    with pytest.raises(ValueError, match="nonnegative"):
        fuse_candidates(old, bad, empty,
            pl.DataFrame(schema={"qid": pl.Int64, "tid": pl.Int64,
                "dense_address": pl.Float32, "dense_address_rank": pl.UInt16}))


def test_embedding_cache_roundtrip_and_stale_input_rejection(tmp_path):
    refs = pl.DataFrame({"rid": [1, 2], "nm": ["one", "two"], "ad": ["1 road", "2 road"],
                         "co": ["us", "us"]})
    queries = pl.DataFrame({"rid": [10], "nm": ["one"], "ad": ["1 road"], "co": ["us"]})
    rng = np.random.default_rng(4)
    rv = rng.normal(size=(2, EMBED_DIM)).astype(np.float32)
    qv = rng.normal(size=(1, EMBED_DIM)).astype(np.float32)
    rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    meta = embedding_cache_meta(refs, queries, "name")
    save_embedding_cache(tmp_path, refs["rid"].to_numpy(), rv.astype(np.float16),
                         queries["rid"].to_numpy(), qv.astype(np.float16), meta)
    loaded = load_embedding_cache(tmp_path, meta)
    assert loaded["refs_ids"].tolist() == [1, 2]
    assert loaded["queries_ids"].tolist() == [10]
    assert loaded["refs"].dtype == np.float16
    changed = queries.with_columns(pl.lit("changed").alias("nm"))
    with pytest.raises(ValueError, match="Stale"):
        load_embedding_cache(tmp_path, embedding_cache_meta(refs, changed, "name"))


def test_neural_dispatch_supports_hf_head_and_innovation_matcher_without_torch():
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
        def float(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.value
        def __getitem__(self, key):
            return Tensor(self.value[key])

    class Head:
        def __call__(self, **tokens):
            assert tokens == {"input_ids": [1, 2]}
            return type("Result", (), {"logits": Tensor([[0.25], [-0.75]])})()

    class Matcher:
        def __call__(self, tokens):
            assert tokens == {"input_ids": [1, 2]}
            return Tensor([0.25, -0.75])

    tokens = {"input_ids": [1, 2]}
    assert _model_scores(Head(), tokens, "base").tolist() == [0.25, -0.75]
    assert _model_scores(Matcher(), tokens, "expert").tolist() == [0.25, -0.75]
