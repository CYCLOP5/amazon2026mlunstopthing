'blocker "skeleton_tfidf": idf-weighted sparse cosine over skeleton tokens'
import os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from er.blocking import BLOCKERS
from er.safe import THREADS, guard


def long_tokens(df: pl.DataFrame, tok: dict) -> pl.DataFrame:
    '(rid, f, h): f 0=name, 1=address; h = 64-bit token hash'
    parts = []

    def uni_bi(col, f, seed_u, seed_b, min_len, use_uni, use_bi):
        t = df.select("rid", pl.col(col).str.split(" ").list.eval(
            pl.element().filter(pl.element().str.len_chars() >= min_len)).alias("t"))
        if use_uni:
            parts.append(t.explode("t").drop_nulls("t")
                          .select("rid", pl.lit(f, pl.UInt8).alias("f"), pl.col("t").hash(seed=seed_u).alias("h")))
        if use_bi:
            parts.append(t.with_columns(pl.col("t").list.slice(1).alias("t2"),
                                        pl.col("t").list.slice(0, pl.col("t").list.len() - 1))
                          .explode("t", "t2").drop_nulls("t2")
                          .select("rid", pl.lit(f, pl.UInt8).alias("f"),
                                  (pl.col("t") + "_" + pl.col("t2")).hash(seed=seed_b).alias("h")))

    uni_bi("name_skel", 0, 11, 23, 1, tok["name_unigrams"], tok["name_bigrams"])
    uni_bi("addr_skel", 1, 17, 19, 2, tok["address_unigrams"], tok["address_bigrams"])
    if tok["compact_name"]:
        parts.append(df.filter(pl.col("name_skel") != "")
                       .select("rid", pl.lit(0, pl.UInt8).alias("f"),
                               pl.col("name_skel").str.replace_all(" ", "").hash(seed=13).alias("h")))
    return pl.concat(parts).unique(["rid", "h"])


def _weights(tok: pl.DataFrame, idf: pl.DataFrame) -> pl.DataFrame:
    'tf=1, weight = idf / ||idf||_field  (each field is unit-normalised)'
    tok = tok.join(idf, on="h", how="inner")
    norm = tok.group_by("rid", "f").agg((pl.col("idf") ** 2).sum().sqrt().alias("nrm"))
    return (tok.join(norm, on=["rid", "f"])
               .with_columns((pl.col("idf") / pl.col("nrm")).cast(pl.Float32).alias("w")).drop("idf", "nrm"))


def _csr(w: pl.DataFrame, n_rows: int, n_cols: int) -> sp.csr_matrix:
    return sp.csr_matrix((w["w"].to_numpy(), (w["rid"].to_numpy(), w["col"].to_numpy())),
                         shape=(n_rows, n_cols), dtype=np.float32)


def rowdot(A: sp.csr_matrix, B: sp.csr_matrix, i: np.ndarray, j: np.ndarray, chunk=1_000_000) -> np.ndarray:
    'out[k] = <a[i[k]], b[j[k]]> for sparse row vectors'
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        a, b = A[i[s:s + chunk]], B[j[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _chunks(df: pl.DataFrame, rows: int):
    for s in range(0, df.height, rows):
        yield s, df.slice(s, rows).with_columns((pl.col("rid") - s).alias("rid"))


def block_country(s1: pl.DataFrame, oth: pl.DataFrame, prm: dict, verbose=True):
    "s1/oth have 'rid' (0..n-1). Yields chunk DataFrames (o_rid, s1_rid, bscore, brank, cos_name, cos_addr)"
    t0 = time.time()
    tok, rows = prm["tokens"], prm["chunk_rows"]
    t1 = long_tokens(s1, tok)
    d1 = t1.group_by("h").agg(pl.len().cast(pl.UInt32).alias("df1"))
    d2 = None
    for _, c in _chunks(oth, rows):
        g = long_tokens(c, tok).group_by("h").agg(pl.len().cast(pl.UInt32).alias("df2"))
        d2 = g if d2 is None else pl.concat([d2, g]).group_by("h").agg(pl.col("df2").sum())
        guard("blocking/df pass")
    n_all = s1.height + oth.height
    idf = (d1.join(d2, on="h", how="full", coalesce=True).fill_null(0)
           .with_columns((np.log((n_all + 1) / (pl.col("df1") + pl.col("df2") + 1).cast(pl.Float64)) + 1.0)
                         .cast(pl.Float32).alias("idf")))
    del d1, d2
    shared = (idf.filter((pl.col("df1") > 0) & (pl.col("df2") > 0))
                 .select("h", (pl.col("df1") <= prm["max_df1"]).alias("blk")).with_row_index("col"))
    V = shared.height
    cost = float(idf.filter((pl.col("df1") > 0) & (pl.col("df1") <= prm["max_df1"]))
                 .select((pl.col("df1").cast(pl.Float64) * pl.col("df2")).sum()).item())
    idf = idf.select("h", "idf")
    w1 = _weights(t1, idf).join(shared, on="h", how="inner")
    del t1
    BT = _csr(w1.filter(pl.col("blk")), s1.height, V).T.tocsr()
    Bn = _csr(w1.filter(pl.col("f") == 0), s1.height, V)
    Ba = _csr(w1.filter(pl.col("f") == 1), s1.height, V)
    del w1
    guard("blocking/s1 matrices")
    if verbose:
        print(f"   vocab={V:,} nnz(S1)={BT.nnz:,} cost={cost:.2e} prep={time.time()-t0:.0f}s", flush=True)
    for s, c in _chunks(oth, rows):
        w2 = _weights(long_tokens(c, tok), idf).join(shared, on="h", how="inner")
        A = _csr(w2.filter(pl.col("blk")), c.height, V)
        An = _csr(w2.filter(pl.col("f") == 0), c.height, V)
        Aa = _csr(w2.filter(pl.col("f") == 1), c.height, V)
        del w2
        C = sp_matmul_topn(A, BT, top_n=prm["retrieve_k"], threshold=0.0, sort=False, n_threads=THREADS).tocsr()
        o_rid = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr))
        s_rid = C.indices.astype(np.int64)
        del C, A
        d = pl.DataFrame({"o_rid": (o_rid + s).astype(np.uint32), "s1_rid": s_rid.astype(np.uint32),
                          "cos_name": rowdot(An, Bn, o_rid, s_rid), "cos_addr": rowdot(Aa, Ba, o_rid, s_rid)})
        del An, Aa, o_rid, s_rid
        d = (d.with_columns((pl.col("cos_name") + pl.col("cos_addr")).alias("bscore"))
              .with_columns(pl.col("bscore").rank("ordinal", descending=True).over("o_rid")
                            .sub(1).cast(pl.UInt8).alias("brank")))
        ka, kn = prm.get("addr_view_k", 0), prm.get("name_view_k", 0)
        if ka or kn:



            d = d.with_columns(
                ((pl.col("brank") >= prm["top_k"])
                 & ((pl.col("cos_addr").rank("ordinal", descending=True).over("o_rid") <= ka)
                    | (pl.col("cos_name").rank("ordinal", descending=True).over("o_rid") <= kn))).alias("view"))
            d = d.filter((pl.col("brank") < prm["top_k"]) | pl.col("view"))
            cols = ["o_rid", "s1_rid", "bscore", "brank", "cos_name", "cos_addr", "view"]
        else:
            d = d.filter(pl.col("brank") < prm["top_k"])
            cols = ["o_rid", "s1_rid", "bscore", "brank", "cos_name", "cos_addr"]
        d = d.select(cols)
        guard("blocking/score chunk")
        if verbose:
            print(f"   chunk {s // rows}: rows {s:,}-{s + c.height:,} pairs={d.height:,} t={time.time()-t0:.0f}s",
                  flush=True)
        yield d


@BLOCKERS.register("skeleton_tfidf")
def skeleton_tfidf(s1: pl.DataFrame, oth: pl.DataFrame, prm: dict, out_path):
    countries = sorted(set(s1["country"].unique().to_list()) | set(oth["country"].unique().to_list()))
    for country in countries:
        a = s1.filter(pl.col("country") == country).with_row_index("rid")
        b = oth.filter(pl.col("country") == country).with_row_index("rid")
        print(f" country={country!r}: s1={a.height:,} others={b.height:,}", flush=True)
        if a.height == 0 or b.height == 0:
            continue
        n_chunks = (b.height + prm["chunk_rows"] - 1) // prm["chunk_rows"]
        if all(os.path.exists(out_path(country, k)) for k in range(n_chunks)):
            print("   already done - skipping", flush=True)
            continue
        ida = a.select(pl.col("rid").alias("s1_rid"), pl.col("entity_id").alias("s1_id"))
        idb = b.select(pl.col("rid").alias("o_rid"), pl.col("entity_id").alias("o_id"))
        for k, d in enumerate(block_country(a, b, prm)):
            extra = ["view"] if "view" in d.columns else []
            (d.join(idb, on="o_rid").join(ida, on="s1_rid")
              .select("s1_id", "o_id", "bscore", "brank", "cos_name", "cos_addr", *extra)
              .write_parquet(out_path(country, k)))
