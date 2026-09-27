import argparse as ap
import hashlib as hh
import heapq as hq
import json
import re
import string as st
import unicodedata as ud
from collections import Counter as ct, defaultdict as dd
from pathlib import Path as path

from eda import f05, pct, rows


tab = str.maketrans({c: " " for c in st.punctuation})


def norm(s):
    return " ".join(ud.normalize("NFKC", s).casefold().translate(tab).split())


def jac(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def tri(s):
    return {s[i:i + 3] for i in range(max(0, len(s) - 2))}


def sig(*xs):
    return hh.blake2b("\t".join(xs).encode(), digest_size=16).digest()


def s1(data, n):
    hs = dd(list)
    df = dd(ct)
    tr = set()
    out = {}
    for sp in ("train", "test"):
        cs = dd(ct)
        ns, ads, rs = ct(), ct(), ct()
        it = rows(data / sp / f"{sp}_source1.tsv")
        next(it)
        for eid, nm, ad, co in it:
            nm, ad = norm(nm), norm(ad)
            k = sig(co, nm, ad)
            cs[co]["n"] += 1
            rs[k] += 1
            ns[(co, nm)] += 1
            ads[(co, ad)] += 1
            if sp == "test":
                cs[co]["train_record_overlap"] += k in tr
                continue
            tr.add(k)
            for fld, s in (("nm", nm), ("ad", ad)):
                df[(co, fld)].update(set(s.split()))
            z = (-int.from_bytes(sig(eid), "big"), eid, nm, ad, co)
            if len(hs[co]) < n:
                hq.heappush(hs[co], z)
            elif z > hs[co][0]:
                hq.heapreplace(hs[co], z)
        out[sp] = {"country": cs,
                   "dup_record_extra": sum(v - 1 for v in rs.values()),
                   "dup_record_groups": sum(v > 1 for v in rs.values()),
                   "dup_name_extra": sum(v - 1 for (co, s), v in ns.items() if s),
                   "dup_addr_extra": sum(v - 1 for (co, s), v in ads.items() if s),
                   "top_names": [[co, s, v] for (co, s), v in ns.most_common(12)],
                   "top_addrs": [[co, s, v] for (co, s), v in ads.most_common(12)]}
        print("anchors", sp, dict(cs), flush=True)
    qs = {eid: (nm, ad, co) for h in hs.values() for _, eid, nm, ad, co in h}
    return qs, df, out


def push(h, v, k):
    if len(h) < k:
        hq.heappush(h, v)
    elif v > h[0]:
        hq.heapreplace(h, v)


def evaluate(qs, gt, ps):
    out = {}
    for co in sorted({v[2] for v in qs.values()}):
        es = [e for e, r in qs.items() if r[2] == co]
        tp = sum(len(gt[e] & ps[e]) for e in es)
        pn = sum(len(ps[e]) for e in es)
        tn = sum(len(gt[e]) for e in es)
        out[co] = {"n": len(es), "pred_links": pn, "true_links": tn, "tp": tp,
                   "macro_f05": sum(f05(gt[e], ps[e]) for e in es) / len(es),
                   "link_precision": tp / pn if pn else None,
                   "link_recall": tp / tn if tn else None,
                   "singleton_n": sum(not gt[e] for e in es),
                   "singleton_fp": sum(not gt[e] and bool(ps[e]) for e in es)}
    return out


def main():
    pa = ap.ArgumentParser()
    root = path(__file__).resolve().parents[1]
    pa.add_argument("--data", type=path, default=root / "student_resource/dataset")
    pa.add_argument("--out", type=path, default=root / "reports/probe.json")
    pa.add_argument("--n", type=int, default=5000)
    pa.add_argument("--q", type=int, default=250)
    pa.add_argument("--check", action="store_true")
    a = pa.parse_args()
    if a.check:
        assert norm("  ACME, Inc. ") == "acme inc"
        assert norm("राम") == "राम"
        assert norm("école") == norm("e\u0301cole")
        assert jac(set(), set()) == 0
        assert jac({"a", "b"}, {"b", "c"}) == 1 / 3
        h = []
        for v in (2, 4, 1, 3):
            push(h, v, 2)
        assert sorted(h) == [3, 4]
        print("checks passed")
        return
    assert a.n > 0 and 0 < a.q <= a.n
    qs, df, du = s1(a.data, a.n)
    gt = {}
    own = {}
    it = rows(a.data / "train/train_ground_truth.tsv")
    next(it)
    for eid, s in it:
        if eid in qs:
            gt[eid] = set(s.split(",")) if s else set()
            for m in gt[eid]:
                assert m not in own
                own[m] = eid
    assert qs.keys() == gt.keys()
    ix = dd(set)
    rs = {}
    for co in sorted({r[2] for r in qs.values()}):
        es = sorted((e for e, r in qs.items() if r[2] == co), key=sig)[:a.q]
        for e in es:
            nm, ad, _ = qs[e]
            rs[e] = (set(nm.split()), set(ad.split()), co)
            for fld, s in (("nm", nm), ("ad", ad)):
                ts = [t for t in set(s.split()) if len(t) >= 3 and any(c.isalpha() for c in t)]
                for t in sorted(ts, key=lambda t: (df[(co, fld)][t], t))[:3]:
                    if df[(co, fld)][t] <= 500:
                        ix[(co, fld, t)].add(e)
    del df
    eq = {fld: dd(set) for fld in ("nm", "ad")}
    ps = {fld: dd(set) for fld in ("nm", "ad", "both")}
    for e, (nm, ad, co) in qs.items():
        if nm:
            eq["nm"][(co, nm)].add(e)
        if ad:
            eq["ad"][(co, ad)].add(e)
    hp = dd(list)
    raw = dd(set)
    cnt = ct()
    pos = dd(ct)
    hist = dd(ct)
    ex = dd(list)
    de = re.compile(r"[\u0900-\u097f]")
    dig = re.compile(r"\d+")
    got = set()
    for j in (2, 3):
        it = rows(a.data / "train" / f"train_source{j}.tsv")
        next(it)
        for eid, nm, ad, co in it:
            cnt[f"s{j}_rows"] += 1
            nm, ad = norm(nm), norm(ad)
            ns, ads = set(nm.split()), set(ad.split())
            ne, ae = eq["nm"].get((co, nm), set()), eq["ad"].get((co, ad), set())
            for fld, es in (("nm", ne), ("ad", ae), ("both", ne & ae)):
                for e in es:
                    ps[fld][e].add(eid)
            hits = set()
            for fld, ts in (("nm", ns), ("ad", ads)):
                for t in ts:
                    hits.update(ix.get((co, fld, t), ()))
            cnt["rare_pairs"] += len(hits)
            for e in hits:
                qn, qa, _ = rs[e]
                sn, sa = jac(qn, ns), jac(qa, ads)
                for fld, s in (("nm", sn), ("ad", sa), ("mix", 0.65 * sn + 0.35 * sa)):
                    push(hp[(e, j, fld)], (s, eid), 50)
                if eid in gt[e]:
                    raw[e].add(eid)
            if eid not in own:
                continue
            got.add(eid)
            e = own[eid]
            qn, qa, qc = qs[e]
            sn, sa = jac(set(qn.split()), ns), jac(set(qa.split()), ads)
            k = f"{qc.lower()}_s{j}"
            z = pos[k]
            z["n"] += 1
            z["country_diff"] += qc != co
            z["nm_eq"] += qn == nm
            z["ad_eq"] += qa == ad and bool(qa)
            z["both_eq"] += qn == nm and qa == ad and bool(qa)
            z["nm_zero_token_overlap"] += sn == 0
            z["ad_zero_token_overlap"] += sa == 0
            z["both_zero_token_overlap"] += sn == 0 and sa == 0
            z["both_weak_token_overlap"] += sn < 0.1 and sa < 0.1
            z["ad_blank_either"] += not qa or not ad
            z["nm_deva_switch"] += bool(de.search(qn)) != bool(de.search(nm))
            qd, rd = set(dig.findall(qa)), set(dig.findall(ad))
            z["ad_digits_both"] += bool(qd and rd)
            z["ad_digits_disjoint"] += bool(qd and rd) and not bool(qd & rd)
            for fld, s in (("nm_jac", sn), ("ad_jac", sa),
                           ("nm_tri", jac(tri(qn), tri(nm))), ("ad_tri", jac(tri(qa), tri(ad)))):
                hist[k + "_" + fld][round(1000 * s)] += 1
            if sn < 0.1 and sa < 0.1 and len(ex[k]) < 8:
                ex[k].append({"q": e, "m": eid, "qn": qn, "qa": qa, "nm": nm, "ad": ad})
        print("targets", j, dict(cnt), flush=True)
    assert got == own.keys(), (len(got), len(own))
    out = {"sample_per_country": a.n, "probe_per_country": a.q,
           "normalization": "nfkc casefold ascii punctuation to space whitespace collapse",
           "s1_audit": du, "counts": cnt, "positive_pairs": pos,
           "positive_quantiles": {k: {str(q): pct(c, q) / 1000 for q in (0.01, 0.1, 0.5, 0.9, 0.99)}
                                  for k, c in hist.items()},
           "weak_examples": ex,
           "exact_baselines": {fld: evaluate(qs, gt, p) for fld, p in ps.items()},
           "rare_probe": {"raw_oracle": evaluate(rs, gt, raw), "topk": {}}}
    for k in (5, 10, 20, 50):
        pred = dd(set)
        for e in rs:
            for j in (2, 3):
                for fld in ("nm", "ad", "mix"):
                    pred[e].update(m for _, m in sorted(hp[(e, j, fld)], reverse=True)[:k])
        oracle = {e: pred[e] & gt[e] for e in rs}
        out["rare_probe"]["topk"][str(k)] = {
            "candidates": sum(map(len, pred.values())),
            "candidate_quantiles": {str(q): pct(ct(map(len, pred.values())), q)
                                    for q in (0.5, 0.9, 0.99, 1.0)},
            "oracle": evaluate(rs, gt, oracle)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
