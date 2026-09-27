import argparse as ap
import csv
import hashlib as hh
import json
import re
from collections import Counter as ct
from pathlib import Path as path


def rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.reader(f, delimiter="\t")


def pct(c, q):
    n = sum(c.values())
    if not n:
        return None
    t = max(1, n * q)
    k = 0
    for x, v in sorted(c.items()):
        k += v
        if k >= t:
            return x


def f05(t, p):
    if not t:
        return float(not p)
    return 1.25 * len(t & p) / (0.25 * len(t) + len(p))


def scan(p):
    it = rows(p)
    hdr = next(it)
    assert hdr == ["entity_id", "business_name", "business_address", "country"], hdr
    ids = set()
    cs = {}
    err = ct()
    px = "S" + p.stem[-1] + "-"
    de = re.compile(r"[\u0900-\u097f]")
    ot = re.compile(r"[\u0980-\u0dff]")
    url = re.compile(r"www\.|https?://|\.(?:com|in|org|net|fr)\b", re.I)
    na = re.compile(r"\b(?:null|nan|none|unknown)\b|n/a", re.I)
    dig = re.compile(r"\d")
    for i, r in enumerate(it, 2):
        assert len(r) == 4, (p.name, i, len(r))
        eid, nm, ad, co = r
        err["dup_id"] += eid in ids
        err["bad_id"] += not eid.startswith(px)
        err["id_space"] += eid != eid.strip()
        ids.add(eid)
        if co not in cs:
            cs[co] = {"n": 0, "nm": ct(), "ad": ct(), "nm_len": ct(), "ad_len": ct()}
        s = cs[co]
        s["n"] += 1
        for k, x in (("nm", nm), ("ad", ad)):
            z = s[k]
            z["blank"] += not x.strip()
            z["non_ascii"] += not x.isascii()
            z["deva"] += bool(de.search(x))
            z["other_indic"] += bool(ot.search(x))
            z["url"] += bool(url.search(x))
            z["na_token"] += bool(na.search(x))
            z["digit"] += bool(dig.search(x))
            z["space_edge"] += x != x.strip()
            s[k + "_len"][len(x)] += 1
        s["both_blank"] = s.get("both_blank", 0) + (not nm.strip() and not ad.strip())
    for s in cs.values():
        for k in ("nm_len", "ad_len"):
            c = s[k]
            s[k] = {str(q): pct(c, q) for q in (0.5, 0.9, 0.95, 0.99, 1.0)}
            s[k]["mean"] = sum(x * n for x, n in c.items()) / s["n"]
    with p.open("rb") as f:
        sha = hh.file_digest(f, "sha256").hexdigest()
    out = {"n": sum(s["n"] for s in cs.values()), "bytes": p.stat().st_size,
           "sha256": sha, "err": err, "country": cs}
    print(p.name, out["n"], {k: v["n"] for k, v in cs.items()}, flush=True)
    return out, ids


def truth(p, ids):
    it = rows(p)
    hdr = next(it)
    assert hdr == ["source1_entity_id", "matched_entity_ids"], hdr
    seen = set()
    tg = {"S2": set(), "S3": set()}
    err = ct()
    hist = ct()
    src = ct()
    for i, r in enumerate(it, 2):
        assert len(r) == 2, (p.name, i, len(r))
        eid, v = r
        ms = v.split(",") if v else []
        err["dup_anchor"] += eid in seen
        err["missing_anchor"] += eid not in ids["train_source1"]
        err["dup_in_list"] += len(ms) - len(set(ms))
        seen.add(eid)
        hist[len(ms)] += 1
        sc = ct(m[:2] for m in ms)
        src[f"{sc['S2']},{sc['S3']}"] += 1
        for m in ms:
            px = m[:2]
            if px not in tg:
                err["bad_target_prefix"] += 1
                continue
            err["reused_target"] += m in tg[px]
            err["missing_target"] += m not in ids["train_source" + px[-1]]
            tg[px].add(m)
    err["unlabeled_anchor"] = len(ids["train_source1"] - seen)
    n = sum(hist.values())
    return {"n": n, "err": err, "match_hist": hist, "source_hist": src,
            "links": sum(k * v for k, v in hist.items()),
            "empty_baseline": hist[0] / n,
            "targets": {px.lower(): {"linked": len(v),
                        "unlinked": len(ids["train_source" + px[-1]] - v)} for px, v in tg.items()}}


def check():
    assert f05(set(), set()) == 1
    assert f05(set(), {"x"}) == 0
    assert f05({"x"}, set()) == 0
    assert f05({"x"}, {"x"}) == 1
    assert abs(f05({"x", "y"}, {"x", "y", "z"}) - 5 / 7) < 1e-12
    assert pct(ct({0: 2, 5: 2}), 0.5) == 0
    assert pct(ct({0: 2, 5: 2}), 0.99) == 5
    assert pct(ct(), 0.5) is None
    print("checks passed")


def labels(data):
    it = rows(data / "train/train_source1.tsv")
    next(it)
    cs = {r[0]: r[3] for r in it}
    own = {}
    hist = {}
    src = {}
    p = data / "train/train_ground_truth.tsv"
    it = rows(p)
    next(it)
    for e, s in it:
        co = cs[e]
        ms = s.split(",") if s else []
        hist.setdefault(co, ct())[len(ms)] += 1
        z = ct(m[:2] for m in ms)
        src.setdefault(co, ct())[f"{z['S2']},{z['S3']}"] += 1
        for m in ms:
            assert m not in own
            own[m] = co
    del cs
    out = {"country": {co: {"n": sum(h.values()), "match_hist": h,
                            "source_hist": src[co], "empty_baseline": h[0] / sum(h.values())}
                       for co, h in hist.items()}, "targets": {}}
    for j in (2, 3):
        ds = {}
        it = rows(data / "train" / f"train_source{j}.tsv")
        next(it)
        for r in it:
            eid, co = r[0], r[3]
            z = ds.setdefault(co, ct())
            z["n"] += 1
            qc = own.get(eid)
            z["linked"] += qc is not None
            z["unlinked"] += qc is None
            z["cross_country_link"] += qc is not None and qc != co
            k = "linked" if qc is not None else "unlinked"
            z[k + "_nm_non_ascii"] += not r[1].isascii()
            z[k + "_ad_blank"] += not r[2].strip()
        out["targets"][f"s{j}"] = ds
        print("label countries", j, ds, flush=True)
    with p.open("rb") as f:
        out["gt_sha256"] = hh.file_digest(f, "sha256").hexdigest()
    return out


def main():
    pa = ap.ArgumentParser()
    root = path(__file__).resolve().parents[1]
    pa.add_argument("--data", type=path, default=root / "student_resource/dataset")
    pa.add_argument("--out", type=path)
    pa.add_argument("--check", action="store_true")
    pa.add_argument("--labels", action="store_true")
    a = pa.parse_args()
    a.out = a.out or root / "reports" / ("labels.json" if a.labels else "eda.json")
    if a.check:
        check()
        return
    if a.labels:
        out = labels(a.data)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    out = {"files": {}}
    ids = {}
    for sp in ("train", "test"):
        for j in (1, 2, 3):
            p = a.data / sp / f"{sp}_source{j}.tsv"
            out["files"][p.stem], ids[p.stem] = scan(p)
    out["truth"] = truth(a.data / "train/train_ground_truth.tsv", ids)
    out["id_overlap"] = {f"s{j}": len(ids[f"train_source{j}"] & ids[f"test_source{j}"])
                         for j in (1, 2, 3)}
    out["all_pairs"] = {sp: len(ids[f"{sp}_source1"]) *
                        (len(ids[f"{sp}_source2"]) + len(ids[f"{sp}_source3"]))
                        for sp in ("train", "test")}
    out["country_pairs"] = {}
    for sp in ("train", "test"):
        fs = [out["files"][f"{sp}_source{j}"]["country"] for j in (1, 2, 3)]
        out["country_pairs"][sp] = sum(v["n"] * sum(f.get(co, {}).get("n", 0) for f in fs[1:])
                                              for co, v in fs[0].items())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out["truth"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
