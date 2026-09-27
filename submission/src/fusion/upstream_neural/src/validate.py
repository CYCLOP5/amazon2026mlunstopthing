#!/usr/bin/env python3
'strict submission checks and exact macro f0.5 scoring'

import argparse
import csv
import sys
import tempfile
from pathlib import Path

try:
    from eda import f05, rows
except ModuleNotFoundError:
    from src.eda import f05, rows


mh = ["source1_entity_id", "matched_entity_ids"]
ch = ["source1_entity_id", "candidate_entity_ids"]
sh = ["entity_id", "business_name", "business_address", "country"]


class ValidationError(ValueError):
    def __init__(self, n, ex):
        self.n = n
        self.ex = ex
        super().__init__(self.text())

    def text(self):
        out = []
        for k, v in sorted(self.n.items()):
            x = ", ".join(self.ex.get(k, ()))
            out.append(f"{k}: {v}" + (f" ({x})" if x else ""))
        return "\n".join(out)


class Errors:
    def __init__(self):
        self.n = {}
        self.ex = {}

    def add(self, k, x=None, n=1):
        self.n[k] = self.n.get(k, 0) + n
        if x is not None and len(self.ex.setdefault(k, [])) < 5:
            self.ex[k].append(str(x))

    def fail(self):
        if self.n:
            raise ValidationError(self.n, self.ex)


def ids(p, hd, pre, er):
    out = set()
    try:
        it = rows(p)
        got = next(it, None)
        if got != hd:
            er.add("invalid source header", p)
            return out
        for i, r in enumerate(it, 2):
            if len(r) != len(hd):
                er.add("malformed source row", f"{p.name}:{i}")
                continue
            x = r[0]
            if not x.startswith(pre):
                er.add("invalid source id", x)
            if x in out:
                er.add("duplicate source id", x)
            out.add(x)
    except (OSError, UnicodeDecodeError, csv.Error) as x:
        er.add("unreadable source file", f"{p}: {x}")
    return out


def split(s, er, a):
    if not s:
        return set()
    xs = s.split(",")
    if len(xs) != len(set(xs)):
        er.add("duplicate id within list", a)
    out = set(xs)
    for x in out:
        if not x.startswith(("S2-", "S3-")):
            er.add("invalid target id prefix", x)
    return out


def output(p, hd, req, tar, er, keep=False, fin=None):
    seen = set()
    out = {} if keep else None
    st = {"rows": 0, "empty": 0, "ids": 0, "sizes": {}}
    try:
        it = rows(p)
        got = next(it, None)
        if got != hd:
            er.add("invalid output header", p)
            return out, st
        for i, r in enumerate(it, 2):
            if len(r) != 2:
                er.add("malformed output row", f"{p.name}:{i}")
                continue
            a, s = r
            st["rows"] += 1
            if a in seen:
                er.add("duplicate source1 row", a)
                continue
            seen.add(a)
            v = split(s, er, a)
            st["ids"] += len(v)
            st["sizes"][len(v)] = st["sizes"].get(len(v), 0) + 1
            if not v:
                st["empty"] += 1
            for x in v:
                if tar is not None and x not in tar:
                    er.add("unknown target id", x)
            if keep:
                out[a] = v
            if fin is not None and a in fin:
                fin[a].difference_update(v)
    except (OSError, UnicodeDecodeError, csv.Error) as x:
        er.add("unreadable output file", f"{p}: {x}")
        return out, st
    for a in req - seen:
        er.add("missing source1 coverage", a)
    for a in seen - req:
        er.add("unknown source1 id", a)
    return out, st


def validate(matching, candidate, test_dir):
    'validate strict test outputs and return aggregate counts'
    er = Errors()
    td = Path(test_dir)
    req = ids(td / "test_source1.tsv", sh, "S1-", er)
    s2 = ids(td / "test_source2.tsv", sh, "S2-", er)
    s3 = ids(td / "test_source3.tsv", sh, "S3-", er)
    tar = s2 | s3
    fin, ms = output(Path(matching), mh, req, tar, er, keep=True)
    if not candidate:
        er.add("missing candidate file")
        cs = {"rows": 0, "empty": 0, "ids": 0}
    else:
        cp = Path(candidate)
        if not cp.is_file():
            er.add("missing candidate file", cp)
            cs = {"rows": 0, "empty": 0, "ids": 0}
        else:
            _, cs = output(cp, ch, req, tar, er, fin=fin)
    if fin is not None:
        for a, v in fin.items():
            if v:
                er.add("final match outside candidate set", a, len(v))
    er.fail()
    hist = cs["sizes"]
    quantiles = {}
    for pct in (50, 95, 99):
        rank = max(1, (len(req) * pct + 99) // 100)
        total = 0
        quantiles[f"p{pct}"] = 0
        for size, count in sorted(hist.items()):
            total += count
            if total >= rank:
                quantiles[f"p{pct}"] = size
                break
    return {
        "source1": len(req),
        "targets": len(tar),
        "matching_rows": ms["rows"],
        "matching_empty": ms["empty"],
        "matching_ids": ms["ids"],
        "candidate_rows": cs["rows"],
        "candidate_empty": cs["empty"],
        "candidate_ids": cs["ids"],
        "candidate_sizes": {
            "mean": cs["ids"] / len(req) if req else 0.0,
            **quantiles,
            "max": max(hist, default=0),
            "quantile_method": "nearest_rank",
            "histogram": hist,
        },
    }


def labels(p, hd, er):
    out = {}
    try:
        it = rows(p)
        got = next(it, None)
        if got != hd:
            er.add("invalid label header", p)
            return out
        for i, r in enumerate(it, 2):
            if len(r) != 2:
                er.add("malformed label row", f"{p.name}:{i}")
                continue
            a, s = r
            if a in out:
                er.add("duplicate source1 row", a)
                continue
            out[a] = split(s, er, a)
    except (OSError, UnicodeDecodeError, csv.Error) as x:
        er.add("unreadable label file", f"{p}: {x}")
    return out


def score(truth, matching):
    'score predictions against training labels with exact macro f0.5'
    er = Errors()
    tr = labels(Path(truth), mh, er)
    pr = labels(Path(matching), mh, er)
    for a in tr.keys() - pr.keys():
        er.add("missing source1 coverage", a)
    for a in pr.keys() - tr.keys():
        er.add("unknown source1 id", a)
    er.fail()
    ss = [f05(t, pr[a]) for a, t in tr.items()]
    tp = sum(len(t & pr[a]) for a, t in tr.items())
    pn = sum(len(pr[a]) for a in tr)
    tn = sum(len(t) for t in tr.values())
    return {
        "anchors": len(tr),
        "singletons": sum(not t for t in tr.values()),
        "true_ids": tn,
        "predicted_ids": pn,
        "true_positive_ids": tp,
        "macro_f05": sum(ss) / len(ss) if ss else 0.0,
    }


def write(p, s):
    p.write_text(s, encoding="utf-8")


def fails(fn, ks):
    try:
        fn()
    except ValidationError as x:
        if not set(ks) <= x.n.keys():
            raise AssertionError(f"expected {ks}, got {x.n}") from x
    else:
        raise AssertionError(f"expected {ks}")


def check():
    'run a small strict checker self-test'
    if f05(set(), set()) != 1:
        raise AssertionError("empty singleton failed")
    if f05({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"}) != 5 / 7:
        raise AssertionError("published metric example failed")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        td = d / "test"
        td.mkdir()
        src = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        write(td / "test_source1.tsv", src + "S1-fr\tn\ta\tFrance\nS1-us\tn\ta\tUS\n")
        write(td / "test_source2.tsv", src + "S2-fr\tn\ta\tFrance\n")
        write(td / "test_source3.tsv", src + "S3-us\tn\ta\tUS\n")
        m = d / "matching.tsv"
        c = d / "candidate.tsv"
        write(m, "source1_entity_id\tmatched_entity_ids\nS1-fr\tS2-fr\nS1-us\t\n")
        write(c, "source1_entity_id\tcandidate_entity_ids\nS1-us\t\nS1-fr\tS2-fr,S3-us\n")
        st = validate(m, c, td)
        if st["matching_empty"] != 1 or st["targets"] != 2:
            raise AssertionError("valid france or empty list failed")
        if st["candidate_sizes"] != {"mean": 1.0, "p50": 0, "p95": 2, "p99": 2, "max": 2,
                                      "quantile_method": "nearest_rank", "histogram": {0: 1, 2: 1}}:
            raise AssertionError("candidate size distribution failed")
        write(m, "source1_entity_id\tmatched_entity_ids\nS1-fr\tS2-fr,S2-fr\nS1-us\t\n")
        fails(lambda: validate(m, c, td), {"duplicate id within list"})
        write(m, "source1_entity_id\tmatched_entity_ids\nS1-fr\tS2-fr\nS1-fr\tS2-fr\nS1-us\t\n")
        fails(lambda: validate(m, c, td), {"duplicate source1 row"})
        write(m, "source1_entity_id\tmatched_entity_ids\nS1-fr\tS2-none\n")
        fails(lambda: validate(m, c, td), {"unknown target id", "missing source1 coverage"})
        write(m, "source1_entity_id\tmatched_entity_ids\nS1-fr\tS2-fr\nS1-us\t\n")
        write(c, "source1_entity_id\tcandidate_entity_ids\nS1-fr\t\n")
        fails(lambda: validate(m, c, td), {"missing source1 coverage", "final match outside candidate set"})
        write(d / "truth.tsv", "source1_entity_id\tmatched_entity_ids\nS1-a\t\nS1-b\tS2-b\n")
        write(d / "pred.tsv", "source1_entity_id\tmatched_entity_ids\nS1-a\t\nS1-b\tS2-b,S3-x\n")
        st = score(d / "truth.tsv", d / "pred.tsv")
        if st["macro_f05"] != (1 + 5 / 9) / 2:
            raise AssertionError("singleton score failed")
        write(d / "pred.tsv", "source1_entity_id\tmatched_entity_ids\nS1-a\t\nS1-x\t\n")
        fails(lambda: score(d / "truth.tsv", d / "pred.tsv"), {"missing source1 coverage", "unknown source1 id"})
    print("check: passed")


def main(argv=None):
    ap = argparse.ArgumentParser(description="strict submission validation and macro f0.5 scoring")
    ap.add_argument("--check", action="store_true", help="run the built-in self-test")
    sp = ap.add_subparsers(dest="cmd")
    vp = sp.add_parser("validate", help="validate matching and candidate outputs")
    vp.add_argument("--matching", required=True)
    vp.add_argument("--candidate", required=True)
    vp.add_argument("--test-dir", required=True)
    xp = sp.add_parser("score", help="score training labels and predictions")
    xp.add_argument("--truth", required=True)
    xp.add_argument("--matching", required=True)
    ns = ap.parse_args(argv)
    try:
        if ns.check:
            check()
            return 0
        if ns.cmd == "validate":
            st = validate(ns.matching, ns.candidate, ns.test_dir)
        elif ns.cmd == "score":
            st = score(ns.truth, ns.matching)
        else:
            ap.error("choose validate, score, or --check")
        for k, v in st.items():
            print(f"{k}: {v}")
        return 0
    except ValidationError as x:
        print(x, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
