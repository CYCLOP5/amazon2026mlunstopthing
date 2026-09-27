'keep, for selected countries, only the matches that a second submission also accepts (removal-only intersection)'
import argparse
import os
import shutil


def read(path, keep):
    d = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s = line.rstrip("\n").split("\t")
            if s[0] in keep:
                d[s[0]] = set(x for x in (s[1].split(",") if len(s) > 1 else []) if x)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source1", required=True)
    ap.add_argument("--base", required=True, help="dir with matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--filter", required=True, help="dir with the second submission's matching_results.tsv")
    ap.add_argument("--countries", default="US,India")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    countries = set(a.countries.split(","))
    with open(a.source1, encoding="utf-8") as f:
        ci = next(f).rstrip("\n").split("\t").index("country")
        sel = {line.split("\t", 1)[0] for line in f if line.rstrip("\n").split("\t")[ci] in countries}
    other = read(os.path.join(a.filter, "matching_results.tsv"), sel)
    os.makedirs(a.out, exist_ok=True)
    removed = 0
    with open(os.path.join(a.base, "matching_results.tsv"), encoding="utf-8") as f, \
            open(os.path.join(a.out, "matching_results.tsv"), "w", encoding="utf-8", newline="") as o:
        o.write(next(f))
        for line in f:
            s = line.rstrip("\n").split("\t")
            if s[0] not in sel:
                o.write(line if line.endswith("\n") else line + "\n")
                continue
            ids = [x for x in (s[1].split(",") if len(s) > 1 else []) if x]
            kept = [x for x in ids if x in other.get(s[0], set())]
            removed += len(ids) - len(kept)
            o.write(s[0] + "\t" + ",".join(kept) + "\n")
    shutil.copyfile(os.path.join(a.base, "candidate_pairs.tsv"), os.path.join(a.out, "candidate_pairs.tsv"))
    print("removed", removed, flush=True)


if __name__ == "__main__":
    main()
