"build a submission by taking each country's rows from a different submission (streaming, low memory)"
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source1", required=True, help="test_source1.tsv (entity_id, ..., country)")
    ap.add_argument("--default", required=True, help="dir with matching_results.tsv + candidate_pairs.tsv for non-French rows")
    ap.add_argument("--france", required=True, help="dir with matching_results.tsv + candidate_pairs.tsv for French rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(a.source1, encoding="utf-8") as f:
        head = next(f).rstrip("\n").split("\t")
        ci = head.index("country")
        fr = {line.split("\t", 1)[0] for line in f if line.rstrip("\n").split("\t")[ci] == "France"}
    for kind in ("matching_results.tsv", "candidate_pairs.tsv"):
        n = {"default": 0, "france": 0}
        with open(os.path.join(a.out, kind), "w", encoding="utf-8", newline="") as out:
            with open(os.path.join(a.default, kind), encoding="utf-8") as src:
                out.write(next(src))
                for line in src:
                    if line.split("\t", 1)[0] not in fr:
                        out.write(line if line.endswith("\n") else line + "\n"); n["default"] += 1
            with open(os.path.join(a.france, kind), encoding="utf-8") as src:
                next(src)
                for line in src:
                    if line.split("\t", 1)[0] in fr:
                        out.write(line if line.endswith("\n") else line + "\n"); n["france"] += 1
        print(kind, n, flush=True)


if __name__ == "__main__":
    main()
