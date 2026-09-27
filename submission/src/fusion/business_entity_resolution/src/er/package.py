'build the final submission zip for one experiment'
import datetime
import glob
import json
import os
import re
import zipfile

from er.config import ROOT


def _read(path, default=""):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else default


def render_doc(paths) -> str:
    doc = _read(os.path.join(ROOT, "docs", "Documentation_template.md"))
    m = json.loads(_read(os.path.join(paths.exp, "metrics.json"), "{}") or "{}")
    table = ["| Candidate set | Pair recall | Oracle macro F0.5 | Pairs (train) |", "|---|---|---|---|"]
    for line in _read(os.path.join(paths.blocking("train"), "eval.txt")).splitlines():
        r = re.match(r"\s*(.+?)\s+pair-recall=([\d.]+)\s+oracle F0.5=([\d.]+)\s+pairs=([\d,]+)", line)
        if r:
            table.append(f"| {r.group(1)} | {float(r.group(2)):.4f} | {float(r.group(3)):.4f} | {r.group(4)} |")
    pred = _read(os.path.join(paths.logs, "predict.log"))
    pm = re.search(r"candidate pairs=([\d,]+)\s+matches=([\d,]+)\s+S1 with >=1 match: ([\d,]+)/([\d,]+)", pred)
    imp = m.get("importance", [])
    tot = sum(v for _, v in imp) or 1
    subs = {
        "DATE": datetime.date.today().isoformat(),
        "EXPERIMENT": m.get("experiment", paths.exp_name),
        "MODEL": m.get("model", "?"),
        "VAL_F05": f"{m['val_f05']:.4f}" if "val_f05" in m else "(not trained yet)",
        "ORACLE_F05": f"{m['oracle_f05']:.4f}" if "oracle_f05" in m else "(not trained yet)",
        "THRESHOLD": f"{m['threshold']:.3f}" if "threshold" in m else "(not trained yet)",
        "BEST_ITER": str(m.get("model_info", {}).get("best_iteration", "")),
        "N_FEATURES": str(m.get("n_features", "")),
        "BLOCKING_TABLE": "\n".join(table) if len(table) > 2 else "(eval_blocking not run)",
        "TEST_CAND_PAIRS": pm.group(1) if pm else "(predict not run)",
        "TEST_MATCHES": pm.group(2) if pm else "(predict not run)",
        "TEST_S1_WITH_MATCH": f"{pm.group(3)} / {pm.group(4)}" if pm else "(predict not run)",
        "TOP_FEATURES": "\n".join(f"| {i + 1} | `{n}` | {100 * v / tot:.1f}% |" for i, (n, v) in enumerate(imp[:15])),
        "THRESHOLD_CURVE": ", ".join(f"{t:.2f}->{f:.4f}" for t, f in m.get("threshold_curve", [])[::2]),
    }
    for k, v in subs.items():
        doc = doc.replace("{{" + k + "}}", v)
    return doc


def build_zip(paths, team: str, out_dir: str) -> str:
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if not os.path.exists(os.path.join(paths.output, f)):
            raise SystemExit(f"missing {os.path.join(paths.output, f)} - run the pipeline for this experiment first")
    zpath = os.path.abspath(os.path.join(out_dir, f"{team}_submission.zip"))
    code = "code/business_entity_resolution"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(os.path.join(paths.output, f), f"output/{f}")
        for f in glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True):
            z.write(f, f"{code}/{os.path.relpath(f, ROOT).replace(os.sep, '/')}")
        for f in glob.glob(os.path.join(ROOT, "configs", "**", "*.toml"), recursive=True):
            z.write(f, f"{code}/{os.path.relpath(f, ROOT).replace(os.sep, '/')}")
        for f in ("README.md", "requirements.txt", os.path.join("docs", "Documentation_template.md")):
            z.write(os.path.join(ROOT, f), f"{code}/{f.replace(os.sep, '/')}")
        for f in glob.glob(os.path.join(paths.logs, "*.log")):
            z.write(f, f"{code}/logs/{os.path.basename(f)}")
        if os.path.exists(os.path.join(paths.exp, "metrics.json")):
            z.write(os.path.join(paths.exp, "metrics.json"), f"{code}/logs/metrics.json")
        z.writestr("Documentation_template.md", render_doc(paths))
    return zpath
