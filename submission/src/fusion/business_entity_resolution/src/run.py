'business entity resolution - single entry point'
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from er.config import ROOT, load_config  # noqa: E402  (light imports only in the orchestrator)
from er.paths import Paths  # noqa: E402
from er.stages import STAGES  # noqa: E402



def run_stage(name: str, cfg: dict, smoke: bool):
    import er.safe  # noqa: F401  thread caps / priority before heavy imports
    from er.stages import get_stage
    split = name.split("_")[-1] if name.split("_")[-1] in ("train", "test") else None
    paths = Paths(cfg, smoke=smoke)
    get_stage(name).run(cfg, paths, split)


def _clear(stage: str, paths: Paths):
    "remove a stage's outputs so a forced re-run recomputes them"
    targets = {"prep": [paths.prep], "blocking_train": [paths.blocking("train")],
               "blocking_test": [paths.blocking("test")],
               "eval_blocking": [os.path.join(paths.blocking("train"), "eval.txt")],
               "features_train": [paths.features("train")], "features_test": [paths.features("test")],
               "train": [os.path.join(paths.exp, "metrics.json"), os.path.join(paths.exp, "model")],
               "predict": [paths.output, os.path.join(paths.exp, "test_scored")]}.get(stage, [])
    for t in targets:
        if os.path.isdir(t):
            shutil.rmtree(t, ignore_errors=True)
        elif os.path.exists(t):
            os.remove(t)


def _is_done(stage: str, cfg: dict, paths: Paths) -> bool:
    from er.stages import get_stage
    split = stage.split("_")[-1] if stage.split("_")[-1] in ("train", "test") else None
    return get_stage(stage).is_done(cfg, paths, split)



def cmd_run(args, cfg):
    paths = Paths(cfg, smoke=args.smoke)
    if args.smoke and not os.path.exists(os.path.join(paths.data, "test", "test_source3.tsv")):
        print("== building smoke dataset (2% of the real data)", flush=True)
        from er.smoke import make_smoke_data
        make_smoke_data(Paths(cfg).data, paths.data, pct=2)
    os.makedirs(paths.logs, exist_ok=True)
    os.makedirs(paths.experiments, exist_ok=True)
    json.dump(cfg, open(os.path.join(paths.exp, "config_resolved.json"), "w"), indent=1)

    forced = set()
    if args.from_stage:
        forced |= set(STAGES[STAGES.index(args.from_stage):])
    if args.force:
        forced |= set(args.force.split(","))
    todo = STAGES
    if args.only:
        todo = args.only.split(",")
        forced |= set(todo)
    for s in forced | set(todo):
        if s not in STAGES:
            sys.exit(f"unknown stage {s!r}; stages: {STAGES}")

    res = cfg.get("resources", {})
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               ER_THREADS=str(res.get("threads", 0)), ER_PRIORITY=str(res.get("priority", "normal")),
               ER_MIN_FREE_GB=str(res.get("min_free_gb", 1.5)))

    cfg_file = os.path.join(paths.logs, "_run_config.json")
    json.dump(cfg, open(cfg_file, "w"))
    print(f"== experiment '{paths.exp_name}'  data={paths.data}\n   work={paths.work}\n   "
          f"cache keys: prep={paths.h_prep} blocking={paths.h_block} features={paths.h_feat}", flush=True)
    t_all = time.time()
    for stage in todo:
        if stage in forced:
            _clear(stage, paths)
        elif _is_done(stage, cfg, paths):
            print(f"== {stage}: cached, skipping", flush=True)
            continue
        t0 = time.time()
        log_path = os.path.join(paths.logs, f"{stage}.log")
        print(f"== {stage}: running (log: {log_path})", flush=True)
        cmd = [sys.executable, "-u", "-W", "ignore", __file__, "_stage", stage, "--_cfg", cfg_file]
        if args.smoke:
            cmd.append("--smoke")
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print("   " + line.rstrip(), flush=True)
                log.write(line)
            rc = proc.wait()
        if rc != 0:
            sys.exit(f"!! stage {stage} failed (exit {rc}) - see {log_path}. "
                     f"Re-run the same command to resume: finished work is kept.")
        print(f"== {stage}: finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"== done in {(time.time() - t_all) / 60:.1f} min. Outputs: {paths.output}")


def cmd_leaderboard(args, cfg):
    board = os.path.join(Paths(cfg, smoke=args.smoke).experiments, "leaderboard.csv")
    if not os.path.exists(board):
        sys.exit("no experiments trained yet")
    import csv
    rows = list(csv.DictReader(open(board, encoding="utf-8")))
    rows.sort(key=lambda r: -float(r["val_f05"]))
    print(f"{'experiment':28s} {'model':12s} {'val F0.5':>9s} {'thr':>6s} {'oracle':>7s} {'#feat':>5s}  trained")
    for r in rows:
        print(f"{r['experiment']:28s} {r['model']:12s} {float(r['val_f05']):9.4f} {float(r['threshold']):6.3f} "
              f"{float(r['oracle_f05']):7.4f} {r['n_features']:>5s}  {r['trained_at']}")


def cmd_package(args, cfg):
    from er.package import build_zip
    paths = Paths(cfg, smoke=args.smoke)
    out = build_zip(paths, args.team, args.out_dir or os.path.join(ROOT, "..", ".."))
    print(f"wrote {out} ({os.path.getsize(out) / 2**20:.0f} MB) from experiment '{paths.exp_name}'")


def cmd_diagnose(args, cfg):
    res = cfg.get("resources", {})
    os.environ.update(ER_THREADS=str(res.get("threads", 0)), ER_PRIORITY=str(res.get("priority", "normal")),
                      ER_MIN_FREE_GB=str(res.get("min_free_gb", 1.5)))
    import er.safe  # noqa: F401  (thread caps before polars is imported)
    from er.diagnose import run
    run(cfg, Paths(cfg, smoke=args.smoke))


def cmd_sweep(args, cfg):
    res = cfg.get("resources", {})
    os.environ.update(ER_THREADS=str(res.get("threads", 0)), ER_PRIORITY=str(res.get("priority", "normal")),
                      ER_MIN_FREE_GB=str(res.get("min_free_gb", 1.5)))
    import er.safe  # noqa: F401
    from er.sweep import run
    run(cfg, Paths(cfg, smoke=args.smoke), apply=args.apply)


def cmd_stack(args, cfg):
    res = cfg.get("resources", {})
    os.environ.update(ER_THREADS=str(res.get("threads", 0)), ER_PRIORITY=str(res.get("priority", "normal")),
                      ER_MIN_FREE_GB=str(res.get("min_free_gb", 1.5)))
    import er.safe  # noqa: F401
    from er.stack.pipeline import run
    run(cfg, Paths(cfg, smoke=args.smoke))


def cmd_list(args, cfg):
    from er.blocking import BLOCKERS
    from er.decision import DECISIONS
    from er.features import FEATURE_GROUPS
    from er.models import MODELS
    for reg in (BLOCKERS, FEATURE_GROUPS, MODELS, DECISIONS):
        print(f"{reg.kind + 's':16s}: {', '.join(reg.names())}")
    print(f"{'stages':16s}: {', '.join(STAGES)}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "_stage":
        stage, cfg_file = sys.argv[2], sys.argv[sys.argv.index("--_cfg") + 1]
        run_stage(stage, json.load(open(cfg_file)), "--smoke" in sys.argv)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "leaderboard", "package", "list", "diagnose", "sweep", "stack"])
    ap.add_argument("-c", "--config", default=None, help="config file (default configs/default.toml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--name", default=None, help="experiment name (overrides [experiment] name)")
    ap.add_argument("--smoke", action="store_true", help="run on a 2%% sample of the data")
    ap.add_argument("--only", default=None, help="run only these stages (comma-separated)")
    ap.add_argument("--from", dest="from_stage", default=None, help="re-run from this stage onwards")
    ap.add_argument("--force", default=None, help="re-run these stages even if cached (comma-separated)")
    ap.add_argument("--team", default="team", help="package: team name for the zip")
    ap.add_argument("--out_dir", default=None, help="package: where to write the zip")
    ap.add_argument("--apply", action="store_true", help="sweep: make the best decision rule the experiment's rule")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg["experiment"]["name"] = args.name
    {"run": cmd_run, "leaderboard": cmd_leaderboard, "package": cmd_package, "list": cmd_list,
     "diagnose": cmd_diagnose, "sweep": cmd_sweep, "stack": cmd_stack}[args.command](args, cfg)


if __name__ == "__main__":
    main()
