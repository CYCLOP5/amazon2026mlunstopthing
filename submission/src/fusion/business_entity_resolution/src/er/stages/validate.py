"Stage validate: run the challenge's utils/validate_submission.py on the experiment output"
import os
import subprocess
import sys

from er.config import ROOT


def find_validator():
    for p in (os.path.join(ROOT, "utils", "validate_submission.py"),
              os.path.join(ROOT, "..", "..", "utils", "validate_submission.py")):
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def is_done(cfg, paths, split=None):
    return False


def run(cfg, paths, split=None):
    v = find_validator()
    if v is None:
        print("validator utils/validate_submission.py not found - skipped")
        return
    rc = subprocess.run([sys.executable, v, "--matching", os.path.join(paths.output, "matching_results.tsv"),
                         "--candidate", os.path.join(paths.output, "candidate_pairs.tsv"),
                         "--test-dir", os.path.join(paths.data, "test")]).returncode
    if rc != 0:
        raise SystemExit("validator reported problems (see above)")
