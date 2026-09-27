'where each stage reads and writes'
import os

from er.config import ROOT, settings_hash

CODE_VERSIONS = {"prep": 2, "blocking": 1, "features": 3}


def _find_data() -> str:
    for cand in (os.path.join(ROOT, "dataset"), os.path.join(ROOT, "..", "..", "dataset")):
        if os.path.isdir(os.path.join(cand, "train")):
            return os.path.abspath(cand)
    return os.path.join(ROOT, "dataset")


class Paths:
    def __init__(self, cfg: dict, smoke: bool = False):
        p = cfg.get("paths", {})
        if smoke:
            base = os.path.join(ROOT, "smoke")
            self.data = os.path.join(base, "dataset")
            self.work = os.path.join(base, "work")
            self.experiments = os.path.join(base, "experiments")
        else:
            self.data = os.path.abspath(p.get("data") or _find_data())
            self.work = os.path.abspath(p.get("work") or os.path.join(ROOT, "work"))
            self.experiments = os.path.abspath(p.get("experiments") or os.path.join(ROOT, "experiments"))
        self.h_prep = settings_hash("prep", CODE_VERSIONS["prep"], cfg["prep"])
        self.h_block = settings_hash(self.h_prep, "blocking", CODE_VERSIONS["blocking"], cfg["blocking"])
        self.h_feat = settings_hash(self.h_block, "features", CODE_VERSIONS["features"],
                                    cfg["candidates"], cfg["features"])
        tr = cfg["training"]
        self.h_sample = settings_hash(tr["sample_pct"], tr["hash_seed"])
        self.exp_name = cfg["experiment"]["name"]


    @property
    def prep(self):
        return os.path.join(self.work, "prep", self.h_prep)

    def prep_file(self, split, src):
        return os.path.join(self.prep, f"{split}_s{src}.parquet")

    def blocking(self, split):
        return os.path.join(self.work, "blocking", self.h_block, split)

    def features(self, split):
        sub = "test" if split == "test" else f"train_{self.h_sample}"
        return os.path.join(self.work, "features", self.h_feat, sub)

    @property
    def exp(self):
        return os.path.join(self.experiments, self.exp_name)

    @property
    def output(self):
        return os.path.join(self.exp, "output")

    @property
    def logs(self):
        return os.path.join(self.exp, "logs")

    def raw(self, split, src):
        name = "train_ground_truth.tsv" if src == "gt" else f"{split}_source{src}.tsv"
        return os.path.join(self.data, split, name)
