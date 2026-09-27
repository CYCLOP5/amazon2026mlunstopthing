'matching models'
import json
import os

import numpy as np

from er.registry import Registry

MODELS = Registry("model")


class BaseModel:
    def __init__(self, params: dict | None = None, fit_options: dict | None = None, threads: int | None = None):
        from er.safe import THREADS
        self.params = dict(params or {})
        self.fit_options = dict(fit_options or {})
        self.threads = threads or THREADS
        self.feature_names = None
        self.info = {}

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_names: list):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, directory: str):
        raise NotImplementedError

    @classmethod
    def load(cls, directory: str):
        raise NotImplementedError

    def importance(self) -> list:
        '[(feature, importance)] sorted descending, if the model has one'
        return []


    def _save_meta(self, directory):
        json.dump({"params": self.params, "fit_options": self.fit_options, "feature_names": self.feature_names,
                   "info": self.info}, open(os.path.join(directory, "model_meta.json"), "w"), indent=1, default=str)

    @staticmethod
    def _load_meta(directory):
        return json.load(open(os.path.join(directory, "model_meta.json")))


from er.models import gbdt, linear  # noqa: E402,F401  (register models)
