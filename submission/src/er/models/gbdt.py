'gradient-boosted tree models: lightgbm (default), xgboost, sklearn histgradientboosting'
import os
import pickle

import numpy as np

from er.models import MODELS, BaseModel


@MODELS.register("lightgbm")
class LightGBMModel(BaseModel):
    '[model.params] -> lightgbm params; [model.fit]: num_boost_round, early_stopping_rounds'

    def fit(self, X, y, X_val, y_val, feature_names):
        import lightgbm as lgb
        self.feature_names = list(feature_names)
        params = {"objective": "binary", "verbose": -1, "num_threads": self.threads, **self.params}
        dtr = lgb.Dataset(X, y, feature_name=self.feature_names, free_raw_data=True)
        dva = lgb.Dataset(X_val, y_val, reference=dtr)
        self.booster = lgb.train(
            params, dtr, num_boost_round=self.fit_options.get("num_boost_round", 3000), valid_sets=[dva],
            callbacks=[lgb.early_stopping(self.fit_options.get("early_stopping_rounds", 100)),
                       lgb.log_evaluation(self.fit_options.get("log_every", 100))])
        self.info["best_iteration"] = self.booster.best_iteration

    def predict(self, X):
        return self.booster.predict(X, num_iteration=self.booster.best_iteration or None, num_threads=self.threads)

    def importance(self):
        gain = self.booster.feature_importance("gain")
        return sorted(zip(self.feature_names, map(float, gain)), key=lambda x: -x[1])

    def save(self, directory):
        self.booster.save_model(os.path.join(directory, "model.txt"), num_iteration=self.booster.best_iteration or None)
        self._save_meta(directory)

    @classmethod
    def load(cls, directory):
        import lightgbm as lgb
        meta = cls._load_meta(directory)
        m = cls(meta["params"], meta["fit_options"])
        m.feature_names, m.info = meta["feature_names"], meta["info"]
        m.booster = lgb.Booster(model_file=os.path.join(directory, "model.txt"))
        return m


@MODELS.register("xgboost")
class XGBoostModel(BaseModel):
    'Needs `pip install xgboost`. [model.params] -> xgboost params (e.g. device = "cuda" for GPU)'

    def fit(self, X, y, X_val, y_val, feature_names):
        import xgboost as xgb
        self.feature_names = list(feature_names)
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                  "nthread": self.threads, **self.params}
        dtr = xgb.DMatrix(X, y, feature_names=self.feature_names)
        dva = xgb.DMatrix(X_val, y_val, feature_names=self.feature_names)
        self.booster = xgb.train(params, dtr, num_boost_round=self.fit_options.get("num_boost_round", 3000),
                                 evals=[(dva, "val")],
                                 early_stopping_rounds=self.fit_options.get("early_stopping_rounds", 100),
                                 verbose_eval=self.fit_options.get("log_every", 100))
        self.info["best_iteration"] = self.booster.best_iteration

    def predict(self, X):
        import xgboost as xgb
        it = self.info.get("best_iteration")
        return self.booster.predict(xgb.DMatrix(X, feature_names=self.feature_names),
                                    iteration_range=(0, it + 1) if it is not None else (0, 0))

    def importance(self):
        g = self.booster.get_score(importance_type="total_gain")
        return sorted(((f, float(g.get(f, 0.0))) for f in self.feature_names), key=lambda x: -x[1])

    def save(self, directory):
        self.booster.save_model(os.path.join(directory, "model.json"))
        self._save_meta(directory)

    @classmethod
    def load(cls, directory):
        import xgboost as xgb
        meta = cls._load_meta(directory)
        m = cls(meta["params"], meta["fit_options"])
        m.feature_names, m.info = meta["feature_names"], meta["info"]
        m.booster = xgb.Booster()
        m.booster.load_model(os.path.join(directory, "model.json"))
        m.booster.set_param({"nthread": m.threads})
        return m


@MODELS.register("sklearn_hgb")
class SklearnHGBModel(BaseModel):
    'sklearn histgradientboostingclassifier (needs scikit-learn). [model.params] -> its constructor'

    def fit(self, X, y, X_val, y_val, feature_names):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.feature_names = list(feature_names)
        self.clf = HistGradientBoostingClassifier(**{"early_stopping": True, "random_state": 7, **self.params})
        self.clf.fit(np.vstack([X, X_val]), np.concatenate([y, y_val]))

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def save(self, directory):
        pickle.dump(self.clf, open(os.path.join(directory, "model.pkl"), "wb"))
        self._save_meta(directory)

    @classmethod
    def load(cls, directory):
        meta = cls._load_meta(directory)
        m = cls(meta["params"], meta["fit_options"])
        m.feature_names, m.info = meta["feature_names"], meta["info"]
        m.clf = pickle.load(open(os.path.join(directory, "model.pkl"), "rb"))
        return m
