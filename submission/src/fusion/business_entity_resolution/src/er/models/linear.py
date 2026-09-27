'linear baseline: standardised logistic regression (needs scikit-learn)'
import os
import pickle

import numpy as np

from er.models import MODELS, BaseModel


@MODELS.register("logreg")
class LogRegModel(BaseModel):
    '[model.params] -> sklearn logisticregression (e.g. c = 1.0). missing values are imputed with 0'

    def fit(self, X, y, X_val, y_val, feature_names):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.feature_names = list(feature_names)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(**{"max_iter": 500, **self.params}))
        self.clf.fit(np.nan_to_num(X), y)

    def predict(self, X):
        return self.clf.predict_proba(np.nan_to_num(X))[:, 1]

    def importance(self):
        coef = np.abs(self.clf[-1].coef_[0])
        return sorted(zip(self.feature_names, map(float, coef)), key=lambda x: -x[1])

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
