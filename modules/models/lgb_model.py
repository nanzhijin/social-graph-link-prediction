"""
LightGBM 分类器封装 — 标准模型接口

每个模型实现: fit → predict_proba → evaluate
GNN 模块实现同样的接口, 无缝接入 ExperimentRunner
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


class LGBModel:
    def __init__(self, name="LGB", params=None):
        self.name = name
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "is_unbalance": True,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self._model = None
        self._feature_cols = None

    def fit(self, X, y, cat_cols=None):
        self._feature_cols = list(X.columns)
        self._cat_cols = []
        X_lgb = X.copy()
        if cat_cols:
            self._cat_cols = [c for c in cat_cols if c in X_lgb.columns]
            for c in self._cat_cols:
                X_lgb[c] = X_lgb[c].astype("category")

        time_cols = [c for c in X_lgb.columns if "first_time" in c or "last_time" in c]
        for tc in time_cols:
            X_lgb[tc] = self._safe_to_ordinal(X_lgb[tc])

        self._model = lgb.LGBMClassifier(**self.params)
        self._model.fit(X_lgb, y, categorical_feature=self._cat_cols or "auto")
        return self

    def predict_proba(self, X):
        X_copy = X[self._feature_cols].copy()
        for c in self._cat_cols:
            if c in X_copy.columns:
                X_copy[c] = X_copy[c].astype("category")
        time_cols = [c for c in X_copy.columns if "first_time" in c or "last_time" in c]
        for tc in time_cols:
            X_copy[tc] = self._safe_to_ordinal(X_copy[tc])
        return self._model.predict_proba(X_copy)[:, 1]

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        return {
            "auc": roc_auc_score(y, proba),
            "pos_rate_pred": float(proba.mean()),
        }

    @property
    def feature_importances(self):
        if self._model is None:
            return None
        return pd.DataFrame({
            "feature": self._feature_cols,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False)

    @staticmethod
    def _safe_to_ordinal(series):
        vals = series.astype(object)
        result = np.full(len(vals), -1.0, dtype=np.float64)
        for i, v in enumerate(vals):
            try:
                if pd.notna(v) and v != -1 and v != "-1":
                    result[i] = float(pd.Timestamp(v).toordinal())
            except (ValueError, TypeError):
                pass
        return result
