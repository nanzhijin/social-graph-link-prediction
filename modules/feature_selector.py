"""
特征选择器模块

标准接口:
    .fit(X, y)          → 训练选择器
    .transform(X)       → 输出缩减后的特征矩阵
    .fit_transform(X,y) → 一步完成
    .selected_features  → 被选中的特征名列表 (SHAP)
    .n_features_out     → 输出维度
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings("ignore")


def _safe_to_ordinal(series):
    """时间戳列 → float epoch days, sentinel 保持 -1"""
    vals = series.astype(object)
    result = np.full(len(vals), -1.0, dtype=np.float64)
    for i, v in enumerate(vals):
        try:
            if pd.notna(v) and v != -1 and v != "-1":
                result[i] = float(pd.Timestamp(v).toordinal())
        except (ValueError, TypeError):
            pass
    return result


class IdentitySelector:
    """全保留 — 不做任何筛选, 用于对照组"""
    def __init__(self, feature_cols=None):
        self._feature_cols = feature_cols

    def fit(self, X, y=None, **kwargs):
        if self._feature_cols is None:
            self._feature_cols = list(X.columns)
        return self

    def transform(self, X, **kwargs):
        return X[self._feature_cols]

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)

    @property
    def selected_features(self):
        return list(self._feature_cols)

    @property
    def n_features_out(self):
        return len(self._feature_cols)


class SHAPSelector:
    """
    SHAP 特征重要性选择器

    用 LightGBM 训练 → 计算 SHAP → 按 mean(|SHAP|) 排序 → 取 top-K

    参数:
        k: 保留的特征数
        lgb_params: LightGBM 参数 (默认基线参数)
        shap_subsample: SHAP 计算的子采样数 (全量太慢, 默认 50k)
        random_state: 随机种子
    """
    def __init__(self, k=50, lgb_params=None, shap_subsample=50000, random_state=42):
        self.k = k
        self.random_state = random_state
        self.shap_subsample = shap_subsample
        self.lgb_params = lgb_params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "is_unbalance": True,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        self._feature_cols = None
        self._importance = None

    def fit(self, X, y, cat_cols=None):
        self._feature_cols = list(X.columns)

        X_lgb = X.copy()
        # 时间戳列转换 (LightGBM 不接受 datetime/object)
        for tc in [c for c in X_lgb.columns if "first_time" in c or "last_time" in c]:
            X_lgb[tc] = _safe_to_ordinal(X_lgb[tc])

        cat_in = []
        if cat_cols:
            cat_in = [c for c in cat_cols if c in X_lgb.columns]
            for c in cat_in:
                X_lgb[c] = X_lgb[c].astype("category")

        model = lgb.LGBMClassifier(**self.lgb_params)
        model.fit(X_lgb, y, categorical_feature=cat_in or "auto")

        # SHAP 计算 (子采样提速)
        n_shap = min(self.shap_subsample, len(X))
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), n_shap, replace=False)
        X_sample = X_lgb.iloc[idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # shap_values shape: (n_samples, n_features) for binary

        self._importance = pd.DataFrame({
            "feature": self._feature_cols,
            "shap_importance": np.abs(shap_values).mean(axis=0),
        }).sort_values("shap_importance", ascending=False)

        self._selected = self._importance.head(self.k)["feature"].tolist()
        self._model = model
        return self

    def transform(self, X):
        return X[self._selected]

    def fit_transform(self, X, y, cat_cols=None):
        return self.fit(X, y, cat_cols).transform(X)

    @property
    def selected_features(self):
        return list(self._selected)

    @property
    def importance_df(self):
        return self._importance

    @property
    def n_features_out(self):
        return self.k


class PCAReducer:
    """
    PCA 降维器

    参数:
        n_components: 输出维度 (int) 或 方差保留比例 (float, 0~1)
        random_state: 随机种子
    """
    def __init__(self, n_components=32, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self._scaler = None
        self._pca = None

    def fit(self, X, y=None, **kwargs):
        # 只对数值列做 PCA
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._num_cols = num_cols

        X_num = X[num_cols].fillna(0)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_num)

        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._pca.fit(X_scaled)

        self._explained_var = self._pca.explained_variance_ratio_.sum()
        return self

    def transform(self, X, **kwargs):
        X_num = X[self._num_cols].fillna(0)
        X_scaled = self._scaler.transform(X_num)
        X_pca = self._pca.transform(X_scaled)
        cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, index=X.index, columns=cols)

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)

    @property
    def explained_variance_ratio(self):
        return self._explained_var

    @property
    def n_features_out(self):
        return self._pca.n_components_ if self._pca else self.n_components


class SHAPPCASelector:
    """
    SHAP → PCA 串行选择器
    SHAP 先筛选 top-K 重要特征 → PCA 进一步降维
    """
    def __init__(self, shap_k=50, pca_n=32, lgb_params=None, random_state=42):
        self.shap = SHAPSelector(k=shap_k, lgb_params=lgb_params, random_state=random_state)
        self.pca = PCAReducer(n_components=pca_n, random_state=random_state)

    def fit(self, X, y, cat_cols=None):
        print(f"  [SHAP→PCA] SHAP 筛选 top-{self.shap.k}...")
        X_shap = self.shap.fit_transform(X, y, cat_cols)
        print(f"  [SHAP→PCA] PCA 降维到 {self.pca.n_components}...")
        self.pca.fit(X_shap)
        print(f"  [SHAP→PCA] 方差保留: {self.pca.explained_variance_ratio:.1%}")
        return self

    def transform(self, X):
        X_shap = self.shap.transform(X)
        return self.pca.transform(X_shap)

    def fit_transform(self, X, y, cat_cols=None):
        self.fit(X, y, cat_cols)
        return self.transform(X)

    @property
    def selected_features(self):
        return self.shap.selected_features

    @property
    def n_features_out(self):
        return self.pca.n_features_out
