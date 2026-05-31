"""
交叉组合实验框架

Config 驱动, 自动跑完所有 FeatureSelector × Model 组合, 产出对比表

Config 格式:
    {
        "name": "A1-SHAP50-LGB",
        "selector": SomeSelector(...),
        "model": SomeModel(...),
        "feature_cols": [...] | None,   # 可选, 预过滤列
        "cat_cols": [...],
        "description": "..."
    }

用法:
    runner = ExperimentRunner(configs=[...])
    runner.run(X_train, y_train, X_valid, y_valid)
    print(runner.results)
"""

import pandas as pd
import numpy as np
import time


class ExperimentRunner:
    def __init__(self, configs):
        """
        configs: list of dict
            每个 dict 包含: name, selector, model, [feature_cols], cat_cols, description
        """
        self.configs = configs
        self.results = None

    def run(self, X_train, y_train, X_valid, y_valid):
        rows = []
        for cfg in self.configs:
            name = cfg["name"]
            selector = cfg["selector"]
            model = cfg["model"]
            cat_cols = cfg.get("cat_cols")
            desc = cfg.get("description", "")

            # 预过滤特征列 (如果指定)
            feature_cols = cfg.get("feature_cols")
            X_tr = X_train if feature_cols is None else X_train[feature_cols]
            X_va = X_valid if feature_cols is None else X_valid[feature_cols]
            cat_in = cat_cols if feature_cols is None else [c for c in cat_cols if c in feature_cols]

            t0 = time.time()

            # Step 1: 特征选择
            sel_name = type(selector).__name__
            X_tr_sel = selector.fit_transform(X_tr, y_train, cat_cols=cat_in)
            X_va_sel = selector.transform(X_va)
            n_feat = (X_tr_sel.shape[1] if isinstance(X_tr_sel, pd.DataFrame)
                      else X_tr_sel.shape[1])

            # Step 2: 训练模型
            # 更新 cat_cols (只保留 selector 保留的列)
            sel_features = (list(X_tr_sel.columns) if isinstance(X_tr_sel, pd.DataFrame)
                           else list(range(n_feat)))
            if isinstance(X_tr_sel, pd.DataFrame) and cat_in:
                cat_sel = [c for c in cat_in if c in sel_features]
            else:
                cat_sel = None

            model.fit(X_tr_sel, y_train, cat_cols=cat_sel)

            # Step 3: 评估
            train_metrics = model.evaluate(X_tr_sel, y_train)
            valid_metrics = model.evaluate(X_va_sel, y_valid)
            elapsed = time.time() - t0

            row = {
                "config": name,
                "selector": sel_name,
                "model": model.name,
                "description": desc,
                "n_features": n_feat,
                "train_auc": train_metrics["auc"],
                "valid_auc": valid_metrics["auc"],
                "time_sec": round(elapsed, 1),
            }

            # 附加 selector 信息
            if hasattr(selector, "explained_variance_ratio"):
                row["pca_var_ratio"] = round(selector.explained_variance_ratio, 3)

            rows.append(row)
            print(f"  [{name}] valid_auc={valid_metrics['auc']:.4f}  "
                  f"n_features={n_feat}  time={elapsed:.1f}s")

        self.results = pd.DataFrame(rows).sort_values("valid_auc", ascending=False)
        return self.results

    @property
    def summary(self):
        """简洁版结果表"""
        if self.results is None:
            return None
        cols = ["config", "selector", "model", "n_features", "valid_auc", "time_sec"]
        return self.results[cols]

    def compare_ab(self, a_name, b_name):
        """
        两组对比: 输出 A/B 的 AUC 差值
        """
        if self.results is None:
            return None
        a = self.results[self.results["config"] == a_name]
        b = self.results[self.results["config"] == b_name]
        if len(a) == 0 or len(b) == 0:
            return None
        delta = b.iloc[0]["valid_auc"] - a.iloc[0]["valid_auc"]
        return {
            "A_config": a_name, "A_auc": a.iloc[0]["valid_auc"],
            "B_config": b_name, "B_auc": b.iloc[0]["valid_auc"],
            "delta_auc": round(delta, 5),
        }
