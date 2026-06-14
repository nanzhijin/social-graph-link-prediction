"""
GNN-B 假设检验 — 品类交叉特征迁移
==================================
GNN-A: GNNModel (SAGE, item-aware)
GNN-B: GNNModelB (SAGE, item-aware + 品类交叉标量×3)

假设: 品类交叉特征 (cate_match_score, top3, overlap)
      能在 GNN 的图结构推理之上提供额外信号
"""
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 3_算法建模/ → GNN/
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
from modules.models.gnn_model import GNNModel
from modules.models.gnn_model_B import GNNModelB

PROCESSED_DIR = _PROJECT_ROOT / "processed"
MODEL_DIR = _PROJECT_ROOT / "3_算法建模"

# 自动检测 CUDA
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  设备: {_DEVICE}")

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

train = pd.read_pickle(PROCESSED_DIR / "train_lgb.pkl")
valid = pd.read_pickle(PROCESSED_DIR / "valid_lgb.pkl")
config = pd.read_pickle(PROCESSED_DIR / "feature_config.pkl")

feature_cols = config["feature_cols"]  # 119
cat_cols = config["cat_cols"]

# 只用 GROUP_BASE 特征 (不含边级朋友圈)
groups = config["feature_groups"]
cols_base = groups["GROUP_BASE"]  # 113 维
label_col = "label"

print(f"  train: {len(train):,}  valid: {len(valid):,}")
print(f"  特征: {len(cols_base)} (GROUP_BASE)")
print(f"  正样本率: {train[label_col].mean()*100:.1f}%")

# ============================================================
# 2. 训练 GNN-A (原版)
# ============================================================
print("\n" + "=" * 60)
print("2. 训练 GNN-A (SAGE + item, 无品类交叉)")
print("=" * 60)

gnn_a = GNNModel(
    name="GNN-A",
    hidden_dim=64,
    num_layers=2,
    gnn_type="sage",
    lr=0.005,
    epochs=50,
    batch_size=8192,
    device=_DEVICE,
    use_item_features=True,
    item_dim=32,
)

X_train_a = train[cols_base + ["inviter_id", "voter_id"]].copy()
y_train = train[label_col]

gnn_a.fit(X_train_a, y_train)

# AUC (用 mask 同步过滤 X 和 y)
from sklearn.metrics import roc_auc_score
proba_a = gnn_a.predict_proba(X_train_a)
mask_a = proba_a > 0
auc_a_train = roc_auc_score(y_train[mask_a], proba_a[mask_a])

X_valid_a = valid[cols_base + ["inviter_id", "voter_id"]].copy()
y_valid = valid[label_col]
proba_a_v = gnn_a.predict_proba(X_valid_a)
mask_av = proba_a_v > 0
auc_a_valid = roc_auc_score(y_valid[mask_av], proba_a_v[mask_av])

print(f"  GNN-A train AUC: {auc_a_train:.4f}")
print(f"  GNN-A valid AUC: {auc_a_valid:.4f}")

# ============================================================
# 3. 训练 GNN-B (+品类交叉)
# ============================================================
print("\n" + "=" * 60)
print("3. 训练 GNN-B (SAGE + item + 品类交叉)")
print("=" * 60)

gnn_b = GNNModelB(
    name="GNN-B",
    hidden_dim=64,
    num_layers=2,
    gnn_type="sage",
    lr=0.005,
    epochs=50,
    batch_size=8192,
    device=_DEVICE,
    use_item_features=True,
    item_dim=32,
)

# X 需包含 extra 特征列 (由 build_train_matrix.py 生成)
extra_cols = GNNModelB.EXTRA_FEATURE_COLS
missing_extra = [c for c in extra_cols if c not in train.columns]
if missing_extra:
    print(f"  ⚠ 缺失新特征列: {missing_extra}")
    print("  请先运行 build_train_matrix.py 生成新特征再跑 GNN-B")
    sys.exit(1)
print(f"  extra 特征: {extra_cols}")

X_train_b = train[cols_base + extra_cols + ["inviter_id", "voter_id"]].copy()
X_valid_b = valid[cols_base + extra_cols + ["inviter_id", "voter_id"]].copy()

gnn_b.fit(X_train_b, y_train)

proba_b = gnn_b.predict_proba(X_train_b)
mask_b = proba_b > 0
auc_b_train = roc_auc_score(y_train[mask_b], proba_b[mask_b])

proba_b_v = gnn_b.predict_proba(X_valid_b)
mask_bv = proba_b_v > 0
auc_b_valid = roc_auc_score(y_valid[mask_bv], proba_b_v[mask_bv])

print(f"  GNN-B train AUC: {auc_b_train:.4f}")
print(f"  GNN-B valid AUC: {auc_b_valid:.4f}")

print(f"\n  ΔAUC (B - A) valid: {auc_b_valid - auc_a_valid:+.4f}")

# ============================================================
# 4. MRR@5 分场景评估
# ============================================================
print("\n" + "=" * 60)
print("4. MRR@5 分场景评估 (GNN-A vs GNN-B)")
print("=" * 60)

# valid 需要包含品类交叉列用于 MRR 评估
valid_eval = valid.copy()
for col in extra_cols:
    if col not in valid_eval.columns:
        valid_eval[col] = 0.0

# GNN-B 的 evaluate_mrr 内部计算品类特征, 同时对比 GNN-A
result_b = gnn_b.evaluate_mrr(valid_eval, model_a=gnn_a, n_queries=500)

# ============================================================
# 5. 保存结果
# ============================================================
print("\n" + "=" * 60)
print("5. 保存结果")
print("=" * 60)

import joblib
results = {
    "auc_a_train": auc_a_train, "auc_a_valid": auc_a_valid,
    "auc_b_train": auc_b_train, "auc_b_valid": auc_b_valid,
    **result_b,
}
pd.to_pickle(results, MODEL_DIR / "gnn_baseline_B_results.pkl")
print(f"  已保存: gnn_baseline_B_results.pkl")

# 保存可调用模型 (单文件: load 后直接 predict)
gnn_a.save(MODEL_DIR / "gnn_a.pt")
gnn_b.save(MODEL_DIR / "gnn_b.pt")

# ============================================================
# 6. 验证: GNN-A 用原始 evaluate_mrr 直接评估 (不经过 GNNModelB)
# ============================================================
print("\n" + "=" * 60)
print("6. 验证: GNN-A 原始 evaluate_mrr (对照基准)")
print("=" * 60)
result_a_direct = gnn_a.evaluate_mrr(valid, n_queries=500)

print("\n" + "=" * 60)
print("GNN-B 假设检验完成!")
print(f"  AUC:  A={auc_a_valid:.4f}  B={auc_b_valid:.4f}  d={auc_b_valid-auc_a_valid:+.4f}")
print(f"  MRR(via GNNModelB):  A={result_b.get('mrr_global_a',0):.5f}  B={result_b['mrr_global']:.5f}")
print(f"        朋友 A={result_b.get('mrr_friend_a',0):.5f} B={result_b['mrr_friend']:.5f}")
print(f"        陌生 A={result_b.get('mrr_stranger_a',0):.5f} B={result_b['mrr_stranger']:.5f}")
print(f"  MRR(GNN-A 原始):     A={result_a_direct['mrr@5']:.5f}  HITS={result_a_direct['hits@5']:.5f}")
print("=" * 60)
