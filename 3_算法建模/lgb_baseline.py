"""
LightGBM 基线建模 — AB 双版本 (含/不含朋友圈边级特征)
========================================================
基线 = 默认参数, 不调参, 只确认数据能学出东西
A 组: 删 6 个边级朋友圈特征 → 113 维
B 组: 全保留 → 119 维
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 非交互后端, 直接存图
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(r"D:\GNN\processed")
FIGURES_DIR = Path(r"D:\GNN\figures")
MODEL_DIR = Path(r"D:\GNN\3_算法建模")

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

train = pd.read_pickle(PROCESSED_DIR / "train_lgb.pkl")
valid = pd.read_pickle(PROCESSED_DIR / "valid_lgb.pkl")
config = pd.read_pickle(PROCESSED_DIR / "feature_config.pkl")

feature_cols = config["feature_cols"]  # 119 特征
cat_cols = config["cat_cols"]          # 14 分类特征

print(f"  训练集: {len(train):,} 行 × {len(feature_cols)} 特征")
print(f"  验证集: {len(valid):,} 行")
print(f"  正样本率: train={train['label'].mean()*100:.1f}%  valid={valid['label'].mean()*100:.1f}%")

# ============================================================
# 2. 定义 AB 两组特征列表
# ============================================================
print("\n" + "=" * 60)
print("2. 构建 AB 特征列表")
print("=" * 60)

# 边级朋友圈特征 — 这 6 列是 AB 测试的实验变量
EDGE_FRIEND_COLS = [
    "is_friend", "share_count_a2b", "share_count_b2a",
    "total_interactions", "last_share_days", "response_rate",
]

# A 组 (基线): 删除边级朋友圈特征
# 注意: 拓扑特征(common_neighbors/jaccard/adamic_adar/pref_attachment) 保留
cols_A = [c for c in feature_cols if c not in EDGE_FRIEND_COLS]

# B 组 (实验): 全保留 119 维
cols_B = list(feature_cols)

# C 组 (实验): 去掉 is_friend(硬开关), 保留其余 5 个连续型边级特征
cols_C = [c for c in feature_cols if c != "is_friend"]

edge_in = [c for c in EDGE_FRIEND_COLS if c in feature_cols]
print(f"  A 组: {len(cols_A)} 维 (删全部6个边级朋友圈特征)")
print(f"  B 组: {len(cols_B)} 维 (全保留, 含 is_friend)")
print(f"  C 组: {len(cols_C)} 维 (删 is_friend, 保留5个连续边级特征)")
print(f"  删除的列: {edge_in}")

# ============================================================
# 3. 准备训练/验证数据
# ============================================================
print("\n" + "=" * 60)
print("3. 准备数据")
print("=" * 60)

X_train_A, y_train = train[cols_A], train["label"]
X_train_B = train[cols_B]
X_train_C = train[cols_C]
X_valid_A, y_valid = valid[cols_A], valid["label"]
X_valid_B = valid[cols_B]
X_valid_C = valid[cols_C]

# 时间戳列 — LightGBM 只接受 int/float/bool
# share_out_first/last 是 datetime64, share_in_first/last 混了 datetime 和 -1 sentinel
# 统一转成 epoch days (float), sentinel 保持 -1
print("\n" + "=" * 60)
print("3b. 转换时间戳列")
print("=" * 60)

time_cols = [c for c in feature_cols if "first_time" in c or "last_time" in c]
print(f"  时间戳列: {len(time_cols)} 个")

def convert_time_cols(df):
    """把时间戳列统一转成 float (epoch ordinal days, sentinel=-1)"""
    for col in time_cols:
        if col not in df.columns:
            continue
        vals = df[col]
        numeric_vals = []
        for v in vals:
            if pd.isna(v) or v == -1 or v == "-1":
                numeric_vals.append(-1.0)
            else:
                try:
                    numeric_vals.append(pd.Timestamp(v).toordinal())
                except:
                    numeric_vals.append(-1.0)
        df[col] = np.array(numeric_vals, dtype=np.float64)
    return df

X_train_A = convert_time_cols(X_train_A)
X_train_B = convert_time_cols(X_train_B)
X_train_C = convert_time_cols(X_train_C)
X_valid_A = convert_time_cols(X_valid_A)
X_valid_B = convert_time_cols(X_valid_B)
X_valid_C = convert_time_cols(X_valid_C)
print(f"  转换完成, 全部转为 float64")
cat_cols_A = [c for c in cat_cols if c in cols_A]
cat_cols_B = [c for c in cat_cols if c in cols_B]
cat_cols_C = [c for c in cat_cols if c in cols_C]

for col in cat_cols_A:
    if col in X_train_A.columns:
        X_train_A[col] = X_train_A[col].astype("category")
        X_valid_A[col] = X_valid_A[col].astype("category")

for col in cat_cols_B:
    if col in X_train_B.columns:
        X_train_B[col] = X_train_B[col].astype("category")
        X_valid_B[col] = X_valid_B[col].astype("category")

for col in cat_cols_C:
    if col in X_train_C.columns:
        X_train_C[col] = X_train_C[col].astype("category")
        X_valid_C[col] = X_valid_C[col].astype("category")

print(f"  A 组分类特征: {len(cat_cols_A)} 个")
print(f"  B 组分类特征: {len(cat_cols_B)} 个")
print(f"  C 组分类特征: {len(cat_cols_C)} 个")

# ============================================================
# 4. 训练 LightGBM 基线 (默认参数)
# ============================================================
print("\n" + "=" * 60)
print("4. 训练 LightGBM 基线")
print("=" * 60)

# 基线参数: 只设必要项, 其余全默认
# is_unbalance=True: 自动处理正负样本不平衡 (正样本 25%)
base_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "is_unbalance": True,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

print("  训练 LGB-A (不含朋友圈)...")
model_a = lgb.LGBMClassifier(**base_params, categorical_feature=cat_cols_A)
model_a.fit(X_train_A, y_train)

print("  训练 LGB-B (含朋友圈, 含 is_friend)...")
model_b = lgb.LGBMClassifier(**base_params, categorical_feature=cat_cols_B)
model_b.fit(X_train_B, y_train)

print("  训练 LGB-C (含朋友圈, 删 is_friend)...")
model_c = lgb.LGBMClassifier(**base_params, categorical_feature=cat_cols_C)
model_c.fit(X_train_C, y_train)

# ============================================================
# 5. 验证集评估
# ============================================================
print("\n" + "=" * 60)
print("5. 验证集评估")
print("=" * 60)

proba_a = model_a.predict_proba(X_valid_A)[:, 1]
proba_b = model_b.predict_proba(X_valid_B)[:, 1]
proba_c = model_c.predict_proba(X_valid_C)[:, 1]

auc_a = roc_auc_score(y_valid, proba_a)
auc_b = roc_auc_score(y_valid, proba_b)
auc_c = roc_auc_score(y_valid, proba_c)

print(f"  LGB-A (不含朋友圈):         AUC = {auc_a:.4f}")
print(f"  LGB-B (含朋友圈+is_friend): AUC = {auc_b:.4f}")
print(f"  LGB-C (含朋友圈, 删is_friend): AUC = {auc_c:.4f}")
print(f"  B vs A 增益: ΔAUC = {auc_b - auc_a:+.4f}")
print(f"  C vs A 增益: ΔAUC = {auc_c - auc_a:+.4f}")

# ============================================================
# 6. 绘制 ROC 曲线
# ============================================================
print("\n" + "=" * 60)
print("6. 绘制 ROC 曲线")
print("=" * 60)

fpr_a, tpr_a, _ = roc_curve(y_valid, proba_a)
fpr_b, tpr_b, _ = roc_curve(y_valid, proba_b)
fpr_c, tpr_c, _ = roc_curve(y_valid, proba_c)

plt.figure(figsize=(8, 6))
plt.plot(fpr_a, tpr_a, label=f"LGB-A (无朋友圈) AUC={auc_a:.4f}", lw=2)
plt.plot(fpr_b, tpr_b, label=f"LGB-B (含is_friend) AUC={auc_b:.4f}", lw=2, alpha=0.6)
plt.plot(fpr_c, tpr_c, label=f"LGB-C (删is_friend) AUC={auc_c:.4f}", lw=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="随机猜测")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("LightGBM Baseline ROC — A/B/C 三组对比", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
roc_path = FIGURES_DIR / "lgb_baseline_roc.png"
plt.savefig(roc_path, dpi=150)
print(f"  已保存: {roc_path}")

# ============================================================
# 7. 特征重要性 Top-20
# ============================================================
print("\n" + "=" * 60)
print("7. 特征重要性 Top-20")
print("=" * 60)

def plot_importance(model, feature_names, title, save_path):
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 7))
    plt.barh(range(len(imp)), imp["importance"].values[::-1])
    plt.yticks(range(len(imp)), imp["feature"].values[::-1], fontsize=9)
    plt.xlabel("Importance (split-based)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return imp

imp_a = plot_importance(model_a, cols_A, "LGB-A Top-20 (无朋友圈)",
                        FIGURES_DIR / "lgb_a_importance.png")
imp_b = plot_importance(model_b, cols_B, "LGB-B Top-20 (含is_friend)",
                        FIGURES_DIR / "lgb_b_importance.png")
imp_c = plot_importance(model_c, cols_C, "LGB-C Top-20 (删is_friend)",
                        FIGURES_DIR / "lgb_c_importance.png")

print(f"  LGB-A Top-5: {list(imp_a['feature'].head(5))}")
print(f"  LGB-B Top-5: {list(imp_b['feature'].head(5))}")
print(f"  LGB-C Top-5: {list(imp_c['feature'].head(5))}")

# ============================================================
# 8. 保存模型
# ============================================================
print("\n" + "=" * 60)
print("8. 保存模型")
print("=" * 60)

import joblib
joblib.dump(model_a, MODEL_DIR / "lgb_baseline_A.pkl")
joblib.dump(model_b, MODEL_DIR / "lgb_baseline_B.pkl")
joblib.dump(model_c, MODEL_DIR / "lgb_baseline_C.pkl")
print(f"  已保存: lgb_baseline_A.pkl, lgb_baseline_B.pkl, lgb_baseline_C.pkl")

# 保存结果摘要
results = {
    "auc_a": auc_a, "auc_b": auc_b, "auc_c": auc_c,
    "delta_b_vs_a": auc_b - auc_a,
    "delta_c_vs_a": auc_c - auc_a,
    "n_features_a": len(cols_A), "n_features_b": len(cols_B), "n_features_c": len(cols_C),
}
pd.to_pickle(results, MODEL_DIR / "lgb_baseline_results.pkl")

print("\n" + "=" * 60)
print(f"AUC 完成! A: {auc_a:.4f} | B: {auc_b:.4f} | C: {auc_c:.4f}")
print(f"  B vs A: {auc_b-auc_a:+.4f}  |  C vs A: {auc_c-auc_a:+.4f}")
print("=" * 60)

# ============================================================
# 9. MRR@5 评估 (竞赛官方指标)
# ============================================================
# 竞赛使用 MRR@5 而非 AUC
# 任务: 给定 (inviter, item, timestamp), 从全体用户中选出 top-5 最可能的 voter
# MRR = mean(1/rank), true voter 排第 n 位得 1/n 分, 不在 top-5 得 0 分
# 官方基线: MRR@5=0.03437  HITS@5=0.09258
# ============================================================
print("\n" + "=" * 60)
print("9. MRR@5 评估 (竞赛官方指标)")
print("=" * 60)
print("  官方基线: MRR@5=0.03437  HITS@5=0.09258")

# --- 9a. 加载参考数据 (重建训练期图结构) ---
print("\n  9a. 加载参考数据...")
df_train_raw = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
df_final_train_raw = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
df_all_raw = pd.concat([df_train_raw, df_final_train_raw], ignore_index=True)
df_all_raw["timestamp"] = pd.to_datetime(df_all_raw["timestamp"])

split_date = pd.Timestamp("2022-10-29")
train_raw = df_all_raw[df_all_raw["timestamp"] <= split_date].copy()

# 商品信息
item_cols_ref = ["item_id", "cate_id", "cate_level1_id", "brand_id", "shop_id"]
item_info = df_all_raw[item_cols_ref].drop_duplicates(subset="item_id")

# 用户画像
df_profile = pd.read_pickle(PROCESSED_DIR / "user_profile_enriched.pkl")
profile_cols = [c for c in df_profile.columns if c != "user_id"]

# 品类传播力
cate_virality = pd.read_pickle(PROCESSED_DIR / "cate_virality.pkl")

# --- 9b. 重建训练期图结构 (拓扑特征用) ---
print("  9b. 重建训练期图结构...")
user_graph_edges = train_raw[["inviter_id", "voter_id"]].drop_duplicates()
total_edges = len(user_graph_edges)
g_inv = user_graph_edges.groupby("inviter_id")["voter_id"].apply(set).to_dict()
g_vot = user_graph_edges.groupby("voter_id")["inviter_id"].apply(set).to_dict()
all_uids = set(g_inv.keys()) | set(g_vot.keys())
user_neighbors = {u: g_inv.get(u, set()) | g_vot.get(u, set()) for u in all_uids}
user_degree = {u: len(user_neighbors[u]) for u in all_uids}
print(f"  训练图: {len(all_uids):,} 节点  {total_edges:,} 边")

# voter 池: 所有训练期出现过的用户
voter_pool = np.unique(np.concatenate([
    train_raw["voter_id"].values, train_raw["inviter_id"].values
]))

# --- 9c. 重建训练期边级特征 ---
print("  9c. 重建边级特征...")
edge_ab = train_raw.groupby(["inviter_id", "voter_id"]).size().reset_index(name="share_count_a2b")
edge_ba = train_raw.groupby(["voter_id", "inviter_id"]).size().reset_index(name="share_count_b2a")
edge_ba = edge_ba.rename(columns={"voter_id": "inviter_id", "inviter_id": "voter_id"})
df_edge_train = edge_ab.merge(edge_ba, on=["inviter_id", "voter_id"], how="outer").fillna(0)
df_edge_train["share_count_a2b"] = df_edge_train["share_count_a2b"].astype(int)
df_edge_train["share_count_b2a"] = df_edge_train["share_count_b2a"].astype(int)
df_edge_train["total_interactions"] = (
    df_edge_train["share_count_a2b"] + df_edge_train["share_count_b2a"])
df_edge_train["is_friend"] = (df_edge_train["share_count_a2b"] > 0).astype(int)
df_edge_train["response_rate"] = np.where(
    df_edge_train["share_count_a2b"] > 0,
    df_edge_train["share_count_b2a"] / df_edge_train["share_count_a2b"],
    0.0)
last_share = train_raw.groupby(["inviter_id", "voter_id"])["timestamp"].max().reset_index()
last_share["last_share_days"] = (split_date - last_share["timestamp"]).dt.days
df_edge_train = df_edge_train.merge(
    last_share[["inviter_id", "voter_id", "last_share_days"]],
    on=["inviter_id", "voter_id"], how="left")
df_edge_train["last_share_days"] = df_edge_train["last_share_days"].fillna(-1)

# --- 9d. 构造 inviter→friends 映射 + 提取验证查询 ---
print("  9d. 准备 inviter→friends 映射...")
# 每个 inviter 在训练期的所有 voter 朋友
inviter_friends = train_raw.groupby("inviter_id")["voter_id"].apply(set).to_dict()
friend_counts = [len(v) for v in inviter_friends.values()]
print(f"  inviter 训练期朋友数: mean={np.mean(friend_counts):.1f}  "
      f"median={np.median(friend_counts):.0f}  max={np.max(friend_counts)}")

valid_pos = valid[valid["label"] == 1][
    ["inviter_id", "item_id", "voter_id", "timestamp"]].copy()
valid_pos = valid_pos.rename(columns={"voter_id": "true_voter_id"})

N_QUERIES = 2000        # 评估查询数
N_RANDOM_PAD = 200      # 在全部朋友外补的随机 voter 数

rng = np.random.default_rng(42)
if len(valid_pos) > N_QUERIES:
    q_idx = rng.choice(len(valid_pos), N_QUERIES, replace=False)
    queries = valid_pos.iloc[q_idx].reset_index(drop=True)
else:
    queries = valid_pos.reset_index(drop=True)
n_q = len(queries)
print(f"  采样 {n_q:,} 个查询")

# --- 9e. 构造候选 voter 集 (inviter 全部训练期朋友 + 随机补充) ---
print("  9e. 构造候选 voter (全部训练期朋友 + 随机补充)...")
# 关键: 把 inviter 的所有训练期朋友都放进候选池
# 这样 is_friend=1 不再稀缺 — 模型必须区分"哪个朋友"而非"是不是朋友"

rows = []
friend_hit_list = []

for i in range(n_q):
    inv = queries.iloc[i]["inviter_id"]
    true_v = queries.iloc[i]["true_voter_id"]

    # 该 inviter 的全部训练期朋友
    friends = inviter_friends.get(inv, set())
    friend_hit_list.append(true_v in friends)

    # 候选 = true_voter + 全部训练期朋友 + 随机补充
    seen = {true_v}
    rows.append({"query_idx": i, "voter_id": true_v, "is_true": 1})

    for fv in friends:
        if fv not in seen:
            seen.add(fv)
            rows.append({"query_idx": i, "voter_id": fv, "is_true": 0})

    # 随机补充到目标数量
    n_existing = len(seen)
    random_samples = rng.choice(voter_pool, size=N_RANDOM_PAD + len(friends),
                                replace=True)
    for v in random_samples:
        if v not in seen:
            seen.add(v)
            rows.append({"query_idx": i, "voter_id": v, "is_true": 0})
            if len(seen) >= n_existing + N_RANDOM_PAD:
                break

true_in_friends = sum(friend_hit_list)
print(f"  true_voter 在训练期朋友中: {true_in_friends}/{n_q} = {true_in_friends/n_q*100:.1f}%")
print(f"  true_voter 不在训练期朋友中: {n_q-true_in_friends}/{n_q} = {(n_q-true_in_friends)/n_q*100:.1f}%")

df_cand = pd.DataFrame(rows)
avg_pool_size = df_cand.groupby("query_idx").size().mean()
print(f"  平均每个查询候选池大小: {avg_pool_size:.0f} "
      f"(含 ~{np.mean(friend_counts):.0f} 朋友 + ~{N_RANDOM_PAD} 随机)")

df_eval = df_cand.merge(
    queries[["inviter_id", "item_id", "timestamp", "true_voter_id"]],
    left_on="query_idx", right_index=True, how="left")
print(f"  候选对总数: {len(df_eval):,}")

# --- 9f. 特征拼接 (复用 build_train_matrix.py 的逻辑) ---
print("  9f. 特征拼接...")

# inviter 画像
inv_p = df_profile[["user_id"] + profile_cols].rename(
    columns={c: f"inviter_{c}" for c in profile_cols})
df_eval = df_eval.merge(inv_p, left_on="inviter_id", right_on="user_id", how="left")
df_eval.drop(columns=["user_id"], inplace=True)

# voter 画像
vot_p = df_profile[["user_id"] + profile_cols].rename(
    columns={c: f"voter_{c}" for c in profile_cols})
df_eval = df_eval.merge(vot_p, left_on="voter_id", right_on="user_id", how="left")
df_eval.drop(columns=["user_id"], inplace=True)

# 商品特征
df_eval = df_eval.merge(item_info, on="item_id", how="left")
df_eval["cate_virality_score"] = df_eval["cate_level1_id"].map(
    cate_virality["cate_virality_score"]).fillna(0.0)

# 边级特征
edge_cols = ["inviter_id", "voter_id", "is_friend", "share_count_a2b",
             "share_count_b2a", "total_interactions", "last_share_days",
             "response_rate"]
df_eval = df_eval.merge(df_edge_train[edge_cols],
                         on=["inviter_id", "voter_id"], how="left")
for c in ["is_friend", "share_count_a2b", "share_count_b2a",
          "total_interactions", "last_share_days", "response_rate"]:
    df_eval[c] = df_eval[c].fillna(0)

# 拓扑特征: 只算训练图中真实存在的配对
real_pairs = df_eval[["inviter_id", "voter_id"]].drop_duplicates()
real_pairs = real_pairs.merge(user_graph_edges,
                               on=["inviter_id", "voter_id"], how="inner")
n_unique = df_eval[["inviter_id", "voter_id"]].drop_duplicates().shape[0]
print(f"  图中配对: {len(real_pairs):,} / {n_unique:,} 唯一对 ({len(real_pairs)/max(n_unique,1)*100:.0f}%)")

cn_vals, jac_vals, aa_vals, pa_vals = [], [], [], []
for _, row in real_pairs.iterrows():
    a, b = row["inviter_id"], row["voter_id"]
    na, nb = user_neighbors.get(a, set()), user_neighbors.get(b, set())
    da, db = user_degree.get(a, 0), user_degree.get(b, 0)
    cn = len(na & nb)
    cn_vals.append(cn)
    union = len(na | nb)
    jac_vals.append(cn / union if union > 0 else 0.0)
    aa_vals.append(sum(1.0 / np.log(max(user_degree.get(z, 1), 2))
                       for z in (na & nb)) if cn > 0 else 0.0)
    pa_vals.append(da * db / max(total_edges, 1))

topo_df = pd.DataFrame({
    "inviter_id": real_pairs["inviter_id"].values,
    "voter_id": real_pairs["voter_id"].values,
    "common_neighbors": cn_vals, "jaccard": jac_vals,
    "adamic_adar": aa_vals, "pref_attachment": pa_vals,
})

for col in ["common_neighbors", "jaccard", "adamic_adar", "pref_attachment"]:
    if col in df_eval.columns:
        df_eval.drop(columns=[col], inplace=True)
df_eval = df_eval.merge(topo_df, on=["inviter_id", "voter_id"], how="left")
for col in ["common_neighbors", "jaccard", "adamic_adar", "pref_attachment"]:
    df_eval[col] = df_eval[col].fillna(0.0)

# 缺失值处理
sentinel_cols = {"days_since_last_share", "days_since_last_receive",
                 "days_since_last_activity", "response_latency_days",
                 "gender", "age"}
for prefix in ["inviter_", "voter_"]:
    for col in profile_cols:
        fc = f"{prefix}{col}"
        if fc not in df_eval.columns:
            continue
        df_eval[fc] = df_eval[fc].fillna(-1 if col in sentinel_cols else 0)

for col in ["cate_id", "cate_level1_id", "brand_id", "shop_id"]:
    if col in df_eval.columns:
        df_eval[col] = df_eval[col].fillna(-1)
for col in ["cate_virality_score"]:
    if col in df_eval.columns:
        df_eval[col] = df_eval[col].fillna(0.0)

# 转换时间戳列
df_eval = convert_time_cols(df_eval)
print(f"  特征矩阵: {df_eval.shape}")

# --- 9g. 三模型打分 + MRR@5 ---
print("  9g. 计算 MRR@5 & HITS@5...")

def compute_mrr(model, feature_list, cat_list, eval_df, label):
    """对候选集打分, 计算 MRR@5 和 HITS@5"""
    feats = [c for c in feature_list if c in eval_df.columns]
    X = eval_df[feats].copy()
    # 补缺失列
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_list]  # 对齐列序

    # 分类特征
    cat_in = [c for c in cat_list if c in X.columns]
    for col in cat_in:
        X[col] = X[col].astype("category")

    scores = model.predict_proba(X)[:, 1]
    eval_df = eval_df.copy()
    eval_df["score"] = scores

    # 按 query_idx 分组排名
    eval_df["rank"] = eval_df.groupby("query_idx")["score"].rank(
        ascending=False, method="first")

    true_mask = eval_df["is_true"] == 1
    ranks = eval_df.loc[true_mask, "rank"]

    mrr = (1.0 / ranks).where(ranks <= 5, 0.0).mean()
    hits = (ranks <= 5).mean()
    return mrr, hits, eval_df

mrr_a, hits_a, _ = compute_mrr(model_a, cols_A, cat_cols_A, df_eval, "A")
mrr_b, hits_b, _ = compute_mrr(model_b, cols_B, cat_cols_B, df_eval, "B")
mrr_c, hits_c, _ = compute_mrr(model_c, cols_C, cat_cols_C, df_eval, "C")

print(f"\n  {'':>12} {'MRR@5':>10} {'HITS@5':>10}")
print(f"  {'─' * 32}")
print(f"  {'官方基线':>12} {0.03437:>10.5f} {0.09258:>10.5f}")
print(f"  {'LGB-A(无朋友圈)':>12} {mrr_a:>10.5f} {hits_a:>10.5f}")
print(f"  {'LGB-B(含is_friend)':>12} {mrr_b:>10.5f} {hits_b:>10.5f}")
print(f"  {'LGB-C(删is_friend)':>12} {mrr_c:>10.5f} {hits_c:>10.5f}")
print(f"\n  MRR 提升 vs 基线: A={mrr_a/0.03437:.1f}x  B={mrr_b/0.03437:.1f}x  C={mrr_c/0.03437:.1f}x")

# 保存 MRR 结果
mrr_results = {
    "mrr_a": mrr_a, "mrr_b": mrr_b, "mrr_c": mrr_c,
    "hits_a": hits_a, "hits_b": hits_b, "hits_c": hits_c,
    "baseline_mrr": 0.03437, "baseline_hits": 0.09258,
    "n_queries": n_q, "n_random_pad": N_RANDOM_PAD,
    "avg_friends": np.mean(friend_counts), "avg_pool_size": avg_pool_size,
}
pd.to_pickle(mrr_results, MODEL_DIR / "lgb_baseline_mrr.pkl")

print("\n" + "=" * 60)
print(f"完成! AUC: A={auc_a:.4f} B={auc_b:.4f} C={auc_c:.4f}")
print(f"      MRR: A={mrr_a:.5f} B={mrr_b:.5f} C={mrr_c:.5f}")
print(f"      官方基线 MRR@5=0.03437  HITS@5=0.09258")
print("=" * 60)
