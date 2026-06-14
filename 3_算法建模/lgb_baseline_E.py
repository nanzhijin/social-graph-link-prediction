"""
LightGBM 模型 E — 假设检验: 时序进化(去rank) + 品类交叉 vs 基线 A
==========================================================
A = GROUP_BASE (113维) — 对照组
E = GROUP_BASE + GROUP_TEMPORAL_without_rank (3) + GROUP_CATEGORY (3) = 119维
     ↑ 删掉 pair_last_share_rank (用-1 sentinel在AUC负采样中注水)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(r"D:\GNN\processed")
FIGURES_DIR = Path(r"D:\GNN\figures")
MODEL_DIR = Path(r"D:\GNN\3_算法建模")

# ============================================================
# 1. 加载数据 & 特征组
# ============================================================
print("=" * 60)
print("1. 加载数据 & 特征组")
print("=" * 60)

train = pd.read_pickle(PROCESSED_DIR / "train_lgb.pkl")
valid = pd.read_pickle(PROCESSED_DIR / "valid_lgb.pkl")
config = pd.read_pickle(PROCESSED_DIR / "feature_config.pkl")

groups = config["feature_groups"]
cat_cols = config["cat_cols"]

cols_A = groups["GROUP_BASE"]                                                          # 113 维
cols_E = (groups["GROUP_BASE"]
          + [c for c in groups["GROUP_TEMPORAL"] if c != "pair_last_share_rank"]
          + groups["GROUP_CATEGORY"])                                                   # 119 维

print(f"  A: {len(cols_A)} 维 (GROUP_BASE)")
print(f"  E: {len(cols_E)} 维 (BASE + TEMPORAL[-rank] + CATEGORY)")
new_features_e = [c for c in cols_E if c not in cols_A]
print(f"  新增特征: {new_features_e}")
print(f"  训练集: {len(train):,} 行  |  验证集: {len(valid):,} 行")
print(f"  正样本率: train={train['label'].mean()*100:.1f}%  valid={valid['label'].mean()*100:.1f}%")

# ============================================================
# 2. 准备数据
# ============================================================
print("\n" + "=" * 60)
print("2. 准备数据 (时间戳转换 + 分类特征)")
print("=" * 60)

X_train_A, y_train = train[cols_A], train["label"]
X_train_E = train[cols_E]
X_valid_A, y_valid = valid[cols_A], valid["label"]
X_valid_E = valid[cols_E]

time_cols = [c for c in cols_E if "first_time" in c or "last_time" in c]
print(f"  时间戳列: {len(time_cols)} 个")


def convert_time_cols(df):
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
                except Exception:
                    numeric_vals.append(-1.0)
        df[col] = np.array(numeric_vals, dtype=np.float64)
    return df


X_train_A = convert_time_cols(X_train_A)
X_train_E = convert_time_cols(X_train_E)
X_valid_A = convert_time_cols(X_valid_A)
X_valid_E = convert_time_cols(X_valid_E)

cat_cols_A = [c for c in cat_cols if c in cols_A]
cat_cols_E = [c for c in cat_cols if c in cols_E]

for col in cat_cols_A:
    X_train_A[col] = X_train_A[col].astype("category")
    X_valid_A[col] = X_valid_A[col].astype("category")

for col in cat_cols_E:
    X_train_E[col] = X_train_E[col].astype("category")
    X_valid_E[col] = X_valid_E[col].astype("category")

print(f"  A 分类特征: {len(cat_cols_A)}  |  E 分类特征: {len(cat_cols_E)}")

# ============================================================
# 3. 训练 LightGBM
# ============================================================
print("\n" + "=" * 60)
print("3. 训练 LightGBM A vs E")
print("=" * 60)

base_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "is_unbalance": True,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

print("  训练 LGB-A (GROUP_BASE, 113维)...")
model_a = lgb.LGBMClassifier(**base_params, categorical_feature=cat_cols_A)
model_a.fit(X_train_A, y_train)

print("  训练 LGB-E (BASE+TEMPORAL[-rank]+CATEGORY, 119维)...")
model_e = lgb.LGBMClassifier(**base_params, categorical_feature=cat_cols_E)
model_e.fit(X_train_E, y_train)

# ============================================================
# 4. AUC 评估
# ============================================================
print("\n" + "=" * 60)
print("4. AUC 评估")
print("=" * 60)

proba_a = model_a.predict_proba(X_valid_A)[:, 1]
proba_e = model_e.predict_proba(X_valid_E)[:, 1]

auc_a = roc_auc_score(y_valid, proba_a)
auc_e = roc_auc_score(y_valid, proba_e)

print(f"  LGB-A (113维): AUC = {auc_a:.4f}")
print(f"  LGB-E (119维,-rank): AUC = {auc_e:.4f}")
print(f"  ΔAUC (E - A):  {auc_e - auc_a:+.4f}")

# ============================================================
# 5. 特征重要性 Top-20 (D 模型)
# ============================================================
print("\n" + "=" * 60)
print("5. 特征重要性 Top-20 (模型 E)")
print("=" * 60)

imp_e = pd.DataFrame({
    "feature": cols_E,
    "importance": model_e.feature_importances_,
}).sort_values("importance", ascending=False)

imp_e["is_new"] = imp_e["feature"].isin(new_features_e)

top20 = imp_e.head(20)
print(f"  {'Feature':<35} {'Importance':>10}  {'':>5}")
print(f"  {'-'*50}")
for _, row in top20.iterrows():
    marker = " ★ NEW" if row["is_new"] else ""
    print(f"  {row['feature']:<35} {row['importance']:>10.0f}{marker}")

new_in_top20 = top20["is_new"].sum()
print(f"\n  新特征进入 Top-20: {new_in_top20} / 6")

plt.figure(figsize=(10, 7))
plt.barh(range(len(top20)), top20["importance"].values[::-1],
         color=["#FF6B6B" if v else "#5B9BD5" for v in top20["is_new"].values[::-1]])
plt.yticks(range(len(top20)), top20["feature"].values[::-1], fontsize=9)
plt.xlabel("Importance (split-based)", fontsize=12)
plt.title("LGB-E Top-20 特征重要性 (红色=新特征, 已删rank)", fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "lgb_e_importance.png", dpi=150)
plt.close()
print(f"  已保存: lgb_e_importance.png")

# ============================================================
# 6. MRR@5 分场景评估
# ============================================================
print("\n" + "=" * 60)
print("6. MRR@5 分场景评估 (朋友组 vs 陌生人组)")
print("=" * 60)

# --- 6a. 重建训练期图结构 ---
print("  6a. 重建训练期图结构...")
df_train_raw = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
df_final_train_raw = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
df_all_raw = pd.concat([df_train_raw, df_final_train_raw], ignore_index=True)
df_all_raw["timestamp"] = pd.to_datetime(df_all_raw["timestamp"])

split_date = pd.Timestamp("2022-10-29")
train_raw = df_all_raw[df_all_raw["timestamp"] <= split_date].copy()

inviter_friends = train_raw.groupby("inviter_id")["voter_id"].apply(set).to_dict()
friend_counts = [len(v) for v in inviter_friends.values()]
print(f"  inviter 训练期朋友数: mean={np.mean(friend_counts):.1f}  "
      f"median={np.median(friend_counts):.0f}  max={np.max(friend_counts)}")

# --- 6b. 准备商品/画像/拓扑参考数据 ---
print("  6b. 准备参考数据...")
item_cols_ref = ["item_id", "cate_id", "cate_level1_id", "brand_id", "shop_id"]
item_info = df_all_raw[item_cols_ref].drop_duplicates(subset="item_id")

df_profile = pd.read_pickle(PROCESSED_DIR / "user_profile_enriched.pkl")
profile_cols = [c for c in df_profile.columns if c != "user_id"]

cate_virality = pd.read_pickle(PROCESSED_DIR / "cate_virality.pkl")

user_graph_edges = train_raw[["inviter_id", "voter_id"]].drop_duplicates()
total_edges = len(user_graph_edges)
g_inv = user_graph_edges.groupby("inviter_id")["voter_id"].apply(set).to_dict()
g_vot = user_graph_edges.groupby("voter_id")["inviter_id"].apply(set).to_dict()
all_uids = set(g_inv.keys()) | set(g_vot.keys())
user_neighbors = {u: g_inv.get(u, set()) | g_vot.get(u, set()) for u in all_uids}
user_degree = {u: len(user_neighbors[u]) for u in all_uids}

voter_pool = np.unique(np.concatenate([
    train_raw["voter_id"].values, train_raw["inviter_id"].values
]))

# --- 6c. 边级特征 ---
print("  6c. 边级特征...")
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
    df_edge_train["share_count_b2a"] / df_edge_train["share_count_a2b"], 0.0)
last_share = train_raw.groupby(["inviter_id", "voter_id"])["timestamp"].max().reset_index()
last_share["last_share_days"] = (split_date - last_share["timestamp"]).dt.days
df_edge_train = df_edge_train.merge(
    last_share[["inviter_id", "voter_id", "last_share_days"]],
    on=["inviter_id", "voter_id"], how="left")
df_edge_train["last_share_days"] = df_edge_train["last_share_days"].fillna(-1)

# --- 6d. 时序进化特征 ---
print("  6d. 时序进化特征...")
midpoint = train_raw["timestamp"].median()
first_half = train_raw[train_raw["timestamp"] <= midpoint]
second_half = train_raw[train_raw["timestamp"] > midpoint]

fh_inv_voters = first_half.groupby("inviter_id")["voter_id"].apply(set).to_dict()
sh_inv_voters = second_half.groupby("inviter_id")["voter_id"].apply(set).to_dict()
inv_temporal = {}
for inv in set(fh_inv_voters.keys()) | set(sh_inv_voters.keys()):
    fh_set = fh_inv_voters.get(inv, set())
    sh_set = sh_inv_voters.get(inv, set())
    inv_temporal[inv] = {
        "inviter_new_voter_ratio": len(sh_set - fh_set) / len(sh_set) if len(sh_set) > 0 else 0.0,
        "inviter_voter_retention": len(fh_set & sh_set) / len(fh_set) if len(fh_set) > 0 else 0.0,
    }
df_inv_temporal = pd.DataFrame.from_dict(inv_temporal, orient="index")
df_inv_temporal.index.name = "inviter_id"
df_inv_temporal = df_inv_temporal.reset_index()

sh_pairs = second_half[["inviter_id", "voter_id"]].drop_duplicates()
sh_pairs["pair_is_recent"] = 1

last_share_rank = train_raw.groupby(["inviter_id", "voter_id"])["timestamp"].max().reset_index()
last_share_rank["pair_last_share_rank"] = last_share_rank.groupby("inviter_id")["timestamp"].rank(
    ascending=False, method="dense")

# --- 6e. 品类交叉特征 ---
print("  6e. 品类交叉特征...")
voter_cate_raw = train_raw[["voter_id", "item_id"]].merge(
    train_raw[["item_id", "cate_level1_id"]].drop_duplicates("item_id"),
    on="item_id", how="left")
voter_cate_count = voter_cate_raw.groupby(["voter_id", "cate_level1_id"]).size().reset_index(name="cate_in_count")
voter_total_in = voter_cate_count.groupby("voter_id")["cate_in_count"].sum().reset_index(name="total_in")
voter_cate_count = voter_cate_count.merge(voter_total_in, on="voter_id")
voter_cate_count["cate_match_score"] = voter_cate_count["cate_in_count"] / voter_cate_count["total_in"]

voter_cate_count["cate_rank"] = voter_cate_count.groupby("voter_id")["cate_in_count"].rank(
    ascending=False, method="dense")
voter_top3 = voter_cate_count[voter_cate_count["cate_rank"] <= 3].copy()
voter_top3["item_cate_in_voter_top3"] = 1

inviter_cate_sets = train_raw.groupby("inviter_id")["cate_level1_id"].apply(set).to_dict()
voter_cate_sets = train_raw.groupby("voter_id")["cate_level1_id"].apply(set).to_dict()

# --- 6f. 采样查询 ---
print("  6f. 采样查询...")
valid_pos = valid[valid["label"] == 1][
    ["inviter_id", "item_id", "voter_id", "timestamp"]].copy()
valid_pos = valid_pos.rename(columns={"voter_id": "true_voter_id"})

N_QUERIES = 500
N_RANDOM_PAD = 200
rng = np.random.default_rng(42)

if len(valid_pos) > N_QUERIES:
    q_idx = rng.choice(len(valid_pos), N_QUERIES, replace=False)
    queries = valid_pos.iloc[q_idx].reset_index(drop=True)
else:
    queries = valid_pos.reset_index(drop=True)
n_q = len(queries)
print(f"  采样 {n_q:,} 个查询")

# --- 6g. 构造候选集 + 标记场景 ---
print("  6g. 构造候选集 + 标记场景...")
rows = []
friend_hit_list = []

for i in range(n_q):
    inv = queries.iloc[i]["inviter_id"]
    true_v = queries.iloc[i]["true_voter_id"]

    friends = inviter_friends.get(inv, set())
    friend_hit_list.append(true_v in friends)

    seen = {true_v}
    rows.append({"query_idx": i, "voter_id": true_v, "is_true": 1})

    for fv in friends:
        if fv not in seen:
            seen.add(fv)
            rows.append({"query_idx": i, "voter_id": fv, "is_true": 0})

    n_existing = len(seen)
    random_samples = rng.choice(voter_pool, size=N_RANDOM_PAD + len(friends), replace=True)
    for v in random_samples:
        if v not in seen:
            seen.add(v)
            rows.append({"query_idx": i, "voter_id": v, "is_true": 0})
            if len(seen) >= n_existing + N_RANDOM_PAD:
                break

true_in_friends = sum(friend_hit_list)
print(f"  true_voter 在训练期朋友中: {true_in_friends}/{n_q} = {true_in_friends/n_q*100:.1f}%")
print(f"  true_voter 不在朋友中 (陌生人/冷启动): {n_q-true_in_friends}/{n_q} = {(n_q-true_in_friends)/n_q*100:.1f}%")

df_cand = pd.DataFrame(rows)
df_eval = df_cand.merge(
    queries[["inviter_id", "item_id", "timestamp", "true_voter_id"]],
    left_on="query_idx", right_index=True, how="left")

df_eval["is_friend_scenario"] = df_eval.apply(
    lambda r: r["true_voter_id"] in inviter_friends.get(r["inviter_id"], set()), axis=1)
true_mask = df_eval["is_true"] == 1
friend_scenario_mask = true_mask & df_eval["is_friend_scenario"]
stranger_scenario_mask = true_mask & (~df_eval["is_friend_scenario"])
print(f"  朋友组 query 数: {friend_scenario_mask.sum()}  |  陌生人组: {stranger_scenario_mask.sum()}")

# --- 6h. 特征拼接 ---
print("  6h. 特征拼接...")

inv_p = df_profile[["user_id"] + profile_cols].rename(
    columns={c: f"inviter_{c}" for c in profile_cols})
df_eval = df_eval.merge(inv_p, left_on="inviter_id", right_on="user_id", how="left")
df_eval.drop(columns=["user_id"], inplace=True)

vot_p = df_profile[["user_id"] + profile_cols].rename(
    columns={c: f"voter_{c}" for c in profile_cols})
df_eval = df_eval.merge(vot_p, left_on="voter_id", right_on="user_id", how="left")
df_eval.drop(columns=["user_id"], inplace=True)

df_eval = df_eval.merge(item_info, on="item_id", how="left")
df_eval["cate_virality_score"] = df_eval["cate_level1_id"].map(
    cate_virality["cate_virality_score"]).fillna(0.0)

edge_cols = ["inviter_id", "voter_id", "is_friend", "share_count_a2b",
             "share_count_b2a", "total_interactions", "last_share_days", "response_rate"]
df_eval = df_eval.merge(df_edge_train[edge_cols], on=["inviter_id", "voter_id"], how="left")
for c in ["is_friend", "share_count_a2b", "share_count_b2a",
          "total_interactions", "last_share_days", "response_rate"]:
    df_eval[c] = df_eval[c].fillna(0)

real_pairs = df_eval[["inviter_id", "voter_id"]].drop_duplicates()
real_pairs = real_pairs.merge(user_graph_edges, on=["inviter_id", "voter_id"], how="inner")
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

# 时序进化
df_eval = df_eval.merge(df_inv_temporal, on="inviter_id", how="left")
df_eval["inviter_new_voter_ratio"] = df_eval["inviter_new_voter_ratio"].fillna(0.0)
df_eval["inviter_voter_retention"] = df_eval["inviter_voter_retention"].fillna(0.0)
df_eval = df_eval.merge(sh_pairs, on=["inviter_id", "voter_id"], how="left")
df_eval["pair_is_recent"] = df_eval["pair_is_recent"].fillna(0).astype(int)
df_eval = df_eval.merge(
    last_share_rank[["inviter_id", "voter_id", "pair_last_share_rank"]],
    on=["inviter_id", "voter_id"], how="left")
df_eval["pair_last_share_rank"] = df_eval["pair_last_share_rank"].fillna(-1)

# 品类交叉
df_eval = df_eval.merge(
    voter_cate_count[["voter_id", "cate_level1_id", "cate_match_score"]],
    on=["voter_id", "cate_level1_id"], how="left")
df_eval["cate_match_score"] = df_eval["cate_match_score"].fillna(0.0)
df_eval = df_eval.merge(
    voter_top3[["voter_id", "cate_level1_id", "item_cate_in_voter_top3"]],
    on=["voter_id", "cate_level1_id"], how="left")
df_eval["item_cate_in_voter_top3"] = df_eval["item_cate_in_voter_top3"].fillna(0).astype(int)

unique_pairs_eval = df_eval[["inviter_id", "voter_id"]].drop_duplicates()
overlap_vals = []
for _, row in unique_pairs_eval.iterrows():
    a, b = row["inviter_id"], row["voter_id"]
    sa, sb = inviter_cate_sets.get(a, set()), voter_cate_sets.get(b, set())
    intersection = len(sa & sb)
    union = len(sa | sb)
    overlap_vals.append(intersection / union if union > 0 else 0.0)
cate_overlap_df = pd.DataFrame({
    "inviter_id": unique_pairs_eval["inviter_id"].values,
    "voter_id": unique_pairs_eval["voter_id"].values,
    "inviter_voter_cate_overlap": overlap_vals,
})
if "inviter_voter_cate_overlap" in df_eval.columns:
    df_eval.drop(columns=["inviter_voter_cate_overlap"], inplace=True)
df_eval = df_eval.merge(cate_overlap_df, on=["inviter_id", "voter_id"], how="left")
df_eval["inviter_voter_cate_overlap"] = df_eval["inviter_voter_cate_overlap"].fillna(0.0)

sentinel_cols = {"days_since_last_share", "days_since_last_receive",
                 "days_since_last_activity", "response_latency_days", "gender", "age"}
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

df_eval = convert_time_cols(df_eval)
print(f"  特征矩阵: {df_eval.shape}")

# --- 6i. 打分 + MRR@5 ---
print("  6i. 计算 MRR@5 (A vs E, 分场景)...")


def compute_mrr_with_scenario(model, feature_list, cat_list, eval_df, label):
    feats = [c for c in feature_list if c in eval_df.columns]
    X = eval_df[feats].copy()
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_list]

    cat_in = [c for c in cat_list if c in X.columns]
    for col in cat_in:
        X[col] = X[col].astype("category")

    scores = model.predict_proba(X)[:, 1]
    _df = eval_df.copy()
    _df["score"] = scores
    _df["rank"] = _df.groupby("query_idx")["score"].rank(ascending=False, method="first")

    true_mask = _df["is_true"] == 1
    _df_true = _df[true_mask].copy()
    _df_true["rr"] = np.where(_df_true["rank"] <= 5, 1.0 / _df_true["rank"], 0.0)

    mrr_global = _df_true["rr"].mean()
    hits_global = (_df_true["rank"] <= 5).mean()

    friend_mask = _df_true["is_friend_scenario"] == True
    mrr_friend = _df_true.loc[friend_mask, "rr"].mean() if friend_mask.sum() > 0 else 0.0
    hits_friend = (_df_true.loc[friend_mask, "rank"] <= 5).mean() if friend_mask.sum() > 0 else 0.0

    stranger_mask = _df_true["is_friend_scenario"] == False
    mrr_stranger = _df_true.loc[stranger_mask, "rr"].mean() if stranger_mask.sum() > 0 else 0.0
    hits_stranger = (_df_true.loc[stranger_mask, "rank"] <= 5).mean() if stranger_mask.sum() > 0 else 0.0

    return {
        "global": (mrr_global, hits_global),
        "friend": (mrr_friend, hits_friend, friend_mask.sum()),
        "stranger": (mrr_stranger, hits_stranger, stranger_mask.sum()),
    }


result_a = compute_mrr_with_scenario(model_a, cols_A, cat_cols_A, df_eval, "A")
result_e = compute_mrr_with_scenario(model_e, cols_E, cat_cols_E, df_eval, "E")

print()
print(f"  {'':>18} {'A (113维)':>12} {'E (119维,-rank)':>16} {'Δ':>10}")
print(f"  {'─'*56}")
mrr_a_g, hits_a_g = result_a["global"]
mrr_e_g, hits_e_g = result_e["global"]
print(f"  {'全局 MRR@5':>18} {mrr_a_g:>12.5f} {mrr_e_g:>16.5f} {mrr_e_g-mrr_a_g:>+10.5f}")
print(f"  {'全局 HITS@5':>18} {hits_a_g:>12.5f} {hits_e_g:>16.5f} {hits_e_g-hits_a_g:>+10.5f}")
print(f"  {'─'*56}")
mrr_a_f, hits_a_f, n_f = result_a["friend"]
mrr_e_f, hits_e_f, _ = result_e["friend"]
print(f"  {'朋友组 MRR@5':>18} {mrr_a_f:>12.5f} {mrr_e_f:>16.5f} {mrr_e_f-mrr_a_f:>+10.5f}  (n={n_f})")
mrr_a_s, hits_a_s, n_s = result_a["stranger"]
mrr_e_s, hits_e_s, _ = result_e["stranger"]
print(f"  {'陌生人组 MRR@5':>18} {mrr_a_s:>12.5f} {mrr_e_s:>16.5f} {mrr_e_s-mrr_a_s:>+10.5f}  (n={n_s})")
print(f"  {'─'*56}")
print(f"  {'朋友组 HITS@5':>18} {hits_a_f:>12.5f} {hits_e_f:>16.5f} {hits_e_f-hits_a_f:>+10.5f}")
print(f"  {'陌生人组 HITS@5':>18} {hits_a_s:>12.5f} {hits_e_s:>16.5f} {hits_e_s-hits_a_s:>+10.5f}")

# ============================================================
# 7. 假设检验判定
# ============================================================
print("\n" + "=" * 60)
print("7. 假设检验判定")
print("=" * 60)

delta_auc = auc_e - auc_a
delta_mrr_global = mrr_e_g - mrr_a_g
delta_mrr_stranger = mrr_e_s - mrr_a_s
delta_mrr_friend = mrr_e_f - mrr_a_f

print(f"  ΔAUC (全局):           {delta_auc:+.4f}")
print(f"  ΔMRR@5 (全局):         {delta_mrr_global:+.5f}")
print(f"  ΔMRR@5 (朋友组):       {delta_mrr_friend:+.5f}")
print(f"  ΔMRR@5 (陌生人组):     {delta_mrr_stranger:+.5f}")

judgments = []
if delta_auc > 0:
    judgments.append("✅ AUC 提升: E > A (非rank注水, 真实提升)")
else:
    judgments.append("⚠️ AUC 未提升: 删rank后D的AUC注水被验证")
if delta_mrr_global > 0:
    judgments.append("✅ MRR@5 全局提升: E > A")
else:
    judgments.append("❌ MRR@5 全局未提升: E ≤ A")
if delta_mrr_stranger > 0:
    judgments.append("✅ 陌生人组 MRR 提升: 品类交叉+时间进化(去rank)有真实冷启动信号")
else:
    judgments.append("❌ 陌生人组 MRR 未提升: 新特征未能改善冷启动")
if delta_mrr_stranger > delta_mrr_friend:
    judgments.append("✅ 陌生人组 Δ > 朋友组 Δ: 新特征是泛化信号, 非变相记忆")
else:
    judgments.append("⚠️ 朋友组 Δ ≥ 陌生人组 Δ")

print()
for j in judgments:
    print(f"  {j}")
print(f"\n  新特征进入 Top-20 (E模型): {new_in_top20}/6")
if new_in_top20 >= 2:
    print("  ✅ 多个新特征进入 Top-20")
else:
    print("  ⚠️ 新特征在 Top-20 中占比较少")

# ============================================================
# 8. 保存模型和结果
# ============================================================
print("\n" + "=" * 60)
print("8. 保存模型和结果")
print("=" * 60)

import joblib

joblib.dump(model_a, MODEL_DIR / "lgb_baseline_E_A.pkl")
joblib.dump(model_e, MODEL_DIR / "lgb_baseline_E.pkl")
print(f"  已保存: lgb_baseline_E_A.pkl, lgb_baseline_E.pkl")

results = {
    "auc_a": auc_a, "auc_e": auc_e, "delta_auc": delta_auc,
    "mrr_global_a": mrr_a_g, "mrr_global_e": mrr_e_g, "delta_mrr_global": delta_mrr_global,
    "mrr_friend_a": mrr_a_f, "mrr_friend_e": mrr_e_f, "delta_mrr_friend": delta_mrr_friend,
    "mrr_stranger_a": mrr_a_s, "mrr_stranger_e": mrr_e_s, "delta_mrr_stranger": delta_mrr_stranger,
    "hits_global_a": hits_a_g, "hits_global_e": hits_e_g,
    "hits_friend_a": hits_a_f, "hits_friend_e": hits_e_f,
    "hits_stranger_a": hits_a_s, "hits_stranger_e": hits_e_s,
    "n_features_a": len(cols_A), "n_features_e": len(cols_E),
    "new_features_top20_e": int(new_in_top20),
    "n_queries": n_q, "n_friend_queries": int(n_f), "n_stranger_queries": int(n_s),
}
pd.to_pickle(results, MODEL_DIR / "lgb_baseline_E_results.pkl")

print("\n" + "=" * 60)
print(f"假设检验完成!")
print(f"  AUC:  A={auc_a:.4f}  E={auc_e:.4f}  Δ={delta_auc:+.4f}")
print(f"  MRR:  A={mrr_a_g:.5f}  E={mrr_e_g:.5f}  Δ={delta_mrr_global:+.5f}")
print(f"        朋友组: A={mrr_a_f:.5f} E={mrr_e_f:.5f} Δ={delta_mrr_friend:+.5f}")
print(f"        陌生人: A={mrr_a_s:.5f} E={mrr_e_s:.5f} Δ={delta_mrr_stranger:+.5f}")
print("=" * 60)
