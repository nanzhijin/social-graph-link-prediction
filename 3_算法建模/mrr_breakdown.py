"""
MRR@5 分场景拆解: true_voter 是/不是训练期朋友
==============================================
量化 is_friend 的"双刃剑"效应:
  场景1: true voter 在训练期朋友中 → is_friend 有帮助
  场景2: true voter 不在训练期朋友中 → is_friend 是误导
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(r"D:\GNN\processed")
MODEL_DIR = Path(r"D:\GNN\3_算法建模")

# ============================================================
# 1. 加载模型
# ============================================================
print("1. 加载模型...")
model_a = joblib.load(MODEL_DIR / "lgb_baseline_A.pkl")
model_b = joblib.load(MODEL_DIR / "lgb_baseline_B.pkl")
model_c = joblib.load(MODEL_DIR / "lgb_baseline_C.pkl")

config = pd.read_pickle(PROCESSED_DIR / "feature_config.pkl")
feature_cols = config["feature_cols"]
cat_cols = config["cat_cols"]

EDGE_FRIEND_COLS = ["is_friend", "share_count_a2b", "share_count_b2a",
                    "total_interactions", "last_share_days", "response_rate"]
cols_A = [c for c in feature_cols if c not in EDGE_FRIEND_COLS]
cols_B = list(feature_cols)
cols_C = [c for c in feature_cols if c != "is_friend"]

cat_cols_A = [c for c in cat_cols if c in cols_A]
cat_cols_B = [c for c in cat_cols if c in cols_B]
cat_cols_C = [c for c in cat_cols if c in cols_C]

# ============================================================
# 2. 重建训练图 + 边级特征
# ============================================================
print("2. 重建训练图...")
df_train_raw = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
df_final_train_raw = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
df_all_raw = pd.concat([df_train_raw, df_final_train_raw], ignore_index=True)
df_all_raw["timestamp"] = pd.to_datetime(df_all_raw["timestamp"])

split_date = pd.Timestamp("2022-10-29")
train_raw = df_all_raw[df_all_raw["timestamp"] <= split_date].copy()

# inviter -> 训练期朋友集合
inviter_friends = train_raw.groupby("inviter_id")["voter_id"].apply(set).to_dict()

# 图结构
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

# 边级特征
edge_ab = train_raw.groupby(["inviter_id", "voter_id"]).size().reset_index(name="share_count_a2b")
edge_ba = train_raw.groupby(["voter_id", "inviter_id"]).size().reset_index(name="share_count_b2a")
edge_ba = edge_ba.rename(columns={"voter_id": "inviter_id", "inviter_id": "voter_id"})
df_edge_train = edge_ab.merge(edge_ba, on=["inviter_id", "voter_id"], how="outer").fillna(0)
df_edge_train["share_count_a2b"] = df_edge_train["share_count_a2b"].astype(int)
df_edge_train["share_count_b2a"] = df_edge_train["share_count_b2a"].astype(int)
df_edge_train["total_interactions"] = df_edge_train["share_count_a2b"] + df_edge_train["share_count_b2a"]
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

# 用户画像
df_profile = pd.read_pickle(PROCESSED_DIR / "user_profile_enriched.pkl")
profile_cols = [c for c in df_profile.columns if c != "user_id"]

# 商品信息
item_cols_ref = ["item_id", "cate_id", "cate_level1_id", "brand_id", "shop_id"]
item_info = df_all_raw[item_cols_ref].drop_duplicates(subset="item_id")

# 品类传播力
cate_virality = pd.read_pickle(PROCESSED_DIR / "cate_virality.pkl")

# ============================================================
# 3. 提取查询并分场景
# ============================================================
print("3. 提取查询...")
valid = pd.read_pickle(PROCESSED_DIR / "valid_lgb.pkl")
valid_pos = valid[valid["label"] == 1].copy()
valid_pos["true_is_friend"] = valid_pos.apply(
    lambda r: r["voter_id"] in inviter_friends.get(r["inviter_id"], set()), axis=1)

n_friend = int(valid_pos["true_is_friend"].sum())
n_nofriend = int((~valid_pos["true_is_friend"]).sum())
print(f"   场景1 (true是朋友): {n_friend:,} ({n_friend/len(valid_pos)*100:.1f}%)")
print(f"   场景2 (true非朋友): {n_nofriend:,} ({n_nofriend/len(valid_pos)*100:.1f}%)")

# 分场景采样
rng = np.random.default_rng(42)
N_EACH = 1000
N_RANDOM_PAD = 200

friend_q = valid_pos[valid_pos["true_is_friend"]]
nofriend_q = valid_pos[~valid_pos["true_is_friend"]]

f_idx = rng.choice(len(friend_q), min(N_EACH, len(friend_q)), replace=False)
nf_idx = rng.choice(len(nofriend_q), min(N_EACH, len(nofriend_q)), replace=False)

scenarios = {
    "场景1: true是训练期朋友": friend_q.iloc[f_idx].reset_index(drop=True),
    "场景2: true不是训练期朋友": nofriend_q.iloc[nf_idx].reset_index(drop=True),
}

# ============================================================
# 4. 分场景评估
# ============================================================
print("4. 分场景评估...")

def convert_time_cols(df):
    time_cols = [c for c in df.columns if "first_time" in c or "last_time" in c]
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

def build_candidates(queries_df, inviter_friends, voter_pool, rng, n_random_pad=200):
    """为每个查询构造候选池: 全部训练期朋友 + 随机补充"""
    rows = []
    for i in range(len(queries_df)):
        row = queries_df.iloc[i]
        inv = row["inviter_id"]
        true_v = row["voter_id"]
        friends = inviter_friends.get(inv, set())

        seen = {true_v}
        rows.append({"query_idx": i, "voter_id": true_v, "is_true": 1})
        for fv in friends:
            if fv not in seen:
                seen.add(fv)
                rows.append({"query_idx": i, "voter_id": fv, "is_true": 0})

        n_existing = len(seen)
        random_samples = rng.choice(voter_pool, size=n_random_pad + len(friends),
                                    replace=True)
        for v in random_samples:
            if v not in seen:
                seen.add(v)
                rows.append({"query_idx": i, "voter_id": v, "is_true": 0})
                if len(seen) >= n_existing + n_random_pad:
                    break
    return pd.DataFrame(rows)

def join_features(df_eval, df_profile, profile_cols, item_info, cate_virality,
                  df_edge_train, user_graph_edges, user_neighbors, user_degree, total_edges):
    """特征拼接 (复用 build_train_matrix 逻辑)"""
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

    # 商品
    df_eval = df_eval.merge(item_info, on="item_id", how="left")
    df_eval["cate_virality_score"] = df_eval["cate_level1_id"].map(
        cate_virality["cate_virality_score"]).fillna(0.0)

    # 边级特征
    edge_cols = ["inviter_id", "voter_id", "is_friend", "share_count_a2b",
                 "share_count_b2a", "total_interactions", "last_share_days", "response_rate"]
    df_eval = df_eval.merge(df_edge_train[edge_cols],
                            on=["inviter_id", "voter_id"], how="left")
    for c in ["is_friend", "share_count_a2b", "share_count_b2a",
              "total_interactions", "last_share_days", "response_rate"]:
        df_eval[c] = df_eval[c].fillna(0)

    # 拓扑特征
    real_pairs = df_eval[["inviter_id", "voter_id"]].drop_duplicates()
    real_pairs = real_pairs.merge(user_graph_edges,
                                  on=["inviter_id", "voter_id"], how="inner")
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

    # 缺失值
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

    df_eval = convert_time_cols(df_eval)
    return df_eval

def compute_mrr(model, feature_list, cat_list, eval_df):
    """计算 MRR@5 和 HITS@5"""
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
    eval_df = eval_df.copy()
    eval_df["score"] = scores
    eval_df["rank"] = eval_df.groupby("query_idx")["score"].rank(
        ascending=False, method="first")
    true_mask = eval_df["is_true"] == 1
    ranks = eval_df.loc[true_mask, "rank"]
    mrr = (1.0 / ranks).where(ranks <= 5, 0.0).mean()
    hits = (ranks <= 5).mean()
    # rank 分布
    rank_dist = {
        "rank1": (ranks == 1).mean(),
        "rank2-3": ((ranks >= 2) & (ranks <= 3)).mean(),
        "rank4-5": ((ranks >= 4) & (ranks <= 5)).mean(),
        "rank>5": (ranks > 5).mean(),
    }
    return mrr, hits, rank_dist

results = {}

for scenario_name, queries_df in scenarios.items():
    print(f"\n   {scenario_name} ({len(queries_df)} 查询)...")

    # 构造候选
    df_cand = build_candidates(queries_df, inviter_friends, voter_pool, rng, N_RANDOM_PAD)
    df_eval = df_cand.merge(
        queries_df[["inviter_id", "item_id", "voter_id", "timestamp"]].rename(
            columns={"voter_id": "true_voter_id"}),
        left_on="query_idx", right_index=True, how="left")
    df_eval = join_features(df_eval, df_profile, profile_cols, item_info,
                            cate_virality, df_edge_train, user_graph_edges,
                            user_neighbors, user_degree, total_edges)

    # 三模型评估
    mrr_a, hits_a, dist_a = compute_mrr(model_a, cols_A, cat_cols_A, df_eval)
    mrr_b, hits_b, dist_b = compute_mrr(model_b, cols_B, cat_cols_B, df_eval)
    mrr_c, hits_c, dist_c = compute_mrr(model_c, cols_C, cat_cols_C, df_eval)

    results[scenario_name] = {
        "A": {"mrr": mrr_a, "hits": hits_a, "dist": dist_a},
        "B": {"mrr": mrr_b, "hits": hits_b, "dist": dist_b},
        "C": {"mrr": mrr_c, "hits": hits_c, "dist": dist_c},
    }

# ============================================================
# 5. 打印结果表
# ============================================================
print("\n" + "=" * 75)
print("MRR@5 分场景拆解 — is_friend 的双刃剑效应")
print("=" * 75)

for scenario_name, model_results in results.items():
    print(f"\n{'─' * 75}")
    print(f"  {scenario_name}")
    print(f"{'─' * 75}")
    print(f"  {'模型':>18} {'MRR@5':>8} {'HITS@5':>8} {'Rank1':>7} {'Rank2-3':>8} {'Rank4-5':>8} {'Rank>5':>8}")
    print(f"  {'─' * 65}")
    for model_name in ["A", "B", "C"]:
        r = model_results[model_name]
        print(f"  {'LGB-'+model_name:>18} {r['mrr']:>8.5f} {r['hits']:>8.5f} "
              f"{r['dist']['rank1']:>7.1%} {r['dist']['rank2-3']:>8.1%} "
              f"{r['dist']['rank4-5']:>8.1%} {r['dist']['rank>5']:>8.1%}")

# 汇总对比
print(f"\n{'═' * 75}")
print("  汇总: 两场景的 MRR@5 对比")
print(f"{'═' * 75}")
print(f"  {'模型':>18} {'场景1(是朋友)':>14} {'场景2(非朋友)':>14} {'差距':>10}")
print(f"  {'─' * 58}")
for model_name in ["A", "B", "C"]:
    s1 = results["场景1: true是训练期朋友"][model_name]["mrr"]
    s2 = results["场景2: true不是训练期朋友"][model_name]["mrr"]
    print(f"  {'LGB-'+model_name:>18} {s1:>14.5f} {s2:>14.5f} {s1-s2:>+10.5f}")

# 关键洞察
print(f"\n{'═' * 75}")
print("  关键洞察")
print(f"{'═' * 75}")
b_s1 = results["场景1: true是训练期朋友"]["B"]["mrr"]
b_s2 = results["场景2: true不是训练期朋友"]["B"]["mrr"]
a_s2 = results["场景2: true不是训练期朋友"]["A"]["mrr"]

print(f"  1. B 在场景1(是朋友)的 MRR={b_s1:.4f}, 场景2(非朋友)暴跌到 {b_s2:.4f}")
print(f"     → is_friend 对 27.6% 的新朋友查询是直接误导")
print(f"  2. A 在场景2(非朋友)的 MRR={a_s2:.4f}, 不受 is_friend 干扰")
print(f"     → 用户画像特征覆盖了朋友圈特征无法处理的冷启动场景")
print(f"  3. 整体 MRR = 72.4% × 场景1_MRR + 27.6% × 场景2_MRR")
for model_name in ["A", "B", "C"]:
    s1 = results["场景1: true是训练期朋友"][model_name]["mrr"]
    s2 = results["场景2: true不是训练期朋友"][model_name]["mrr"]
    weighted = 0.724 * s1 + 0.276 * s2
    print(f"     LGB-{model_name}: 0.724×{s1:.4f} + 0.276×{s2:.4f} = {weighted:.4f}")
