"""
用户画像升级：从"事实标签"到"模型标签"
在现有 profile 基础上补充工业级特征（共13组）
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy
from pathlib import Path

print("=" * 60)
print("升级用户画像表 (工业化特征补充)")
print("=" * 60)

PROCESSED_DIR = Path(r"/processed")

# 时间切分: 只用训练期数据算画像, 防止验证期信息泄漏到特征里
# 训练期 = 2022-10-29 之前, 验证期 = 2022-10-29 之后 (竞赛自然分界)
SPLIT_DATE = pd.Timestamp("2022-10-29")
REF_DATE = SPLIT_DATE  # 画像截止于训练期最后一天

# ── 加载 ──
df_profile = pd.read_pickle(PROCESSED_DIR / "user_profile.pkl")
df_train = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
df_final_train = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
df_train_all = pd.concat([df_train, df_final_train], ignore_index=True)
df_train_all["timestamp"] = pd.to_datetime(df_train_all["timestamp"])

# ★ 只保留训练期数据 — 防止数据泄漏
df_train_all = df_train_all[df_train_all["timestamp"] <= SPLIT_DATE]

print(f"基础画像: {df_profile.shape[0]:,} 用户")
print(f"训练期数据 (≤{SPLIT_DATE.date()}): {df_train_all.shape[0]:,} 行")

# ═══════════════════════════════════════════════════════════
# 1. 时间新鲜度（Recency）
#    → 距训练截止日多久没活动了？流失信号
# ═══════════════════════════════════════════════════════════
print("\n[1/13] 计算时间新鲜度...")

last_as_inviter = df_train_all.groupby("inviter_id")["timestamp"].max()
last_as_voter   = df_train_all.groupby("voter_id")["timestamp"].max()

df_profile["days_since_last_share"] = (
    REF_DATE - df_profile["user_id"].map(last_as_inviter)
).dt.days
df_profile["days_since_last_receive"] = (
    REF_DATE - df_profile["user_id"].map(last_as_voter)
).dt.days

# 最近一次活动（不管是分享还是被分享）
df_profile["days_since_last_activity"] = df_profile[
    ["days_since_last_share", "days_since_last_receive"]
].min(axis=1)

# 填充：没有该角色的填 -1 表示"从未"
df_profile["days_since_last_share"]   = df_profile["days_since_last_share"].fillna(-1)
df_profile["days_since_last_receive"] = df_profile["days_since_last_receive"].fillna(-1)

# ═══════════════════════════════════════════════════════════
# 2. 角色分（Role Score）
#    → 0~1 连续值，越接近 1 越是纯分享者，越接近 0 越是纯消费者
# ═══════════════════════════════════════════════════════════
print("[2/13] 计算角色分...")

total_interactions = (
    df_profile["share_out_count"].fillna(0) +
    df_profile["share_in_count"].fillna(0)
)
df_profile["role_score"] = np.where(
    total_interactions > 0,
    df_profile["share_out_count"].fillna(0) / total_interactions,
    0.5  # 无活动用户给 0.5（中性）
)

# ═══════════════════════════════════════════════════════════
# 3. 交互深度（Repeat Ratio）
#    → 是"广撒网"还是"固定搭子"？
# ═══════════════════════════════════════════════════════════
print("[3/13] 计算交互深度...")

out_count = df_profile["share_out_count"].fillna(0)
in_count  = df_profile["share_in_count"].fillna(0)
out_unique = df_profile["share_out_unique_voter"].fillna(0)
in_unique  = df_profile["share_in_unique_inviter"].fillna(0)

# 重复分享率：超过1个不同对象才算
df_profile["out_repeat_ratio"] = np.where(
    out_count > 0,
    (out_count - out_unique) / out_count,
    0.0
)
df_profile["in_repeat_ratio"] = np.where(
    in_count > 0,
    (in_count - in_unique) / in_count,
    0.0
)

# 平均每人分享/被分享次数
df_profile["avg_out_per_voter"] = np.where(
    out_unique > 0, out_count / out_unique, 0.0
)
df_profile["avg_in_per_inviter"] = np.where(
    in_unique > 0, in_count / in_unique, 0.0
)

# ═══════════════════════════════════════════════════════════
# 4. 时间规律（Weekend / Night）
#    → 行为是否有时间偏好？
# ═══════════════════════════════════════════════════════════
print("[4/13] 计算时间规律...")

df_train_all["day_of_week"] = df_train_all["timestamp"].dt.dayofweek
df_train_all["hour"]        = df_train_all["timestamp"].dt.hour
df_train_all["is_weekend"]  = (df_train_all["day_of_week"] >= 5).astype(int)
df_train_all["is_night"]    = ((df_train_all["hour"] >= 22) | (df_train_all["hour"] <= 5)).astype(int)

# 按 inviter 聚合
inviter_time = df_train_all.groupby("inviter_id").agg(
    total_actions=("timestamp", "count"),
    weekend_actions=("is_weekend", "sum"),
    night_actions=("is_night", "sum"),
).fillna(0)

# 按 voter 聚合
voter_time = df_train_all.groupby("voter_id").agg(
    total_actions_in=("timestamp", "count"),
    weekend_actions_in=("is_weekend", "sum"),
    night_actions_in=("is_night", "sum"),
).fillna(0)

# 合并两个角色
user_time = pd.merge(
    inviter_time, voter_time,
    left_index=True, right_index=True, how="outer"
).fillna(0)

user_time["total_all"] = user_time["total_actions"] + user_time["total_actions_in"]
user_time["weekend_all"] = user_time["weekend_actions"] + user_time["weekend_actions_in"]
user_time["night_all"] = user_time["night_actions"] + user_time["night_actions_in"]

user_time["weekend_ratio"] = np.where(
    user_time["total_all"] > 0,
    user_time["weekend_all"] / user_time["total_all"],
    0.0
)
user_time["night_ratio"] = np.where(
    user_time["total_all"] > 0,
    user_time["night_all"] / user_time["total_all"],
    0.0
)

df_profile = df_profile.merge(
    user_time[["weekend_ratio", "night_ratio"]],
    left_on="user_id", right_index=True, how="left"
)
df_profile["weekend_ratio"] = df_profile["weekend_ratio"].fillna(0.0)
df_profile["night_ratio"]   = df_profile["night_ratio"].fillna(0.0)

# ═══════════════════════════════════════════════════════════
# 5. 品类偏好熵（Category Entropy + Concentration）
#    → 兴趣广泛 vs 专注某类？
# ═══════════════════════════════════════════════════════════
print("[5/13] 计算品类偏好...")

# inviter 侧：每个用户分享各品类的次数
cate_dist_out = df_train_all.groupby(["inviter_id", "cate_level1_id"]).size().reset_index(name="count")

def compute_cate_features(df, id_col, count_col):
    """计算每个用户的品类熵和集中度"""
    # 每个用户的品类分布
    user_totals = df.groupby(id_col)[count_col].sum()
    df["ratio"] = df[count_col] / df[id_col].map(user_totals)

    # 熵
    def shannon_entropy(ratios):
        ratios = ratios[ratios > 0]
        if len(ratios) <= 1:
            return 0.0
        return float(entropy(ratios.values, base=np.e))

    entropy_series = df.groupby(id_col)["ratio"].apply(shannon_entropy)

    # 品类数
    cate_count = df.groupby(id_col).size()

    # Top 品类占比
    top_ratio = df.groupby(id_col)["ratio"].max()

    return pd.DataFrame({
        "cate_entropy_out": entropy_series,
        "cate_diversity_out": cate_count,
        "top_cate_ratio_out": top_ratio,
    })

cate_features = compute_cate_features(cate_dist_out, "inviter_id", "count")

df_profile = df_profile.merge(
    cate_features, left_on="user_id", right_index=True, how="left"
)
df_profile["cate_entropy_out"]    = df_profile["cate_entropy_out"].fillna(0.0)
df_profile["cate_diversity_out"]  = df_profile["cate_diversity_out"].fillna(0).astype(int)
df_profile["top_cate_ratio_out"]  = df_profile["top_cate_ratio_out"].fillna(0.0)

# ═══════════════════════════════════════════════════════════
# 6. 商品/品牌多样性
#    → 分享内容广度
# ═══════════════════════════════════════════════════════════
print("[6/13] 计算内容多样性...")

# 用已有的 share_out_unique_item / share_out_count
df_profile["item_diversity_out"] = np.where(
    df_profile["share_out_count"].fillna(0) > 0,
    df_profile["share_out_unique_item"].fillna(0) / df_profile["share_out_count"].fillna(0),
    0.0
)
df_profile["item_diversity_in"] = np.where(
    df_profile["share_in_count"].fillna(0) > 0,
    df_profile["share_in_unique_item"].fillna(0) / df_profile["share_in_count"].fillna(0),
    0.0
)

# ═══════════════════════════════════════════════════════════
# 7. 行为爆发度（Burstiness）
#    → 集中爆发 vs 稳定均匀？
#    Burstiness = (σ - μ) / (σ + μ), ∈ [-1, 1]
#    > 0 = bursty, < 0 = steady
# ═══════════════════════════════════════════════════════════
print("[7/13] 计算行为爆发度...")

# inviter 侧：每日分享数的均值和标准差
daily_out = df_train_all.groupby(["inviter_id", df_train_all["timestamp"].dt.date]).size()
daily_stats = daily_out.groupby("inviter_id").agg(["mean", "std"]).fillna(0)
daily_stats.columns = ["daily_mean", "daily_std"]

daily_stats["burstiness_out"] = np.where(
    (daily_stats["daily_mean"] + daily_stats["daily_std"]) > 0,
    (daily_stats["daily_std"] - daily_stats["daily_mean"]) /
    (daily_stats["daily_std"] + daily_stats["daily_mean"]),
    0.0
)

# voter 侧
daily_in = df_train_all.groupby(["voter_id", df_train_all["timestamp"].dt.date]).size()
daily_stats_in = daily_in.groupby("voter_id").agg(["mean", "std"]).fillna(0)
daily_stats_in.columns = ["daily_mean_in", "daily_std_in"]
daily_stats_in["burstiness_in"] = np.where(
    (daily_stats_in["daily_mean_in"] + daily_stats_in["daily_std_in"]) > 0,
    (daily_stats_in["daily_std_in"] - daily_stats_in["daily_mean_in"]) /
    (daily_stats_in["daily_std_in"] + daily_stats_in["daily_mean_in"]),
    0.0
)

df_profile = df_profile.merge(
    daily_stats[["burstiness_out"]], left_on="user_id", right_index=True, how="left"
)
df_profile = df_profile.merge(
    daily_stats_in[["burstiness_in"]], left_on="user_id", right_index=True, how="left"
)
df_profile["burstiness_out"] = df_profile["burstiness_out"].fillna(0.0)
df_profile["burstiness_in"]  = df_profile["burstiness_in"].fillna(0.0)

# ═══════════════════════════════════════════════════════════
# 8. 分享成功率（Share Success Rate）
#    → 用户发出的分享中，多大比例获得了对方的回流响应？
# ═══════════════════════════════════════════════════════════
print("[8/13] 计算分享成功率...")

edges_fwd = df_train_all[['inviter_id', 'voter_id']].drop_duplicates()
edges_rev = edges_fwd.rename(columns={'inviter_id': 'voter_id', 'voter_id': 'inviter_id'})
reciprocal_edges = edges_fwd.merge(edges_rev, on=['inviter_id', 'voter_id'], how='inner')
reciprocal_edges['is_reciprocal'] = 1
edges_tagged = edges_fwd.merge(
    reciprocal_edges[['inviter_id', 'voter_id', 'is_reciprocal']],
    on=['inviter_id', 'voter_id'], how='left'
)
edges_tagged['is_reciprocal'] = edges_tagged['is_reciprocal'].fillna(0).astype(int)

inviter_success = edges_tagged.groupby('inviter_id')['is_reciprocal'].agg(['sum', 'count'])
inviter_success['share_success_rate'] = inviter_success['sum'] / inviter_success['count']

df_profile = df_profile.merge(
    inviter_success[['share_success_rate']], left_on='user_id', right_index=True, how='left'
)
df_profile['share_success_rate'] = df_profile['share_success_rate'].fillna(0.0)

# ═══════════════════════════════════════════════════════════
# 9. 社交网络增长趋势（Network Growth Rate）
#    → 社交圈在扩张(>1)还是在收缩(<1)？
# ═══════════════════════════════════════════════════════════
print("[9/13] 计算社交网络增长趋势...")

first_interaction = df_train_all.groupby(
    ['inviter_id', 'voter_id']
)['timestamp'].min().reset_index()

midpoint = df_train_all['timestamp'].median()
new_in_first = first_interaction[
    first_interaction['timestamp'] <= midpoint
].groupby('inviter_id').size().rename('first_half_new')
new_in_second = first_interaction[
    first_interaction['timestamp'] > midpoint
].groupby('inviter_id').size().rename('second_half_new')

growth_df = pd.concat([new_in_first, new_in_second], axis=1).fillna(0)
growth_df['network_growth_rate'] = (
    growth_df['second_half_new'] / (growth_df['first_half_new'] + 1)
)

df_profile = df_profile.merge(
    growth_df[['network_growth_rate']], left_on='user_id', right_index=True, how='left'
)
df_profile['network_growth_rate'] = df_profile['network_growth_rate'].fillna(0.0)

# ═══════════════════════════════════════════════════════════
# 10. 活跃度趋势（Activity Trend）
#     → 分享行为在走上坡(>0)还是下坡(<0)？
# ═══════════════════════════════════════════════════════════
print("[10/13] 计算活跃度趋势...")

from scipy.stats import linregress

daily_out = df_train_all.groupby(
    ['inviter_id', df_train_all['timestamp'].dt.date]
).size().reset_index()
daily_out.columns = ['inviter_id', 'date', 'daily_count']

def compute_slope(group):
    if len(group) < 3:
        return 0.0
    x = pd.to_datetime(group['date']).map(pd.Timestamp.toordinal).values
    y = group['daily_count'].values
    slope, _, _, _, _ = linregress(x, y)
    return slope

slopes = daily_out.groupby('inviter_id').apply(compute_slope, include_groups=False).rename('activity_trend')
df_profile = df_profile.merge(
    slopes, left_on='user_id', right_index=True, how='left'
)
df_profile['activity_trend'] = df_profile['activity_trend'].fillna(0.0)

# ═══════════════════════════════════════════════════════════
# 11. 响应延迟（Response Latency）
#     → 从第一次被分享到第一次主动分享，间隔几天？
#       负值 = 先分享再被分享(主动型)，正值 = 先被分享再分享(响应型)
# ═══════════════════════════════════════════════════════════
print("[11/13] 计算响应延迟...")

first_received = df_train_all.groupby('voter_id')['timestamp'].min()
first_sent = df_train_all.groupby('inviter_id')['timestamp'].min()

latency = pd.DataFrame({'first_received': first_received, 'first_sent': first_sent})
latency['response_latency_days'] = (
    (latency['first_sent'] - latency['first_received']).dt.days
)

df_profile = df_profile.merge(
    latency[['response_latency_days']], left_on='user_id', right_index=True, how='left'
)
df_profile['response_latency_days'] = df_profile['response_latency_days'].fillna(-999)

# ═══════════════════════════════════════════════════════════
# 12. 品牌集中度（Brand Concentration）
#     → 用户是否对特定品牌忠诚？（与品类熵正交）
# ═══════════════════════════════════════════════════════════
print("[12/13] 计算品牌集中度...")

brand_dist_out = df_train_all.groupby(
    ['inviter_id', 'brand_id']
).size().reset_index(name='count')

brand_user_totals = brand_dist_out.groupby('inviter_id')['count'].sum()
brand_dist_out['ratio'] = brand_dist_out['count'] / brand_dist_out['inviter_id'].map(brand_user_totals)

def compute_entropy(ratios):
    ratios = ratios[ratios > 0]
    if len(ratios) <= 1:
        return 0.0
    return float(entropy(ratios.values, base=np.e))

brand_entropy = brand_dist_out.groupby('inviter_id')['ratio'].apply(compute_entropy).rename('brand_entropy_out')
brand_diversity = brand_dist_out.groupby('inviter_id').size().rename('brand_diversity_out')
brand_top = brand_dist_out.groupby('inviter_id')['ratio'].max().rename('top_brand_ratio_out')

brand_features = pd.concat([brand_entropy, brand_diversity, brand_top], axis=1)

df_profile = df_profile.merge(brand_features, left_on='user_id', right_index=True, how='left')
df_profile['brand_entropy_out']   = df_profile['brand_entropy_out'].fillna(0.0)
df_profile['brand_diversity_out'] = df_profile['brand_diversity_out'].fillna(0).astype(int)
df_profile['top_brand_ratio_out'] = df_profile['top_brand_ratio_out'].fillna(0.0)

# ═══════════════════════════════════════════════════════════
# 13. 商品社交传播力（Item Social Virality）
#     → 品类级别的分享热度，建模时join到商品侧
# ═══════════════════════════════════════════════════════════
print("[13/13] 计算品类社交传播力...")

cate_share_freq = df_train_all.groupby('cate_level1_id').size().rename('cate_share_freq')
cate_virality_score = (
    (cate_share_freq - cate_share_freq.min()) /
    (cate_share_freq.max() - cate_share_freq.min())
).rename('cate_virality_score')

cate_virality = pd.DataFrame({
    'cate_level1_id': cate_share_freq.index,
    'cate_share_freq': cate_share_freq.values,
    'cate_virality_score': cate_virality_score.values,
}).set_index('cate_level1_id')

cate_virality.to_pickle(PROCESSED_DIR / 'cate_virality.pkl')

# ═══════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════
ENRICHED_PATH = PROCESSED_DIR / "user_profile_enriched.pkl"
df_profile.to_pickle(ENRICHED_PATH)

# 输出摘要
new_cols = [
    "days_since_last_activity", "role_score",
    "out_repeat_ratio", "in_repeat_ratio",
    "avg_out_per_voter", "avg_in_per_inviter",
    "weekend_ratio", "night_ratio",
    "cate_entropy_out", "cate_diversity_out", "top_cate_ratio_out",
    "item_diversity_out", "item_diversity_in",
    "burstiness_out", "burstiness_in",
    "share_success_rate", "network_growth_rate",
    "activity_trend", "response_latency_days",
    "brand_entropy_out", "brand_diversity_out", "top_brand_ratio_out",
]

print(f"\n{'='*60}")
print(f"升级完成: {df_profile.shape[0]:,} 用户 × {df_profile.shape[1]} 列")
print(f"新增 {len(new_cols)} 个特征:")
for c in new_cols:
    if c in df_profile.columns:
        vals = df_profile[c]
        if c == "response_latency_days":
            vals = vals[vals != -999]  # exclude sentinel
        else:
            vals = vals.dropna()
        if len(vals) > 0:
            print(f"  {c:30s}  mean={vals.mean():.4f}  median={vals.median():.4f}  nonzero={(vals!=0).mean()*100:.0f}%")
print(f"{'='*60}")
print(f"已保存至: {ENRICHED_PATH}")
print(f"品类传播力: {PROCESSED_DIR / 'cate_virality.pkl'} ({len(cate_virality)} 品类)")
print(f"总列数: {df_profile.shape[1]}")
