"""
================================================================================
标签构建 + 特征矩阵 — 数据预处理 (完整注释版)
================================================================================

做什么：
  把原始分享记录变成 LightGBM 能直接训练的表格。

为什么需要这一步：
  - 原始数据只有"谁分享了什么给谁"，没有"没分享"的记录（负样本）
  - 模型需要同时看到正样本和负样本才能学会区分
  - 每条记录需要拼接上用户画像、商品特征、边特征、图拓扑特征

输入 (processed/ 目录下的文件):
  share_train.pkl, share_final_train.pkl  -- 分享行为记录
  share_final_test.pkl                    -- 竞赛测试集
  user_profile_enriched.pkl               -- 用户画像 (53维)
  edge_friend_features.pkl                -- 朋友圈边级特征
  cate_virality.pkl                       -- 品类社交传播力

输出 (processed/ 目录):
  train_lgb.pkl      -- LightGBM 训练矩阵 (特征列 + label列)
  valid_lgb.pkl      -- LightGBM 验证矩阵
  test_queries.pkl   -- 测试集查询 (只有 inviter, item, timestamp, 等预测)
  feature_config.pkl -- 特征列名分类 (cat_cols, num_cols, feature_cols)

产物用途:
  train_lgb.pkl → X_train, y_train → LightGBM 训练
  valid_lgb.pkl → X_valid, y_valid → LightGBM 验证
  test_queries.pkl → 最终预测时用: 对每个查询构造候选 voter, 喂给模型打分
================================================================================
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 路径常量: 所有中间文件都在 processed/ 下
PROCESSED_DIR = Path(r"D:\GNN\processed")

# 随机种子: 保证每次运行结果一致
# 固定随机种子，就是固定随机序列，用同样的负样本，来保持同样的训练结果，保证复现能力
SEED = 42

# 负采样比例: 每 1 个正样本配 3 个负样本
# 为什么是 3? — 常见做法是 1:1 到 1:5, 3 是保守折中
# LightGBM 可以用 scale_pos_weight 或 is_unbalance 进一步处理
NEG_RATIO = 3


# ═══════════════════════════════════════════════════════════════
# 1. 加载所有数据
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

# share_train:         初赛训练集, 60万行, 时间范围 2021-12 ~ 2022-10
# share_final_train:   决赛训练集, 12万行, 时间范围 2022-10 ~ 2022-12
# share_final_test:    决赛测试集, 只有查询没有答案 (inviter, item, timestamp)
df_train = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
df_final_train = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
df_final_test = pd.read_pickle(PROCESSED_DIR / "share_final_test.pkl")

# user_profile_enriched: 每个用户的 52 维特征 (除去 user_id 列)
# 包含: 基础属性/活跃度/角色分/时间规律/品类偏好/品牌集中度 等
df_profile = pd.read_pickle(PROCESSED_DIR / "user_profile_enriched.pkl")

# edge_friend_features 不再直接从 pkl 加载
# 改从 train_raw 重新计算, 防止验证期信息泄漏到边级特征

# cate_virality: 每个品类在不同用户中被分享的热度分数 (0~1)
cate_virality = pd.read_pickle(PROCESSED_DIR / "cate_virality.pkl")

# 合并初赛+决赛训练数据 → 用于时间切分和特征统计
df_all = pd.concat([df_train, df_final_train], ignore_index=True)
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
df_final_test["timestamp"] = pd.to_datetime(df_final_test["timestamp"])
print(f"  总训练数据: {len(df_all):,} 行")


# ═══════════════════════════════════════════════════════════════
# 2. 标签构建 (Label Construction)
# ═══════════════════════════════════════════════════════════════
# 这一步做三件事:
#   a) 时间切分: 把数据分成训练集和验证集 (按时间, 不是随机)
#   b) 确定正样本: 实际发生的分享 → label=1
#   c) 构造负样本: 对每条正样本, 随机抽 3 个"没回流"的 voter → label=0
#
# 为什么要按时间切分而不是随机切?
#   社交分享行为在时间上是相关的 — 用"过去"预测"未来"
#   随机切会导致模型看到"未来数据"再去预测"过去" → 过拟合假象
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. 标签构建 (向量化负采样)")
print("=" * 60)

# 时间分界线: share_train 和 share_final_train 自然断开于 2022-10-29
# 训练集 = 2022-10-29 之前的数据 (约60万条)
# 验证集 = 2022-10-29 之后的数据 (约12万条)
# 这样做验证才符合真实预测场景: 用历史数据训练, 用未来数据验证
split_date = pd.Timestamp("2022-10-29")
train_raw = df_all[df_all["timestamp"] <= split_date].copy()
valid_raw = df_all[df_all["timestamp"] > split_date].copy()

# —— 正样本: 真实发生过的分享 ——
# 从训练数据中提取 (inviter, item, voter, timestamp) 四元组
# 这些是"标准答案" → label 标为 1
pos_train = train_raw[["inviter_id", "item_id", "voter_id", "timestamp"]].copy()
pos_valid = valid_raw[["inviter_id", "item_id", "voter_id", "timestamp"]].copy()
pos_train["label"] = 1  # 标记为正样本
pos_valid["label"] = 1
print(f"  正样本: train={len(pos_train):,}  valid={len(pos_valid):,}")

# —— 负样本所需的 voter 池 ——
# 只用训练期用户 (train_raw), 防止验证期信息泄漏
# 验证期新用户不在池里 → 但负采样不需要他们 (他们只出现在验证正样本里)
voter_pool = np.unique(np.concatenate([
    train_raw["voter_id"].values, train_raw["inviter_id"].values
]))
print(f"  voter 池: {len(voter_pool):,}")


def fast_neg_sampling(pos_df, pool, neg_ratio, seed_offset, exclude_sets):
    """
    向量化负采样 — 为每条正样本生成 neg_ratio 条负样本

    算法逻辑:
      1. 给 voter 池建一个 {voter_id → 索引} 的字典 (O(1) 查找)
      2. 对每条正样本, 找到"该 (inviter, item) 对有过的所有真实 voter"的池索引
         — 不只是当前查询的那一个, 而是所有发生过回流的 voter
         — 排除列表通常只有 1 个元素 (87.3% 的情况)
      3. 一次性随机生成 N × (neg_ratio + 10) 个随机索引
         — 多生成 10 个是"安全余量", 防止过滤后不够
      4. 对每一行:
         - 过滤掉排除列表中的 ALL 真实 voter (用 np.isin)
         - 这样就避免了"同一商品被A分享给多人时, 把另一个真实voter误标为负样本"
      5. 用 np.repeat 一次性构造最终 DataFrame

    参数:
      pos_df:       正样本 DataFrame
      pool:         全局 voter ID 池 (numpy array)
      neg_ratio:    每条正样本生成几个负样本
      seed_offset:  随机种子偏移量
      exclude_sets: list of numpy arrays —
                    exclude_sets[i] = 第i行需要排除的所有 voter 在 pool 中的索引

    返回:
      负样本 DataFrame (列: inviter_id, item_id, voter_id, timestamp, label=0)
    """
    # 训练用42，valid用43
    rng = np.random.default_rng(SEED + seed_offset)
    n = len(pos_df)               # 正样本数量
    pool_size = len(pool)         # voter 池大小

    # Step 1: 建映射字典 {voter_id字符串 → 数组索引}
    #         这样查找 voter 的索引位置只需要一次字典查询
    voter2idx = {v: i for i, v in enumerate(pool)}

    # Step 2: 构建每行的"排除索引数组"
    #         exclude_idxs[i] = array([索引1, 索引2, ...])
    #         这一行需要排除的所有 voter (该 inviter-item 对的所有真实回流者)
    exclude_idxs = []
    for voters in exclude_sets:
        # 只保留池内存在的 voter: 验证期新用户不在池里, 跳过即可
        # 他们不可能被负采样抽到, 所以不需要排除
        valid_voters = [voter2idx[v] for v in voters if v in voter2idx]
        exclude_idxs.append(np.array(valid_voters, dtype=np.int32))

    # Step 3: 一次性随机采样
    #         samples[i, :] = 第 i 条正样本从池中抽到的候选索引
    samples = rng.integers(0, pool_size, size=(n, neg_ratio + 10), dtype=np.int32)

    # Step 4: 对每一行过滤掉排除列表中的所有真实 voter
    neg_voters = np.empty((n, neg_ratio), dtype=pool.dtype)
    for i in range(n):
        row = samples[i]
        exclude = exclude_idxs[i]                     # 这一行要排除的所有索引
        mask = ~np.isin(row, exclude)                 # 过滤 ALL 真实 voter
        filtered = row[mask]                          # 剩余的候选
        if len(filtered) >= neg_ratio:
            neg_voters[i] = pool[filtered[:neg_ratio]]
        else:
            # 极罕见情况: 过滤后不够 (碰巧多次抽到要排除的人)
            # → 补抽额外候选, 确保足够
            extra = rng.integers(0, pool_size, size=neg_ratio * 2, dtype=np.int32)
            extra_mask = ~np.isin(extra, exclude)
            extra_filtered = extra[extra_mask][:neg_ratio - len(filtered)]
            combined = np.concatenate([filtered, extra_filtered])[:neg_ratio]
            neg_voters[i] = pool[combined]

    # Step 5: 用 np.repeat 一次性构造 DataFrame
    result = pd.DataFrame({
        "inviter_id": np.repeat(pos_df["inviter_id"].values, neg_ratio),
        "item_id":    np.repeat(pos_df["item_id"].values, neg_ratio),
        "voter_id":   neg_voters.ravel(),
        "timestamp":  np.repeat(pos_df["timestamp"].values, neg_ratio),
        "label": 0,
    })
    return result


# —— 为负采样准备排除列表 ——
# 对每个 (inviter, item) 对, 找出 ALL 发生过回流的 voter
# 负采样时会把这些 voter 全部排除, 避免"同一商品被分享给多人"时产生假负样本
# 87.3% 的 (inviter, item) 对只有 1 个 voter → 排除列表只有 1 个人, 跟原来一样
# 12.7% 有多个 voter → 全部排除, 标签更干净
def build_exclude_sets(pos_df):
    pair_voters_df = pos_df.groupby(['inviter_id', 'item_id'])['voter_id'].apply(set).reset_index()
    pair_voters_df.columns = ['inviter_id', 'item_id', 'all_voters']
    merged = pos_df[['inviter_id', 'item_id']].merge(pair_voters_df, on=['inviter_id', 'item_id'], how='left')
    return merged['all_voters']

pos_train_exclude = build_exclude_sets(pos_train)
pos_valid_exclude = build_exclude_sets(pos_valid)

# 生成训练和验证的负样本 (传入排除列表)
neg_train = fast_neg_sampling(pos_train, voter_pool, NEG_RATIO, 0, pos_train_exclude)
neg_valid = fast_neg_sampling(pos_valid, voter_pool, NEG_RATIO, 1, pos_valid_exclude)

# 合并正负样本 → 最终训练表
# 每行是一个 (inviter, item, voter) 组合 + label
# label=1 表示真实回流过, label=0 表示没有回流 (随机负样本)
train_labeled = pd.concat([pos_train, neg_train], ignore_index=True)
valid_labeled = pd.concat([pos_valid, neg_valid], ignore_index=True)
print(f"  train: {len(train_labeled):,} (正{len(pos_train):,} + 负{len(neg_train):,})")
print(f"  valid: {len(valid_labeled):,} (正{len(pos_valid):,} + 负{len(neg_valid):,})")


# ═══════════════════════════════════════════════════════════════
# 3. 特征拼接 (Feature Joining)
# ═══════════════════════════════════════════════════════════════
# 现在每行只有 (inviter_id, item_id, voter_id, timestamp, label) 5列
# 需要把用户画像、商品特征、边特征 都拼到这行上
#
# 拼接逻辑:
#   inviter 侧: 用 inviter_id 去 user_profile 表里查这个人的 52 个特征
#               列名加前缀 "inviter_" → inviter_share_out_count, inviter_role_score ...
#   voter 侧:   同理, 列名加前缀 "voter_"
#   商品侧:     用 item_id 查品类/品牌/店铺, 再用 cate_level1_id 查 virality 分数
#   边级特征:   用 (inviter_id, voter_id) 去 edge_friend_features 查朋友圈关系
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. 特征拼接")
print("=" * 60)

# 用户画像列名 (去掉 user_id 因为不需要作为特征)
profile_cols = [c for c in df_profile.columns if c != "user_id"]

# —— 拼接 inviter 侧用户画像 ——
# 例: user_profile 有列 "share_out_count", join 后变成 "inviter_share_out_count"
inviter_profile = df_profile[["user_id"] + profile_cols].copy()
inviter_profile = inviter_profile.rename(columns={c: f"inviter_{c}" for c in profile_cols})
train_labeled = train_labeled.merge(inviter_profile, left_on="inviter_id", right_on="user_id", how="left")
valid_labeled = valid_labeled.merge(inviter_profile, left_on="inviter_id", right_on="user_id", how="left")
train_labeled.drop(columns=["user_id"], inplace=True)  # user_id 列 join 完了不需要
valid_labeled.drop(columns=["user_id"], inplace=True)

# —— 拼接 voter 侧用户画像 ——
# 同理, 但列名前缀是 "voter_"
voter_profile = df_profile[["user_id"] + profile_cols].copy()
voter_profile = voter_profile.rename(columns={c: f"voter_{c}" for c in profile_cols})
train_labeled = train_labeled.merge(voter_profile, left_on="voter_id", right_on="user_id", how="left")
valid_labeled = valid_labeled.merge(voter_profile, left_on="voter_id", right_on="user_id", how="left")
train_labeled.drop(columns=["user_id"], inplace=True)
valid_labeled.drop(columns=["user_id"], inplace=True)

# —— 拼接商品特征 ——
# item_info: 每个 item 的原始属性 (cate_id, brand_id, shop_id...)
item_cols = ["item_id", "cate_id", "cate_level1_id", "brand_id", "shop_id"]
item_info = df_all[item_cols].drop_duplicates(subset="item_id")
train_labeled = train_labeled.merge(item_info, on="item_id", how="left")
valid_labeled = valid_labeled.merge(item_info, on="item_id", how="left")

# cate_virality_score: 品类级别的社交传播力 (该品类被分享的频繁程度, 0~1)
# 例: 手机壳(cate=16)的 virality 可能很高, 工业零件(cate=91)可能很低
for d in [train_labeled, valid_labeled]:
    d["cate_virality_score"] = d["cate_level1_id"].map(
        cate_virality["cate_virality_score"]
    ).fillna(0.0)  # 没见过的品类填 0

# —— 拼接朋友圈边级特征 ——
# ★ 从 train_raw 实时计算, 不加载预计算的 pkl (防止验证期信息泄漏)
# 边级特征只基于训练期数据:
#   is_friend:       训练期里 A 是否分享过给 B
#   share_count_a2b: 训练期里 A→B 的分享次数
#   share_count_b2a: 训练期里 B→A 的分享次数
#   total_interactions: share_count_a2b + share_count_b2a
#   response_rate:   训练期里 B 回流 A 的比例 (B→A次数 / A→B次数)
#   last_share_days: 训练期里 A→B 最近一次分享距今训练截止日的天数
print("  从训练期数据实时计算边级特征...")
edge_ab = train_raw.groupby(["inviter_id", "voter_id"]).size().reset_index(name="share_count_a2b")
edge_ba = train_raw.groupby(["voter_id", "inviter_id"]).size().reset_index(name="share_count_b2a")
edge_ba = edge_ba.rename(columns={"voter_id": "inviter_id", "inviter_id": "voter_id"})
# 合并双向计数
df_edge_train = edge_ab.merge(edge_ba, on=["inviter_id", "voter_id"], how="outer").fillna(0)
df_edge_train["share_count_a2b"] = df_edge_train["share_count_a2b"].astype(int)
df_edge_train["share_count_b2a"] = df_edge_train["share_count_b2a"].astype(int)
df_edge_train["total_interactions"] = df_edge_train["share_count_a2b"] + df_edge_train["share_count_b2a"]
df_edge_train["is_friend"] = (df_edge_train["share_count_a2b"] > 0).astype(int)
# response_rate: B 对 A 的回流比例
df_edge_train["response_rate"] = np.where(
    df_edge_train["share_count_a2b"] > 0,
    df_edge_train["share_count_b2a"] / df_edge_train["share_count_a2b"],
    0.0
)
# last_share_days: A→B 最近一次分享距训练截止日的天数
last_share = train_raw.groupby(["inviter_id", "voter_id"])["timestamp"].max().reset_index()
last_share["last_share_days"] = (split_date - last_share["timestamp"]).dt.days
df_edge_train = df_edge_train.merge(
    last_share[["inviter_id", "voter_id", "last_share_days"]],
    on=["inviter_id", "voter_id"], how="left"
)
df_edge_train["last_share_days"] = df_edge_train["last_share_days"].fillna(-1)

edge_feat_cols = ["inviter_id", "voter_id", "is_friend", "share_count_a2b",
                  "share_count_b2a", "total_interactions", "last_share_days", "response_rate"]
train_labeled = train_labeled.merge(df_edge_train[edge_feat_cols], on=["inviter_id", "voter_id"], how="left")
valid_labeled = valid_labeled.merge(df_edge_train[edge_feat_cols], on=["inviter_id", "voter_id"], how="left")

# 边级特征缺失值 → 填 0 (验证期新出现的配对没有训练期交互历史)
for d in [train_labeled, valid_labeled]:
    for c in ["is_friend", "share_count_a2b", "share_count_b2a",
              "total_interactions", "last_share_days", "response_rate"]:
        d[c] = d[c].fillna(0)
    # last_share_days -1 sentinel → 没有交互历史
    d["last_share_days"] = d["last_share_days"].replace(-1, -1).fillna(-1)

print(f"  拼接后: train={train_labeled.shape}  valid={valid_labeled.shape}")


# ═══════════════════════════════════════════════════════════════
# 4. 图拓扑特征 (Graph Topology Features)
# ═══════════════════════════════════════════════════════════════
# 拓扑特征衡量两个用户在社交图中的"结构关系"
# 不需要知道他们实际互动过没有, 只看他们在图中的位置
#
# 四个特征:
#   common_neighbors   — A和B的共同邻居数 (共享多少联系人)
#   jaccard           — 共同邻居 / 总邻居 (归一化的重叠度, 0~1)
#   adamic_adar       — 给度数低的共同邻居更高权重
#                        (两个人都认识一个"冷门人物" > 都认识"大佬")
#   pref_attachment   — A的度 × B的度 / 总边数
#                        (度数高的人之间更可能产生新链接)
#
# 为什么只算唯一配对?
#   280万行里很多 (inviter, voter) 会重复出现 (同两个人, 不同商品)
#   拓扑特征只取决于 (A, B) 两个人, 跟商品无关
#   → 先去重, 算 ~20万对, merge 回 280万行, 比逐行算快 10 倍以上
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. 图拓扑特征 (唯一对策略)")
print("=" * 60)

# —— 构建用户-用户图 ——
# 从训练数据里提取所有出现过的 (A→B) 边, 去重
user_graph_edges = train_raw[["inviter_id", "voter_id"]].drop_duplicates()
total_edges = len(user_graph_edges)

# 构建邻接字典: {user_id: 所有跟它有过边的人的集合}
# 分别从 inviter 角度 (谁分享给了谁) 和 voter 角度 (谁被谁分享了)
# 然后合并成完整邻居集合 (双向, 不区分方向)
g_inv = user_graph_edges.groupby("inviter_id")["voter_id"].apply(set).to_dict()
g_vot = user_graph_edges.groupby("voter_id")["inviter_id"].apply(set).to_dict()
all_uids = set(g_inv.keys()) | set(g_vot.keys())
user_neighbors = {u: g_inv.get(u, set()) | g_vot.get(u, set()) for u in all_uids}
user_degree = {u: len(user_neighbors[u]) for u in all_uids}  # 每个用户的度
print(f"  用户节点: {len(all_uids):,}  图边: {total_edges:,}")


def build_pair_topo(pairs, neighbors, degree, tot_e):
    """
    为一批 (inviter, voter) 配对计算 4 个图拓扑特征

    参数:
      pairs:     DataFrame, 列 = [inviter_id, voter_id], 每一行是一对用户
      neighbors: dict, {user_id: set(邻居集合)}
      degree:    dict, {user_id: 邻居数量}
      tot_e:     图中总边数 (用于 pref_attachment 归一化)

    返回:
      DataFrame, 列 = [inviter_id, voter_id, common_neighbors,
                        jaccard, adamic_adar, pref_attachment]
      可以直接 merge 回原表
    """
    cn_list, jac_list, aa_list, pa_list = [], [], [], []
    for _, row in pairs.iterrows():
        a, b = row["inviter_id"], row["voter_id"]

        # 取 A 和 B 各自的邻居集合 (没有邻居就是空集)
        na, nb = neighbors.get(a, set()), neighbors.get(b, set())
        da, db = degree.get(a, 0), degree.get(b, 0)

        # common_neighbors: A和B共有的邻居数量
        cn = len(na & nb)
        cn_list.append(cn)

        # jaccard: 共同邻居数 / 总邻居数
        # 例: A认识{甲,乙,丙}, B认识{乙,丙,丁} → 交集{乙,丙}=2 / 并集{甲,乙,丙,丁}=4 = 0.5
        union = len(na | nb)
        jac_list.append(cn / union if union > 0 else 0.0)

        # adamic_adar: 对每个共同邻居 z, 加 1/log(z的度)
        # 度越低的共同邻居权重越大 — "冷门联系人重合"比"都认识大佬"更有信号
        aa = sum(1.0 / np.log(max(degree.get(z, 1), 2)) for z in (na & nb)) if cn > 0 else 0.0
        aa_list.append(aa)

        # pref_attachment: 度数乘积
        # 度数高的人之间更容易产生新边 ("富人更富")
        pa_list.append(da * db / max(tot_e, 1))

    return pd.DataFrame({
        "inviter_id": pairs["inviter_id"].values,
        "voter_id": pairs["voter_id"].values,
        "common_neighbors": cn_list,
        "jaccard": jac_list,
        "adamic_adar": aa_list,
        "pref_attachment": pa_list,
    })


# 对训练和验证分别处理: 只算图中真实边的拓扑, 其他对填 0
# 为什么只算真实边?
#   负采样引入大量随机 (inviter, voter) 对 → 唯一对数量接近总行数(200万+)
#   但随机配对在图中没有边 → CN/Jaccard/AA 天然为 0
#   所以只需算真实图边(~20万对), 其余直接填 0, 精度不损失, 速度快 10 倍
datasets = {"train": train_labeled, "valid": valid_labeled}
for name in ["train", "valid"]:
    df_lbl = datasets[name]

    # 只对图中真实存在的边算拓扑
    real_pairs = df_lbl[["inviter_id", "voter_id"]].drop_duplicates()
    real_pairs = real_pairs.merge(
        user_graph_edges,
        on=["inviter_id", "voter_id"],
        how="inner"  # ← 只保留图中真实存在的配对
    )
    n_real = len(real_pairs)
    n_unique = df_lbl[["inviter_id", "voter_id"]].drop_duplicates().shape[0]
    print(f"  {name}: {n_real:,} 真实对 / {n_unique:,} 唯一对 "
          f"→ 只算 {n_real/n_unique*100:.0f}%")

    topo = build_pair_topo(real_pairs, user_neighbors, user_degree, total_edges)

    # 删除旧列, merge 新计算结果 (非真实边 = NaN → 后面统一填 0)
    topo_cols = ["inviter_id", "voter_id", "common_neighbors", "jaccard",
                 "adamic_adar", "pref_attachment"]
    for col in ["common_neighbors", "jaccard", "adamic_adar", "pref_attachment"]:
        if col in df_lbl.columns:
            df_lbl.drop(columns=[col], inplace=True)
    df_lbl = df_lbl.merge(topo[topo_cols], on=["inviter_id", "voter_id"], how="left")
    # 非图中边: CN/Jaccard/AA = 0, pref_attachment 从 degree 计算
    for col in ["common_neighbors", "jaccard", "adamic_adar"]:
        df_lbl[col] = df_lbl[col].fillna(0.0)
    # pref_attachment 对非图中边也有值: deg(a)*deg(b)/total_e
    df_lbl["pref_attachment"] = df_lbl["pref_attachment"].fillna(0.0)
    datasets[name] = df_lbl

train_labeled = datasets["train"]
valid_labeled = datasets["valid"]


# ═══════════════════════════════════════════════════════════
# 5. 缺失值处理 (Missing Value Handling)
# ═══════════════════════════════════════════════════════════
# 原则: 不掩盖"缺失"这个信号本身
#  - 数值特征缺失 → 填 0 (表示"没有")
#  - 类别/身份特征缺失 → 填 -1 (表示"未知", 与 0/1 区分开)
#  - 时间相关缺失 → 填 -1 (表示"从未发生过", 与正常值区分)
#
# 为什么不用均值/中位数填充?
#   因为"缺失"本身是一个重要信号:
#     "从未分享过"的人 vs "分享了但成功率很低"的人 → 两种人, 不该混为一谈
#   如果用均值填充, 等于说"你没做过的事, 我当你做过平均水平"→ 信号丢失
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. 缺失值处理")
print("=" * 60)

# 需要填 -1 的列 (sentinel = 缺失本身有含义)
sentinel_cols = {"days_since_last_share", "days_since_last_receive",
                 "days_since_last_activity", "response_latency_days",
                 "gender", "age"}

for df_lbl in [train_labeled, valid_labeled]:
    # inviter 侧和 voter 侧的缺失值分别处理
    for prefix in ["inviter_", "voter_"]:
        for col in profile_cols:
            fc = f"{prefix}{col}"       # 完整列名: inviter_share_out_count
            if fc not in df_lbl.columns:
                continue
            fill_val = -1 if col in sentinel_cols else 0
            df_lbl[fc] = df_lbl[fc].fillna(fill_val)

    # 商品侧特征缺失
    for col in ["cate_id", "cate_level1_id", "brand_id", "shop_id"]:
        if col in df_lbl.columns:
            df_lbl[col] = df_lbl[col].fillna(-1)

    # 传播力/拓扑特征缺失
    for col in ["cate_virality_score", "common_neighbors", "jaccard",
                "adamic_adar", "pref_attachment"]:
        if col in df_lbl.columns:
            df_lbl[col] = df_lbl[col].fillna(0.0)

# 验证: 严禁余留任何 NaN
# 如果这里不是 0, 说明有地方没处理到 → LightGBM 会报错
print(f"  train NaN: {train_labeled.isnull().sum().sum()}")
print(f"  valid NaN: {valid_labeled.isnull().sum().sum()}")


# ═══════════════════════════════════════════════════════════════
# 6. 特征分类 & 保存
# ═══════════════════════════════════════════════════════════════
# 为 LightGBM 区分哪些列是分类特征, 哪些是数值特征
# LightGBM 需要显式告知 categorical_feature 列表
# 它会自动对分类特征做直方图分桶, 不需要手动 One-Hot 编码
#
# 产物:
#   train_lgb.pkl      → 训练用
#   valid_lgb.pkl      → 验证用 (观察是否过拟合)
#   test_queries.pkl   → 最终预测用
#   feature_config.pkl → 特征元信息 (建模脚本直接读取)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. 保存")
print("=" * 60)

# id_cols: 不是特征的列 (不喂给模型)
id_cols = ["inviter_id", "item_id", "voter_id", "timestamp", "label"]

# feature_cols: 所有特征列 = 全部列 - id_cols
feature_cols = [c for c in train_labeled.columns if c not in id_cols]

# cat_cols: LightGBM 需要特殊处理的分类特征
# 包括: 商品ID类, 用户性别/年龄/等级 (虽然数值但本质是类别), 布尔标签
cat_cols = sorted([c for c in feature_cols if c in {
    "cate_id", "cate_level1_id", "brand_id", "shop_id",
    "inviter_gender", "voter_gender", "inviter_age", "voter_age",
    "inviter_level", "voter_level",
    "inviter_is_active_sharer", "voter_is_active_sharer",
    "inviter_is_active_target", "voter_is_active_target",
}])

# num_cols: 其余全是数值特征
num_cols = [c for c in feature_cols if c not in cat_cols]

# 保存特征配置 (后续建模脚本直接读取)
pd.to_pickle({
    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "id_cols": id_cols,
}, PROCESSED_DIR / "feature_config.pkl")

# 保存训练/验证矩阵
train_labeled.to_pickle(PROCESSED_DIR / "train_lgb.pkl")
valid_labeled.to_pickle(PROCESSED_DIR / "valid_lgb.pkl")

# 测试集: 只保留查询信息 (inviter, item, timestamp)
# 不含"真实答案" — 这是竞赛提交用的
df_final_test[["inviter_id", "item_id", "timestamp"]].to_pickle(PROCESSED_DIR / "test_queries.pkl")

print(f"  train_lgb:     {len(train_labeled):,} × {len(train_labeled.columns)} cols")
print(f"  valid_lgb:     {len(valid_labeled):,} × {len(valid_labeled.columns)} cols")
print(f"  test_queries:  {len(df_final_test):,} 查询")
print(f"  features:      {len(feature_cols)}  ({len(cat_cols)} cat + {len(num_cols)} num)")
print("\n" + "=" * 60)
print("完成! 下一步: 导入 LightGBM, 读取 train_lgb.pkl 开始训练")
print("=" * 60)
