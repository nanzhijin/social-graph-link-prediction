# 社交图谱动态链接预测

CAAI-BDSC2023 竞赛 Task 2：基于历史分享行为数据，预测未来时间窗口内的用户-商品分享链路。

## 方法概述

**LightGBM + GNN 双路线**，覆盖表格特征和图结构两种数据视角：

| 路线 | 方法 | 核心思路 |
|------|------|---------|
| 路线一 | LightGBM (A→E 五版本) | 113→120 维特征，消融实验 + 假设检验 |
| 路线二 | GNN GraphSAGE (A/B) | 图结构推理，B 版迁移 LGB-E 新特征到图中 |

### 特征组定义 (2026-06-14 新增)

```
GROUP_BASE      (113): 52×2 画像 + 5 商品 + 4 拓扑
GROUP_EDGE      (  6): is_friend, share_count_a2b/b2a, total_interactions,
                        last_share_days, response_rate
GROUP_TEMPORAL  (  4): inviter_new_voter_ratio, inviter_voter_retention,
                        pair_is_recent, pair_last_share_rank
GROUP_CATEGORY  (  3): cate_match_score, item_cate_in_voter_top3,
                        inviter_voter_cate_overlap
```

## 关键结果

### LightGBM 全系列 (统一评估: N_QUERIES=500, 候选池=训练期朋友+随机)

| 模型 | 特征组 | Dim | AUC | MRR@5 | HITS@5 | 说明 |
|------|--------|-----|-----|-------|--------|------|
| 官方基线 | — | — | — | 0.0344 | 0.0926 | 竞赛官方 baseline |
| **LGB-A** | BASE | 113 | **0.8957** | **0.5523** | 0.6100 | 对照组，行为统计 |
| LGB-B | BASE+EDGE | 119 | 0.8606 | 0.4291 | 0.7275 | 边特征反降 AUC |
| LGB-C | BASE+EDGE-is_friend | 118 | 0.8555 | 0.4560 | 0.6930 | 删硬标签 |
| LGB-D | BASE+TEMP+CAT | 120 | 0.9479 | 0.3650 | 0.5680 | ⚠️ rank 注水AUC |
| **LGB-E** ★ | BASE+TEMP[-rank]+CAT | 119 | 0.8840 | 0.4482 | **0.7600** | 去 rank 后真实版 |

### GNN 全系列 (统一评估: N_QUERIES=500, 仅正样本)

| 模型 | 架构 | AUC | MRR@5 | HITS@5 | 说明 |
|------|------|-----|-------|--------|------|
| GNN (无item) | SAGE | 0.9604 | 0.2137 | 0.3600 | 纯图结构 |
| DySAT (无item) | DySAT | 0.9567 | 0.2219 | 0.3700 | 时序注意力 |
| DySAT + Item | DySAT + ItemEncoder | 0.9790 | 0.3111 | 0.5245 | 时序信号弱，-5% vs GNN+Item |
| **GNN-A** | SAGE + ItemEncoder | 0.9853 | **0.3517** | 0.5480 | 原版 GNN 基线 |
| **GNN-B** ★ | SAGE + ItemEncoder + Extra(6d) | **0.9889** | **0.3824** | **0.6100** | 迁移 LGB-E 6 特征 (时序×3 + 品类×3) |

### MRR@5 分场景拆解 (全模型对比)

| 场景 | 占比 | LGB-A | LGB-E | GNN-A | GNN-B | 最优 |
|------|------|:-----:|:-----:|:-----:|:-----:|:----:|
| 全局 | 100% | **0.55** | 0.45 | 0.35 | 0.38 | LGB-A |
| 朋友组 (Seen) | 74.6% | **0.74** | 0.52 | 0.28 | 0.34 | LGB-A |
| 陌生人组 (Unseen) | 25.4% | 0.00 | 0.23 | **0.56** | 0.52 | **GNN-A** |

**核心洞察 (2026-06-14 更新):**

- **GNN-A 陌生人 MRR 0.56** 碾压所有模型（LGB-A 0.00 / LGB-E 0.23 / GNN-B 0.52），HITS@5 0.65 意味着 **65% 的陌生人能被排进前 5**
- LGB-A 朋友 MRR 0.74 是单场景最高记录，is_friend 一招制胜
- **GNN-B 全局 MRR 0.38 (+8.7%)，但提升全部集中在朋友侧 (+20%)，陌生人反而 −7.8%**
- 最优组合不变: **朋友用 LGB-A + 陌生人用 GNN-A → 理论 MRR ≈ 0.69**
- GNN-B 验证了"标量拼 MLP"无法突破陌生人瓶颈，正确的增量方向是边权重消息传递（5/20 设计，尚未实现）

### 优化历史

| 日期 | 实验 | 结论 |
|------|------|------|
| 2026-05 | LGB A/B/C 消融 | 朋友圈特征 (is_friend) 是过滤器非排序器 |
| 2026-06-14 | LGB-D (时序+品类, 含rank) | AUC +0.05 但 MRR 崩塌，rank 在负采样中注水 |
| 2026-06-14 | LGB-E (去 rank) | 陌生人 MRR 0→0.23，HITS 0→0.57，但全局 MRR 未超 A |
| 2026-06-14 | GNN-B (6特征迁移) 云部署 | 全局 MRR 0.38 (+8.7%)，朋友 +20%，陌生人 −7.8% |
| 2026-06-14 | MRR 评估修复 | 发现并修复 label 不过滤负样本导致 MRR 被拖死 10 倍的 bug |

---

## 项目目录结构

```
GNN/
│
├── README.md                          # 本文件
├── LICENSE                            # MIT
├── .gitignore
│
├── info/                              # 📦 原始数据 (6个JSON)
│   ├── user_info.json                 #    115,849 用户 × 4 静态属性
│   ├── item_info.json                 #    438,164 商品 × 5 ID属性
│   ├── item_share_train_info.json     #    602,679 行 初赛训练集
│   ├── item_share_preliminary_test_info.json  # 118,424 行 初赛测试
│   ├── item_share_final_train_info.json       # 118,424 行 决赛训练
│   └── item_share_final_test_info.json        # 115,414 行 决赛测试
│
├── 0_方案设计/                         # 📋 设计文档 & 面试准备
│   ├── 数据流向.md                     #    ★ 完整数据流设计 (本README的详细版)
│   ├── 模型数据维度规格书.md           #    LightGBM 119维 + GNN 图结构 完整规格
│   ├── 阶段性报告.md                   #    最终结果汇总 & 时间线
│   ├── AB实验设计方案（优化）.md       #    AB 测试从"调参工具"到"实验设计书"
│   ├── AB测试设计思路.md               #    AB 测试早期思考
│   ├── 路径设置.md                     #    数据路径配置说明
│   ├── 基于GNN深度学习+传统机器学习方法的双方案电商社交图谱动态链接预测系统.md  # 项目论文
│   ├── SQL面试真题集.md                #    供应链 SQL 面试题
│   ├── SQL窗口函数刷题.md              #    窗口函数专项
│   ├── SQL真题集_填空训练.md           #    填空训练
│   ├── 多多买菜面试.md                 #    多多买菜面试文档
│   ├── 米哈游面试.md                   #    米哈游面试文档
│   ├── 拼多多模拟面试反馈.md           #    模拟面试记录
│   ├── 米哈游模拟面试反馈.md           #    模拟面试记录
│   └── enrich_profile.py               #    用户画像工业化升级 (独立脚本)
│
├── 1_用户画像设计/                     # 🔬 EDA & 特征工程 (Jupyter Notebooks)
│   ├── 1_1数据加载与合并.ipynb         #    JSON → share_merged.pkl (95万行)
│   ├── 1_2探索性数据分析和用户画像前瞻.ipynb  # EDA + 46维用户画像 + 16张图
│   ├── 1_3阅读pkl.ipynb                #    中间产物检查
│   ├── 1_4朋友圈构建与交互可视化.ipynb #    朋友圈字典 + 边级特征 + 双向率
│   └── 用户画像维度体系-北极星指标驱动.md   # 特征设计理念
│
├── 2_标签构造和数据清洗/               # 🏷️ 训练矩阵构建
│   └── build_train_matrix.py           #    ★ 核心预处理脚本 (572行)
│                                       #    时间切分→负采样→特征拼接→拓扑→保存
│
├── 3_算法建模/                         # 🤖 模型训练 & 评估
│   ├── lgb_baseline.py                 #    LightGBM A/B/C 三模型 + MRR@5
│   ├── mrr_breakdown.py                #    分场景 MRR 拆解
│   ├── lgb_baseline_A.pkl              #    A模型 (113维, 无朋友圈)
│   ├── lgb_baseline_B.pkl              #    B模型 (119维, 含is_friend)
│   ├── lgb_baseline_C.pkl              #    C模型 (118维, 删is_friend)
│   ├── lgb_baseline_results.pkl        #    AUC 结果摘要
│   └── lgb_baseline_mrr.pkl            #    MRR 结果摘要
│
├── modules/                            # 🧩 模块化框架 (可替换+交叉组合)
│   ├── __init__.py                     #    导出全部类
│   ├── feature_selector.py            #    4种特征选择器 (223行)
│   │                                   #    Identity / SHAP / PCA / SHAP+PCA
│   ├── experiment.py                   #    Config驱动交叉组合 + AB对比 (121行)
│   └── models/
│       ├── __init__.py                 #    模型导出
│       ├── lgb_model.py               #    LightGBM 标准接口 (82行)
│       ├── gnn_model.py               #    GraphSAGE/GCN/GAT + ItemEncoder (524行)
│       └── dysat_model.py             #    多快照 + TemporalAttention (461行)
│
├── processed/                          # 💾 中间产物 (pkl, 不入git)
│   ├── share_merged.pkl                #    四数据集合并总表 (95万行)
│   ├── share_train.pkl                 #    初赛训练集
│   ├── share_prelim_test.pkl           #    初赛测试集
│   ├── share_final_train.pkl           #    决赛训练集
│   ├── share_final_test.pkl            #    决赛测试集
│   ├── user_profile.pkl                #    基础用户画像 (29维)
│   ├── user_profile_enriched.pkl       #    ★ 最终用户画像 (53维)
│   ├── cate_virality.pkl              #    品类传播力分数
│   ├── friend_circle.pkl              #    朋友圈字典
│   ├── edge_friend_features.pkl       #    边级朋友圈特征
│   ├── feature_config.pkl             #    特征列名分类 (cat/num/id)
│   ├── train_lgb.pkl                  #    ★ LightGBM 训练矩阵 (~240万行)
│   ├── valid_lgb.pkl                  #    ★ LightGBM 验证矩阵 (~48万行)
│   ├── test_queries.pkl               #    测试集查询
│   ├── gnn_diag_state.pkl             #    GNN 诊断状态
│   └── gnn_item_results.pkl           #    GNN+item 结果
│
├── figures/                            # 📊 可视化 (20张图)
│   ├── 1_dataset_distribution.png      #    数据集分布
│   ├── 2_missing_values.png            #    缺失值分析
│   ├── 3_time_analysis.png             #    时间趋势
│   ├── 4_user_static_features.png      #    用户静态特征
│   ├── 5_inviter_vs_voter.png          #    Inviter vs Voter 对比
│   ├── 6_item_features.png             #    商品特征
│   ├── 7_user_behaviors.png            #    用户行为
│   ├── 8_item_popularity.png           #    商品流行度
│   ├── 9_graph_degree_distribution.png #    图度分布
│   ├── 10_pagerank_top20.png           #    PageRank Top-20
│   ├── 11_edge_topology_features.png   #    边拓扑特征
│   ├── 12_user_profile_dashboard.png   #    用户画像仪表盘
│   ├── 13_friend_circle_distribution.png    # 朋友圈分布
│   ├── 14_edge_friend_features.png     #    边级朋友圈特征
│   ├── 15_friend_weight_evidence.png   #    朋友圈加权证据
│   ├── 16_bidirectional_ratio.png      #    双向好友率
│   ├── lgb_baseline_roc.png            #    ROC 曲线 (A/B/C)
│   ├── lgb_a_importance.png            #    LGB-A 特征重要性
│   ├── lgb_b_importance.png            #    LGB-B 特征重要性
│   └── lgb_c_importance.png            #    LGB-C 特征重要性
│
├── 过期可视化图像回收站/               # 🗑️ 开发期测试图 (不关注)
│
├── lib/                                # 📚 前端可视化依赖 (vis-network)
│   ├── vis-9.1.2/
│   └── tom-select/
│
└── 00_技术验证/                         # 🧪 早期技术验证 (时间序列学习)
    └── 时间序列分析/
        ├── 1_创建时间序列.py
        ├── 2_设置日期为索引.py
        ├── 3_重采样.py
        ├── 4_1_滚动计算移动平均.py
        ├── 5_时序数据可视化.py
        ├── 6_时序特征提取.py
        └── 7_股票数据分析实战.py
```

---

## 数据流 Pipeline

```
                              ┌──────────────────────────────────────────┐
                              │            📦 info/ (6个原始JSON)           │
                              │  user_info  │  item_info  │  share × 4    │
                              └──────────────┬───────────────────────────┘
                                             │
                    ╔════════════════════════╧════════════════════════╗
                    ║        阶段 1: 数据加载与合并 (1_1)              ║
                    ║  3次 LEFT JOIN: 分享表 ⋈ inviter ⋈ voter ⋈ item ║
                    ╚════════════════════════╤════════════════════════╝
                                             │
                              ┌──────────────▼──────────────┐
                              │  processed/share_merged.pkl │
                              │  954,941 行 × 16 列          │
                              └──────────────┬──────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
╔══════════════════════════╗  ╔══════════════════════════╗  ╔══════════════════════════╗
║  阶段 2a: EDA + 用户画像  ║  ║  阶段 2b: 朋友圈构建      ║  ║  阶段 2c: 工业化特征升级  ║
║  (1_2 notebook)          ║  ║  (1_4 notebook)          ║  ║  (enrich_profile.py)    ║
║                          ║  ║                          ║  ║                          ║
║  行为聚合 (out/in)       ║  ║  inviter→voters 映射      ║  ║  时间切分 → 无泄漏计算    ║
║  图拓扑 (degree/PR)      ║  ║  双向率分析               ║  ║  +15个连续值特征          ║
║  品类偏好统计            ║  ║  边级特征计算             ║  ║  (角色分/爆发度/响应延迟)  ║
╚═══════════╤══════════════╝  ╚═══════════╤══════════════╝  ╚═══════════╤══════════════╝
            │                             │                             │
            ▼                             ▼                             ▼
┌───────────────────────┐   ┌────────────────────────┐   ┌──────────────────────────────┐
│ user_profile.pkl      │   │ friend_circle.pkl       │   │ user_profile_enriched.pkl ★  │
│ 111,770 用户 × 29 维  │   │ edge_friend_features    │   │ 111,770 用户 × 53 维         │
└───────────────────────┘   └────────────────────────┘   │ cate_virality.pkl            │
                                                         └──────────────┬───────────────┘
                                                                        │
                    ╔═══════════════════════════════════════════════════╧══════════════════╗
                    ║                    阶段 3: 训练矩阵构建 (build_train_matrix.py)          ║
                    ║                                                                        ║
                    ║  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────┐  ║
                    ║  │ 时间切分     │   │ 负采样        │   │ 特征拼接     │   │ 图拓扑     │  ║
                    ║  │             │   │              │   │             │   │           │  ║
                    ║  │ 2022-10-29  │ → │ 向量化 1:3   │ → │ inviter画像  │ → │ CN/Jaccard│  ║
                    ║  │ 前→训练     │   │ 排除全部真   │   │ +voter画像   │   │ AA/PA     │  ║
                    ║  │ 后→验证     │   │ 实voter      │   │ +商品+边特征 │   │ 唯一对策略 │  ║
                    ║  └─────────────┘   └──────────────┘   └─────────────┘   └───────────┘  ║
                    ╚══════════════════════════════════════════╤════════════════════════════╝
                                                               │
                              ┌────────────────────────────────┼────────────────────────────┐
                              │                                │                            │
                              ▼                                ▼                            ▼
              ┌───────────────────────┐    ┌───────────────────────────┐    ┌──────────────────┐
              │ train_lgb.pkl        │    │ valid_lgb.pkl             │    │ feature_config   │
              │ ~240万行 × ~124列     │    │ ~48万行 × ~124列          │    │ .pkl             │
              │ (75% 训练)            │    │ (25% 验证, 未来时间)       │    │ cat/num/id 分类  │
              └───────────┬───────────┘    └───────────┬───────────────┘    └──────────────────┘
                          │                            │
        ┌─────────────────┴────────────────────────────┴─────────────────┐
        │                                                                 │
        ▼                                                                 ▼
╔═══════════════════════════════════╗          ╔══════════════════════════════════════╗
║  路线一: LightGBM                 ║          ║  路线二: GNN (GraphSAGE) + DySAT     ║
║  (lgb_baseline.py)               ║          ║  (modules/models/)                  ║
║                                   ║          ║                                      ║
║  ┌─────────────────────────────┐  ║          ║  ┌────────────────────────────────┐  ║
║  │ A/B/C 三组消融实验           │  ║          ║  │ 图构建                         │  ║
║  │ A: 113维 (删全部边特征)      │  ║          ║  │ 109K节点 + 603K有向边           │  ║
║  │ B: 119维 (全保留)            │  ║          ║  │ 节点特征: user_profile 53维     │  ║
║  │ C: 118维 (单独删is_friend)   │  ║          ║  │                                │  ║
║  └─────────────────────────────┘  ║          ║  │ ┌──────────┐   ┌─────────────┐  │  ║
║                                   ║          ║  │ │ GNNModel │   │ DySATModel  │  │  ║
║  ┌─────────────────────────────┐  ║          ║  │ │          │   │             │  │  ║
║  │ 评估: AUC + MRR@5            │  ║          ║  │ │ 单图     │   │ 2快照累积图  │  │  ║
║  │ MRR: 候选池=好友+200随机     │  ║          ║  │ │ GraphSAGE│   │ +TemporalAttn│  │  ║
║  └─────────────────────────────┘  ║          ║  │ │ 2层64维  │   │ (MHA 4头)   │  │  ║
║                                   ║          ║  │ └──────────┘   └─────────────┘  │  ║
║  输出:                            ║          ║  │        │               │          │  ║
║  ├─ lgb_baseline_{A,B,C}.pkl     ║          ║  │        └───────┬───────┘          │  ║
║  ├─ lgb_baseline_results.pkl     ║          ║  │                │                  │  ║
║  ├─ lgb_baseline_mrr.pkl         ║          ║  │         ItemEncoder                │  ║
║  └─ figures/lgb_*.png            ║          ║  │   (cate+brand+shop Emb→32维)       │  ║
║                                   ║          ║  │                │                  │  ║
╚═══════════╤═══════════════════════╝          ║  └────────────────┼──────────────────┘  ║
            │                                  ║                   │                     ║
            │                                  ║           LinkPredictor                 ║
            │                                  ║    concat(emb_u, emb_v, item_emb)       ║
            │                                  ║         → MLP → score                  ║
            │                                  ╚═══════════════════╤═════════════════════╝
            │                                                      │
            └──────────────────────────┬───────────────────────────┘
                                       │
                                       ▼
                         ╔═════════════════════════════╗
                         ║  阶段 5: 评估 & 分场景拆解   ║
                         ║                             ║
                         ║  AUC: 分类 vs 随机?          ║
                         ║  MRR@5: 排序能力 (竞赛指标)   ║
                         ║  场景1: 好友召回 (72.4%)     ║
                         ║  场景2: 冷启动 (27.6%)       ║
                         ╚═════════════════════════════╝
```

### Pipeline 阶段速查

| 阶段 | 文件 | 输入 | 输出 | 核心操作 |
|------|------|------|------|----------|
| **1. 加载** | `1_1数据加载与合并.ipynb` | 6个JSON | `share_merged.pkl` (95万行) | 3次 LEFT JOIN |
| **2a. EDA** | `1_2...ipynb` | share_merged | 16张图 + `user_profile.pkl` (29维) | 行为聚合/图拓扑/品类偏好 |
| **2b. 朋友圈** | `1_4...ipynb` | share_train | `friend_circle.pkl` + `edge_friend_features.pkl` | 朋友圈字典/双向率 |
| **2c. 画像升级** | `enrich_profile.py` | 训练期数据 | `user_profile_enriched.pkl` (53维) + `cate_virality.pkl` | +15连续特征, 时间切分防泄漏 |
| **3. 训练矩阵** | `build_train_matrix.py` | 上述全部pkl | `train_lgb.pkl` (240万行) + `valid_lgb.pkl` (48万行) + `feature_config.pkl` | 时间切分→负采样1:3→特征拼接→图拓扑→缺失值 |
| **4a. LGB建模** | `lgb_baseline.py` | train/valid_lgb | 3个模型pkl + ROC/重要性图 + MRR | A/B/C 消融, AUC+MRR@5 |
| **4b. GNN建模** | `gnn_model.py` | 图结构+user_profile | GNN节点嵌入 + 预测分数 | GraphSAGE+ItemEncoder+LinkPredictor |
| **4c. DySAT** | `dysat_model.py` | 2快照图 | 时序聚合嵌入 | Multi-head Self-Attention跨时间步 |
| **5. 评估** | `mrr_breakdown.py` | 模型预测 | 分场景MRR表 | 按true_voter是否训练期好友拆分 |

### 模块化框架接口

`modules/` 下的所有模型和选择器共享统一接口，支持 `ExperimentRunner` 交叉组合：

```
FeatureSelector          Model (任一)
├─ fit(X,y)              ├─ fit(X,y)
├─ transform(X)          ├─ predict_proba(X)
├─ fit_transform(X,y)    ├─ evaluate(X,y) → {"auc", "pos_rate_pred"}
└─ selected_features     └─ evaluate_mrr(valid_df) → {"mrr@5", "hits@5"}
```

`ExperimentRunner(configs)` 自动完成 `选择器→模型→评估` 的完整流水线，产出对比表。

---

## 核心发现

1. **AUC ≠ MRR**：GNN AUC 0.98+ 但 MRR 0.35。图结构擅长分类不擅长排序。BCE loss 优化链接存在性，不优化候选人相对排序。
2. **LGB + GNN 场景互补**：LGB-A 对老朋友一招制胜（MRR 0.74），GNN-A 对陌生人冷启动碾压（MRR 0.56, HITS 65%）。双模型组合理论可达 MRR 0.69。
3. **标量拼 MLP 对陌生人无效**：GNN-B 的 6 特征（时序+品类）只帮了朋友（+20%），陌生人反而退步（−8%）。extra 占 166 维的 3.6% 被 embedding 淹没。
4. **边权重消息传递是正确方向**：朋友圈信号应该作为 SAGEConv 的 edge_weight 做加权传播（5/20 设计），而非拼进分类器。该方案至今未实现。
5. **DySAT 时序信号极弱**：快照间距仅38天，71%用户度数无变化，Temporal Attention 基本负优化。
6. **MRR 评估必须过滤负样本**：valid_lgb 75% 为负样，不过滤会导致 MRR 被拖死 10 倍（0.08 vs 0.35）。

---

## 模型部署 (NaNaGi 可调用)

### 模型文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `3_算法建模/gnn_a.pt` | 49MB | GNN-A 完整模型 (陌生人专家, MRR 0.56) |
| `3_算法建模/gnn_b.pt` | 49MB | GNN-B 完整模型 (全局门卫, AUC 0.989) |
| `3_算法建模/lgb_baseline_A.pkl` | ~2MB | LGB-A 完整模型 (朋友专家, MRR 0.74) |

### 调用方式

**GNN 模型：**
```python
from modules.models.gnn_model import GNNModel
model = GNNModel.load("3_算法建模/gnn_a.pt")
scores = model.predict_proba(df)  # 需要 inviter_id, voter_id, cate_id, brand_id, shop_id, cate_virality_score
```

**LGB 模型：**
```python
import joblib
model = joblib.load("3_算法建模/lgb_baseline_A.pkl")
# 用模型自带的特征名取列 + 时间列转数字 + 分类列标 category
scores = model.predict_proba(X)[:, 1]
```

### 三模型联合预测验证 (10条随机抽样)

```
label  LGB-A   GNN-A   GNN-B
  0    0.0000  0.0000  0.0000
  0    0.0000  0.0000  0.0000
  0    0.0000  0.0002  0.0000
  1    0.0000  1.0000  1.0000   ← GNN双命中，LGB-A 漏了
  0    0.0000  0.0467  0.0014
  0    0.0000  0.0000  0.0000
  0    0.0000  0.0800  0.0002
  1    0.9999  0.9986  0.9593   ← 三模型全命中
  0    0.0000  0.0210  0.0005
  0    0.0000  0.0000  0.0000
```

**关键发现：**
- 第4行：GNN-A/GNN-B 双命中正样本，LGB-A 漏了 → **GNN 对陌生人冷启动的互补价值**
- 第8行：三模型全命中，高置信度一致 → 朋友场景三模型协同
- GNN-B 对负样本压制最狠（均值 0.0003 vs GNN-A 0.0148 vs LGB-A 0.0000）→ **GNN-B 的粗筛/门卫价值**
- LGB-A 对负样本同样压得接近零 → 适合朋友入口的精排

---

## 上线娜娜吉注意事项

### 数据预处理（必读）

LGB-A 模型调用前必须做三步预处理——

**LGB-A 推理包装函数：**
```python
import joblib, pandas as pd, numpy as np

def predict_lgb_a(df, model_path="3_算法建模/lgb_baseline_A.pkl"):
    model = joblib.load(model_path)
    
    # 1. 只用模型训练时的特征列
    X = df[model.feature_name_].copy()
    
    # 2. 时间列转 ordinal 数字
    for c in X.columns:
        if 'first_time' in c or 'last_time' in c:
            vals = []
            for v in X[c]:
                if pd.isna(v) or v == -1 or v == '-1' or v == 0:
                    vals.append(-1.0)
                else:
                    try: vals.append(float(pd.Timestamp(v).toordinal()))
                    except: vals.append(-1.0)
            X[c] = np.array(vals, dtype=np.float64)
    
    # 3. 分类列标 category，数值列填 0
    cat_feat = model.get_params().get('categorical_feature', [])
    for col in X.columns:
        if col in cat_feat:
            X[col] = X[col].fillna(-1).astype(int).astype('category')
        else:
            X[col] = X[col].fillna(0).astype(np.float64)
    
    return model.predict_proba(X)[:, 1]
```

**GNN 推理包装函数：**
```python
from modules.models.gnn_model import GNNModel

def predict_gnn(df, model_path):
    model = GNNModel.load(model_path)
    cols = ['inviter_id', 'voter_id', 'cate_id', 'brand_id', 'shop_id', 'cate_virality_score']
    return model.predict_proba(df[cols])
```

### 三模型分工（生产环境调用顺序）

```
1. GNN-B 先验过滤: p_link < θ → 直接丢弃（省算力）
2. 业务路由:
   - 朋友入口 → LGB-A 精排
   - 陌生人入口 → GNN-A 精排
```

### 已知坑

| 坑 | 影响 | 解决 |
|---|---|---|
| LGB categorical dtype 不匹配 | sklearn wrapper 报错 | 用 `model.feature_name_` + category dtype |
| 时间列为 datetime/object | LGB 不吃非数值型 | `toordinal()` 转换 |
| 特征列数与训练不一致 | 数据重跑后列数可能变 | 永远用 `model.feature_name_`，不自己推断 |
| GNN 需要 item 特征列 | predict 时缺列会报错 | 必须传入 `cate_id, brand_id, shop_id, cate_virality_score` |

---

## 上线路线图 (NaNaGi 落地)

### 分业务场景路由架构

产品层两个入口天然分流，不需要额外学路由器：

```
朋友推荐入口                        陌生人探索入口
candidates = 2-hop + 同校/同司       candidates = 兴趣/活动/跨社群
    ↓                                      ↓
  LGB-A 精排                            GNN-A 精排
  (朋友 MRR 0.74)                       (陌生人 MRR 0.56)
    ↓                                      ↓
          ┌──────────────────────────────┐
          │    GNN-B Prior Filter (共享)  │
          │    全局 p_link 截断/校准      │
          └──────────────────────────────┘
```

### GNN-B = 全局先验过滤器（Prior Filter）

GNN-B 的 AUC 0.989 不参与精排竞争，下沉为两路线共享的门卫：

```
全量候选 → GNN-B p_link(u,v) → 扔掉 p_link < θ 的 → 精排模型处理 → Top-5
```

| 角色 | 负责模型 | 核心指标 |
|------|:--------:|:--------:|
| 朋友精排 | LGB-A | 朋友 MRR 0.74, HITS@5 0.89 |
| 陌生人精排 | GNN-A | 陌生人 MRR 0.56, HITS@5 0.65 |
| 全局粗筛/门卫 | GNN-B | AUC 0.989, 全局 MRR 0.38 |

### 审计清单 (来自外部评审)

#### ✅ 已做对

| 维度 | 说明 |
|------|------|
| 时间切分 | split_date=2022-10-29，训练图不含验证边 → random-edge-removal 会虚高 10-30 点 |
| candidate 协议 | friends + 200 random + label==1 过滤 → 比 global random-pair AUC 更贴近真实 |
| 分场景建模 | 朋友/陌生人分开评估 → 两套生成机制天生不同，统一模型反而不如分工 |
| 基线 ablations | GNN 无item (0.96 AUC)、LGB-A 无图特征 (0.896) → 图结构增量可度量 |
| 负样本过滤 | candidate 含训练期朋友占位 → 陌生人要在朋友抢前排的条件下排序，难度设定诚实 |

#### ⚠️ 需要补充

| 事项 | 优先级 | 说明 |
|------|:------:|------|
| 补 Common Neighbors / Adamic-Adar baseline | 🔴 P0 | 纯拓扑启发式在同一 split 上跑 AUC → 如果 >0.92，0.989 就是拓扑天花板不是泄漏；写进论文/报告主动"解魅" |
| 明确 HITS@k 的 k 值 | 🟡 P1 | 目前所有 HITS 都是 @5，报告里写 "HITS: 0.89" 应写成 "HITS@5: 0.89" |
| 评估协议文档化 | 🟡 P1 | 时间切分方式、候选池 size、负采样协议、filtered evaluation → 论文 Methodology 一节 |
| GNN-B 截断阈值 θ 设定 | 🟡 P1 | 一个人平均多少 candidate → p_link Top-K 截到多少 → 精排算力节省多少 |
| 陌生人 MRR 需要"下一个动作"叙事 | 🟢 P2 | 0.35→0.56 是图结构能做的极限，接下来靠 temporal interaction / 跨社群共现 / 品类偏好独立通路上分 |
| LGB-E 朋友 HITS 补测 | 🟢 P2 | mrr_breakdown.py 有 rank 分布数据，补跑一次即可 |

#### 🔴 AUC 0.989 的防御性叙事

社交网络结构天花板效应 → 不是泄漏，是诚实的时间切分 + 高聚类系数图的自然结果：

- 训练期 ≥3 个共同好友 → 未来连边概率指数级高于随机
- 纯图结构（零特征）GNN 就已 AUC 0.96
- 时间切分 ≠ random edge removal，后者会虚高 10-30 点
- AUC 在你的数据上区分度低（0.96→0.989 只差 0.029）→ **不要当核心贡献指标，MRR/HITS@5 才是**

论文/报告里建议主动放这样一张表：

```
Method                         AUC     MRR@5   HITS@5
Common Neighbors (纯拓扑)      ~0.93     —        —
LGB-A (无图特征)               0.896    0.55     0.61
LGB-A (朋友专家)                 —      0.74     0.89
GNN-A (陌生人专家)             0.985    0.35     0.65
GNN-B (全局门卫)               0.989    0.38     0.61
```

主动摆出来 = 你在驾驭指标，不是被 AUC 绑架。

#### 论文级核心论点（一句）

> Friend-link formation is dominated by local triadic closure efficiently captured by feature-engineered gradient boosting, while stranger-link formation requires multi-hop relational diffusion that benefits from GNN message passing — therefore a router-based hybrid outperforms any single homogeneous model.

### NaNaGi 落地待办

- [ ] P0: 补 CN/AA 拓扑 baseline 跑 AUC
- [ ] P0: 确定 GNN-B prior filter 的截断阈值 θ
- [ ] P1: 两个入口的 candidate pool 构建规则精确文档化（5-6 条规则/数据源清单）
- [ ] P1: 评估协议完整文档（split/负采样/HITS@k/filtered eval）
- [ ] P2: 边权重消息传递 GNN-B 实现（5/20 原案）
- [ ] P2: 品类偏好独立通路（不与 MLP 拼接，直接加权候选池）
- [ ] P3: GNN-B embedding → ANN 检索召回（替代随机 200 候选）

## 技术栈

Python · LightGBM · PyTorch Geometric · SHAP · PCA · Pandas · NumPy · joblib · scikit-learn

## 数据规格

| 项目 | 数值 |
|------|------|
| 用户数 | 115,849 |
| 商品数 | 438,164 |
| 分享记录 | 721,103 (训练) |
| 时间跨度 | 2021-12-27 ~ 2023-02-28 |
| 训练集 | ~240万行 (1:3 正负比) |
| 验证集 | ~48万行 (未来时间, 无泄漏) |
| LightGBM 特征 | 119维 (14 cat + 105 num) |
| GNN 节点 | 109,158 |
| GNN 边 | 602,606 有向边 |
| GNN embedding | 64维 |
