# 社交图谱动态链接预测

CAAI-BDSC2023 竞赛 Task 2：基于历史分享行为数据，预测未来时间窗口内的用户-商品分享链路。

## 方法概述

**LightGBM + GNN 双路线**，覆盖表格特征和图结构两种数据视角：

| 路线 | 方法 | 核心思路 |
|------|------|---------|
| 路线一 | LightGBM | 119维特征（用户画像/图拓扑/边特征/品类交叉），梯度提升树分类 |
| 路线二 | GNN (GraphSAGE) + DySAT | 对社交图直接建模，GraphSAGE学习节点嵌入，DySAT引入时序注意力 |

## 关键结果

| 模型 | Valid AUC | MRR@5 | HITS@5 | 说明 |
|------|-----------|-------|--------|------|
| 官方基线 | — | 0.0344 | 0.0926 | 竞赛官方 baseline |
| **LGB-A** (113维) | **0.8957** | **0.5606** | 0.6045 | **主力模型**，不含朋友圈特征 |
| LGB-B (119维) | 0.8606 | 0.4291 | 0.7275 | 边特征反而降低 AUC |
| LGB-C (118维) | 0.8555 | 0.4560 | 0.6930 | 单独删 is_friend |
| GNN (无item) | 0.9604 | 0.2137 | 0.3600 | 纯图结构 |
| **GNN + Item** | **0.9847** | **0.3273** | 0.5415 | +53% MRR vs 无item |
| DySAT (无item) | 0.9567 | 0.2219 | 0.3700 | +3.8% vs GNN |
| DySAT + Item | 0.9790 | 0.3111 | 0.5245 | -5.0% vs GNN+item |

**MRR 分场景拆解：**

| 场景 | 占比 | LGB-A MRR | GNN+Item MRR | 诊断 |
|------|------|-----------|-------------|------|
| 场景1：好友召回 | 72.4% | **0.74** | 0.25 | LGB 的 is_friend 一招制胜 |
| 场景2：冷启动 | 27.6% | 0.09 | **0.54** | GNN 图嵌入碾压 LGB |

LGB 在好友推荐上靠边特征一招制胜；GNN 在冷启动场景通过图嵌入碾压 LGB。**两者互补。**

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

1. **AUC ≠ MRR**：GNN 的 AUC 极高（0.98）但 MRR 低（0.33）。图结构擅长分类不擅长排序。损失函数（BCE）和共享 item embedding 导致 GNN 无法学习候选人的相对排序。
2. **边特征是双刃剑**：is_friend 在场景1 MRR=0.74（决定性），但对 AUC 是负贡献——正负样本各25%都有 is_friend=1，分类面被污染。
3. **缺失 per-(voter, item) 交互特征**：119维里没有一个特征回答"这个候选人是否适合这个具体商品"。单特征规则（voter 是否接收过该品类）就能达到 MRR=0.18。
4. **DySAT 时序信号极弱**：快照间距仅38天，71%用户度数无变化，Temporal Attention 基本是负优化。
5. **GNN ItemEncoder 是关键**：加入商品特征后 GNN MRR 从 0.21→0.33 (+53%)，证明 item-aware 预测对排序至关重要。

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
