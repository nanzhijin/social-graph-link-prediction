# 社交图谱动态链接预测

CAAI-BDSC2023 竞赛 Task 2：基于历史分享行为数据，预测未来时间窗口内的用户-商品分享链路。

## 方法概述

**LightGBM + GNN 双路线**，覆盖表格特征和图结构两种数据视角：

| 路线 | 方法 | 核心思路 |
|------|------|---------|
| 路线一 | LightGBM | 119维特征（用户画像/图拓扑/边特征/品类交叉），梯度提升树分类 |
| 路线二 | GNN (GraphSAGE) + DySAT | 对社交图直接建模，GraphSAGE学习节点嵌入，DySAT引入时序注意力 |

## 关键结果

| 模型 | AUC | MRR@5 | 说明 |
|------|-----|-------|------|
| LGB-A (113维) | 0.8957 | 0.5606 | 主力模型，不含朋友圈特征 |
| GNN + Item | 0.9847 | 0.3273 | 图模型，冷启动场景 MRR=0.54 远高于 LGB |
| DySAT + Item | 0.9790 | 0.3111 | 时序GNN，时序信号弱（快照间仅新增7%边） |

**MRR 分场景拆解：**

| 场景 | LGB-A MRR | GNN+Item MRR |
|------|-----------|-------------|
| 场景1：好友召回（72.4%） | **0.74** | 0.25 |
| 场景2：冷启动（27.6%） | 0.09 | **0.54** |

LGB 在好友推荐上靠边特征（is_friend 等）一招制胜；GNN 在冷启动场景通过图嵌入碾压 LGB。两者互补。

## 项目结构

```
├── modules/                     # 模型框架
│   ├── models/
│   │   ├── lgb_model.py         # LightGBM 标准接口
│   │   ├── gnn_model.py         # GraphSAGE/GCN/GAT + LinkPredictor + ItemEncoder
│   │   └── dysat_model.py       # 多快照 + TemporalAttention
│   ├── feature_selector.py      # 4种特征选择器 (Identity/SHAP/PCA/SHAP+PCA)
│   └── experiment.py            # Config驱动交叉组合 + AB对比
├── 0_方案设计/                   # 设计文档与面试准备
│   ├── 模型数据维度规格书.md
│   ├── 阶段性报告.md
│   ├── AB实验设计方案.md
│   ├── 数据流向.md
│   └── SQL面试真题集.md
└── work.ipynb                   # EDA + 特征工程 + 用户画像
```

## 核心发现

1. **AUC ≠ MRR**：GNN 的 AUC 极高（0.98）但 MRR 低（0.33）。图结构擅长分类不擅长排序。损失函数（BCE）和共享 item embedding 导致 GNN 无法学习候选人的相对排序。
2. **边特征是双刃剑**：is_friend 在场景1 MRR=0.74（决定性），但对 AUC 是负贡献——正负样本各25%都有 is_friend=1，分类面被污染。
3. **缺失 per-(voter, item) 交互特征**：119维里没有一个特征回答"这个候选人是否适合这个具体商品"。单特征规则（voter 是否接收过该品类）就能达到 MRR=0.18。
4. **DySAT 时序信号极弱**：快照间距仅38天，71%用户度数无变化，Temporal Attention 基本是负优化。

## 技术栈

Python · LightGBM · PyTorch Geometric · SHAP · PCA · Pandas · NumPy
