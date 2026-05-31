"""
GNN 链接预测模型 v3 — item-aware LinkPredictor
GraphSAGE/GCN/GAT 编码器 + Item Encoder + MLP Link Predictor

标准模型接口: fit → predict_proba → evaluate → evaluate_mrr
与 ExperimentRunner 无缝集成
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(r"D:\GNN\processed")

ITEM_CAT_COLS = ["cate_id", "brand_id", "shop_id"]
ITEM_NUM_COLS = ["cate_virality_score"]


class LinkPredictor(nn.Module):
    """链接预测 MLP — 支持 item 特征注入"""
    def __init__(self, user_dim, item_dim=0, hidden_dim=64):
        super().__init__()
        self.item_dim = item_dim
        in_dim = user_dim * 2 + item_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, emb_u, emb_v, item_feat=None):
        if item_feat is not None:
            x = torch.cat([emb_u, emb_v, item_feat], dim=-1)
        else:
            x = torch.cat([emb_u, emb_v], dim=-1)
        return self.net(x).squeeze(-1)


class ItemEncoder(nn.Module):
    """商品特征编码器 — 类别 Embedding + 数值特征 → 统一向量"""
    def __init__(self, vocab_sizes, embed_dim=8, out_dim=32):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(vs + 1, embed_dim) for vs in vocab_sizes
        ])
        n_cat = len(vocab_sizes)
        in_dim = n_cat * embed_dim + 1  # +1 for numerical (cate_virality_score)
        self.proj = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, cat_ids, num_feat):
        # cat_ids: list of (batch,) LongTensors  [cate_id, brand_id, shop_id]
        # num_feat: (batch, 1) FloatTensor
        feats = [emb(cat_ids[i]) for i, emb in enumerate(self.embeds)]
        feats.append(num_feat)
        x = torch.cat(feats, dim=-1)
        return self.proj(x)


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, num_layers=2, gnn_type="sage"):
        super().__init__()
        Conv = {"sage": SAGEConv, "gcn": GCNConv, "gat": GATConv}[gnn_type]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            din = in_dim if i == 0 else hidden_dim
            dout = out_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(Conv(din, dout))
            self.bns.append(nn.BatchNorm1d(dout))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = F.relu(bn(conv(x, edge_index)))
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        return x


class GNNModel:
    def __init__(self, name="GNN", hidden_dim=64, num_layers=2, gnn_type="sage",
                 lr=0.005, epochs=50, batch_size=8192, device="cpu",
                 node_feature_dim=None, profile_cols=None,
                 use_item_features=True, item_dim=32):
        self.name = name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.node_feature_dim = node_feature_dim
        self.profile_cols = profile_cols
        self.use_item_features = use_item_features
        self.item_dim = item_dim

        self._data = None
        self._node_to_idx = None
        self._idx_to_node = None
        self._encoder = None
        self._predictor = None
        self._n_nodes = 0

        # item 编码
        self._item_encoder = None
        self._item_cat_maps = None
        self._item_vocab_sizes = None
        self._item_cat_cols = []
        self._item_num_cols = []
        self._has_item_features = False

    # ═══════════════════════════════════════════════════════════
    # 图构建
    # ═══════════════════════════════════════════════════════════
    def _build_graph(self):
        print("  [GNN] 构建图...")
        df_train = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
        df_final_train = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
        df_all = pd.concat([df_train, df_final_train], ignore_index=True)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        split_date = pd.Timestamp("2022-10-29")
        train_raw = df_all[df_all["timestamp"] <= split_date]

        df_profile = pd.read_pickle(PROCESSED_DIR / "user_profile_enriched.pkl")
        all_profile_cols = [c for c in df_profile.columns if c != "user_id"]
        if self.profile_cols is not None:
            profile_cols = [c for c in self.profile_cols if c in all_profile_cols]
            print(f"  [GNN] 节点特征: {len(profile_cols)} 维 (精选)")
        else:
            profile_cols = all_profile_cols

        all_users = set(train_raw["inviter_id"].unique()) | set(train_raw["voter_id"].unique())
        self._node_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
        self._idx_to_node = {i: u for u, i in self._node_to_idx.items()}
        self._n_nodes = len(self._node_to_idx)
        n_feat = len(profile_cols)
        print(f"  [GNN] 节点: {self._n_nodes:,}  边: {len(train_raw):,}  特征: {n_feat}d")

        # 节点特征矩阵
        profile_idx = {u: i for i, u in enumerate(df_profile["user_id"])}
        time_cols = [c for c in profile_cols if "first_time" in c or
                     "last_time" in c or "days_since" in c]
        df_profile_clean = df_profile[profile_cols].fillna(0).copy()
        for tc in time_cols:
            if tc in df_profile_clean.columns:
                df_profile_clean[tc] = df_profile_clean[tc].apply(
                    lambda v: float(pd.Timestamp(v).toordinal())
                    if pd.notna(v) and v != -1 and v != "-1" and v != 0 else -1.0)
        profile_arr = df_profile_clean.values.astype(np.float32)

        node_feat = np.zeros((self._n_nodes, n_feat), dtype=np.float32)
        for user_id, idx in self._node_to_idx.items():
            if user_id in profile_idx:
                node_feat[idx] = profile_arr[profile_idx[user_id]]

        feat_mean = node_feat.mean(axis=0, keepdims=True)
        feat_std = node_feat.std(axis=0, keepdims=True) + 1e-8
        node_feat = (node_feat - feat_mean) / feat_std

        if self.node_feature_dim is not None and self.node_feature_dim < n_feat:
            pca = PCA(n_components=self.node_feature_dim, random_state=42)
            node_feat = pca.fit_transform(node_feat)
            print(f"  [GNN] PCA: {n_feat}d → {self.node_feature_dim}d  "
                  f"var={pca.explained_variance_ratio_.sum():.1%}")

        # 边索引 (有向)
        edges = train_raw[["inviter_id", "voter_id"]].drop_duplicates()
        src = [self._node_to_idx[u] for u in edges["inviter_id"]
               if u in self._node_to_idx]
        dst = [self._node_to_idx[v] for v in edges["voter_id"]
               if v in self._node_to_idx]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        self._data = Data(
            x=torch.tensor(node_feat, dtype=torch.float32),
            edge_index=edge_index,
        )

    # ═══════════════════════════════════════════════════════════
    # Item 特征处理
    # ═══════════════════════════════════════════════════════════
    def _build_item_vocab(self, X):
        """扫描 X 中的 item 特征列, 建立类别→id 映射"""
        cat_maps = []
        vocab_sizes = []
        found_cat = []
        found_num = []

        for col in ITEM_CAT_COLS:
            if col in X.columns:
                vals = X[col].fillna(-1).astype(int).values
                unique_vals = sorted(np.unique(vals))
                mapping = {v: i for i, v in enumerate(unique_vals)}
                cat_maps.append(mapping)
                vocab_sizes.append(len(unique_vals))
                found_cat.append(col)

        for col in ITEM_NUM_COLS:
            if col in X.columns:
                found_num.append(col)

        self._item_cat_maps = cat_maps
        self._item_vocab_sizes = vocab_sizes
        self._item_cat_cols = found_cat
        self._item_num_cols = found_num
        self._has_item_features = len(found_cat) > 0

        if self._has_item_features:
            print(f"  [GNN] Item 特征: cat={found_cat} vocab={vocab_sizes} "
                  f"num={found_num}")
        else:
            print("  [GNN] ⚠ 未检测到 item 特征列, 回退到纯用户模式")

    def _extract_item_tensors(self, X, valid_mask=None):
        """从 DataFrame 提取 item 特征的 tensor, 返回 (cat_tensors, num_tensor)"""
        if not self._has_item_features:
            return None, None

        idxs = np.arange(len(X))
        if valid_mask is not None:
            idxs = idxs[valid_mask]

        cat_tensors = []
        for i, col in enumerate(self._item_cat_cols):
            raw = X[col].iloc[idxs].fillna(-1).astype(int).values
            mapping = self._item_cat_maps[i]
            default = len(mapping)
            mapped = np.array([mapping.get(v, default) for v in raw], dtype=np.int64)
            cat_tensors.append(torch.tensor(mapped, dtype=torch.long, device=self.device))

        num_col = self._item_num_cols[0]
        num_vals = X[num_col].iloc[idxs].fillna(0.0).astype(np.float32).values
        num_tensor = torch.tensor(num_vals, dtype=torch.float32, device=self.device).unsqueeze(-1)

        return cat_tensors, num_tensor

    def _item_feat_for_query(self, row):
        """从单行 (Series) 提取 item 特征 tensor, 返回 (cat_tensors, num_tensor) 各为 size=1"""
        if not self._has_item_features:
            return None, None

        cat_tensors = []
        for i, col in enumerate(self._item_cat_cols):
            raw = int(row[col]) if pd.notna(row[col]) else -1
            mapping = self._item_cat_maps[i]
            idx = mapping.get(raw, len(mapping))
            cat_tensors.append(torch.tensor([idx], dtype=torch.long, device=self.device))

        num_col = self._item_num_cols[0]
        num_val = float(row[num_col]) if pd.notna(row[num_col]) else 0.0
        num_tensor = torch.tensor([[num_val]], dtype=torch.float32, device=self.device)

        return cat_tensors, num_tensor

    # ═══════════════════════════════════════════════════════════
    # 训练 (mini-batch, 解耦: GNN forward 每 epoch 一次)
    # ═══════════════════════════════════════════════════════════
    def fit(self, X, y, cat_cols=None, **kwargs):
        if self._data is None:
            self._build_graph()

        # 首次调用时构建 item vocab
        if self.use_item_features and self._item_encoder is None:
            self._build_item_vocab(X)
            if self._has_item_features:
                self._item_encoder = ItemEncoder(
                    self._item_vocab_sizes, embed_dim=8, out_dim=self.item_dim
                ).to(self.device)

        inv_idx = np.array([self._node_to_idx.get(u, -1) for u in X["inviter_id"]])
        vot_idx = np.array([self._node_to_idx.get(v, -1) for v in X["voter_id"]])
        valid = (inv_idx >= 0) & (vot_idx >= 0)
        inv_idx = inv_idx[valid]
        vot_idx = vot_idx[valid]
        y_all = np.array(y, dtype=np.float32)[valid]
        n_samples = len(inv_idx)
        print(f"  [GNN] 训练样本: {n_samples:,} ({valid.sum()/len(valid)*100:.1f}%)")

        # 提取 item 特征
        item_cat, item_num = self._extract_item_tensors(X, valid)

        in_dim = self._data.x.shape[1]
        self._encoder = GNNEncoder(in_dim, self.hidden_dim, self.hidden_dim,
                                   self.num_layers, self.gnn_type).to(self.device)
        item_dim_actual = self.item_dim if self._has_item_features else 0
        self._predictor = LinkPredictor(self.hidden_dim, item_dim_actual,
                                        self.hidden_dim).to(self.device)
        data = self._data.to(self.device)

        params = list(self._encoder.parameters()) + list(self._predictor.parameters())
        if self._has_item_features:
            params += list(self._item_encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)

        # 打乱
        perm = np.random.permutation(n_samples)
        inv_idx_shuf = inv_idx[perm]
        vot_idx_shuf = vot_idx[perm]
        y_shuf = y_all[perm]

        inv_t = torch.tensor(inv_idx_shuf, dtype=torch.long, device=self.device)
        vot_t = torch.tensor(vot_idx_shuf, dtype=torch.long, device=self.device)
        y_t = torch.tensor(y_shuf, dtype=torch.float32, device=self.device)

        if self._has_item_features:
            item_cat_shuf = [cat[perm] for cat in item_cat]
            item_num_shuf = item_num[perm]

        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(1, self.epochs + 1):
            self._encoder.train(); self._predictor.train()
            if self._has_item_features:
                self._item_encoder.train()

            # Full GNN forward (retain grad for backward)
            emb_all = self._encoder(data.x, data.edge_index)  # (N, D)

            # Accumulate loss across mini-batches
            total_loss = torch.tensor(0.0, device=self.device)
            for b in range(n_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n_samples)

                if self._has_item_features:
                    item_emb = self._item_encoder(
                        [cat[start:end] for cat in item_cat_shuf],
                        item_num_shuf[start:end],
                    )
                    scores = self._predictor(
                        emb_all[inv_t[start:end]],
                        emb_all[vot_t[start:end]],
                        item_feat=item_emb,
                    )
                else:
                    scores = self._predictor(
                        emb_all[inv_t[start:end]],
                        emb_all[vot_t[start:end]],
                    )
                loss = F.binary_cross_entropy(scores, y_t[start:end])
                total_loss = total_loss + loss * ((end - start) / n_samples)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            avg_loss = total_loss.item()

            if epoch % 10 == 0 or epoch == 1 or epoch == self.epochs:
                with torch.no_grad():
                    pos_mask = y_all > 0.5
                    if pos_mask.sum() > 0 and (~pos_mask).sum() > 0:
                        emb_all_eval = self._encoder(data.x, data.edge_index)
                        inv_t_all = torch.tensor(inv_idx, dtype=torch.long, device=self.device)
                        vot_t_all = torch.tensor(vot_idx, dtype=torch.long, device=self.device)
                        if self._has_item_features:
                            item_emb_eval = self._item_encoder(item_cat, item_num)
                            all_scores = self._predictor(
                                emb_all_eval[inv_t_all], emb_all_eval[vot_t_all],
                                item_feat=item_emb_eval,
                            ).cpu().numpy()
                        else:
                            all_scores = self._predictor(
                                emb_all_eval[inv_t_all], emb_all_eval[vot_t_all],
                            ).cpu().numpy()
                        auc = roc_auc_score(y_all, all_scores)
                    else:
                        auc = float('nan')
                print(f"  [GNN] epoch {epoch:3d}/{self.epochs}  "
                      f"loss={avg_loss:.4f}  auc={auc:.4f}")

        return self

    # ═══════════════════════════════════════════════════════════
    # 预测 & 评估
    # ═══════════════════════════════════════════════════════════
    def predict_proba(self, X):
        self._encoder.eval(); self._predictor.eval()
        if self._has_item_features:
            self._item_encoder.eval()
        with torch.no_grad():
            emb = self._encoder(self._data.x.to(self.device),
                               self._data.edge_index.to(self.device))
            inv_idx = np.array([self._node_to_idx.get(u, -1) for u in X["inviter_id"]])
            vot_idx = np.array([self._node_to_idx.get(v, -1) for v in X["voter_id"]])
            valid_mask = (inv_idx >= 0) & (vot_idx >= 0)

            scores = np.zeros(len(X), dtype=np.float32)
            if valid_mask.sum() > 0:
                inv_t = torch.tensor(inv_idx[valid_mask], dtype=torch.long, device=self.device)
                vot_t = torch.tensor(vot_idx[valid_mask], dtype=torch.long, device=self.device)
                if self._has_item_features:
                    item_cat, item_num = self._extract_item_tensors(X, valid_mask)
                    item_emb = self._item_encoder(item_cat, item_num)
                    scores[valid_mask] = self._predictor(
                        emb[inv_t], emb[vot_t], item_feat=item_emb).cpu().numpy()
                else:
                    scores[valid_mask] = self._predictor(
                        emb[inv_t], emb[vot_t]).cpu().numpy()
        return scores

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        valid_mask = proba > 0  # 只评估图上有的用户对
        return {
            "auc": roc_auc_score(y[valid_mask], proba[valid_mask]),
            "pos_rate_pred": float(proba.mean()),
        }

    def evaluate_mrr(self, valid_df, model_a=None, n_queries=2000):
        """MRR@5 评估 — item-aware"""
        print(f"  [MRR] 评估 {n_queries} 个查询...")

        df_train_raw = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
        df_final_train_raw = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
        df_all_raw = pd.concat([df_train_raw, df_final_train_raw], ignore_index=True)
        df_all_raw["timestamp"] = pd.to_datetime(df_all_raw["timestamp"])
        split_date = pd.Timestamp("2022-10-29")
        train_raw = df_all_raw[df_all_raw["timestamp"] <= split_date]

        inviter_friends = train_raw.groupby("inviter_id")["voter_id"].apply(set).to_dict()

        # 保留 item 列用于后续查询
        query_cols = ["inviter_id", "item_id", "voter_id", "timestamp"]
        if self._has_item_features:
            for col in self._item_cat_cols + self._item_num_cols:
                if col in valid_df.columns:
                    query_cols.append(col)
        valid_pos = valid_df[query_cols].copy()
        valid_pos = valid_pos.rename(columns={"voter_id": "true_voter_id"})
        rng = np.random.default_rng(42)
        n_q = min(n_queries, len(valid_pos))
        q_idx = rng.choice(len(valid_pos), n_q, replace=False)
        queries = valid_pos.iloc[q_idx].reset_index(drop=True)

        # 预计算所有节点 embedding
        self._encoder.eval(); self._predictor.eval()
        if self._has_item_features:
            self._item_encoder.eval()
        with torch.no_grad():
            emb_all = self._encoder(self._data.x.to(self.device),
                                    self._data.edge_index.to(self.device))

        voter_pool = np.unique(np.concatenate([
            train_raw["voter_id"].values, train_raw["inviter_id"].values
        ]))
        pool_idx_map = {v: i for i, v in enumerate(voter_pool)}
        pool_emb = torch.zeros((len(voter_pool), emb_all.shape[1]), device=self.device)
        for v, i in pool_idx_map.items():
            if v in self._node_to_idx:
                pool_emb[i] = emb_all[self._node_to_idx[v]]

        ranks = []
        for i in range(n_q):
            inv = queries.iloc[i]["inviter_id"]
            true_v = queries.iloc[i]["true_voter_id"]
            friends = inviter_friends.get(inv, set())

            candidates = {true_v}
            candidates.update(friends)
            while len(candidates) < 200:
                candidates.add(voter_pool[rng.integers(0, len(voter_pool))])
            cand_list = list(candidates)

            cand_idx = [self._node_to_idx.get(v, -1) for v in cand_list]
            valid_c = [(j, idx) for j, idx in enumerate(cand_idx) if idx >= 0]
            if not valid_c:
                ranks.append(999)
                continue

            j_map, idx_list = zip(*valid_c)
            cand_emb = emb_all[torch.tensor(idx_list, dtype=torch.long, device=self.device)]

            if inv not in self._node_to_idx:
                ranks.append(999)
                continue
            inv_emb = emb_all[self._node_to_idx[inv]].unsqueeze(0).expand(len(idx_list), -1)

            # item 特征: 查询商品特征, 广播到所有候选
            n_cand = len(idx_list)
            if self._has_item_features:
                q_cat, q_num = self._item_feat_for_query(queries.iloc[i])
                item_cat_expanded = [
                    t.expand(n_cand) for t in q_cat
                ]
                item_num_expanded = q_num.expand(n_cand, -1)
                item_emb = self._item_encoder(item_cat_expanded, item_num_expanded)
            else:
                item_emb = None

            with torch.no_grad():
                scores = self._predictor(inv_emb, cand_emb, item_feat=item_emb).cpu().numpy()
            sorted_indices = np.argsort(-scores)
            rank_map = {cand_list[j_map[j]]: r + 1 for r, j in enumerate(sorted_indices)}

            rank = rank_map.get(true_v, 999)
            ranks.append(rank)

        ranks = np.array(ranks, dtype=np.float64)
        mrr = float(np.mean(np.where(ranks <= 5, 1.0 / ranks, 0.0)))
        hits = float(np.mean(ranks <= 5))
        print(f"  [MRR] MRR@5={mrr:.5f}  HITS@5={hits:.5f}")
        return {"mrr@5": mrr, "hits@5": hits}

    @property
    def node_embeddings(self):
        self._encoder.eval()
        with torch.no_grad():
            emb = self._encoder(self._data.x.to(self.device),
                               self._data.edge_index.to(self.device))
        return emb.cpu().numpy()
