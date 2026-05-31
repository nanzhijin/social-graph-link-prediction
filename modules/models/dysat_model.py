"""
DySAT 动态图链接预测模型 v2 — item-aware

GNN per snapshot → Multi-head Self-Attention across time → Link Predictor

架构:
    Snapshot G₁ ─→ GNN(shared) ─→ emb₁ ─┐
    Snapshot G₂ ─→ GNN(shared) ─→ emb₂ ─┼─→ Temporal Attention ─→ z_v
                                          │    (学时间步权重)
    节点静态特征 ─────────────────────────┘

标准模型接口: fit → predict_proba → evaluate → evaluate_mrr
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(r"D:\GNN\processed")

from modules.models.gnn_model import (
    GNNEncoder, LinkPredictor, ItemEncoder,
    ITEM_CAT_COLS, ITEM_NUM_COLS,
)


class TemporalAttention(nn.Module):
    """
    Multi-head Self-Attention 跨时间步聚合

    输入: (N, T, D) — N 个节点, T 个时间步, D 维 embedding
    输出: (N, D)    — 每个节点的时序聚合 embedding
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            batch_first=True, dropout=0.1
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, 10, dim) * 0.02)

    def forward(self, x):
        N, T, D = x.shape
        x = x + self.pos_encoding[:, :T, :]
        out, _ = self.attn(x, x, x)
        return out.mean(dim=1)  # (N, D)


class DySATModel:
    """
    DySAT 动态图链接预测 — item-aware

    参数:
        hidden_dim: GNN 隐藏维度
        num_layers: GNN 层数
        gnn_type: 'sage' | 'gcn' | 'gat'
        num_heads: 时序 attention 头数
        lr/epochs/batch_size: 训练参数
        use_item_features: 是否使用商品特征
        item_dim: 商品特征编码维度
    """
    def __init__(self, name="DySAT", hidden_dim=64, num_layers=2, gnn_type="sage",
                 num_heads=4, lr=0.003, epochs=50, batch_size=8192, device="cpu",
                 use_item_features=True, item_dim=32):
        self.name = name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.num_heads = num_heads
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.use_item_features = use_item_features
        self.item_dim = item_dim

        self._snapshots = []
        self._node_to_idx = None
        self._idx_to_node = None
        self._n_nodes = 0
        self._encoder = None
        self._temp_attn = None
        self._predictor = None

        # item 编码 (复用 GNNModel 的 ItemEncoder)
        self._item_encoder = None
        self._item_cat_maps = None
        self._item_vocab_sizes = None
        self._item_cat_cols = []
        self._item_num_cols = []
        self._has_item_features = False

    # ═══════════════════════════════════════════════════════════
    # 多快照图构建
    # ═══════════════════════════════════════════════════════════
    def _build_snapshots(self):
        """构建两个累积快照图"""
        print("  [DySAT] 构建快照图...")

        df_train = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
        df_final_train = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
        df_all = pd.concat([df_train, df_final_train], ignore_index=True)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

        df_profile = pd.read_pickle(PROCESSED_DIR / "user_profile_enriched.pkl")
        profile_cols = [c for c in df_profile.columns if c != "user_id"]

        t1 = pd.Timestamp("2022-10-29")
        t2 = df_all["timestamp"].max()

        all_users = set(df_all["inviter_id"].unique()) | set(df_all["voter_id"].unique())
        self._node_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
        self._idx_to_node = {i: u for u, i in self._node_to_idx.items()}
        self._n_nodes = len(self._node_to_idx)

        time_cols = [c for c in profile_cols if "first_time" in c or
                     "last_time" in c or "days_since" in c]
        df_profile_clean = df_profile[profile_cols].fillna(0).copy()
        for tc in time_cols:
            if tc in df_profile_clean.columns:
                df_profile_clean[tc] = df_profile_clean[tc].apply(
                    lambda v: float(pd.Timestamp(v).toordinal())
                    if pd.notna(v) and v != -1 and v != "-1" and v != 0 else -1.0)
        profile_arr = df_profile_clean.values.astype(np.float32)

        n_feat = len(profile_cols)
        profile_idx = {u: i for i, u in enumerate(df_profile["user_id"])}
        node_feat = np.zeros((self._n_nodes, n_feat), dtype=np.float32)
        for user_id, idx in self._node_to_idx.items():
            if user_id in profile_idx:
                node_feat[idx] = profile_arr[profile_idx[user_id]]
        feat_mean = node_feat.mean(axis=0, keepdims=True)
        feat_std = node_feat.std(axis=0, keepdims=True) + 1e-8
        node_feat = (node_feat - feat_mean) / feat_std
        x = torch.tensor(node_feat, dtype=torch.float32)

        for snap_name, cutoff in [("Snapshot1", t1), ("Snapshot2", t2)]:
            snap_data = df_all[df_all["timestamp"] <= cutoff]
            edges = snap_data[["inviter_id", "voter_id"]].drop_duplicates()
            src = [self._node_to_idx[u] for u in edges["inviter_id"]
                   if u in self._node_to_idx]
            dst = [self._node_to_idx[v] for v in edges["voter_id"]
                   if v in self._node_to_idx]
            ei = torch.tensor([src, dst], dtype=torch.long)
            data = Data(x=x.clone(), edge_index=ei)
            self._snapshots.append(data)
            print(f"  [DySAT] {snap_name}: {len(src):,} 边 (≤{cutoff.date()})")

    # ═══════════════════════════════════════════════════════════
    # Item 特征处理 (与 GNNModel 一致)
    # ═══════════════════════════════════════════════════════════
    def _build_item_vocab(self, X):
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
            print(f"  [DySAT] Item 特征: cat={found_cat} vocab={vocab_sizes} "
                  f"num={found_num}")
        else:
            print("  [DySAT] ⚠ 未检测到 item 特征列, 回退到纯用户模式")

    def _extract_item_tensors(self, X, valid_mask=None):
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
    # 训练
    # ═══════════════════════════════════════════════════════════
    def fit(self, X, y, cat_cols=None, **kwargs):
        if not self._snapshots:
            self._build_snapshots()

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
        print(f"  [DySAT] 训练样本: {n_samples:,}")

        # 提取 item 特征
        item_cat, item_num = self._extract_item_tensors(X, valid)

        # 模型初始化
        in_dim = self._snapshots[0].x.shape[1]
        self._encoder = GNNEncoder(in_dim, self.hidden_dim, self.hidden_dim,
                                   self.num_layers, self.gnn_type).to(self.device)
        self._temp_attn = TemporalAttention(self.hidden_dim,
                                            self.num_heads).to(self.device)
        item_dim_actual = self.item_dim if self._has_item_features else 0
        self._predictor = LinkPredictor(self.hidden_dim, item_dim_actual,
                                        self.hidden_dim).to(self.device)

        snapshots = [s.to(self.device) for s in self._snapshots]

        params = (list(self._encoder.parameters()) +
                  list(self._temp_attn.parameters()) +
                  list(self._predictor.parameters()))
        if self._has_item_features:
            params += list(self._item_encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)

        inv_t = torch.tensor(inv_idx, dtype=torch.long, device=self.device)
        vot_t = torch.tensor(vot_idx, dtype=torch.long, device=self.device)
        y_t = torch.tensor(y_all, dtype=torch.float32, device=self.device)
        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(1, self.epochs + 1):
            self._encoder.train(); self._temp_attn.train(); self._predictor.train()
            if self._has_item_features:
                self._item_encoder.train()

            # Full forward on all snapshots (retain grad)
            snap_embs = []
            for s in snapshots:
                snap_embs.append(self._encoder(s.x, s.edge_index))
            all_embs = torch.stack(snap_embs, dim=1)      # (N, T, D)
            temporal_emb = self._temp_attn(all_embs)       # (N, D)

            total_loss = torch.tensor(0.0, device=self.device)
            for b in range(n_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n_samples)

                if self._has_item_features:
                    item_emb = self._item_encoder(
                        [cat[start:end] for cat in item_cat],
                        item_num[start:end],
                    )
                    scores = self._predictor(
                        temporal_emb[inv_t[start:end]],
                        temporal_emb[vot_t[start:end]],
                        item_feat=item_emb,
                    )
                else:
                    scores = self._predictor(
                        temporal_emb[inv_t[start:end]],
                        temporal_emb[vot_t[start:end]],
                    )
                loss = F.binary_cross_entropy(scores, y_t[start:end])
                total_loss = total_loss + loss * ((end - start) / n_samples)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            avg_loss = total_loss.item()

            if epoch % 10 == 0 or epoch == 1 or epoch == self.epochs:
                self._encoder.eval(); self._temp_attn.eval(); self._predictor.eval()
                if self._has_item_features:
                    self._item_encoder.eval()
                with torch.no_grad():
                    snap_embs_eval = []
                    for s in snapshots:
                        snap_embs_eval.append(self._encoder(s.x, s.edge_index))
                    all_embs_eval = torch.stack(snap_embs_eval, dim=1)
                    temp_emb_eval = self._temp_attn(all_embs_eval)
                    if self._has_item_features:
                        item_emb_eval = self._item_encoder(item_cat, item_num)
                        all_scores = self._predictor(
                            temp_emb_eval[inv_t], temp_emb_eval[vot_t],
                            item_feat=item_emb_eval,
                        ).cpu().numpy()
                    else:
                        all_scores = self._predictor(
                            temp_emb_eval[inv_t], temp_emb_eval[vot_t],
                        ).cpu().numpy()
                    auc = roc_auc_score(y_all, all_scores) if len(np.unique(y_all)) > 1 else float('nan')
                print(f"  [DySAT] epoch {epoch:3d}/{self.epochs}  "
                      f"loss={avg_loss:.4f}  auc={auc:.4f}")

        return self

    # ═══════════════════════════════════════════════════════════
    # 预测 & 评估
    # ═══════════════════════════════════════════════════════════
    def _get_temporal_embeddings(self):
        """获取时序节点 embedding"""
        self._encoder.eval(); self._temp_attn.eval()
        with torch.no_grad():
            snap_embs = []
            for s in self._snapshots:
                snap_embs.append(
                    self._encoder(s.x.to(self.device), s.edge_index.to(self.device)))
            all_embs = torch.stack(snap_embs, dim=1)
            return self._temp_attn(all_embs)

    def predict_proba(self, X):
        emb = self._get_temporal_embeddings()
        inv_idx = np.array([self._node_to_idx.get(u, -1) for u in X["inviter_id"]])
        vot_idx = np.array([self._node_to_idx.get(v, -1) for v in X["voter_id"]])
        valid_mask = (inv_idx >= 0) & (vot_idx >= 0)

        scores = np.zeros(len(X), dtype=np.float32)
        if valid_mask.sum() > 0:
            inv_t = torch.tensor(inv_idx[valid_mask], dtype=torch.long, device=self.device)
            vot_t = torch.tensor(vot_idx[valid_mask], dtype=torch.long, device=self.device)
            self._predictor.eval()
            if self._has_item_features:
                self._item_encoder.eval()
            with torch.no_grad():
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
        valid_mask = proba > 0
        return {
            "auc": roc_auc_score(y[valid_mask], proba[valid_mask]),
            "pos_rate_pred": float(proba.mean()),
        }

    def evaluate_mrr(self, valid_df, n_queries=2000):
        """MRR@5 评估 — item-aware"""
        print(f"  [MRR-DySAT] 评估 {n_queries} 个查询...")

        df_train_raw = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
        df_final_train_raw = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
        df_all_raw = pd.concat([df_train_raw, df_final_train_raw], ignore_index=True)
        df_all_raw["timestamp"] = pd.to_datetime(df_all_raw["timestamp"])
        split_date = pd.Timestamp("2022-10-29")
        train_raw = df_all_raw[df_all_raw["timestamp"] <= split_date]
        inviter_friends = train_raw.groupby("inviter_id")["voter_id"].apply(set).to_dict()

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

        emb_all = self._get_temporal_embeddings()
        voter_pool = np.unique(np.concatenate([
            train_raw["voter_id"].values, train_raw["inviter_id"].values
        ]))

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
            if not valid_c or inv not in self._node_to_idx:
                ranks.append(999)
                continue

            j_map, idx_list = zip(*valid_c)
            cand_emb = emb_all[torch.tensor(idx_list, dtype=torch.long, device=self.device)]
            inv_emb = emb_all[self._node_to_idx[inv]].unsqueeze(0).expand(len(idx_list), -1)

            n_cand = len(idx_list)
            if self._has_item_features:
                q_cat, q_num = self._item_feat_for_query(queries.iloc[i])
                item_cat_expanded = [t.expand(n_cand) for t in q_cat]
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
        print(f"  [MRR-DySAT] MRR@5={mrr:.5f}  HITS@5={hits:.5f}")
        return {"mrr@5": mrr, "hits@5": hits}
