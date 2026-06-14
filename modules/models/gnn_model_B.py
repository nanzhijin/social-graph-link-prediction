"""
GNN 链接预测模型 B — 继承 GNNModel, 增加 E 模型的新特征
==========================================================
v3 的 GNN-A 仅用 [emb_u | emb_v | item_emb]
GNN-B 追加 6 个 LGB-E 新特征 (去 rank):
  时序进化 (3):
    inviter_new_voter_ratio  — inviter 后半段新 voter 占比
    inviter_voter_retention  — inviter 前半段 voter 保留率
    pair_is_recent           — 该对是否在后半段出现 0/1
  品类交叉 (3):
    cate_match_score         — voter 对该 item 品类的偏好占比
    item_cate_in_voter_top3  — 该品类是否在 voter Top3 偏好中
    inviter_voter_cate_overlap — inviter-voter 品类 Jaccard

改动: 仅重写 LinkPredictor (加 extra_dim=6) + 三个方法
       fit / predict_proba / evaluate_mrr
       其余图构建/ItemEncoder 完全复用 GNNModel
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models.gnn_model import GNNModel, ItemEncoder, GNNEncoder, LinkPredictor as _BaseLinkPredictor

import os as _os
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # modules/models/ → GNN/
PROCESSED_DIR = _PROJECT_ROOT / "processed"


class LinkPredictorB(nn.Module):
    """链接预测 MLP — 支持 item 特征 + 品类交叉标量"""
    def __init__(self, user_dim, item_dim=0, extra_dim=0, hidden_dim=64):
        super().__init__()
        self.item_dim = item_dim
        self.extra_dim = extra_dim
        in_dim = user_dim * 2 + item_dim + extra_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, emb_u, emb_v, item_feat=None, extra_feat=None):
        parts = [emb_u, emb_v]
        if item_feat is not None:
            parts.append(item_feat)
        if extra_feat is not None:
            parts.append(extra_feat)
        x = torch.cat(parts, dim=-1)
        return self.net(x).squeeze(-1)


class GNNModelB(GNNModel):
    """GNN-B: 继承 GNNModel, 增加品类交叉特征 + 时序进化特征"""

    def save(self, path):
        """保存完整模型到单文件"""
        state = {
            "_type": "GNNModelB",
            "encoder": self._encoder.state_dict(),
            "predictor": self._predictor.state_dict(),
            "node_to_idx": self._node_to_idx,
            "data": self._data,
            "config": {
                "name": self.name, "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers, "gnn_type": self.gnn_type,
                "use_item_features": self.use_item_features, "item_dim": self.item_dim,
                "node_feature_dim": self.node_feature_dim,
            },
            "_extra_dim": self._extra_dim,
            "_extra_mean": self._extra_mean,
            "_extra_std": self._extra_std,
            "extra_feature_cols": self.EXTRA_FEATURE_COLS,
        }
        if self._has_item_features:
            state["item_encoder"] = self._item_encoder.state_dict()
            state["item_cat_maps"] = self._item_cat_maps
            state["item_vocab_sizes"] = self._item_vocab_sizes
            state["item_cat_cols"] = self._item_cat_cols
            state["item_num_cols"] = self._item_num_cols
        torch.save(state, path)
        print(f"  [{self.name}] 已保存到 {path}")

    @classmethod
    def _load_state(cls, state, device="cpu"):
        cfg = state["config"]
        extra_cols = state.get("extra_feature_cols", cls.EXTRA_FEATURE_COLS)
        model = cls(**cfg, device=device, extra_feature_cols=extra_cols)
        model._node_to_idx = state["node_to_idx"]
        model._data = state["data"]
        model._n_nodes = len(model._node_to_idx)
        model._idx_to_node = {i: u for u, i in model._node_to_idx.items()}
        model._extra_dim = state["_extra_dim"]
        model._extra_mean = state["_extra_mean"]
        model._extra_std = state["_extra_std"]

        # 重建 encoder
        in_dim = state["data"].x.shape[1]
        model._encoder = GNNEncoder(in_dim, model.hidden_dim, model.hidden_dim,
                                     model.num_layers, model.gnn_type).to(device)
        model._encoder.load_state_dict(state["encoder"])

        # 重建 predictor (LinkPredictorB)
        item_dim = model.item_dim if cfg["use_item_features"] else 0
        model._predictor = LinkPredictorB(
            model.hidden_dim, item_dim, model._extra_dim, model.hidden_dim
        ).to(device)
        model._predictor.load_state_dict(state["predictor"])

        # 重建 item encoder
        if model.use_item_features and "item_encoder" in state:
            model._item_cat_maps = state["item_cat_maps"]
            model._item_vocab_sizes = state["item_vocab_sizes"]
            model._item_cat_cols = state["item_cat_cols"]
            model._item_num_cols = state["item_num_cols"]
            model._has_item_features = len(model._item_cat_cols) > 0
            model._item_encoder = ItemEncoder(
                model._item_vocab_sizes, embed_dim=8, out_dim=model.item_dim
            ).to(device)
            model._item_encoder.load_state_dict(state["item_encoder"])

        model._encoder.eval()
        model._predictor.eval()
        if model._has_item_features:
            model._item_encoder.eval()
        return model

    EXTRA_FEATURE_COLS = [
        # 时序进化 (3)
        "inviter_new_voter_ratio",
        "inviter_voter_retention",
        "pair_is_recent",
        # 品类交叉 (3)
        "cate_match_score",
        "item_cate_in_voter_top3",
        "inviter_voter_cate_overlap",
    ]

    def __init__(self, name="GNN-B", **kwargs):
        # 提取 GNN-B 特有参数
        self._extra_dim = len(self.EXTRA_FEATURE_COLS)
        # 如果传了 extra_feature_cols 就用传的, 否则用默认的
        extra_cols = kwargs.pop("extra_feature_cols", None)
        if extra_cols is not None:
            self.EXTRA_FEATURE_COLS = extra_cols
            self._extra_dim = len(extra_cols)
        super().__init__(name=name, **kwargs)
        self._extra_mean = None
        self._extra_std = None

    def _extract_extra_features(self, X, valid_mask=None):
        """从 DataFrame 提取品类交叉特征, 返回归一化 tensor"""
        idxs = np.arange(len(X))
        if valid_mask is not None:
            idxs = idxs[valid_mask]
        vals = []
        for col in self.EXTRA_FEATURE_COLS:
            if col in X.columns:
                vals.append(X[col].iloc[idxs].fillna(0.0).astype(np.float32).values)
            else:
                vals.append(np.zeros(len(idxs), dtype=np.float32))
        arr = np.stack(vals, axis=1)
        # 归一化 (用训练时的 mean/std)
        if self._extra_mean is not None:
            arr = (arr - self._extra_mean) / (self._extra_std + 1e-8)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def fit(self, X, y, cat_cols=None, **kwargs):
        # 计算训练集归一化参数
        vals = []
        for col in self.EXTRA_FEATURE_COLS:
            if col in X.columns:
                vals.append(X[col].fillna(0.0).astype(np.float32).values)
            else:
                vals.append(np.zeros(len(X), dtype=np.float32))
        arr = np.stack(vals, axis=1)
        self._extra_mean = arr.mean(axis=0, keepdims=True)
        self._extra_std = arr.std(axis=0, keepdims=True) + 1e-8

        # 调父类构建图
        if self._data is None:
            self._build_graph()

        # 首次构建 item vocab
        if self.use_item_features and self._item_encoder is None:
            self._build_item_vocab(X)

        # 提取用户索引
        inv_idx = np.array([self._node_to_idx.get(u, -1) for u in X["inviter_id"]])
        vot_idx = np.array([self._node_to_idx.get(v, -1) for v in X["voter_id"]])
        valid = (inv_idx >= 0) & (vot_idx >= 0)
        inv_idx = inv_idx[valid]
        vot_idx = vot_idx[valid]
        y_all = np.array(y, dtype=np.float32)[valid]
        n_samples = len(inv_idx)
        print(f"  [GNN-B] 训练样本: {n_samples:,}  extra_dim={self._extra_dim}")

        # 提取 item + extra 特征
        item_cat, item_num = self._extract_item_tensors(X, valid)
        extra_t = self._extract_extra_features(X, valid)

        # 创建 item encoder (从 GNNModel.fit 迁移)
        if self._has_item_features:
            self._item_encoder = ItemEncoder(
                self._item_vocab_sizes, embed_dim=8, out_dim=self.item_dim
            ).to(self.device)

        in_dim = self._data.x.shape[1]
        self._encoder = GNNEncoder(in_dim, self.hidden_dim, self.hidden_dim,
                                   self.num_layers, self.gnn_type).to(self.device)
        item_dim_actual = self.item_dim if self._has_item_features else 0
        self._predictor = LinkPredictorB(
            self.hidden_dim, item_dim_actual, self._extra_dim, self.hidden_dim
        ).to(self.device)
        data = self._data.to(self.device)

        params = list(self._encoder.parameters()) + list(self._predictor.parameters())
        if self._has_item_features:
            params += list(self._item_encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)

        perm = np.random.permutation(n_samples)
        inv_t = torch.tensor(inv_idx[perm], dtype=torch.long, device=self.device)
        vot_t = torch.tensor(vot_idx[perm], dtype=torch.long, device=self.device)
        y_t = torch.tensor(y_all[perm], dtype=torch.float32, device=self.device)
        extra_t = extra_t[perm]

        if self._has_item_features:
            item_cat_shuf = [cat[perm] for cat in item_cat]
            item_num_shuf = item_num[perm]

        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(1, self.epochs + 1):
            self._encoder.train(); self._predictor.train()
            if self._has_item_features:
                self._item_encoder.train()

            emb_all = self._encoder(data.x, data.edge_index)

            total_loss = torch.tensor(0.0, device=self.device)
            for b in range(n_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n_samples)

                item_emb = None
                if self._has_item_features:
                    item_emb = self._item_encoder(
                        [cat[start:end] for cat in item_cat_shuf],
                        item_num_shuf[start:end],
                    )
                scores = self._predictor(
                    emb_all[inv_t[start:end]],
                    emb_all[vot_t[start:end]],
                    item_feat=item_emb,
                    extra_feat=extra_t[start:end],
                )
                loss = F.binary_cross_entropy(scores, y_t[start:end])
                total_loss = total_loss + loss * ((end - start) / n_samples)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == 1 or epoch == self.epochs:
                with torch.no_grad():
                    pos_mask = y_all > 0.5
                    if pos_mask.sum() > 0 and (~pos_mask).sum() > 0:
                        emb_all_eval = self._encoder(data.x, data.edge_index)
                        inv_t_all = torch.tensor(inv_idx, dtype=torch.long, device=self.device)
                        vot_t_all = torch.tensor(vot_idx, dtype=torch.long, device=self.device)
                        extra_eval = self._extract_extra_features(X, valid)
                        item_emb_eval = None
                        if self._has_item_features:
                            item_emb_eval = self._item_encoder(item_cat, item_num)
                        all_scores = self._predictor(
                            emb_all_eval[inv_t_all], emb_all_eval[vot_t_all],
                            item_feat=item_emb_eval, extra_feat=extra_eval,
                        ).cpu().numpy()
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(y_all, all_scores)
                    else:
                        auc = float('nan')
                print(f"  [GNN-B] epoch {epoch:3d}/{self.epochs}  "
                      f"loss={total_loss.item():.4f}  auc={auc:.4f}")

        return self

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
                extra_t = self._extract_extra_features(X, valid_mask)
                item_emb = None
                if self._has_item_features:
                    item_cat, item_num = self._extract_item_tensors(X, valid_mask)
                    item_emb = self._item_encoder(item_cat, item_num)
                scores[valid_mask] = self._predictor(
                    emb[inv_t], emb[vot_t],
                    item_feat=item_emb, extra_feat=extra_t,
                ).cpu().numpy()
        return scores

    def evaluate_mrr(self, valid_df, model_a=None, n_queries=500):
        """MRR@5 评估 — item-aware + 品类交叉 + 分场景拆解"""
        print(f"  [MRR-GNN-B] 评估 {n_queries} 个查询 (含品类交叉)...")

        # ---- 重建训练期图 + 计算品类交叉特征 ----
        df_train_raw = pd.read_pickle(PROCESSED_DIR / "share_train.pkl")
        df_final_train_raw = pd.read_pickle(PROCESSED_DIR / "share_final_train.pkl")
        df_all_raw = pd.concat([df_train_raw, df_final_train_raw], ignore_index=True)
        df_all_raw["timestamp"] = pd.to_datetime(df_all_raw["timestamp"])
        split_date = pd.Timestamp("2022-10-29")
        train_raw = df_all_raw[df_all_raw["timestamp"] <= split_date]

        inviter_friends = train_raw.groupby("inviter_id")["voter_id"].apply(set).to_dict()

        # 时序进化特征 (同 lgb_baseline_D.py 的 6d 节)
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
        sh_pairs = second_half[["inviter_id", "voter_id"]].drop_duplicates()

        # ---- 预构建 O(1) 查询 dict (避免循环内 pandas filter) ----
        recent_pair_set = set()
        for inv_id, vot_id in sh_pairs.values:
            recent_pair_set.add((inv_id, vot_id))

        # 品类交叉特征 (同 lgb_baseline_D.py 的 6e 节)
        voter_cate_raw = train_raw[["voter_id", "item_id"]].merge(
            train_raw[["item_id", "cate_level1_id"]].drop_duplicates("item_id"),
            on="item_id", how="left")
        voter_cate_count = voter_cate_raw.groupby(["voter_id", "cate_level1_id"]).size().reset_index(name="cate_in_count")
        voter_total_in = voter_cate_count.groupby("voter_id")["cate_in_count"].sum().reset_index(name="total_in")
        voter_cate_count = voter_cate_count.merge(voter_total_in, on="voter_id")
        voter_cate_count["cate_match_score"] = voter_cate_count["cate_in_count"] / voter_cate_count["total_in"]

        voter_cate_count["cate_rank"] = voter_cate_count.groupby("voter_id")["cate_in_count"].rank(ascending=False, method="dense")
        voter_top3 = voter_cate_count[voter_cate_count["cate_rank"] <= 3].copy()
        voter_top3["item_cate_in_voter_top3"] = 1

        # ---- 预构建 O(1) cate dict ----
        voter_cate_score_dict = {}
        for voter_id, cate_id, score in voter_cate_count[
            ["voter_id", "cate_level1_id", "cate_match_score"]
        ].values:
            voter_cate_score_dict[(voter_id, cate_id)] = score

        voter_top3_set = set()
        for voter_id, cate_id in voter_top3[["voter_id", "cate_level1_id"]].values:
            voter_top3_set.add((voter_id, cate_id))

        inviter_cate_sets = train_raw.groupby("inviter_id")["cate_level1_id"].apply(set).to_dict()
        voter_cate_sets = train_raw.groupby("voter_id")["cate_level1_id"].apply(set).to_dict()

        # ---- 准备查询 ----
        query_cols = ["inviter_id", "item_id", "voter_id", "timestamp", "cate_level1_id"]
        if self._has_item_features:
            for col in self._item_cat_cols + self._item_num_cols:
                if col in valid_df.columns and col not in query_cols:
                    query_cols.append(col)
        # 只评估正样本 (有真实分享关系的查询)
        valid_pos = valid_df[valid_df["label"] == 1][query_cols].copy()
        valid_pos = valid_pos.rename(columns={"voter_id": "true_voter_id"})
        rng = np.random.default_rng(42)
        n_q = min(n_queries, len(valid_pos))
        q_idx = rng.choice(len(valid_pos), n_q, replace=False)
        queries = valid_pos.iloc[q_idx].reset_index(drop=True)

        # ---- 预计算 GNN-B 节点 embedding ----
        self._encoder.eval(); self._predictor.eval()
        if self._has_item_features:
            self._item_encoder.eval()
        with torch.no_grad():
            emb_all_b = self._encoder(self._data.x.to(self.device),
                                      self._data.edge_index.to(self.device))

        # ---- 预计算 GNN-A 节点 embedding (用 A 自己的 encoder) ----
        if model_a is not None:
            model_a._encoder.eval()
            with torch.no_grad():
                emb_all_a = model_a._encoder(
                    model_a._data.x.to(self.device),
                    model_a._data.edge_index.to(self.device))

        voter_pool = np.unique(np.concatenate([
            train_raw["voter_id"].values, train_raw["inviter_id"].values
        ]))

        ranks_a = []  # A 模型 (无 extra)
        ranks_b = []  # B 模型 (有 extra)
        friend_hit_list = []

        for i in range(n_q):
            inv = queries.iloc[i]["inviter_id"]
            true_v = queries.iloc[i]["true_voter_id"]
            cate_id = queries.iloc[i]["cate_level1_id"]
            friends = inviter_friends.get(inv, set())
            friend_hit_list.append(true_v in friends)

            candidates = {true_v}
            candidates.update(friends)
            while len(candidates) < 200:
                candidates.add(voter_pool[rng.integers(0, len(voter_pool))])
            cand_list = list(candidates)

            cand_idx = [self._node_to_idx.get(v, -1) for v in cand_list]
            valid_c = [(j, idx) for j, idx in enumerate(cand_idx) if idx >= 0]
            if not valid_c or inv not in self._node_to_idx:
                ranks_b.append(999)
                if model_a is not None:
                    ranks_a.append(999)
                continue

            j_map, idx_list = zip(*valid_c)
            idx_t = torch.tensor(idx_list, dtype=torch.long, device=self.device)
            cand_emb_b = emb_all_b[idx_t]
            inv_emb_b = emb_all_b[self._node_to_idx[inv]].unsqueeze(0).expand(len(idx_list), -1)
            # GNN-A 用自己 encoder 的 embedding
            if model_a is not None:
                cand_emb_a = emb_all_a[idx_t]
                inv_emb_a = emb_all_a[self._node_to_idx[inv]].unsqueeze(0).expand(len(idx_list), -1)
            n_c = len(idx_list)

            # item 特征 — GNN-B 用自己的 item encoder
            item_emb_b = None
            if self._has_item_features:
                q_cat, q_num = self._item_feat_for_query(queries.iloc[i])
                item_cat_expanded = [t.expand(n_c) for t in q_cat]
                item_num_expanded = q_num.expand(n_c, -1)
                item_emb_b = self._item_encoder(item_cat_expanded, item_num_expanded)

            # item 特征 — GNN-A 用自己的 item encoder
            item_emb_a = None
            if model_a is not None and model_a._has_item_features:
                q_cat_a, q_num_a = model_a._item_feat_for_query(queries.iloc[i])
                item_cat_expanded_a = [t.expand(n_c) for t in q_cat_a]
                item_num_expanded_a = q_num_a.expand(n_c, -1)
                item_emb_a = model_a._item_encoder(item_cat_expanded_a, item_num_expanded_a)

            # ---- GNN-B extra 特征 (6维: 时序×3 + 品类×3) ----
            # 只对图中存在的候选 (j_map) 计算, 索引用 k (0..n_c-1)
            extra_arr = np.zeros((n_c, 6), dtype=np.float32)
            for k, j in enumerate(j_map):
                cid = cand_list[j]
                # 时序: inviter_new_voter_ratio, inviter_voter_retention
                td = inv_temporal.get(inv, {})
                extra_arr[k, 0] = td.get("inviter_new_voter_ratio", 0.0)
                extra_arr[k, 1] = td.get("inviter_voter_retention", 0.0)
                # 时序: pair_is_recent (dict O(1))
                extra_arr[k, 2] = 1 if (inv, cid) in recent_pair_set else 0
                # 品类: cate_match_score (dict O(1))
                cid_int = int(cate_id) if pd.notna(cate_id) else -1
                extra_arr[k, 3] = voter_cate_score_dict.get((cid, cid_int), 0.0)
                # 品类: item_cate_in_voter_top3 (set O(1))
                extra_arr[k, 4] = 1 if (cid, cid_int) in voter_top3_set else 0
                # 品类: inviter_voter_cate_overlap
                sa = inviter_cate_sets.get(inv, set())
                sb = voter_cate_sets.get(cid, set())
                inter = len(sa & sb)
                union = len(sa | sb)
                extra_arr[k, 5] = inter / union if union > 0 else 0.0

            # 归一化
            if self._extra_mean is not None:
                extra_arr = (extra_arr - self._extra_mean) / (self._extra_std + 1e-8)
            extra_tensor = torch.tensor(extra_arr, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                # B 模型打分 (含 extra, 用 B 的 embedding + B 的 item encoder)
                scores_b = self._predictor(
                    inv_emb_b, cand_emb_b, item_feat=item_emb_b, extra_feat=extra_tensor
                ).cpu().numpy()
                sorted_b = np.argsort(-scores_b)
                rank_map_b = {cand_list[j_map[j]]: r + 1 for r, j in enumerate(sorted_b)}
                ranks_b.append(rank_map_b.get(true_v, 999))

                # A 模型打分 (无 extra, 用 A 自己的 embedding + A 自己的 item encoder)
                if model_a is not None:
                    scores_a = model_a._predictor(
                        inv_emb_a, cand_emb_a, item_feat=item_emb_a
                    ).cpu().numpy()
                    sorted_a = np.argsort(-scores_a)
                    rank_map_a = {cand_list[j_map[j]]: r + 1 for r, j in enumerate(sorted_a)}
                    ranks_a.append(rank_map_a.get(true_v, 999))

        # ---- 统计 ----
        ranks_b_arr = np.array(ranks_b, dtype=np.float64)
        mrr_b_g = float(np.mean(np.where(ranks_b_arr <= 5, 1.0 / ranks_b_arr, 0.0)))
        hits_b_g = float(np.mean(ranks_b_arr <= 5))

        friend_mask = np.array(friend_hit_list)
        mrrs_b = np.where(ranks_b_arr <= 5, 1.0 / ranks_b_arr, 0.0)
        mrr_b_f = float(np.mean(mrrs_b[friend_mask])) if friend_mask.sum() > 0 else 0.0
        mrr_b_s = float(np.mean(mrrs_b[~friend_mask])) if (~friend_mask).sum() > 0 else 0.0
        hits_b_f = float(np.mean(ranks_b_arr[friend_mask] <= 5)) if friend_mask.sum() > 0 else 0.0
        hits_b_s = float(np.mean(ranks_b_arr[~friend_mask] <= 5)) if (~friend_mask).sum() > 0 else 0.0

        result = {
            "mrr_global": mrr_b_g, "hits_global": hits_b_g,
            "mrr_friend": mrr_b_f, "hits_friend": hits_b_f, "n_friend": int(friend_mask.sum()),
            "mrr_stranger": mrr_b_s, "hits_stranger": hits_b_s, "n_stranger": int((~friend_mask).sum()),
        }

        if model_a is not None:
            ranks_a_arr = np.array(ranks_a, dtype=np.float64)
            mrr_a_g = float(np.mean(np.where(ranks_a_arr <= 5, 1.0 / ranks_a_arr, 0.0)))
            hits_a_g = float(np.mean(ranks_a_arr <= 5))
            mrrs_a = np.where(ranks_a_arr <= 5, 1.0 / ranks_a_arr, 0.0)
            mrr_a_f = float(np.mean(mrrs_a[friend_mask])) if friend_mask.sum() > 0 else 0.0
            mrr_a_s = float(np.mean(mrrs_a[~friend_mask])) if (~friend_mask).sum() > 0 else 0.0
            hits_a_f = float(np.mean(ranks_a_arr[friend_mask] <= 5)) if friend_mask.sum() > 0 else 0.0
            hits_a_s = float(np.mean(ranks_a_arr[~friend_mask] <= 5)) if (~friend_mask).sum() > 0 else 0.0
            result.update({
                "mrr_global_a": mrr_a_g, "hits_global_a": hits_a_g,
                "mrr_friend_a": mrr_a_f, "hits_friend_a": hits_a_f,
                "mrr_stranger_a": mrr_a_s, "hits_stranger_a": hits_a_s,
            })

            # 打印对比表
            print()
            print(f"  {'':>18} {'GNN-A':>12} {'GNN-B (品类交叉)':>18} {'Δ':>10}")
            print(f"  {'─'*58}")
            print(f"  {'全局 MRR@5':>18} {mrr_a_g:>12.5f} {mrr_b_g:>18.5f} {mrr_b_g-mrr_a_g:>+10.5f}")
            print(f"  {'全局 HITS@5':>18} {hits_a_g:>12.5f} {hits_b_g:>18.5f} {hits_b_g-hits_a_g:>+10.5f}")
            print(f"  {'─'*58}")
            print(f"  {'朋友组 MRR@5':>18} {mrr_a_f:>12.5f} {mrr_b_f:>18.5f} {mrr_b_f-mrr_a_f:>+10.5f}  (n={int(friend_mask.sum())})")
            print(f"  {'陌生人组 MRR@5':>18} {mrr_a_s:>12.5f} {mrr_b_s:>18.5f} {mrr_b_s-mrr_a_s:>+10.5f}  (n={int((~friend_mask).sum())})")
            print(f"  {'─'*58}")
            print(f"  {'朋友组 HITS@5':>18} {hits_a_f:>12.5f} {hits_b_f:>18.5f} {hits_b_f-hits_a_f:>+10.5f}")
            print(f"  {'陌生人组 HITS@5':>18} {hits_a_s:>12.5f} {hits_b_s:>18.5f} {hits_b_s-hits_a_s:>+10.5f}")
        else:
            print(f"  [MRR-GNN-B] MRR@5={mrr_b_g:.5f}  HITS@5={hits_b_g:.5f}")

        return result
