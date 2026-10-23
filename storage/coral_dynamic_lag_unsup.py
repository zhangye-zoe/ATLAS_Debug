#!/usr/bin/env python3
"""
CORAL-Optimized with Batch Fix
==============================

Goals
-----
1. Stronger cell-level multi-omics fusion
2. Source-only cross-omics prediction
3. Adaptive lag learning without velocity kinetic priors
4. Better dynamic downstream support (T3/T4)
5. Explicit batch-effect removal in shared / fused latent space

Main additions vs previous optimized version
--------------------------------------------
- Gradient Reversal Layer (GRL)
- Batch discriminator on fused latent h
- Batch discriminator on shared latents of each modality
- Keeps leakage-free source-only prediction
- Keeps lag-conditioned latent vector field
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances, roc_auc_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset

# optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_write_h5ad(adata: ad.AnnData, path: os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    adata.write_h5ad(str(path))


def to_dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def maybe_log1p(x: np.ndarray, assume_logged: bool) -> np.ndarray:
    if assume_logged:
        return x.astype(np.float32, copy=False)
    return np.log1p(np.clip(x, a_min=0, a_max=None)).astype(np.float32, copy=False)


def intersect_strings(*lists: Sequence[str]) -> List[str]:
    if not lists:
        return []
    out = set(lists[0])
    for arr in lists[1:]:
        out &= set(arr)
    return sorted(out)


def union_strings(*lists: Sequence[str]) -> List[str]:
    out = set()
    for arr in lists:
        out |= set(arr)
    return sorted(out)


def save_npz_dict(arrays: Dict[str, np.ndarray], path: os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(1)
    denom = w.sum().clamp_min(1.0) * pred.shape[1]
    return ((pred - target) ** 2 * w).sum() / denom


def entropy_loss(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ent = -(alpha * torch.log(alpha + eps)).sum(dim=1)
    return ent.mean()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def pearson_cellwise(pred: np.ndarray, true: np.ndarray) -> float:
    vals = [safe_corr(pred[i], true[i]) for i in range(pred.shape[0])]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def pearson_genewise(pred: np.ndarray, true: np.ndarray) -> float:
    vals = [safe_corr(pred[:, j], true[:, j]) for j in range(pred.shape[1])]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def knn_graph_indices(x: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    n_neighbors = int(min(n_neighbors, max(2, x.shape[0] - 1)))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    _, idx = nbrs.kneighbors(x)
    return idx


def localize_global_knn(batch_global_idx: np.ndarray, global_knn: np.ndarray, min_neighbors: int = 4) -> np.ndarray:
    pos = {int(g): i for i, g in enumerate(batch_global_idx.tolist())}
    local_lists: List[List[int]] = []
    batch_set = set(pos.keys())
    for g in batch_global_idx.tolist():
        neigh = [pos[int(j)] for j in global_knn[int(g)].tolist() if int(j) in batch_set and int(j) != int(g)]
        if len(neigh) < min_neighbors:
            fallback = [i for i in range(len(batch_global_idx)) if i != pos[int(g)]]
            neigh = fallback[:max(min_neighbors, 1)]
        local_lists.append(neigh[:max(min_neighbors, 1)])
    k = min(len(x) for x in local_lists) if local_lists else 1
    if k <= 0:
        return np.zeros((len(batch_global_idx), 1), dtype=np.int64)
    return np.asarray([x[:k] for x in local_lists], dtype=np.int64)


def smoothness_loss(latent: torch.Tensor, knn_idx_local: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(knn_idx_local, dtype=torch.long, device=latent.device)
    center = latent.unsqueeze(1).expand(-1, idx.shape[1], -1)
    neigh = latent[idx]
    return ((center - neigh) ** 2).mean()


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return torch.tensor(0.0, device=x.device)
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    kxx = torch.exp(-xx / (2 * sigma * sigma))
    kyy = torch.exp(-yy / (2 * sigma * sigma))
    kxy = torch.exp(-xy / (2 * sigma * sigma))
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def sinkhorn_transport(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 30) -> torch.Tensor:
    n, m = cost.shape
    a = torch.full((n,), 1.0 / max(n, 1), device=cost.device)
    b = torch.full((m,), 1.0 / max(m, 1), device=cost.device)
    K = torch.exp(-cost / epsilon).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v).clamp_min(1e-12)
        v = b / (K.t() @ u).clamp_min(1e-12)
    T = torch.diag(u) @ K @ torch.diag(v)
    return T / T.sum().clamp_min(1e-12)


def weighted_pair_loss(x: torch.Tensor, y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, y, p=2).pow(2)
    return (T * dist).sum()


def cosine_velocity_alignment(h: torch.Tensor, v: torch.Tensor, neigh: torch.Tensor) -> torch.Tensor:
    disp = neigh - h.unsqueeze(1)
    v_rep = v.unsqueeze(1).expand_as(disp)
    cos = F.cosine_similarity(v_rep, disp, dim=-1)
    return cos


def foscttm_from_embeddings(z1: np.ndarray, z2: np.ndarray) -> float:
    assert z1.shape[0] == z2.shape[0]
    d = pairwise_distances(z1, z2)
    ranks = []
    for i in range(d.shape[0]):
        true_d = d[i, i]
        ranks.append(np.mean(d[i] < true_d))
    return float(np.mean(ranks))


def paired_embedding_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    assert z1.shape[0] == z2.shape[0]
    return float(np.linalg.norm(z1 - z2, axis=1).mean())


def build_transition_matrix_from_velocity(
    h: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    n_neighbors: int = 30,
    beta_time: float = 2.0,
    beta_vel: float = 4.0,
) -> np.ndarray:
    idx = knn_graph_indices(h, n_neighbors=n_neighbors + 1)
    n = h.shape[0]
    P = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neigh = idx[i, 1:]
        disp = h[neigh] - h[i]
        disp_norm = np.linalg.norm(disp, axis=1) + 1e-8
        cos = (disp @ v[i]) / (disp_norm * (np.linalg.norm(v[i]) + 1e-8))
        dt = t[neigh] - t[i]
        dh = np.sum(disp ** 2, axis=1)
        score = beta_vel * cos + beta_time * dt - 0.5 * dh
        score = score - np.max(score)
        p = np.exp(score)
        p = p / np.clip(p.sum(), 1e-8, None)
        P[i, neigh] = p
    return P


def estimate_pseudotime_from_latent(h: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    adata = ad.AnnData(X=h)
    sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, max(3, h.shape[0] - 1)), use_rep="X")
    sc.tl.diffmap(adata)
    diffmap = np.asarray(adata.obsm["X_diffmap"])
    root_idx = int(np.argmin(diffmap[:, 0]))
    adata.uns["iroot"] = root_idx
    try:
        sc.tl.dpt(adata)
    except Exception:
        pass
    if "dpt_pseudotime" in adata.obs.columns:
        pt = np.asarray(adata.obs["dpt_pseudotime"]).astype(np.float32)
    else:
        pt = diffmap[:, 0].astype(np.float32)
        pt = pt - np.nanmin(pt)
        denom = np.nanmax(pt)
        if denom > 0:
            pt = pt / denom
    if np.isnan(pt).any():
        pt = np.nan_to_num(pt, nan=np.nanmedian(pt))
    return pt


def build_lag_bank(
    latent: np.ndarray,
    pseudotime: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    max_lag_bins: int = 4,
    k_per_bin: int = 6,
) -> np.ndarray:
    n, d = latent.shape
    bins = np.linspace(0, 1, max_lag_bins + 2)
    lag_bank = np.zeros((n, max_lag_bins, d), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(30, max(2, n - 1))).fit(latent)
    _, idx = nn.kneighbors(latent)
    if source_mask is None:
        source_mask = np.ones(n, dtype=bool)
    source_mask = np.asarray(source_mask).astype(bool)

    for i in range(n):
        pi = pseudotime[i]
        valid_past = source_mask & (pseudotime < pi)
        for b in range(max_lag_bins):
            lo = max(0.0, pi - bins[b + 1])
            hi = max(0.0, pi - bins[b])
            cand = np.where(valid_past & (pseudotime >= lo) & (pseudotime < hi))[0]
            if cand.size == 0:
                cand = np.array([j for j in idx[i].tolist() if source_mask[j] and pseudotime[j] < pi], dtype=int)
            if cand.size == 0:
                lag_bank[i, b] = latent[i]
                continue
            if cand.size > k_per_bin:
                c_h = latent[cand]
                dists = np.sum((c_h - latent[i]) ** 2, axis=1)
                cand = cand[np.argsort(dists)[:k_per_bin]]
            lag_bank[i, b] = latent[cand].mean(axis=0)
    return lag_bank


# =============================================================================
# Batch adversarial modules
# =============================================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class BatchDiscriminator(nn.Module):
    def __init__(self, input_dim: int, n_batches: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_batches),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Evaluation helpers
# =============================================================================

def _find_first_existing_key(container, candidates, kind="key", raise_error: bool = False):
    for k in candidates:
        if k in container:
            return k
    if raise_error:
        raise KeyError(f"Could not find any valid {kind}. Tried: {candidates}")
    return None


def _resolve_label_key(adata, preferred=None):
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["celltype", "cell_type", "annotation", "annot", "labels", "leiden"])
    return _find_first_existing_key(adata.obs.columns, candidates, kind="label key", raise_error=False)


def interclass_pairwise_accuracy(pred, ref, weighted=True):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[valid], ref[valid]
    correct, total = 0.0, 0.0
    n = len(pred)
    for i in range(n):
        for j in range(i + 1, n):
            if ref[i] == ref[j]:
                continue
            w = abs(ref[i] - ref[j]) if weighted else 1.0
            is_correct = (pred[i] < pred[j]) if ref[i] < ref[j] else (pred[j] < pred[i])
            correct += w * float(is_correct)
            total += w
    return correct / total if total > 0 else np.nan


def interclass_pairwise_auc(pred, ref, weighted=True):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[valid], ref[valid]
    classes = np.sort(np.unique(ref))
    aucs, weights = [], []
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            mask = (ref == a) | (ref == b)
            y = (ref[mask] == b).astype(int)
            s = pred[mask]
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, s)
            w = abs(b - a) if weighted else 1.0
            aucs.append(auc)
            weights.append(w)
    if len(aucs) == 0:
        return np.nan
    return float(np.average(np.asarray(aucs), weights=np.asarray(weights)))


def compute_t3_metrics(adata, time_key: str, reference_time_key: Optional[str] = None, label_key: Optional[str] = None, stage_order_map: Optional[dict] = None):
    pred = np.asarray(adata.obs[time_key]).astype(float)
    if reference_time_key is not None and reference_time_key in adata.obs.columns:
        ref = np.asarray(adata.obs[reference_time_key]).astype(float)
    else:
        if label_key is None:
            label_key = _resolve_label_key(adata)
        if stage_order_map is None:
            return {"POA": np.nan, "POAUC": np.nan, "n_eval": 0}
        labels = adata.obs[label_key].astype(str)
        ref = labels.map(stage_order_map).to_numpy(dtype=float)
    valid = np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() < 3:
        return {"POA": np.nan, "POAUC": np.nan, "n_eval": int(valid.sum())}
    pred_v, ref_v = pred[valid], ref[valid]
    return {
        "POA": float(interclass_pairwise_accuracy(pred_v, ref_v, weighted=True)),
        "POAUC": float(interclass_pairwise_auc(pred_v, ref_v, weighted=True)),
        "n_eval": int(valid.sum()),
    }


def compute_t4_metrics(adata, transition_matrix: np.ndarray, lineage_key: str):
    if lineage_key is None or lineage_key not in adata.obs.columns:
        return {"F1": np.nan, "AUROC": np.nan, "n_eval": 0, "status": "skipped_no_lineage_key"}
    y_true_str = adata.obs[lineage_key].astype(str).to_numpy()
    class_order = sorted(pd.unique(y_true_str))
    if len(class_order) < 2:
        return {"F1": np.nan, "AUROC": np.nan, "n_eval": 0, "status": "skipped_less_than_2_classes"}
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_true = np.array([class_to_idx[x] for x in y_true_str], dtype=int)
    t = np.asarray(adata.obs.get("CORAL_time", np.linspace(0, 1, adata.n_obs))).astype(float)
    terminal_mask = np.zeros(adata.n_obs, dtype=bool)
    for cls in class_order:
        cls_idx = np.where(y_true_str == cls)[0]
        top_k = max(1, int(0.1 * len(cls_idx)))
        keep = cls_idx[np.argsort(t[cls_idx])[-top_k:]]
        terminal_mask[keep] = True
    proba = np.zeros((adata.n_obs, len(class_order)), dtype=float)
    for i, cls in enumerate(class_order):
        cls_terminal = terminal_mask & (y_true_str == cls)
        if cls_terminal.sum() == 0:
            continue
        proba[:, i] = transition_matrix[:, cls_terminal].sum(axis=1)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    proba = proba / row_sum
    y_pred = proba.argmax(axis=1)
    f1 = f1_score(y_true, y_pred, average="macro")
    if len(class_order) == 2:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        y_bin = label_binarize(y_true, classes=np.arange(len(class_order)))
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    return {"F1": float(f1), "AUROC": float(auc), "n_eval": int(adata.n_obs), "status": "ok"}


# =============================================================================
# Data
# =============================================================================

@dataclasses.dataclass
class ModalitySpec:
    name: str
    path: str
    assume_logged: bool = True
    align_weight: float = 1.0


@dataclasses.dataclass
class LagEdge:
    source: str
    target: str
    weight: float = 1.0


class MultimodalRatioDataset(Dataset):
    def __init__(self, modalities: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], cell_ids: Sequence[str], batch_ids: Sequence[int]):
        self.modalities = {k: v.astype(np.float32, copy=False) for k, v in modalities.items()}
        self.masks = {k: v.astype(np.float32, copy=False) for k, v in masks.items()}
        self.modality_names = sorted(self.modalities.keys())
        self.cell_ids = np.asarray(cell_ids)
        self.batch_ids = np.asarray(batch_ids).astype(np.int64)
        self.n_obs = len(self.cell_ids)

    def __len__(self) -> int:
        return self.n_obs

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "modalities": {m: torch.from_numpy(self.modalities[m][idx]) for m in self.modality_names},
            "masks": {m: torch.tensor(self.masks[m][idx], dtype=torch.float32) for m in self.modality_names},
            "batch_id": torch.tensor(self.batch_ids[idx], dtype=torch.long),
        }


def collate_multimodal(batch: List[Dict[str, object]]) -> Dict[str, object]:
    modality_names = sorted(batch[0]["modalities"].keys())
    return {
        "index": torch.stack([b["index"] for b in batch], dim=0),
        "modalities": {m: torch.stack([b["modalities"][m] for b in batch], dim=0) for m in modality_names},
        "masks": {m: torch.stack([b["masks"][m] for b in batch], dim=0) for m in modality_names},
        "batch_id": torch.stack([b["batch_id"] for b in batch], dim=0),
    }


def parse_manifest(path: os.PathLike):
    raw = load_json(path)
    modalities = {
        name: ModalitySpec(
            name=name,
            path=cfg["path"],
            assume_logged=cfg.get("assume_logged", True),
            align_weight=float(cfg.get("align_weight", 1.0)),
        )
        for name, cfg in raw["modalities"].items()
    }
    alignment_pairs = [tuple(x) for x in raw.get("alignment_pairs", [])]
    prediction_pairs = [tuple(x) for x in raw.get("prediction_pairs", [])]
    lag_edges = [LagEdge(source=e["source"], target=e["target"], weight=float(e.get("weight", 1.0))) for e in raw.get("lag_graph", [])]
    label_key = raw.get("label_key")
    future_label_key = raw.get("future_label_key")
    reference_time_key = raw.get("reference_time_key")
    stage_order_map = raw.get("stage_order_map")
    return modalities, alignment_pairs, prediction_pairs, lag_edges, label_key, future_label_key, reference_time_key, stage_order_map


def load_or_rebuild_modality(path: Path, global_path: str, cells: Sequence[str]) -> ad.AnnData:
    if path.exists():
        return sc.read_h5ad(str(path))
    global_adata = sc.read_h5ad(str(global_path))
    common = [c for c in cells if c in global_adata.obs_names]
    if not common:
        raise ValueError(f"No requested cells found in global file: {global_path}")
    return global_adata[common].copy()


def _build_obs_union(adatas: Dict[str, ad.AnnData], union_ids: Sequence[str]) -> pd.DataFrame:
    union_ids = pd.Index([str(x) for x in union_ids], name="cell_id")
    merged = pd.DataFrame(index=union_ids)
    merged["cell_id"] = union_ids.astype(str)
    for m, a in adatas.items():
        obs = a.obs.copy()
        obs.index = obs.index.astype(str)
        obs = obs[~obs.index.duplicated(keep="first")]
        keep_cols = [c for c in obs.columns if c not in merged.columns]
        if keep_cols:
            merged = merged.join(obs[keep_cols], how="left")
        merged[f"has_{m}"] = merged.index.isin(obs.index).astype(int)
    return merged


def build_multimodal_ratio_dataset(split_dir: os.PathLike, modality_specs: Dict[str, ModalitySpec], label_key: Optional[str] = None, standardize: bool = True):
    split_dir = Path(split_dir)
    split_info = load_json(split_dir / "split_info.json")
    train_cells = split_info["train_cells"]
    val_cells = split_info.get("val_cells", split_info.get("val_query_cells", split_info.get("val_query_atac_cells", [])))
    if len(train_cells) == 0 or len(val_cells) == 0:
        raise ValueError(f"{split_dir}: train_cells or val_cells empty")

    train_adatas: Dict[str, ad.AnnData] = {}
    val_adatas: Dict[str, ad.AnnData] = {}
    feature_sets_train: List[List[str]] = []
    feature_sets_val: List[List[str]] = []

    for m, spec in modality_specs.items():
        train_candidates = [split_dir / f"train_{m}.h5ad"]
        val_candidates = [split_dir / f"val_{m}.h5ad"]
        if m in ["rna", "s"]:
            train_candidates.extend([split_dir / "train_rna_ref.h5ad"])
            val_candidates.extend([split_dir / "val_true_rna.h5ad"])
        elif m == "atac":
            train_candidates.extend([split_dir / "train_atac_activity.h5ad", split_dir / "train_atac_full.h5ad"])
            val_candidates.extend([split_dir / "val_atac_activity.h5ad", split_dir / "val_query_atac.h5ad"])
        train_path = next((p for p in train_candidates if p.exists()), train_candidates[0])
        val_path = next((p for p in val_candidates if p.exists()), val_candidates[0])
        train_adatas[m] = load_or_rebuild_modality(train_path, spec.path, train_cells)
        val_adatas[m] = load_or_rebuild_modality(val_path, spec.path, val_cells)
        feature_sets_train.append(train_adatas[m].var_names.tolist())
        feature_sets_val.append(val_adatas[m].var_names.tolist())

    common_features = intersect_strings(*feature_sets_train, *feature_sets_val)
    if len(common_features) < 32:
        raise ValueError(f"Too few common features across selected modalities: {len(common_features)}")

    train_union = sorted(union_strings(*[a.obs_names.tolist() for a in train_adatas.values()]))
    val_union = sorted(union_strings(*[a.obs_names.tolist() for a in val_adatas.values()]))
    train_pos = {c: i for i, c in enumerate(train_union)}
    val_pos = {c: i for i, c in enumerate(val_union)}

    train_arrays: Dict[str, np.ndarray] = {}
    train_masks: Dict[str, np.ndarray] = {}
    val_arrays: Dict[str, np.ndarray] = {}
    val_masks: Dict[str, np.ndarray] = {}
    val_arrays_raw: Dict[str, np.ndarray] = {}
    stats: Dict[str, dict] = {}

    for m, spec in modality_specs.items():
        tr = train_adatas[m][:, common_features].copy()
        va = val_adatas[m][:, common_features].copy()
        x_tr_raw = maybe_log1p(to_dense(tr.X), spec.assume_logged)
        x_va_raw = maybe_log1p(to_dense(va.X), spec.assume_logged)

        tr_full = np.zeros((len(train_union), len(common_features)), dtype=np.float32)
        va_full = np.zeros((len(val_union), len(common_features)), dtype=np.float32)
        va_raw_full = np.zeros((len(val_union), len(common_features)), dtype=np.float32)
        tr_mask = np.zeros(len(train_union), dtype=np.float32)
        va_mask = np.zeros(len(val_union), dtype=np.float32)

        for j, c in enumerate(tr.obs_names.tolist()):
            tr_full[train_pos[c]] = x_tr_raw[j]
            tr_mask[train_pos[c]] = 1.0
        for j, c in enumerate(va.obs_names.tolist()):
            va_full[val_pos[c]] = x_va_raw[j]
            va_raw_full[val_pos[c]] = x_va_raw[j]
            va_mask[val_pos[c]] = 1.0

        if standardize and tr_mask.sum() > 0:
            mu = tr_full[tr_mask > 0].mean(axis=0, keepdims=True)
            sd = tr_full[tr_mask > 0].std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            tr_full = ((tr_full - mu) / sd).astype(np.float32)
            va_full = ((va_full - mu) / sd).astype(np.float32)
            stats[m] = {"mean": mu.squeeze(0).astype(np.float32), "std": sd.squeeze(0).astype(np.float32)}
        else:
            stats[m] = {"mean": np.zeros(len(common_features), dtype=np.float32), "std": np.ones(len(common_features), dtype=np.float32)}

        train_arrays[m] = tr_full
        train_masks[m] = tr_mask
        val_arrays[m] = va_full
        val_arrays_raw[m] = va_raw_full
        val_masks[m] = va_mask

    labels = None
    val_obs_union = _build_obs_union(val_adatas, val_union)
    if label_key is not None and label_key in val_obs_union.columns:
        try:
            labels = val_obs_union.loc[val_union, label_key].astype(str).to_numpy()
        except Exception:
            labels = None

    train_obs_union = _build_obs_union(train_adatas, train_union)
    if "batch" not in train_obs_union.columns:
        raise ValueError("No 'batch' column found in training observations.")
    train_batch_series = train_obs_union.loc[train_union, "batch"].astype(str)
    batch_categories = sorted(train_batch_series.unique())
    batch_to_idx = {b: i for i, b in enumerate(batch_categories)}
    train_batch_ids = train_batch_series.map(batch_to_idx).to_numpy(dtype=np.int64)

    ds_train = MultimodalRatioDataset(train_arrays, train_masks, train_union, train_batch_ids)
    meta = {
        "split_info": split_info,
        "train_cell_ids": np.asarray(train_union),
        "val_cell_ids": np.asarray(val_union),
        "common_features": np.asarray(common_features),
        "val_modalities": val_arrays,
        "val_modalities_raw": val_arrays_raw,
        "val_masks": val_masks,
        "labels": labels,
        "label_key": label_key,
        "stats": stats,
        "val_obs_union": val_obs_union,
        "batch_categories": batch_categories,
        "train_batch_ids": train_batch_ids,
    }
    return ds_train, meta, val_adatas


def inverse_standardize(x: np.ndarray, stats: dict) -> np.ndarray:
    mu = np.asarray(stats["mean"], dtype=np.float32)
    sd = np.asarray(stats["std"], dtype=np.float32)
    return x * sd[None, :] + mu[None, :]


# =============================================================================
# Model
# =============================================================================

class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.1, last_activation: bool = False, ln: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or last_activation:
                if ln:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SplitEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], shared_dim: int, private_dim: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = MLP([input_dim, *hidden_dims], dropout=dropout, last_activation=True)
        hdim = hidden_dims[-1]
        self.shared_head = nn.Linear(hdim, shared_dim)
        self.private_head = nn.Linear(hdim, private_dim)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        zs = F.normalize(self.shared_head(h), dim=1)
        zp = self.private_head(h)
        return zs, zp


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = MLP([input_dim, *hidden_dims, output_dim], dropout=dropout, last_activation=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MaskAwareFusion(nn.Module):
    def __init__(self, shared_dim: int, n_modalities: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_modalities * shared_dim + n_modalities, max(shared_dim, 32)),
            nn.GELU(),
            nn.Linear(max(shared_dim, 32), n_modalities),
        )

    def forward(self, z_stack: torch.Tensor, mask_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, m, d = z_stack.shape
        gate_in = torch.cat([z_stack.reshape(b, m * d), mask_stack], dim=1)
        logits = self.gate(gate_in)
        logits = logits.masked_fill(mask_stack <= 0, -1e9)
        weights = F.softmax(logits, dim=1)
        h = torch.sum(weights.unsqueeze(-1) * z_stack, dim=1)
        return h, weights


class TimeHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h))


class DirectedLagModule(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, current_h: torch.Tensor, source_bank: torch.Tensor):
        b, k, d = source_bank.shape
        cur = current_h.unsqueeze(1).expand(b, k, d)
        diff = cur - source_bank
        score_in = torch.cat([cur, source_bank, diff], dim=-1)
        alpha = F.softmax(self.score(score_in).squeeze(-1), dim=1)
        ctx = torch.sum(alpha.unsqueeze(-1) * source_bank, dim=1)
        delta = self.update(torch.cat([current_h, ctx], dim=1))
        return ctx, delta, alpha


class VelocityFieldHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, lag_delta: torch.Tensor):
        return self.net(torch.cat([h, lag_delta], dim=1))


class CORALOptimizedBatchFix(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, int],
        prediction_pairs: Sequence[Tuple[str, str]],
        lag_edges: Sequence[LagEdge],
        n_batches: int,
        shared_dim: int = 32,
        private_dim: int = 16,
        enc_hidden: Sequence[int] = (256, 128),
        dec_hidden: Sequence[int] = (128, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_names = sorted(input_dims.keys())
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.n_batches = n_batches

        self.encoders = nn.ModuleDict({m: SplitEncoder(d, enc_hidden, shared_dim, private_dim, dropout) for m, d in input_dims.items()})
        self.current_decoders = nn.ModuleDict({m: Decoder(shared_dim + private_dim, dec_hidden, d, dropout) for m, d in input_dims.items()})
        self.prediction_heads = nn.ModuleDict({f"{src}__to__{tgt}": Decoder(shared_dim + private_dim, dec_hidden, input_dims[tgt], dropout) for src, tgt in prediction_pairs})
        # Dynamic lag-prediction heads are generic for any manifest edge:
        # ATAC->RNA, unspliced->spliced, RNA->protein, etc.
        self.lag_prediction_heads = nn.ModuleDict({
            f"{e.source}__to__{e.target}": Decoder(shared_dim, dec_hidden, input_dims[e.target], dropout)
            for e in lag_edges
            if e.target in input_dims
        })
        self.fusion = MaskAwareFusion(shared_dim=shared_dim, n_modalities=len(self.modality_names))
        self.shared_refine = MLP([shared_dim, shared_dim, shared_dim], dropout=dropout, last_activation=False)
        self.time_head = TimeHead(shared_dim)

        self.lag_modules = nn.ModuleDict({f"{e.source}__to__{e.target}": DirectedLagModule(shared_dim) for e in lag_edges})
        self.field_heads = nn.ModuleDict({f"{e.source}__to__{e.target}": VelocityFieldHead(shared_dim) for e in lag_edges})
        self.edge_weights = {f"{e.source}__to__{e.target}": float(e.weight) for e in lag_edges}
        self.target_to_edges: Dict[str, List[str]] = defaultdict(list)
        for e in lag_edges:
            self.target_to_edges[e.target].append(f"{e.source}__to__{e.target}")

        # batch discriminators
        self.batch_disc_h = BatchDiscriminator(shared_dim, n_batches, hidden_dim=128, dropout=dropout)
        self.batch_disc_shared = BatchDiscriminator(shared_dim, n_batches, hidden_dim=128, dropout=dropout)

    def encode(self, modalities: Dict[str, torch.Tensor]):
        shared, private = {}, {}
        for m in self.modality_names:
            hs, hp = self.encoders[m](modalities[m])
            shared[m], private[m] = hs, hp
        return shared, private

    def fuse_shared(self, shared: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]):
        z_stack = torch.stack([shared[m] for m in self.modality_names], dim=1)
        mask_stack = torch.stack([masks[m] for m in self.modality_names], dim=1)
        h, fusion_weights = self.fusion(z_stack, mask_stack)
        h = self.shared_refine(h)
        return h, fusion_weights

    def decode_current(self, h_shared: torch.Tensor, private: Dict[str, torch.Tensor]):
        out = {}
        for m in self.modality_names:
            out[m] = self.current_decoders[m](torch.cat([h_shared, private[m]], dim=1))
        return out

    def decode_prediction(self, shared: Dict[str, torch.Tensor], private: Dict[str, torch.Tensor]):
        out = {}
        for key, head in self.prediction_heads.items():
            src, tgt = key.split("__to__")
            out[key] = head(torch.cat([shared[src], private[src]], dim=1))
        return out

    def apply_directed_lag(self, h: torch.Tensor, lag_banks: Dict[str, torch.Tensor]):
        lag_contexts: Dict[str, torch.Tensor] = {}
        lag_delta_by_edge: Dict[str, torch.Tensor] = {}
        lag_alpha: Dict[str, torch.Tensor] = {}
        target_velocity: Dict[str, torch.Tensor] = {}

        for target in self.modality_names:
            edge_names = self.target_to_edges.get(target, [])
            proposals = []
            weights = []
            for edge_name in edge_names:
                source = edge_name.split("__to__")[0]
                if source not in lag_banks:
                    continue
                ctx, delta, alpha = self.lag_modules[edge_name](h, lag_banks[source])
                vel = self.field_heads[edge_name](h, delta)
                proposals.append(vel)
                weights.append(self.edge_weights[edge_name])
                lag_contexts[edge_name] = ctx
                lag_delta_by_edge[edge_name] = delta
                lag_alpha[edge_name] = alpha
            if proposals:
                w = torch.tensor(weights, dtype=h.dtype, device=h.device)
                w = w / w.sum().clamp_min(1e-8)
                stacked = torch.stack(proposals, dim=1)
                target_velocity[target] = torch.sum(stacked * w.view(1, -1, 1), dim=1)
            else:
                target_velocity[target] = torch.zeros_like(h)
        return target_velocity, lag_contexts, lag_delta_by_edge, lag_alpha

    def forward(self, modalities: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], lag_banks: Optional[Dict[str, torch.Tensor]] = None):
        shared, private = self.encode(modalities)
        h, fusion_weights = self.fuse_shared(shared, masks)
        t = self.time_head(h)
        current_rec = self.decode_current(h, private)
        pred_rec = self.decode_prediction(shared, private)

        out = {
            "shared": shared,
            "private": private,
            "h": h,
            "t": t,
            "fusion_weights": fusion_weights,
            "current_rec": current_rec,
            "pred_rec": pred_rec,
        }

        if lag_banks is not None and len(self.lag_modules) > 0:
            target_velocity, lag_contexts, lag_delta_by_edge, lag_alpha = self.apply_directed_lag(h, lag_banks)
            out["target_velocity"] = target_velocity
            out["lag_contexts"] = lag_contexts
            out["lag_delta_by_edge"] = lag_delta_by_edge
            out["lag_alpha"] = lag_alpha
            lag_pred_rec = {}
            for edge_name, ctx in lag_contexts.items():
                if edge_name in self.lag_prediction_heads:
                    lag_pred_rec[edge_name] = self.lag_prediction_heads[edge_name](ctx)
            out["lag_pred_rec"] = lag_pred_rec

        return out


# =============================================================================
# Training
# =============================================================================

@dataclasses.dataclass
class TrainConfig:
    output_dir: str
    epochs_phase1: int = 80
    epochs_phase2: int = 60
    epochs_phase3: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    shared_dim: int = 32
    private_dim: int = 16
    dropout: float = 0.1

    mmd_weight: float = 0.2
    align_weight: float = 1.0
    recon_weight: float = 1.0
    pred_weight: float = 1.5

    topo_weight: float = 0.2
    time_cons_weight: float = 0.4
    forward_weight: float = 0.5
    dyn_contrast_weight: float = 1.0
    lag_weight: float = 0.5
    lag_entropy_weight: float = 0.05
    fusion_entropy_weight: float = 0.02
    align_contrast_weight: float = 0.5

    # batch fix
    batch_adv_weight: float = 1.0
    batch_shared_weight: float = 0.7
    grl_lambda: float = 1.0

    grad_clip: float = 5.0
    device: str = "cuda"
    num_workers: int = 0
    save_every: int = 25
    topo_k: int = 20
    topo_min_local: int = 4
    bank_warmup_epochs: int = 5
    lag_bins: int = 4
    future_margin: float = 0.02

    # Phase-3 unsupervised dynamic improvements. These do not use labels.
    lag_pred_weight: float = 0.3
    time_spread_weight: float = 0.05
    time_smooth_weight: float = 0.05
    time_std_target: float = 0.18
    time_head_blend: float = 0.35
    transition_weight: float = 0.05


@dataclasses.dataclass
class FreezePlan:
    freeze_encoders: bool = False
    freeze_fusion: bool = False
    freeze_prediction: bool = False
    freeze_dynamics: bool = False


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def apply_freeze_plan(model: CORALOptimizedBatchFix, plan: FreezePlan) -> None:
    set_requires_grad(model.encoders, not plan.freeze_encoders)
    set_requires_grad(model.fusion, not plan.freeze_fusion)
    set_requires_grad(model.shared_refine, not plan.freeze_fusion)
    set_requires_grad(model.current_decoders, not plan.freeze_encoders)
    set_requires_grad(model.prediction_heads, not plan.freeze_prediction)
    set_requires_grad(model.time_head, not plan.freeze_dynamics)
    set_requires_grad(model.lag_modules, not plan.freeze_dynamics)
    set_requires_grad(model.field_heads, not plan.freeze_dynamics)
    if hasattr(model, "lag_prediction_heads"):
        set_requires_grad(model.lag_prediction_heads, not plan.freeze_dynamics)
    set_requires_grad(model.batch_disc_h, True)
    set_requires_grad(model.batch_disc_shared, True)


def build_alignment_losses(out, modality_specs, alignment_pairs, masks):
    shared = out["shared"]
    device = next(iter(shared.values())).device
    mmd_total = torch.tensor(0.0, device=device)
    align_total = torch.tensor(0.0, device=device)
    contrast_total = torch.tensor(0.0, device=device)

    pairs = list(alignment_pairs) if alignment_pairs else [(a, b) for i, a in enumerate(sorted(shared.keys())) for b in sorted(shared.keys())[i + 1:]]

    for a, b in pairs:
        paired_mask = (masks[a] > 0) & (masks[b] > 0)
        if paired_mask.sum() < 2:
            continue
        za = shared[a][paired_mask]
        zb = shared[b][paired_mask]

        mmd_ab = compute_mmd_rbf(za, zb, sigma=1.0)
        cost = torch.cdist(za, zb, p=2).pow(2)
        T = sinkhorn_transport(cost, epsilon=0.05, n_iter=20)
        align_ab = weighted_pair_loss(za, zb, T)

        logits = za @ zb.t() / 0.2
        labels = torch.arange(logits.shape[0], device=device)
        contrast_ab = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        w = 0.5 * (modality_specs[a].align_weight + modality_specs[b].align_weight)
        mmd_total = mmd_total + w * mmd_ab
        align_total = align_total + w * align_ab
        contrast_total = contrast_total + w * contrast_ab

    return mmd_total, align_total, contrast_total


def build_reconstruction_loss(out, x, masks, modality_names):
    recon = torch.tensor(0.0, device=next(iter(x.values())).device)
    for m in modality_names:
        recon = recon + masked_mse(out["current_rec"][m], x[m], masks[m])
    return recon


def build_prediction_loss(out, x, masks, prediction_pairs):
    loss = torch.tensor(0.0, device=next(iter(x.values())).device)
    n_used = 0
    for src, tgt in prediction_pairs:
        key = f"{src}__to__{tgt}"
        paired_mask = ((masks[src] > 0) & (masks[tgt] > 0)).float()
        if paired_mask.sum() < 1:
            continue
        loss = loss + masked_mse(out["pred_rec"][key], x[tgt], paired_mask)
        n_used += 1
    if n_used == 0:
        return torch.tensor(0.0, device=next(iter(x.values())).device)
    return loss / n_used


def build_batch_adv_loss(model, out, batch_ids, masks, modality_names, grl_lambda=1.0):
    device = batch_ids.device
    loss_h = torch.tensor(0.0, device=device)
    loss_s = torch.tensor(0.0, device=device)

    h_rev = grad_reverse(out["h"], grl_lambda)
    logits_h = model.batch_disc_h(h_rev)
    loss_h = F.cross_entropy(logits_h, batch_ids)

    s_losses = []
    for m in modality_names:
        obs_mask = masks[m] > 0
        if obs_mask.sum() < 2:
            continue
        z_rev = grad_reverse(out["shared"][m][obs_mask], grl_lambda)
        logits_s = model.batch_disc_shared(z_rev)
        s_losses.append(F.cross_entropy(logits_s, batch_ids[obs_mask]))

    if len(s_losses) > 0:
        loss_s = torch.stack(s_losses).mean()

    return loss_h, loss_s


def _minmax01_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def build_global_dynamic_artifacts(model: CORALOptimizedBatchFix, ds: MultimodalRatioDataset, cfg: TrainConfig, device: torch.device):
    """Build unsupervised pseudo-dynamic artifacts for phase3.

    Uses a blend of model time_head and DPT on the fused latent. No cell-type,
    stage-order, or future label is used for training.
    """
    model.eval()
    with torch.no_grad():
        all_mod_t = {m: torch.from_numpy(ds.modalities[m]).to(device) for m in ds.modality_names}
        all_mask_t = {m: torch.from_numpy(ds.masks[m]).to(device) for m in ds.modality_names}
        out_all = model(all_mod_t, all_mask_t)
        h_np = out_all["h"].detach().cpu().numpy()
        t_head_np = _minmax01_np(out_all["t"].detach().cpu().numpy().reshape(-1))
        t_dpt_np = _minmax01_np(estimate_pseudotime_from_latent(h_np, n_neighbors=cfg.topo_k))
        w = float(np.clip(cfg.time_head_blend, 0.0, 1.0))
        t_np = _minmax01_np(w * t_head_np + (1.0 - w) * t_dpt_np)
        global_knn = knn_graph_indices(h_np, n_neighbors=cfg.topo_k + 1)
        lag_banks_np = {
            m: build_lag_bank(
                h_np,
                t_np,
                source_mask=ds.masks[m] > 0,
                max_lag_bins=cfg.lag_bins,
                k_per_bin=6,
            )
            for m in ds.modality_names
        }
    return h_np, t_np, global_knn, lag_banks_np


def build_lag_prediction_loss(out, x, masks, cfg: TrainConfig):
    """Predict current target modality from selected source lag context.

    This makes adaptive lag learning predictive and modality-agnostic without
    using labels. It is active only for lag edges present in the manifest.
    """
    device = next(iter(x.values())).device
    loss = torch.tensor(0.0, device=device)
    n_used = 0
    if "lag_pred_rec" not in out or len(out["lag_pred_rec"]) == 0:
        return loss
    for edge_name, pred in out["lag_pred_rec"].items():
        if "__to__" not in edge_name:
            continue
        _, tgt = edge_name.split("__to__", 1)
        if tgt not in x or tgt not in masks:
            continue
        obs_mask = (masks[tgt] > 0).float()
        if obs_mask.sum() < 2:
            continue
        loss = loss + masked_mse(pred, x[tgt], obs_mask)
        n_used += 1
    return loss / n_used if n_used > 0 else torch.tensor(0.0, device=device)


def build_time_regularization_losses(out, batch_idx_np, global_knn, cfg: TrainConfig, device: torch.device):
    """Light unsupervised anti-collapse + local smoothness for time_head."""
    t = out["t"].reshape(-1)
    target_std = torch.tensor(cfg.time_std_target, dtype=t.dtype, device=device)
    time_spread = F.relu(target_std - torch.std(t))
    time_smooth = torch.tensor(0.0, device=device)
    if global_knn is not None and t.numel() > 2:
        knn_local = localize_global_knn(batch_idx_np, global_knn, min_neighbors=cfg.topo_min_local)
        idx = torch.as_tensor(knn_local, dtype=torch.long, device=device)
        neigh_t = t[idx]
        time_smooth = torch.mean(torch.abs(t.unsqueeze(1) - neigh_t))
    return time_spread, time_smooth


def build_transition_consistency_loss(out, batch_idx_np, global_knn, global_t, cfg: TrainConfig, device: torch.device):
    """Encourage h + velocity to move toward pseudo-future local neighbors."""
    if global_knn is None or "target_velocity" not in out or len(out["target_velocity"]) == 0:
        return torch.tensor(0.0, device=device)
    knn_local = localize_global_knn(batch_idx_np, global_knn, min_neighbors=cfg.topo_min_local)
    idx = torch.as_tensor(knn_local, dtype=torch.long, device=device)
    neigh_h = out["h"][idx]
    t_batch = torch.as_tensor(global_t[batch_idx_np], dtype=out["h"].dtype, device=device).reshape(-1)
    neigh_t = t_batch[idx]
    future_mask = (neigh_t > (t_batch.unsqueeze(1) + cfg.future_margin)).float()
    if future_mask.sum() < 1:
        return torch.tensor(0.0, device=device)
    future_center = (future_mask.unsqueeze(-1) * neigh_h).sum(dim=1) / future_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    valid = (future_mask.sum(dim=1) > 0).float().unsqueeze(1)
    all_vel = torch.stack(list(out["target_velocity"].values()), dim=1).mean(dim=1)
    h_next = out["h"] + 0.1 * all_vel
    return (((h_next - future_center) ** 2) * valid).sum() / (valid.sum().clamp_min(1.0) * out["h"].shape[1])


def build_dynamic_losses(out, batch_idx_np, global_knn, global_t, cfg: TrainConfig, device: torch.device):
    topo = torch.tensor(0.0, device=device)
    time_cons = torch.tensor(0.0, device=device)
    forward = torch.tensor(0.0, device=device)
    lag_entropy = torch.tensor(0.0, device=device)
    lag_mag = torch.tensor(0.0, device=device)
    dyn_contrast = torch.tensor(0.0, device=device)
    time_spread = torch.tensor(0.0, device=device)
    time_smooth = torch.tensor(0.0, device=device)
    transition = torch.tensor(0.0, device=device)

    if global_knn is None:
        return topo, time_cons, forward, lag_entropy, lag_mag, dyn_contrast, time_spread, time_smooth, transition

    knn_local = localize_global_knn(batch_idx_np, global_knn, min_neighbors=cfg.topo_min_local)
    topo = smoothness_loss(out["h"], knn_local)

    t_batch = torch.as_tensor(global_t[batch_idx_np], dtype=out["h"].dtype, device=device).unsqueeze(1)
    idx = torch.as_tensor(knn_local, dtype=torch.long, device=device)
    neigh_h = out["h"][idx]
    neigh_t = t_batch.squeeze(1)[idx]

    if "target_velocity" in out and len(out["target_velocity"]) > 0:
        all_vel = torch.stack(list(out["target_velocity"].values()), dim=1).mean(dim=1)
        cos = cosine_velocity_alignment(out["h"], all_vel, neigh_h)
        future_mask = (neigh_t > (t_batch + cfg.future_margin)).float()
        past_mask = (neigh_t < (t_batch - cfg.future_margin)).float()

        pos = (cos * future_mask).sum() / future_mask.sum().clamp_min(1.0) if future_mask.sum() > 0 else torch.tensor(0.0, device=device)
        neg = (cos * past_mask).sum() / past_mask.sum().clamp_min(1.0) if past_mask.sum() > 0 else torch.tensor(0.0, device=device)

        dyn_contrast = F.relu(0.2 - pos + neg)
        forward = F.relu(0.1 - pos)
        lag_mag = all_vel.norm(dim=1).mean()

        future_center = (future_mask.sum(dim=1, keepdim=True) > 0).float()
        future_t_mean = (future_mask * neigh_t).sum(dim=1, keepdim=True) / future_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        time_cons = (((t_batch - future_t_mean) ** 2) * future_center).sum() / future_center.sum().clamp_min(1.0)
        transition = build_transition_consistency_loss(out, batch_idx_np, global_knn, global_t, cfg, device)

    if "lag_alpha" in out and len(out["lag_alpha"]) > 0:
        lag_entropy_terms = [entropy_loss(alpha) for alpha in out["lag_alpha"].values()]
        lag_entropy = torch.stack(lag_entropy_terms).mean()

    time_spread, time_smooth = build_time_regularization_losses(out, batch_idx_np, global_knn, cfg, device)
    return topo, time_cons, forward, lag_entropy, lag_mag, dyn_contrast, time_spread, time_smooth, transition


def run_training_phase(
    model: CORALOptimizedBatchFix,
    ds: MultimodalRatioDataset,
    modality_specs: Dict[str, ModalitySpec],
    alignment_pairs,
    prediction_pairs,
    cfg: TrainConfig,
    writer: SummaryWriter,
    phase_name: str,
    epochs: int,
    start_epoch: int,
    freeze_plan: FreezePlan,
):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    apply_freeze_plan(model, freeze_plan)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    history = []
    best_loss = math.inf
    best_ckpt = None
    global_knn = None
    global_t = None
    lag_banks_np = None

    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch + local_epoch - 1

        if phase_name == "phase3":
            _, global_t, global_knn, lag_banks_np = build_global_dynamic_artifacts(model, ds, cfg, device)

        model.train()
        epoch_stats = defaultdict(float)

        for batch in loader:
            batch_idx_np = batch["index"].detach().cpu().numpy()
            x = {m: batch["modalities"][m].to(device) for m in ds.modality_names}
            masks = {m: batch["masks"][m].to(device) for m in ds.modality_names}
            batch_ids = batch["batch_id"].to(device)

            lag_banks_batch = None
            if phase_name == "phase3" and lag_banks_np is not None and len(model.lag_modules) > 0:
                lag_banks_batch = {m: torch.from_numpy(lag_banks_np[m][batch_idx_np]).to(device) for m in ds.modality_names}

            out = model(x, masks, lag_banks=lag_banks_batch)

            recon = build_reconstruction_loss(out, x, masks, ds.modality_names)
            pred = build_prediction_loss(out, x, masks, prediction_pairs)
            mmd, align, align_contrast = build_alignment_losses(out, modality_specs, alignment_pairs, masks)
            fusion_entropy = entropy_loss(out["fusion_weights"])
            batch_adv_h, batch_adv_s = build_batch_adv_loss(
                model,
                out,
                batch_ids=batch_ids,
                masks=masks,
                modality_names=ds.modality_names,
                grl_lambda=cfg.grl_lambda,
            )

            topo = torch.tensor(0.0, device=device)
            time_cons = torch.tensor(0.0, device=device)
            forward = torch.tensor(0.0, device=device)
            lag_entropy = torch.tensor(0.0, device=device)
            lag_mag = torch.tensor(0.0, device=device)
            dyn_contrast = torch.tensor(0.0, device=device)
            time_spread = torch.tensor(0.0, device=device)
            time_smooth = torch.tensor(0.0, device=device)
            transition = torch.tensor(0.0, device=device)
            lag_pred = torch.tensor(0.0, device=device)

            if phase_name == "phase3":
                topo, time_cons, forward, lag_entropy, lag_mag, dyn_contrast, time_spread, time_smooth, transition = build_dynamic_losses(
                    out, batch_idx_np, global_knn, global_t, cfg, device
                )
                lag_pred = build_lag_prediction_loss(out, x, masks, cfg)

            if phase_name == "phase1":
                loss = (
                    cfg.recon_weight * recon
                    + cfg.mmd_weight * mmd
                    + cfg.align_weight * align
                    + cfg.align_contrast_weight * align_contrast
                    + cfg.fusion_entropy_weight * fusion_entropy
                    + cfg.batch_adv_weight * batch_adv_h
                    + cfg.batch_shared_weight * batch_adv_s
                )
            elif phase_name == "phase2":
                loss = (
                    cfg.recon_weight * recon
                    + cfg.pred_weight * pred
                    + 0.5 * cfg.mmd_weight * mmd
                    + 0.5 * cfg.align_weight * align
                    + cfg.align_contrast_weight * align_contrast
                    + cfg.fusion_entropy_weight * fusion_entropy
                    + cfg.batch_adv_weight * batch_adv_h
                    + cfg.batch_shared_weight * batch_adv_s
                )
            else:
                loss = (
                    cfg.recon_weight * recon
                    + cfg.pred_weight * pred
                    + 0.5 * cfg.mmd_weight * mmd
                    + 0.5 * cfg.align_weight * align
                    + cfg.align_contrast_weight * align_contrast
                    + cfg.topo_weight * topo
                    + cfg.time_cons_weight * time_cons
                    + cfg.forward_weight * forward
                    + cfg.dyn_contrast_weight * dyn_contrast
                    + cfg.lag_entropy_weight * lag_entropy
                    + cfg.lag_weight * lag_mag
                    + cfg.lag_pred_weight * lag_pred
                    + cfg.time_spread_weight * time_spread
                    + cfg.time_smooth_weight * time_smooth
                    + cfg.transition_weight * transition
                    + cfg.fusion_entropy_weight * fusion_entropy
                    + cfg.batch_adv_weight * batch_adv_h
                    + cfg.batch_shared_weight * batch_adv_s
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            epoch_stats["loss"] += float(loss.detach().cpu())
            epoch_stats["recon"] += float(recon.detach().cpu())
            epoch_stats["pred"] += float(pred.detach().cpu())
            epoch_stats["mmd"] += float(mmd.detach().cpu())
            epoch_stats["align"] += float(align.detach().cpu())
            epoch_stats["align_contrast"] += float(align_contrast.detach().cpu())
            epoch_stats["topo"] += float(topo.detach().cpu())
            epoch_stats["time_cons"] += float(time_cons.detach().cpu())
            epoch_stats["forward"] += float(forward.detach().cpu())
            epoch_stats["dyn_contrast"] += float(dyn_contrast.detach().cpu())
            epoch_stats["lag_entropy"] += float(lag_entropy.detach().cpu())
            epoch_stats["lag_mag"] += float(lag_mag.detach().cpu())
            epoch_stats["lag_pred"] += float(lag_pred.detach().cpu())
            epoch_stats["time_spread"] += float(time_spread.detach().cpu())
            epoch_stats["time_smooth"] += float(time_smooth.detach().cpu())
            epoch_stats["transition"] += float(transition.detach().cpu())
            epoch_stats["fusion_entropy"] += float(fusion_entropy.detach().cpu())
            epoch_stats["batch_adv_h"] += float(batch_adv_h.detach().cpu())
            epoch_stats["batch_adv_s"] += float(batch_adv_s.detach().cpu())

        denom = max(len(loader), 1)
        row = {k: v / denom for k, v in epoch_stats.items()}
        row["epoch"] = epoch
        row["phase"] = phase_name
        history.append(row)

        for k, v in row.items():
            if k not in ["epoch", "phase"]:
                writer.add_scalar(f"{phase_name}/{k}", v, epoch)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (local_epoch % cfg.save_every == 0) or (local_epoch == epochs):
            ckpt_path = Path(cfg.output_dir) / f"{phase_name}_epoch_{epoch}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "phase": phase_name}, ckpt_path)

        print(
            f"[{phase_name}] epoch={epoch} "
            f"loss={row.get('loss', 0):.4f} "
            f"recon={row.get('recon', 0):.4f} "
            f"pred={row.get('pred', 0):.4f} "
            f"align={row.get('align', 0):.4f} "
            f"batchh={row.get('batch_adv_h', 0):.4f} "
            f"batchs={row.get('batch_adv_s', 0):.4f} "
            f"dyn={row.get('dyn_contrast', 0):.4f} "
            f"topo={row.get('topo', 0):.4f} "
            f"lagmag={row.get('lag_mag', 0):.4f} "
            f"lagpred={row.get('lag_pred', 0):.4f} "
            f"tspread={row.get('time_spread', 0):.4f}"
        )

    if best_ckpt is not None:
        model.load_state_dict(best_ckpt)
    return model, history, start_epoch + epochs


# =============================================================================
# Inference and evaluation
# =============================================================================

def infer_on_arrays(
    model: CORALOptimizedBatchFix,
    arrays: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    device: torch.device,
    lag_banks_np: Optional[Dict[str, np.ndarray]] = None,
):
    model.eval()
    with torch.no_grad():
        x = {m: torch.from_numpy(arrays[m]).to(device) for m in sorted(arrays.keys())}
        ms = {m: torch.from_numpy(masks[m]).to(device) for m in sorted(masks.keys())}
        lag_banks = None
        if lag_banks_np is not None and len(model.lag_modules) > 0:
            lag_banks = {m: torch.from_numpy(lag_banks_np[m]).to(device) for m in sorted(lag_banks_np.keys())}
        out = model(x, ms, lag_banks=lag_banks)
        pred = {k: v.detach().cpu().numpy() for k, v in out["pred_rec"].items()}
        current = {k: v.detach().cpu().numpy() for k, v in out["current_rec"].items()}
        h = out["h"].detach().cpu().numpy()
        t_head = _minmax01_np(out["t"].detach().cpu().numpy().reshape(-1))
        t_dpt = _minmax01_np(estimate_pseudotime_from_latent(h, n_neighbors=20))
        t = _minmax01_np(0.5 * t_head + 0.5 * t_dpt)
        shared = {k: v.detach().cpu().numpy() for k, v in out["shared"].items()}
        private = {k: v.detach().cpu().numpy() for k, v in out["private"].items()}
        target_velocity = None
        lag_pred_rec = None
        if "target_velocity" in out:
            target_velocity = {k: v.detach().cpu().numpy() for k, v in out["target_velocity"].items()}
        if "lag_pred_rec" in out:
            lag_pred_rec = {k: v.detach().cpu().numpy() for k, v in out["lag_pred_rec"].items()}
    return {
        "pred": pred,
        "current": current,
        "h": h,
        "t": t,
        "t_head": t_head,
        "t_dpt": t_dpt,
        "shared": shared,
        "private": private,
        "target_velocity": target_velocity,
        "lag_pred_rec": lag_pred_rec,
    }


def resolve_query_ids_for_pair(meta, src: str, tgt: str) -> np.ndarray:
    split_info = meta["split_info"]
    candidates = [f"val_query_{src}_cells", f"val_query_{src}_cell_ids"]
    if src == "atac":
        candidates.append("val_query_atac_cells")
    if src in ["rna", "s"]:
        candidates.append("val_query_rna_cells")
    candidates.extend(["val_query_cells", "val_cells"])
    for k in candidates:
        if k in split_info and len(split_info[k]) > 0:
            return np.asarray([str(x) for x in split_info[k]])
    return meta["val_cell_ids"].astype(str)


def evaluate_t1(meta, infer_res, alignment_pairs, output_dir: Path):
    rows = []
    for a, b in alignment_pairs:
        mask = (meta["val_masks"][a] > 0) & (meta["val_masks"][b] > 0)
        if mask.sum() < 2:
            continue
        za = infer_res["shared"][a][mask]
        zb = infer_res["shared"][b][mask]
        rows.append({
            "pair": f"{a}->{b}",
            "FOSCTTM": foscttm_from_embeddings(za, zb),
            "PED": paired_embedding_distance(za, zb),
            "n": int(mask.sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "T1_metrics.csv", index=False)
    return df


def evaluate_t2(meta, infer_res, prediction_pairs, output_dir: Path):
    rows = []
    for src, tgt in prediction_pairs:
        key = f"{src}__to__{tgt}"
        query_ids = resolve_query_ids_for_pair(meta, src, tgt)
        query_set = set(query_ids.tolist())
        val_ids = np.asarray(meta["val_cell_ids"]).astype(str)
        id_mask = np.asarray([cid in query_set for cid in val_ids], dtype=bool)
        mask = id_mask & (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)
        if mask.sum() < 2:
            mask = (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)
        if mask.sum() < 2:
            continue

        pred_std = infer_res["pred"][key][mask]
        true_std = meta["val_modalities"][tgt][mask]
        true_raw = meta["val_modalities_raw"][tgt][mask]
        pred_raw = inverse_standardize(pred_std, meta["stats"][tgt])

        rows.append({
            "pair": key,
            "eval_mode": "query_style",
            "pearson_cell_mean_std": pearson_cellwise(pred_std, true_std),
            "pearson_gene_mean_std": pearson_genewise(pred_std, true_std),
            "rmse_std": rmse(pred_std, true_std),
            "pearson_cell_mean_raw": pearson_cellwise(pred_raw, true_raw),
            "pearson_gene_mean_raw": pearson_genewise(pred_raw, true_raw),
            "rmse_raw": rmse(pred_raw, true_raw),
            "n": int(mask.sum()),
        })

        cell_ids = meta["val_cell_ids"][mask]
        pd.DataFrame(pred_std, index=cell_ids, columns=meta["common_features"]).to_csv(output_dir / f"pred_{key}_std.csv")
        pd.DataFrame(pred_raw, index=cell_ids, columns=meta["common_features"]).to_csv(output_dir / f"pred_{key}_raw.csv")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "T2_metrics.csv", index=False)
    return df


def build_eval_anndata(meta, infer_res, val_adatas, label_key: Optional[str], future_label_key: Optional[str]):
    obs_union = meta["val_obs_union"].copy()
    obs_union.index = obs_union.index.astype(str)
    obs = obs_union.loc[np.asarray(meta["val_cell_ids"]).astype(str)].copy()
    adata = ad.AnnData(X=infer_res["h"], obs=obs)
    adata.obs_names = np.asarray(meta["val_cell_ids"]).astype(str)
    adata.obsm["X_CORAL"] = infer_res["h"]
    adata.obs["CORAL_time"] = infer_res["t"]
    if infer_res.get("target_velocity") is not None and len(infer_res["target_velocity"]) > 0:
        mean_v = np.mean(np.stack(list(infer_res["target_velocity"].values()), axis=0), axis=0)
        adata.layers["CORAL_velocity"] = mean_v.astype(np.float32)
    if label_key is not None and label_key in obs.columns:
        adata.obs[label_key] = obs[label_key].astype(str).values
    if future_label_key is not None and future_label_key in obs.columns:
        adata.obs[future_label_key] = obs[future_label_key].astype(str).values
    return adata


def evaluate_t3_t4(meta, infer_res, val_adatas, label_key, future_label_key, reference_time_key, stage_order_map, output_dir: Path):
    adata_eval = build_eval_anndata(meta, infer_res, val_adatas, label_key, future_label_key)

    if infer_res.get("target_velocity") is not None and len(infer_res["target_velocity"]) > 0:
        mean_v = np.mean(np.stack(list(infer_res["target_velocity"].values()), axis=0), axis=0)
    else:
        mean_v = np.zeros_like(infer_res["h"])

    P = build_transition_matrix_from_velocity(
        adata_eval.obsm["X_CORAL"],
        mean_v,
        adata_eval.obs["CORAL_time"].to_numpy(dtype=float),
        n_neighbors=30,
    )
    adata_eval.obsp["CORAL_transition_matrix"] = sp.csr_matrix(P)

    try:
        t3 = compute_t3_metrics(
            adata_eval,
            time_key="CORAL_time",
            reference_time_key=reference_time_key,
            label_key=label_key,
            stage_order_map=stage_order_map,
        )
    except Exception as e:
        t3 = {"POA": np.nan, "POAUC": np.nan, "n_eval": 0, "status": f"failed:{type(e).__name__}:{e}"}

    try:
        if future_label_key is not None:
            t4 = compute_t4_metrics(adata_eval, P, lineage_key=future_label_key)
        else:
            t4 = {"F1": np.nan, "AUROC": np.nan, "n_eval": 0, "status": "skipped_no_future_label_key"}
    except Exception as e:
        t4 = {"F1": np.nan, "AUROC": np.nan, "n_eval": 0, "status": f"failed:{type(e).__name__}:{e}"}

    pd.DataFrame([t3]).to_csv(output_dir / "T3_metrics.csv", index=False)
    pd.DataFrame([t4]).to_csv(output_dir / "T4_metrics.csv", index=False)
    safe_write_h5ad(adata_eval, output_dir / "CORAL_eval_embeddings.h5ad")
    np.save(output_dir / "CORAL_transition_matrix.npy", P)
    return t3, t4


def save_eval_diagnostics(meta, infer_res, prediction_pairs, output_dir: Path):
    diag = {"n_val_cells": int(len(meta["val_cell_ids"])), "prediction_pairs": []}
    for src, tgt in prediction_pairs:
        key = f"{src}__to__{tgt}"
        query_ids = resolve_query_ids_for_pair(meta, src, tgt)
        query_set = set(query_ids.tolist())
        val_ids = np.asarray(meta["val_cell_ids"]).astype(str)
        id_mask = np.asarray([cid in query_set for cid in val_ids], dtype=bool)
        mask = id_mask & (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)
        pred_std = infer_res["pred"][key][mask] if mask.sum() > 0 else np.zeros((0, len(meta["common_features"])), dtype=np.float32)
        true_std = meta["val_modalities"][tgt][mask] if mask.sum() > 0 else np.zeros((0, len(meta["common_features"])), dtype=np.float32)
        diag["prediction_pairs"].append({
            "pair": key,
            "query_ids_requested": int(len(query_ids)),
            "cells_used": int(mask.sum()),
            "pred_std_mean": float(np.nanmean(pred_std)) if pred_std.size > 0 else None,
            "true_std_mean": float(np.nanmean(true_std)) if true_std.size > 0 else None,
            "pred_std_std": float(np.nanstd(pred_std)) if pred_std.size > 0 else None,
            "true_std_std": float(np.nanstd(true_std)) if true_std.size > 0 else None,
        })
    save_json(diag, output_dir / "eval_diagnostics.json")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CORAL-Optimized with batch fix")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--epochs-phase1", type=int, default=80)
    parser.add_argument("--epochs-phase2", type=int, default=60)
    parser.add_argument("--epochs-phase3", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--shared-dim", type=int, default=32)
    parser.add_argument("--private-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--mmd-weight", type=float, default=0.2)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--pred-weight", type=float, default=1.5)

    parser.add_argument("--topo-weight", type=float, default=0.2)
    parser.add_argument("--time-cons-weight", type=float, default=0.4)
    parser.add_argument("--forward-weight", type=float, default=0.5)
    parser.add_argument("--dyn-contrast-weight", type=float, default=1.0)
    parser.add_argument("--lag-weight", type=float, default=0.5)
    parser.add_argument("--lag-entropy-weight", type=float, default=0.05)
    parser.add_argument("--fusion-entropy-weight", type=float, default=0.02)
    parser.add_argument("--align-contrast-weight", type=float, default=0.5)

    parser.add_argument("--batch-adv-weight", type=float, default=1.0)
    parser.add_argument("--batch-shared-weight", type=float, default=0.7)
    parser.add_argument("--grl-lambda", type=float, default=1.0)

    parser.add_argument("--topo-k", type=int, default=20)
    parser.add_argument("--lag-bins", type=int, default=4)
    parser.add_argument("--lag-pred-weight", type=float, default=0.3)
    parser.add_argument("--time-spread-weight", type=float, default=0.05)
    parser.add_argument("--time-smooth-weight", type=float, default=0.05)
    parser.add_argument("--time-std-target", type=float, default=0.18)
    parser.add_argument("--time-head-blend", type=float, default=0.35)
    parser.add_argument("--transition-weight", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    cfg = TrainConfig(
        output_dir=args.output_dir,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        epochs_phase3=args.epochs_phase3,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        dropout=args.dropout,
        mmd_weight=args.mmd_weight,
        align_weight=args.align_weight,
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        topo_weight=args.topo_weight,
        time_cons_weight=args.time_cons_weight,
        forward_weight=args.forward_weight,
        dyn_contrast_weight=args.dyn_contrast_weight,
        lag_weight=args.lag_weight,
        lag_entropy_weight=args.lag_entropy_weight,
        fusion_entropy_weight=args.fusion_entropy_weight,
        align_contrast_weight=args.align_contrast_weight,
        batch_adv_weight=args.batch_adv_weight,
        batch_shared_weight=args.batch_shared_weight,
        grl_lambda=args.grl_lambda,
        device=args.device,
        topo_k=args.topo_k,
        lag_bins=args.lag_bins,
        lag_pred_weight=args.lag_pred_weight,
        time_spread_weight=args.time_spread_weight,
        time_smooth_weight=args.time_smooth_weight,
        time_std_target=args.time_std_target,
        time_head_blend=args.time_head_blend,
        transition_weight=args.transition_weight,
    )
    save_json(dataclasses.asdict(cfg), Path(cfg.output_dir) / "train_config.json")

    modalities, alignment_pairs, prediction_pairs, lag_edges, label_key, future_label_key, reference_time_key, stage_order_map = parse_manifest(args.manifest)
    ds_train, meta, val_adatas = build_multimodal_ratio_dataset(args.split_dir, modalities, label_key=label_key, standardize=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(Path(cfg.output_dir) / "tensorboard"))

    input_dims = {m: ds_train.modalities[m].shape[1] for m in ds_train.modality_names}
    n_batches = len(meta["batch_categories"])

    model = CORALOptimizedBatchFix(
        input_dims=input_dims,
        prediction_pairs=prediction_pairs,
        lag_edges=lag_edges,
        n_batches=n_batches,
        shared_dim=cfg.shared_dim,
        private_dim=cfg.private_dim,
        dropout=cfg.dropout,
    ).to(device)

    epoch_ptr = 1
    model, hist1, epoch_ptr = run_training_phase(
        model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
        phase_name="phase1", epochs=cfg.epochs_phase1, start_epoch=epoch_ptr,
        freeze_plan=FreezePlan(freeze_encoders=False, freeze_fusion=False, freeze_prediction=False, freeze_dynamics=True),
    )
    model, hist2, epoch_ptr = run_training_phase(
        model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
        phase_name="phase2", epochs=cfg.epochs_phase2, start_epoch=epoch_ptr,
        freeze_plan=FreezePlan(freeze_encoders=False, freeze_fusion=False, freeze_prediction=False, freeze_dynamics=True),
    )
    model, hist3, epoch_ptr = run_training_phase(
        model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
        phase_name="phase3", epochs=cfg.epochs_phase3, start_epoch=epoch_ptr,
        freeze_plan=FreezePlan(freeze_encoders=False, freeze_fusion=False, freeze_prediction=False, freeze_dynamics=False),
    )

    hist = pd.concat([pd.DataFrame(hist1), pd.DataFrame(hist2), pd.DataFrame(hist3)], axis=0, ignore_index=True)
    hist.to_csv(Path(cfg.output_dir) / "training_history.csv", index=False)
    torch.save({"model": model.state_dict()}, Path(cfg.output_dir) / "CORAL_optimized_batchfix_final.pt")

    infer_res = infer_on_arrays(model, meta["val_modalities"], meta["val_masks"], device, lag_banks_np=None)
    np.save(Path(cfg.output_dir) / "val_h.npy", infer_res["h"])
    np.save(Path(cfg.output_dir) / "val_time.npy", infer_res["t"])
    if "t_head" in infer_res:
        np.save(Path(cfg.output_dir) / "val_time_head.npy", infer_res["t_head"])
    if "t_dpt" in infer_res:
        np.save(Path(cfg.output_dir) / "val_time_dpt.npy", infer_res["t_dpt"])
    save_npz_dict(infer_res["shared"], Path(cfg.output_dir) / "val_shared_latents.npz")
    save_npz_dict(infer_res["private"], Path(cfg.output_dir) / "val_private_latents.npz")
    if infer_res.get("target_velocity") is not None:
        save_npz_dict(infer_res["target_velocity"], Path(cfg.output_dir) / "val_target_velocity.npz")
    if infer_res.get("lag_pred_rec") is not None:
        save_npz_dict(infer_res["lag_pred_rec"], Path(cfg.output_dir) / "val_lag_predictions.npz")

    t1_df = evaluate_t1(meta, infer_res, alignment_pairs, Path(cfg.output_dir))
    t2_df = evaluate_t2(meta, infer_res, prediction_pairs, Path(cfg.output_dir))
    t3, t4 = evaluate_t3_t4(meta, infer_res, val_adatas, label_key, future_label_key, reference_time_key, stage_order_map, Path(cfg.output_dir))
    save_eval_diagnostics(meta, infer_res, prediction_pairs, Path(cfg.output_dir))

    summary = {
        "T1": t1_df.to_dict(orient="records"),
        "T2": t2_df.to_dict(orient="records"),
        "T3": t3,
        "T4": t4,
        "batch_categories": meta["batch_categories"],
    }
    save_json(summary, Path(cfg.output_dir) / "summary_metrics.json")

    print("\n===== CORAL-Optimized-BatchFix training finished =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()