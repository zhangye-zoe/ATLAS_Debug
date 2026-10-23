#!/usr/bin/env python3
"""
Train CORAL across ratio splits and evaluate T1/T2/T3 directly.

Compared with the previous T1/T2 ratio-loop script, this version adds direct T3
trajectory evaluation without running Monocle3. The workflow is:

1. Train CORAL on each ratio split.
2. Infer fused latent embedding h and pseudotime t on validation cells.
3. Evaluate:
   - T1: true RNA vs predicted RNA embedding matching, scMRDR-style.
   - T2: true RNA vs predicted RNA expression recovery.
   - T3: direct trajectory ordering from CORAL_time against stage_order labels.

T3 metrics:
- PairAcc / POA: weighted pairwise ordering accuracy across different stages.
- PairAUC / POAUC: weighted pairwise AUC across different stage pairs.

Typical usage for PBMC:
python coral_ratio_loop_T1T2T3_direct.py \
  --coral-module /data5/zhangye/CORAL/coral_optimized_batchfix.py \
  --manifest /data5/zhangye/CORAL/pbmc_rna_atac_manifest.json \
  --split-root /data5/zhangye/scMRDR/input/PBMC/preprocessed_input/results_ratio_loop \
  --out-root /data5/zhangye/CORAL_T1T2T3/PBMC \
  --label-key cell_type \
  --stage-order-json '{"HSPC":0,"B naive":1,"CD4 Naive":1,"CD8 Naive":1,"Eryth":1,"B intermediate":2,"CD4 TCM":2,"CD8 TCM":2,"CD14 Mono":2,"CD16 Mono":2,"cDC1":2,"cDC2":2,"pDC":2,"CD4 Proliferating":2,"NK Proliferating":2,"B memory":3,"Plasmablast":3,"CD4 TEM":3,"CD8 TEM":3,"Treg":3,"MAIT":3,"NK":3,"dnT":3,"gdT":3}' \
  --run-phase3 --epochs-phase3 80

If labels are not already in the split h5ad obs, add e.g.:
  --label-csv /data5/zhangye/scMRDR/input/PBMC/raw_input/pbmc_azimuth_annotations.csv \
  --label-csv-cell-col cell_id \
  --label-csv-label-col predicted.celltype.l2 \
  --label-key cell_type
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, roc_auc_score
import torch


# =========================================================
# 0. utilities
# =========================================================

def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_cor(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size <= 1 or y.size <= 1:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def hit_at_k(cross_dist: np.ndarray, k: int = 5) -> float:
    nq = cross_dist.shape[0]
    hits = []
    for i in range(nq):
        ord_idx = np.argsort(cross_dist[i, :])[: min(k, cross_dist.shape[1])]
        hits.append(i in ord_idx)
    return float(np.mean(hits)) if len(hits) > 0 else np.nan


def save_h5ad_safe(adata: ad.AnnData, path: os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(path))


def load_json_maybe(x: str | None):
    """
    Load JSON from either:
    1) a JSON string, e.g. '{"HSPC":0,"B naive":1}'
    2) a JSON file path, e.g. './stage_order_pbmc.json'

    Important:
    Do not call Path(x).exists() before checking whether x is a JSON string,
    otherwise a long JSON string may be interpreted as a file path and trigger
    OSError: [Errno 36] File name too long.
    """
    if x is None:
        return None

    x = str(x).strip()
    if x == "":
        return None

    # Case 1: direct JSON string
    if x.startswith("{") or x.startswith("["):
        return json.loads(x)

    # Case 2: JSON file path
    p = Path(x)
    if p.exists():
        return json.loads(p.read_text())

    # Fallback: try parsing as JSON
    return json.loads(x)


def read_id_file(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(str(s))
    return ids


# =========================================================
# 1. import CORAL module dynamically
# =========================================================

def load_coral_module(module_path: str):
    module_path = str(module_path)
    spec = importlib.util.spec_from_file_location("coral_user_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =========================================================
# 2. T1/T2 evaluation, scMRDR-style
# =========================================================

def evaluate_t2_from_true_pred(pred_mat: np.ndarray, true_mat: np.ndarray) -> pd.DataFrame:
    cell_corr = [safe_cor(pred_mat[i], true_mat[i]) for i in range(pred_mat.shape[0])]
    gene_corr = [safe_cor(pred_mat[:, j], true_mat[:, j]) for j in range(pred_mat.shape[1])]
    rmse = float(np.sqrt(np.mean((pred_mat - true_mat) ** 2)))
    return pd.DataFrame(
        {
            "metric": ["pearson_cell_mean", "pearson_gene_mean", "rmse"],
            "value": [float(np.nanmean(cell_corr)), float(np.nanmean(gene_corr)), rmse],
        }
    )


def evaluate_t1_from_true_pred(
    pred_mat: np.ndarray,
    true_mat: np.ndarray,
    common_cells: Sequence[str],
    common_features: Sequence[str],
    outdir: Path,
    seed: int = 1234,
) -> pd.DataFrame:
    true_cells_by_features = np.asarray(true_mat, dtype=float)
    pred_cells_by_features = np.asarray(pred_mat, dtype=float)

    mix = np.vstack([true_cells_by_features, pred_cells_by_features])
    n_components = min(30, mix.shape[0] - 1, mix.shape[1])
    if n_components < 2:
        raise ValueError("Too few dimensions to run PCA for T1 evaluation.")

    pca = PCA(n_components=n_components, random_state=seed)
    emb = pca.fit_transform(mix)
    n_q = true_cells_by_features.shape[0]
    true_emb = emb[:n_q]
    pred_emb = emb[n_q:]

    cross_dist = pairwise_distances(true_emb, pred_emb)

    fos, ped = [], []
    for i in range(cross_dist.shape[0]):
        true_d = cross_dist[i, i]
        fos.append(np.mean(cross_dist[i] < true_d))
        ped.append(true_d)

    metrics = pd.DataFrame(
        {
            "metric": ["FOSCTTM", "PED", "Top1_ACC", "Top5_ACC", "Top10_ACC"],
            "value": [
                float(np.mean(fos)),
                float(np.mean(ped)),
                hit_at_k(cross_dist, 1),
                hit_at_k(cross_dist, 5),
                hit_at_k(cross_dist, 10),
            ],
        }
    )
    metrics.to_csv(outdir / "T1_metrics_scMRDR_style.csv", index=False)

    pca_df = pd.DataFrame(
        emb[:, : min(5, emb.shape[1])],
        columns=[f"PC{i+1}" for i in range(min(5, emb.shape[1]))],
    )
    pca_df["group"] = ["True_RNA"] * n_q + ["Pred_from_ATAC"] * n_q
    pca_df["cell"] = [f"true_{c}" for c in common_cells] + [f"pred_{c}" for c in common_cells]
    pca_df.to_csv(outdir / "pca_true_pred_coords.csv", index=False)
    return metrics


# =========================================================
# 3. T3 direct metrics
# =========================================================

def _find_first_existing_key(container, candidates, kind="key", raise_error: bool = False):
    for k in candidates:
        if k in container:
            return k
    if raise_error:
        raise KeyError(f"Could not find any valid {kind}. Tried: {candidates}")
    return None


def resolve_label_key(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["celltype", "cell_type", "annotation", "annot", "labels", "leiden"])
    return _find_first_existing_key(df.columns, candidates, kind="label key", raise_error=False)


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


def orient_time_by_stage(pred: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Orient time so earliest stages have lower time than latest stages."""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() < 3:
        return pred, False
    classes = np.sort(np.unique(ref[valid]))
    if len(classes) < 2:
        return pred, False
    early = classes[0]
    late = classes[-1]
    early_mean = np.nanmean(pred[valid & (ref == early)])
    late_mean = np.nanmean(pred[valid & (ref == late)])
    if np.isfinite(early_mean) and np.isfinite(late_mean) and early_mean > late_mean:
        return 1.0 - pred, True
    return pred, False


def compute_t3_direct_metrics(
    obs_df: pd.DataFrame,
    pred_time: np.ndarray,
    label_key: Optional[str],
    stage_order_map: Optional[dict],
    reference_time_key: Optional[str] = None,
    eval_mask: Optional[np.ndarray] = None,
    orient_by_stage: bool = True,
) -> dict:
    if eval_mask is None:
        eval_mask = np.ones(len(obs_df), dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask).astype(bool)

    pred = np.asarray(pred_time, dtype=float).copy()

    if reference_time_key is not None and reference_time_key in obs_df.columns:
        ref = pd.to_numeric(obs_df[reference_time_key], errors="coerce").to_numpy(dtype=float)
        reference_name = reference_time_key
        final_label_key = None
        missing_stage_labels = []
    else:
        final_label_key = resolve_label_key(obs_df, label_key)
        if final_label_key is None:
            return {
                "PairAcc": np.nan,
                "PairAUC": np.nan,
                "POA": np.nan,
                "POAUC": np.nan,
                "n_eval": 0,
                "status": "failed_no_label_key",
            }
        if stage_order_map is None:
            return {
                "PairAcc": np.nan,
                "PairAUC": np.nan,
                "POA": np.nan,
                "POAUC": np.nan,
                "n_eval": 0,
                "label_key": final_label_key,
                "status": "failed_no_stage_order_map",
            }
        labels = obs_df[final_label_key].astype(str)
        ref = labels.map(stage_order_map).to_numpy(dtype=float)
        reference_name = f"ordinal({final_label_key})"
        observed = set(labels.dropna().astype(str).unique().tolist())
        missing_stage_labels = sorted([x for x in observed if x not in set(stage_order_map.keys())])

    flipped = False
    if orient_by_stage and reference_time_key is None:
        pred, flipped = orient_time_by_stage(pred, ref)

    valid = eval_mask & np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() < 3:
        out = {
            "PairAcc": np.nan,
            "PairAUC": np.nan,
            "POA": np.nan,
            "POAUC": np.nan,
            "n_eval": int(valid.sum()),
            "pred_time_key": "CORAL_time",
            "reference": reference_name,
            "time_flipped": bool(flipped),
            "status": "skipped_too_few_valid_cells",
        }
    else:
        pred_v, ref_v = pred[valid], ref[valid]
        pair_acc = float(interclass_pairwise_accuracy(pred_v, ref_v, weighted=True))
        pair_auc = float(interclass_pairwise_auc(pred_v, ref_v, weighted=True))
        out = {
            "PairAcc": pair_acc,
            "PairAUC": pair_auc,
            "POA": pair_acc,
            "POAUC": pair_auc,
            "n_eval": int(valid.sum()),
            "pred_time_key": "CORAL_time",
            "reference": reference_name,
            "time_flipped": bool(flipped),
            "status": "ok",
        }

    if final_label_key is not None:
        out["label_key"] = final_label_key
    if missing_stage_labels:
        out["n_unmapped_labels"] = int(len(missing_stage_labels))
        out["unmapped_labels_preview"] = ";".join(missing_stage_labels[:20])
    return out


def merge_label_csv_into_meta(
    meta: dict,
    label_csv: Optional[str],
    cell_col: str = "cell_id",
    label_col: str = "cell_type",
    output_label_key: str = "cell_type",
) -> None:
    if label_csv is None:
        return
    path = Path(label_csv)
    if not path.exists():
        raise FileNotFoundError(f"label_csv not found: {path}")
    df = pd.read_csv(path)
    if cell_col not in df.columns:
        raise KeyError(f"{cell_col!r} not found in label_csv columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(f"{label_col!r} not found in label_csv columns: {list(df.columns)}")

    label_df = df[[cell_col, label_col]].copy()
    label_df[cell_col] = label_df[cell_col].astype(str)
    label_df = label_df.drop_duplicates(subset=cell_col).set_index(cell_col)
    label_df = label_df.rename(columns={label_col: output_label_key})

    obs = meta["val_obs_union"].copy()
    obs.index = obs.index.astype(str)
    obs = obs.join(label_df, how="left", rsuffix="_from_csv")

    # If the target column already existed, fill its missing values from CSV.
    csv_col = f"{output_label_key}_from_csv"
    if csv_col in obs.columns:
        obs[output_label_key] = obs[output_label_key].where(obs[output_label_key].notna(), obs[csv_col])
        obs = obs.drop(columns=[csv_col])

    meta["val_obs_union"] = obs
    meta["labels"] = obs.loc[np.asarray(meta["val_cell_ids"]).astype(str), output_label_key].astype(str).to_numpy() \
        if output_label_key in obs.columns else None
    meta["label_key"] = output_label_key


def save_coral_eval_h5ad(meta: dict, infer_res: dict, obs_df: pd.DataFrame, outdir: Path) -> None:
    adata_eval = ad.AnnData(X=np.asarray(infer_res["h"], dtype=np.float32), obs=obs_df.copy())
    adata_eval.obs_names = np.asarray(meta["val_cell_ids"]).astype(str)
    adata_eval.obsm["X_CORAL"] = np.asarray(infer_res["h"], dtype=np.float32)
    adata_eval.obs["CORAL_time"] = np.asarray(infer_res["t"], dtype=float)
    save_h5ad_safe(adata_eval, outdir / "CORAL_eval_embeddings_T3.h5ad")


# =========================================================
# 4. helpers for ratio split logic
# =========================================================

def resolve_ratio_labels(split_root: Path, requested: Sequence[str] | None) -> List[str]:
    if requested and len(requested) > 0:
        return list(requested)
    labels = sorted([p.name for p in split_root.iterdir() if p.is_dir() and p.name.startswith("single_")])
    return labels


def pick_primary_prediction_pair(prediction_pairs: Sequence[Tuple[str, str]]) -> Tuple[str, str]:
    if len(prediction_pairs) == 0:
        raise ValueError("manifest prediction_pairs is empty. Static T2 requires at least one src->tgt pair.")
    return tuple(prediction_pairs[0])


def get_query_mask(meta: dict, src: str, tgt: str) -> np.ndarray:
    split_info = meta["split_info"]
    val_ids = np.asarray(meta["val_cell_ids"]).astype(str)

    candidates = [f"val_query_{src}_cells", f"val_query_{src}_cell_ids"]
    if src == "atac":
        candidates.append("val_query_atac_cells")
    if src in ["rna", "s"]:
        candidates.append("val_query_rna_cells")
    candidates.extend(["val_query_cells", "val_cells"])

    for k in candidates:
        if k in split_info and len(split_info[k]) > 0:
            qset = set(str(x) for x in split_info[k])
            return np.asarray([cid in qset for cid in val_ids], dtype=bool)
    return np.ones(len(val_ids), dtype=bool)


def get_t3_eval_mask(meta: dict, t3_cell_set: str, fixed_cell_ids: Optional[List[str]]) -> np.ndarray:
    val_ids = np.asarray(meta["val_cell_ids"]).astype(str)
    split_info = meta["split_info"]

    if fixed_cell_ids is not None:
        keep = set(str(x) for x in fixed_cell_ids)
        return np.asarray([x in keep for x in val_ids], dtype=bool)

    if t3_cell_set == "val_all":
        return np.ones(len(val_ids), dtype=bool)
    if t3_cell_set == "val_atac":
        ids = split_info.get("val_query_atac_cells", split_info.get("val_atac_cells", []))
        keep = set(str(x) for x in ids)
        return np.asarray([x in keep for x in val_ids], dtype=bool)
    if t3_cell_set == "val_paired":
        ids = split_info.get("val_paired_cells", [])
        keep = set(str(x) for x in ids)
        if len(keep) == 0:
            return np.ones(len(val_ids), dtype=bool)
        return np.asarray([x in keep for x in val_ids], dtype=bool)
    raise ValueError(f"Unknown --t3-cell-set: {t3_cell_set}")


def save_prediction_h5ad(pred_raw: np.ndarray, cell_ids: Sequence[str], feature_names: Sequence[str], out_path: Path):
    adata = ad.AnnData(X=np.asarray(pred_raw, dtype=np.float32))
    adata.obs_names = np.asarray(cell_ids).astype(str)
    adata.var_names = np.asarray(feature_names).astype(str)
    save_h5ad_safe(adata, out_path)


def build_val_lag_banks_if_possible(coral, infer_res: dict, meta: dict, cfg, model) -> Optional[Dict[str, np.ndarray]]:
    """Build validation lag banks for dynamic inference after phase3, if supported."""
    if not hasattr(model, "lag_modules") or len(model.lag_modules) == 0:
        return None
    if not hasattr(coral, "build_lag_bank"):
        return None
    h = np.asarray(infer_res["h"], dtype=np.float32)
    t = np.asarray(infer_res["t"], dtype=np.float32)
    lag_bins = int(getattr(cfg, "lag_bins", 4))
    out = {}
    for m in sorted(meta["val_modalities"].keys()):
        out[m] = coral.build_lag_bank(
            h,
            t,
            source_mask=np.asarray(meta["val_masks"][m]) > 0,
            max_lag_bins=lag_bins,
            k_per_bin=6,
        )
    return out


# =========================================================
# 5. one-ratio training + T1/T2/T3 evaluation
# =========================================================

def train_one_ratio(
    coral,
    split_dir: Path,
    outdir: Path,
    manifest_path: Path,
    cfg,
    static_only: bool,
    cli_label_key: Optional[str],
    cli_reference_time_key: Optional[str],
    cli_stage_order_map: Optional[dict],
    label_csv: Optional[str],
    label_csv_cell_col: str,
    label_csv_label_col: str,
    t3_cell_set: str,
    fixed_cell_ids: Optional[List[str]],
    orient_t3_by_stage: bool,
):
    outdir.mkdir(parents=True, exist_ok=True)

    modalities, alignment_pairs, prediction_pairs, lag_edges, label_key, future_label_key, reference_time_key, stage_order_map = coral.parse_manifest(manifest_path)
    if cli_label_key is not None:
        label_key = cli_label_key
    if cli_reference_time_key is not None:
        reference_time_key = cli_reference_time_key
    if cli_stage_order_map is not None:
        stage_order_map = cli_stage_order_map

    ds_train, meta, val_adatas = coral.build_multimodal_ratio_dataset(
        split_dir,
        modalities,
        label_key=label_key,
        standardize=True,
    )

    if label_csv is not None:
        merge_label_csv_into_meta(
            meta,
            label_csv=label_csv,
            cell_col=label_csv_cell_col,
            label_col=label_csv_label_col,
            output_label_key=label_key or "cell_type",
        )
        label_key = label_key or "cell_type"

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    writer = coral.SummaryWriter(log_dir=str(outdir / "tensorboard"))

    input_dims = {m: ds_train.modalities[m].shape[1] for m in ds_train.modality_names}

    if hasattr(coral, "CORALOptimizedBatchFix"):
        n_batches = len(meta.get("batch_categories", []))
        if n_batches <= 0:
            n_batches = 1
        model = coral.CORALOptimizedBatchFix(
            input_dims=input_dims,
            prediction_pairs=prediction_pairs,
            lag_edges=lag_edges,
            n_batches=n_batches,
            shared_dim=cfg.shared_dim,
            private_dim=cfg.private_dim,
            dropout=cfg.dropout,
        ).to(device)
    elif hasattr(coral, "CORALOptimized"):
        model = coral.CORALOptimized(
            input_dims=input_dims,
            prediction_pairs=prediction_pairs,
            lag_edges=lag_edges,
            shared_dim=cfg.shared_dim,
            private_dim=cfg.private_dim,
            dropout=cfg.dropout,
        ).to(device)
    elif hasattr(coral, "CORAL"):
        model = coral.CORAL(
            input_dims=input_dims,
            prediction_pairs=prediction_pairs,
            lag_edges=lag_edges,
            shared_dim=cfg.shared_dim,
            private_dim=cfg.private_dim,
            dropout=cfg.dropout,
        ).to(device)
    else:
        raise AttributeError("Cannot find a supported CORAL model class in the imported module.")

    epoch_ptr = 1
    model, hist1, epoch_ptr = coral.run_training_phase(
        model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
        phase_name="phase1", epochs=cfg.epochs_phase1, start_epoch=epoch_ptr,
        freeze_plan=coral.FreezePlan(
            freeze_encoders=False,
            freeze_fusion=False,
            freeze_prediction=False,
            freeze_dynamics=True,
        ),
    )
    model, hist2, epoch_ptr = coral.run_training_phase(
        model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
        phase_name="phase2", epochs=cfg.epochs_phase2, start_epoch=epoch_ptr,
        freeze_plan=coral.FreezePlan(
            freeze_encoders=False,
            freeze_fusion=False,
            freeze_prediction=False,
            freeze_dynamics=True,
        ),
    )

    hist_parts = [pd.DataFrame(hist1), pd.DataFrame(hist2)]

    if not static_only and cfg.epochs_phase3 > 0:
        model, hist3, epoch_ptr = coral.run_training_phase(
            model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
            phase_name="phase3", epochs=cfg.epochs_phase3, start_epoch=epoch_ptr,
            freeze_plan=coral.FreezePlan(
                freeze_encoders=False,
                freeze_fusion=False,
                freeze_prediction=False,
                freeze_dynamics=False,
            ),
        )
        hist_parts.append(pd.DataFrame(hist3))

    hist = pd.concat(hist_parts, axis=0, ignore_index=True)
    hist.to_csv(outdir / "training_history.csv", index=False)
    torch.save({"model": model.state_dict()}, outdir / "CORAL_T1T2T3_final.pt")

    # First inference. This always gives h and CORAL_time.
    infer_res = coral.infer_on_arrays(model, meta["val_modalities"], meta["val_masks"], device)

    # If phase3/dynamic modules exist, run a second validation inference with validation lag banks
    # so target_velocity is available for diagnostics; T3 still uses CORAL_time.
    val_lag_banks = build_val_lag_banks_if_possible(coral, infer_res, meta, cfg, model)
    if val_lag_banks is not None:
        infer_res = coral.infer_on_arrays(model, meta["val_modalities"], meta["val_masks"], device, lag_banks_np=val_lag_banks)

    np.save(outdir / "val_h.npy", infer_res["h"])
    np.save(outdir / "val_time.npy", infer_res["t"])
    coral.save_npz_dict(infer_res["shared"], outdir / "val_shared_latents.npz")
    coral.save_npz_dict(infer_res["private"], outdir / "val_private_latents.npz")
    if infer_res.get("target_velocity") is not None:
        coral.save_npz_dict(infer_res["target_velocity"], outdir / "val_target_velocity.npz")

    # Keep the model module's native T1/T2 exports too.
    if hasattr(coral, "evaluate_t1"):
        try:
            coral.evaluate_t1(meta, infer_res, alignment_pairs, outdir)
        except Exception as e:
            warnings.warn(f"coral.evaluate_t1 failed: {repr(e)}")
    if hasattr(coral, "evaluate_t2"):
        try:
            coral.evaluate_t2(meta, infer_res, prediction_pairs, outdir)
        except Exception as e:
            warnings.warn(f"coral.evaluate_t2 failed: {repr(e)}")

    # Direct T3 evaluation from CORAL_time.
    val_ids = np.asarray(meta["val_cell_ids"]).astype(str)
    obs_union = meta["val_obs_union"].copy()
    obs_union.index = obs_union.index.astype(str)
    obs_df = obs_union.loc[val_ids].copy()
    obs_df["CORAL_time"] = np.asarray(infer_res["t"], dtype=float)
    obs_df["cell_id"] = val_ids

    t3_eval_mask = get_t3_eval_mask(meta, t3_cell_set=t3_cell_set, fixed_cell_ids=fixed_cell_ids)
    t3 = compute_t3_direct_metrics(
        obs_df=obs_df,
        pred_time=infer_res["t"],
        label_key=label_key,
        stage_order_map=stage_order_map,
        reference_time_key=reference_time_key,
        eval_mask=t3_eval_mask,
        orient_by_stage=orient_t3_by_stage,
    )
    pd.DataFrame([t3]).to_csv(outdir / "T3_metrics_direct.csv", index=False)
    obs_df.to_csv(outdir / "T3_eval_cells.csv", index=False)
    save_coral_eval_h5ad(meta, infer_res, obs_df, outdir)

    # Static T1/T2 query evaluation.
    src, tgt = pick_primary_prediction_pair(prediction_pairs)
    pair_key = f"{src}__to__{tgt}"

    query_mask = get_query_mask(meta, src, tgt)
    paired_mask = query_mask & (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)
    if paired_mask.sum() < 2:
        paired_mask = (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)
    if paired_mask.sum() < 2:
        raise ValueError(f"Not enough paired/query cells for static evaluation on {src}->{tgt}")

    pred_std = infer_res["pred"][pair_key][paired_mask]
    true_raw = meta["val_modalities_raw"][tgt][paired_mask]
    pred_raw = coral.inverse_standardize(pred_std, meta["stats"][tgt]) if hasattr(coral, "inverse_standardize") else pred_std

    cell_ids = np.asarray(meta["val_cell_ids"])[paired_mask].astype(str)
    feature_names = np.asarray(meta["common_features"]).astype(str)

    pred_df = pd.DataFrame(pred_raw, index=cell_ids, columns=feature_names)
    true_df = pd.DataFrame(true_raw, index=cell_ids, columns=feature_names)
    pred_df.to_csv(outdir / "pred_rna_val.csv")
    true_df.to_csv(outdir / "true_rna_val.csv")
    save_prediction_h5ad(pred_raw, cell_ids, feature_names, outdir / "pred_rna_val.h5ad")
    save_prediction_h5ad(true_raw, cell_ids, feature_names, outdir / "true_rna_val.h5ad")

    t2_metrics = evaluate_t2_from_true_pred(pred_raw, true_raw)
    t2_metrics.to_csv(outdir / "T2_metrics_scMRDR_style.csv", index=False)

    t1_metrics = evaluate_t1_from_true_pred(
        pred_mat=pred_raw,
        true_mat=true_raw,
        common_cells=cell_ids,
        common_features=feature_names,
        outdir=outdir,
        seed=1234,
    )

    t1_map = dict(zip(t1_metrics["metric"], t1_metrics["value"]))
    t2_map = dict(zip(t2_metrics["metric"], t2_metrics["value"]))
    split_info = meta["split_info"]

    row = {
        "ratio_label": split_dir.name,
        "single_frac": split_info.get("single_frac", np.nan),
        "train_paired": len(split_info.get("train_paired_cells", [])),
        "train_rna_only": len(split_info.get("train_rna_only_cells", [])),
        "train_atac_only": len(split_info.get("train_atac_only_cells", [])),
        "val_paired": len(split_info.get("val_paired_cells", [])),
        "val_rna_only": len(split_info.get("val_rna_only_cells", [])),
        "val_atac_only": len(split_info.get("val_atac_only_cells", [])),
        "query_cells": int(len(cell_ids)),
        "common_features": int(len(feature_names)),
        "prediction_pair": pair_key,
        "cellwise_pearson_mean": t2_map["pearson_cell_mean"],
        "genewise_pearson_mean": t2_map["pearson_gene_mean"],
        "rmse": t2_map["rmse"],
        "foscttm": t1_map["FOSCTTM"],
        "ped": t1_map["PED"],
        "top1_acc": t1_map["Top1_ACC"],
        "top5_acc": t1_map["Top5_ACC"],
        "top10_acc": t1_map["Top10_ACC"],
        "t3_pairacc": t3.get("PairAcc", np.nan),
        "t3_pairauc": t3.get("PairAUC", np.nan),
        "t3_poa": t3.get("POA", np.nan),
        "t3_poauc": t3.get("POAUC", np.nan),
        "t3_n_eval": t3.get("n_eval", 0),
        "t3_status": t3.get("status", "unknown"),
        "t3_label_key": t3.get("label_key", label_key),
        "t3_reference": t3.get("reference", None),
        "t3_time_flipped": t3.get("time_flipped", False),
    }

    with open(outdir / "summary_one_ratio.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    with open(outdir / "summary_t3.json", "w", encoding="utf-8") as f:
        json.dump(t3, f, indent=2, ensure_ascii=False)

    writer.close()
    return row


# =========================================================
# 6. main ratio-loop entry
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Train CORAL across ratio splits and evaluate T1/T2/T3 directly")
    parser.add_argument("--coral-module", type=str, required=True, help="Path to the dynamic CORAL python file")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--ratio-labels", nargs="*", default=None, help="Optional specific ratio labels, e.g. single_000 single_020")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--epochs-phase1", type=int, default=80)
    parser.add_argument("--epochs-phase2", type=int, default=60)
    parser.add_argument("--epochs-phase3", type=int, default=0)
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
    parser.add_argument("--run-phase3", action="store_true", help="Enable dynamic phase3. Recommended for T3 direct evaluation.")

    # T3 arguments
    parser.add_argument("--label-key", type=str, default=None, help="Cell type / stage label key. Overrides manifest label_key.")
    parser.add_argument("--reference-time-key", type=str, default=None, help="Optional numeric reference time key. If set, stage_order_json is not required.")
    parser.add_argument("--stage-order-json", type=str, default=None, help="JSON string or JSON file path mapping label -> ordinal stage.")
    parser.add_argument("--t3-cell-set", type=str, default="val_all", choices=["val_all", "val_atac", "val_paired"])
    parser.add_argument("--fixed-cell-ids", type=str, default=None, help="Optional text file, one cell_id per line, for fixed T3 eval set across ratios.")
    parser.add_argument("--no-orient-t3-by-stage", action="store_true", help="Do not flip CORAL_time using earliest/latest stage means.")

    # Optional external labels, useful for PBMC Azimuth annotations.
    parser.add_argument("--label-csv", type=str, default=None, help="Optional CSV with cell labels to merge into val obs.")
    parser.add_argument("--label-csv-cell-col", type=str, default="cell_id")
    parser.add_argument("--label-csv-label-col", type=str, default="cell_type")

    args = parser.parse_args()

    coral = load_coral_module(args.coral_module)
    coral.set_seed(args.seed)

    split_root = Path(args.split_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)

    ratio_labels = resolve_ratio_labels(split_root, args.ratio_labels)
    if len(ratio_labels) == 0:
        raise ValueError(f"No ratio folders found under: {split_root}")

    stage_order_map = load_json_maybe(args.stage_order_json)
    fixed_cell_ids = read_id_file(args.fixed_cell_ids)

    cfg_kwargs = dict(
        output_dir=str(out_root),
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
        lag_weight=args.lag_weight,
        lag_entropy_weight=args.lag_entropy_weight,
        device=args.device,
        topo_k=args.topo_k,
        lag_bins=args.lag_bins,
    )

    optional_fields = {
        "dyn_contrast_weight": args.dyn_contrast_weight,
        "fusion_entropy_weight": args.fusion_entropy_weight,
        "align_contrast_weight": args.align_contrast_weight,
        "batch_adv_weight": args.batch_adv_weight,
        "batch_shared_weight": args.batch_shared_weight,
        "grl_lambda": args.grl_lambda,
    }

    train_config_fields = set(getattr(coral.TrainConfig, "__dataclass_fields__", {}).keys())
    for k, v in optional_fields.items():
        if k in train_config_fields:
            cfg_kwargs[k] = v

    all_summary: List[dict] = []

    for ratio_label in ratio_labels:
        split_dir = split_root / ratio_label
        outdir = out_root / ratio_label
        outdir.mkdir(parents=True, exist_ok=True)

        if not split_dir.exists():
            warnings.warn(f"Missing split directory: {split_dir}. Skipping.")
            continue

        print("\n============================")
        print("Running ratio:", ratio_label)
        print("Split dir:", split_dir)
        print("Out dir:", outdir)
        print("============================")

        try:
            cfg_kwargs["output_dir"] = str(outdir)
            cfg = coral.TrainConfig(**cfg_kwargs)
            coral.save_json(dataclasses.asdict(cfg), outdir / "train_config.json")
            if stage_order_map is not None:
                coral.save_json(stage_order_map, outdir / "stage_order_map_used.json")

            row = train_one_ratio(
                coral=coral,
                split_dir=split_dir,
                outdir=outdir,
                manifest_path=manifest_path,
                cfg=cfg,
                static_only=(not args.run_phase3),
                cli_label_key=args.label_key,
                cli_reference_time_key=args.reference_time_key,
                cli_stage_order_map=stage_order_map,
                label_csv=args.label_csv,
                label_csv_cell_col=args.label_csv_cell_col,
                label_csv_label_col=args.label_csv_label_col,
                t3_cell_set=args.t3_cell_set,
                fixed_cell_ids=fixed_cell_ids,
                orient_t3_by_stage=(not args.no_orient_t3_by_stage),
            )
            all_summary.append(row)
            print("Finished:", ratio_label)
            print(pd.DataFrame(all_summary).tail(1))

        except Exception as e:
            warnings.warn(f"Failed on {ratio_label}: {repr(e)}")
            with open(outdir / "error.txt", "w", encoding="utf-8") as f:
                f.write(repr(e) + "\n")
            continue

    if len(all_summary) > 0:
        summary_df = pd.DataFrame(all_summary)
        summary_path = out_root / "summary_all_ratios_CORAL_T1T2T3_direct.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\nAll finished. Summary saved to:")
        print(summary_path)
        print(summary_df)
    else:
        print("No successful ratio runs.")


if __name__ == "__main__":
    main()
