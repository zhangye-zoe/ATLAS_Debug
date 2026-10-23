#!/usr/bin/env python3
"""
Train CORAL static fusion across ratio splits using the dynamic CORAL model code.

What this script does
---------------------
1. Reuses the ratio-split reading logic from the user's static baseline workflow
   (split directories such as single_000, single_020, ... containing
   train_rna_ref.h5ad / train_atac_activity.h5ad / val_true_rna.h5ad /
   val_atac_activity.h5ad / split_info.json).
2. Uses the CORAL model implementation from the user's dynamic model file
   (the uploaded batch-fix / optimized CORAL code).
3. Trains CORAL in a *static fusion* mode: phase1 + phase2 only by default,
   while keeping phase3 optional and disabled by default.
4. Evaluates T1/T2 in the *static scMRDR-style* used by the user's ratio-loop
   baseline script:
      - T1: true RNA vs predicted RNA embeddings (PCA on concatenated true/pred),
            FOSCTTM / PED / Top1 / Top5 / Top10.
      - T2: cell-wise Pearson / gene-wise Pearson / RMSE.
5. Loops over all ratio folders and saves a single summary CSV.

Typical usage
-------------
python coral_static_ratio_loop.py \
  --coral-module /data5/zhangye/CORAL/coral_optimized_batchfix.py \
  --manifest /data5/zhangye/CORAL/pbmc_rna_atac_manifest.json \
  --split-root /data5/zhangye/scMRDR/input/PBMC/preprocessed_input/results_ratio_loop \
  --out-root /data5/zhangye/CORAL_static/PBMC

Notes
-----
- This script expects prediction_pairs to contain the static translation direction
  you want to evaluate, e.g. atac->rna or atac->s.
- For fairness with the user's prior static baseline, summary_all_ratios contains
  the same high-level ratio bookkeeping fields and static T1/T2 metrics.
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
from typing import Dict, List, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import torch


# =========================================================
# 0. utilities used by the static ratio-loop workflow
# =========================================================

def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_cor(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1 or y.size <= 1:
        return np.nan
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
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


def maybe_log1p_np(x: np.ndarray, assume_logged: bool) -> np.ndarray:
    if assume_logged:
        return x.astype(np.float32, copy=False)
    return np.log1p(np.clip(x, a_min=0, a_max=None)).astype(np.float32, copy=False)


def get_common_names(*arrays: Sequence[str]) -> List[str]:
    if len(arrays) == 0:
        return []
    common = set(arrays[0])
    for arr in arrays[1:]:
        common &= set(arr)
    return sorted(common)


def save_h5ad_safe(adata: ad.AnnData, path: os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(path))


# =========================================================
# 1. import the user's CORAL module dynamically
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
# 2. scMRDR-style static evaluation
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

    fos = []
    ped = []
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
# 3. helpers to reuse the static split logic
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


def save_prediction_h5ad(pred_raw: np.ndarray, cell_ids: Sequence[str], feature_names: Sequence[str], out_path: Path):
    adata = ad.AnnData(X=np.asarray(pred_raw, dtype=np.float32))
    adata.obs_names = np.asarray(cell_ids).astype(str)
    adata.var_names = np.asarray(feature_names).astype(str)
    save_h5ad_safe(adata, out_path)


# =========================================================
# 4. one-ratio training using dynamic model as static model
# =========================================================

def train_one_ratio(
    coral,
    split_dir: Path,
    outdir: Path,
    manifest_path: Path,
    cfg,
    static_only: bool = True,
):
    outdir.mkdir(parents=True, exist_ok=True)

    modalities, alignment_pairs, prediction_pairs, lag_edges, label_key, future_label_key, reference_time_key, stage_order_map = coral.parse_manifest(manifest_path)

    ds_train, meta, val_adatas = coral.build_multimodal_ratio_dataset(
        split_dir,
        modalities,
        label_key=label_key,
        standardize=True,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    writer = coral.SummaryWriter(log_dir=str(outdir / "tensorboard"))

    input_dims = {m: ds_train.modalities[m].shape[1] for m in ds_train.modality_names}

    # prefer batch-fix model if available; otherwise fall back to optimized / fixed CORAL model
    if hasattr(coral, "CORALOptimizedBatchFix"):
        n_batches = len(meta.get("batch_categories", []))
        if n_batches <= 0:
            # fallback to 1 if batch info absent
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
    torch.save({"model": model.state_dict()}, outdir / "CORAL_static_final.pt")

    # inference
    infer_res = coral.infer_on_arrays(model, meta["val_modalities"], meta["val_masks"], device)
    np.save(outdir / "val_h.npy", infer_res["h"])
    np.save(outdir / "val_time.npy", infer_res["t"])
    coral.save_npz_dict(infer_res["shared"], outdir / "val_shared_latents.npz")
    coral.save_npz_dict(infer_res["private"], outdir / "val_private_latents.npz")

    # keep the model's own T1/T2 exports too
    if hasattr(coral, "evaluate_t1"):
        try:
            coral.evaluate_t1(meta, infer_res, alignment_pairs, outdir)
        except Exception:
            pass
    if hasattr(coral, "evaluate_t2"):
        try:
            coral.evaluate_t2(meta, infer_res, prediction_pairs, outdir)
        except Exception:
            pass

    src, tgt = pick_primary_prediction_pair(prediction_pairs)
    pair_key = f"{src}__to__{tgt}"

    query_mask = get_query_mask(meta, src, tgt)
    paired_mask = query_mask & (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)
    if paired_mask.sum() < 2:
        paired_mask = (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)
    if paired_mask.sum() < 2:
        raise ValueError(f"Not enough paired/query cells for static evaluation on {src}->{tgt}")

    pred_std = infer_res["pred"][pair_key][paired_mask]
    true_std = meta["val_modalities"][tgt][paired_mask]
    true_raw = meta["val_modalities_raw"][tgt][paired_mask]
    pred_raw = coral.inverse_standardize(pred_std, meta["stats"][tgt]) if hasattr(coral, "inverse_standardize") else pred_std

    cell_ids = np.asarray(meta["val_cell_ids"])[paired_mask].astype(str)
    feature_names = np.asarray(meta["common_features"]).astype(str)

    # static baseline-style saved objects
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
    }

    with open(outdir / "summary_one_ratio.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    writer.close()
    return row


# =========================================================
# 5. main ratio-loop entry
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Train static CORAL across ratio splits using dynamic CORAL model code")
    parser.add_argument("--coral-module", type=str, required=True, help="Path to the dynamic CORAL python file (e.g. coral_optimized_batchfix.py)")
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
    parser.add_argument("--run-phase3", action="store_true", help="Enable dynamic phase3 as an ablation. Default is static-only.")
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

    cfg_kwargs = dict(
        output_dir=str(out_root),  # overwritten per ratio below
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

    # optional fields that exist only in optimized/batchfix configs
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

            row = train_one_ratio(
                coral=coral,
                split_dir=split_dir,
                outdir=outdir,
                manifest_path=manifest_path,
                cfg=cfg,
                static_only=(not args.run_phase3),
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
        summary_path = out_root / "summary_all_ratios_CORAL_static.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\nAll finished. Summary saved to:")
        print(summary_path)
        print(summary_df)
    else:
        print("No successful ratio runs.")


if __name__ == "__main__":
    main()
