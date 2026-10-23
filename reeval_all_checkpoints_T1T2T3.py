#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-evaluate all CORAL checkpoint .pt files on T1/T2/T3.

This script:
1. Reads all .pt checkpoint files from one trained output directory.
2. Rebuilds the same CORAL model architecture.
3. Loads each checkpoint.
4. Runs validation inference.
5. Recomputes:
   - Native T1/T2 from the CORAL module, if available.
   - scMRDR-style T1: true target RNA vs predicted target RNA PCA matching.
   - scMRDR-style T2: true target RNA vs predicted target RNA expression recovery.
   - Direct T3: POA / POAUC based on CORAL_time.
6. Saves per-checkpoint results and a stability summary.

Example:
python /data5/zhangye/CORAL/reeval_all_checkpoints_T1T2T3.py \
  --coral-module /data5/zhangye/CORAL/coral_optimized_batchfix.py \
  --helper-module /data5/zhangye/CORAL/coral_ratio_loop_T1T2T3_direct.py \
  --manifest /data5/zhangye/CORAL/bmmc_rna_atac_manifest.json \
  --split-dir /data5/zhangye/scMRDR/input/BMMC/preprocessed_input/results_ratio_loop/single_000 \
  --checkpoint-dir /data5/zhangye/CORAL/output/BMMC_timehead_fast/single_000 \
  --out-dir /data5/zhangye/CORAL/output/BMMC_timehead_fast/single_000/checkpoint_reeval_T1T2T3 \
  --label-key cell_type \
  --t3-time-mode head \
  --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch


# =========================================================
# 1. dynamic import
# =========================================================

def load_py_module(module_path: str, module_name: str):
    module_path = str(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# =========================================================
# 2. checkpoint utilities
# =========================================================

def parse_ckpt_name(path: Path) -> Tuple[int, str]:
    """
    Return sortable key: (epoch, phase_rank).
    Supports names like:
      phase1_epoch_25.pt
      phase2_epoch_125.pt
      phase3_epoch_200.pt
      CORAL_optimized_batchfix_final.pt
      CORAL_T1T2T3_final.pt
    """
    name = path.name

    m = re.search(r"(phase\d+)_epoch_(\d+)\.pt$", name)
    if m:
        phase = m.group(1)
        epoch = int(m.group(2))
        return epoch, phase

    m = re.search(r"epoch[_\-]?(\d+)", name)
    if m:
        return int(m.group(1)), "unknown"

    if "final" in name.lower():
        return 10**9, "final"

    return 10**8, "unknown"


def list_checkpoints(checkpoint_dir: Path, include_final: bool = True) -> List[Path]:
    pts = sorted(checkpoint_dir.glob("*.pt"), key=parse_ckpt_name)
    if not include_final:
        pts = [p for p in pts if "final" not in p.name.lower()]
    if len(pts) == 0:
        raise FileNotFoundError(f"No .pt checkpoint files found under: {checkpoint_dir}")
    return pts


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ckpt_obj:
                return ckpt_obj[k]
    return ckpt_obj


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove DataParallel-style 'module.' prefix if present.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def safe_load_checkpoint(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = extract_state_dict(ckpt)
    state = clean_state_dict_keys(state)

    try:
        model.load_state_dict(state, strict=True)
        load_status = "strict_ok"
    except RuntimeError as e:
        warnings.warn(
            f"Strict loading failed for {ckpt_path.name}; retrying strict=False.\n{repr(e)}"
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        load_status = f"non_strict_ok; missing={len(missing)}; unexpected={len(unexpected)}"

    epoch = None
    phase = None
    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", None)
        phase = ckpt.get("phase", None)

    if epoch is None:
        epoch, phase_from_name = parse_ckpt_name(ckpt_path)
        if epoch >= 10**8:
            epoch = None
        if phase is None:
            phase = phase_from_name

    return {
        "epoch": epoch,
        "phase": phase,
        "load_status": load_status,
    }


# =========================================================
# 3. config / model rebuild
# =========================================================

def load_json_maybe(path: Optional[str]):
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path)


def make_cfg(coral, args, checkpoint_dir: Path, out_dir: Path):
    """
    Reconstruct TrainConfig.

    Priority:
    1. checkpoint_dir/train_config.json if available
    2. command-line defaults
    """
    raw_cfg = {}
    cfg_path = checkpoint_dir / "train_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

    # Override fields needed for re-evaluation.
    raw_cfg["output_dir"] = str(out_dir)
    raw_cfg["device"] = args.device

    # Add fallback values if train_config.json is incomplete.
    fallback = dict(
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
        topo_k=args.topo_k,
        lag_bins=args.lag_bins,
        dynamic_time_source=args.training_time_source,
        eval_all_time_candidates=args.eval_all_time_candidates,
    )

    for k, v in fallback.items():
        raw_cfg.setdefault(k, v)

    cfg_fields = set(getattr(coral.TrainConfig, "__dataclass_fields__", {}).keys())
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in cfg_fields}

    return coral.TrainConfig(**cfg_kwargs)


def build_model(coral, ds_train, meta, prediction_pairs, lag_edges, cfg, device):
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
        raise AttributeError("Cannot find CORAL model class in coral module.")

    return model


# =========================================================
# 4. inference helpers
# =========================================================

def build_val_lag_banks_if_possible(coral, infer_res: dict, meta: dict, cfg, model):
    """
    Same logic as the training/eval script:
    after first validation inference, build validation lag banks and infer again.
    """
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


def infer_checkpoint(coral, model, meta, cfg, device, args):
    compute_dpt = bool(args.eval_all_time_candidates or args.t3_time_mode == "dpt")
    infer_time_source = args.t3_time_mode if args.t3_time_mode in {"head", "dpt"} else "head"

    try:
        infer_res = coral.infer_on_arrays(
            model,
            meta["val_modalities"],
            meta["val_masks"],
            device,
            time_source=infer_time_source,
            compute_dpt=compute_dpt,
        )
    except TypeError:
        infer_res = coral.infer_on_arrays(
            model,
            meta["val_modalities"],
            meta["val_masks"],
            device,
        )

    if args.use_val_lag_banks:
        val_lag_banks = build_val_lag_banks_if_possible(coral, infer_res, meta, cfg, model)
        if val_lag_banks is not None:
            try:
                infer_res = coral.infer_on_arrays(
                    model,
                    meta["val_modalities"],
                    meta["val_masks"],
                    device,
                    lag_banks_np=val_lag_banks,
                    time_source=infer_time_source,
                    compute_dpt=compute_dpt,
                )
            except TypeError:
                infer_res = coral.infer_on_arrays(
                    model,
                    meta["val_modalities"],
                    meta["val_masks"],
                    device,
                    lag_banks_np=val_lag_banks,
                )

    return infer_res


# =========================================================
# 5. metric helpers
# =========================================================

def get_scalar_from_metric_df(df: pd.DataFrame, key_candidates: Sequence[str]):
    if df is None or len(df) == 0:
        return np.nan

    # style 1: metric/value
    if "metric" in df.columns and "value" in df.columns:
        tmp = df.set_index("metric")["value"].to_dict()
        for k in key_candidates:
            if k in tmp:
                return tmp[k]

    # style 2: columns directly
    for k in key_candidates:
        if k in df.columns:
            return df[k].iloc[0]

    return np.nan


def inverse_standardize_fallback(coral, x: np.ndarray, stats: dict) -> np.ndarray:
    if hasattr(coral, "inverse_standardize"):
        return coral.inverse_standardize(x, stats)

    mu = np.asarray(stats["mean"], dtype=np.float32)
    sd = np.asarray(stats["std"], dtype=np.float32)
    return x * sd[None, :] + mu[None, :]


def evaluate_static_scmrdr_style(coral, helper, meta, infer_res, prediction_pairs, outdir: Path, seed: int):
    """
    T1/T2 based on primary prediction pair, e.g. atac -> rna.
    This matches the ratio-loop helper:
      - T1: PCA(true target, predicted target) + FOSCTTM/PED/TopK.
      - T2: Pearson/RMSE between predicted target and true target.
    """
    src, tgt = helper.pick_primary_prediction_pair(prediction_pairs)
    pair_key = f"{src}__to__{tgt}"

    if pair_key not in infer_res["pred"]:
        raise KeyError(f"Prediction pair {pair_key} not found in infer_res['pred'].")

    query_mask = helper.get_query_mask(meta, src, tgt)
    paired_mask = query_mask & (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)

    if paired_mask.sum() < 2:
        paired_mask = (meta["val_masks"][src] > 0) & (meta["val_masks"][tgt] > 0)

    if paired_mask.sum() < 2:
        raise ValueError(f"Not enough paired/query cells for static evaluation on {src}->{tgt}")

    pred_std = infer_res["pred"][pair_key][paired_mask]
    true_raw = meta["val_modalities_raw"][tgt][paired_mask]
    pred_raw = inverse_standardize_fallback(coral, pred_std, meta["stats"][tgt])

    val_ids = np.asarray(meta["val_cell_ids"]).astype(str)
    cell_ids = val_ids[paired_mask]
    feature_names = np.asarray(meta["common_features"]).astype(str)

    t1_df = helper.evaluate_t1_from_true_pred(
        pred_raw,
        true_raw,
        common_cells=cell_ids,
        common_features=feature_names,
        outdir=outdir,
        seed=seed,
    )

    t2_df = helper.evaluate_t2_from_true_pred(
        pred_raw,
        true_raw,
    )
    t2_df.to_csv(outdir / "T2_metrics_scMRDR_style.csv", index=False)

    return {
        "prediction_pair": pair_key,
        "query_cells": int(paired_mask.sum()),
        "common_features": int(len(feature_names)),
        "foscttm_scmrdr": get_scalar_from_metric_df(t1_df, ["FOSCTTM"]),
        "ped_scmrdr": get_scalar_from_metric_df(t1_df, ["PED"]),
        "top1_acc_scmrdr": get_scalar_from_metric_df(t1_df, ["Top1_ACC"]),
        "top5_acc_scmrdr": get_scalar_from_metric_df(t1_df, ["Top5_ACC"]),
        "top10_acc_scmrdr": get_scalar_from_metric_df(t1_df, ["Top10_ACC"]),
        "pearson_cell_mean_scmrdr": get_scalar_from_metric_df(t2_df, ["pearson_cell_mean"]),
        "pearson_gene_mean_scmrdr": get_scalar_from_metric_df(t2_df, ["pearson_gene_mean"]),
        "rmse_scmrdr": get_scalar_from_metric_df(t2_df, ["rmse"]),
    }


def evaluate_t3_direct(coral, helper, meta, infer_res, val_adatas, label_key, future_label_key,
                       reference_time_key, stage_order_map, outdir: Path, args):
    """
    Build eval AnnData if the CORAL module supports it; otherwise use infer_res['t'].
    Then compute selected T3 metric and all available time-candidate metrics.
    """
    val_ids = np.asarray(meta["val_cell_ids"]).astype(str)
    obs_union = meta["val_obs_union"].copy()
    obs_union.index = obs_union.index.astype(str)

    adata_eval = None
    obs_df = None

    if hasattr(coral, "build_eval_anndata"):
        try:
            adata_eval = coral.build_eval_anndata(
                meta,
                infer_res,
                val_adatas,
                label_key,
                future_label_key,
                t3_time_mode=args.t3_time_mode,
                root_labels=args.root_labels,
                graph_k=args.graph_k,
                eval_all_time_candidates=args.eval_all_time_candidates,
            )
            obs_df = adata_eval.obs.copy()
            obs_df["cell_id"] = val_ids

            if "CORAL_time" not in obs_df.columns:
                obs_df["CORAL_time"] = np.asarray(infer_res["t"], dtype=float)

            adata_eval.write_h5ad(str(outdir / "CORAL_eval_embeddings_T3.h5ad"))

            if "CORAL_time_info" in adata_eval.uns:
                with open(outdir / "CORAL_time_info.json", "w", encoding="utf-8") as f:
                    json.dump(adata_eval.uns["CORAL_time_info"], f, indent=2, ensure_ascii=False)

        except TypeError as e:
            warnings.warn(
                "coral.build_eval_anndata exists but does not support these arguments; "
                f"falling back to infer_res['t']. Error: {repr(e)}"
            )

    if obs_df is None:
        obs_df = obs_union.loc[val_ids].copy()
        obs_df["CORAL_time"] = np.asarray(infer_res["t"], dtype=float)
        obs_df["cell_id"] = val_ids

    t3_eval_mask = helper.get_t3_eval_mask(
        meta,
        t3_cell_set=args.t3_cell_set,
        fixed_cell_ids=None,
    )

    t3_pred = pd.to_numeric(obs_df["CORAL_time"], errors="coerce").to_numpy(dtype=float)

    t3 = helper.compute_t3_direct_metrics(
        obs_df=obs_df,
        pred_time=t3_pred,
        label_key=label_key,
        stage_order_map=stage_order_map,
        reference_time_key=reference_time_key,
        eval_mask=t3_eval_mask,
        orient_by_stage=(not args.no_orient_t3_by_stage),
    )

    t3["t3_time_mode"] = args.t3_time_mode
    t3["root_labels"] = ";".join([str(x) for x in args.root_labels]) if args.root_labels else None
    t3["graph_k"] = int(args.graph_k)

    candidate_rows = []
    for tk in [
        "CORAL_time",
        "CORAL_time_head",
        "CORAL_time_dpt",
        "CORAL_time_graph_auto",
        "CORAL_time_graph_root",
    ]:
        if tk in obs_df.columns:
            tt = pd.to_numeric(obs_df[tk], errors="coerce").to_numpy(dtype=float)
            cand = helper.compute_t3_direct_metrics(
                obs_df=obs_df,
                pred_time=tt,
                label_key=label_key,
                stage_order_map=stage_order_map,
                reference_time_key=reference_time_key,
                eval_mask=t3_eval_mask,
                orient_by_stage=(not args.no_orient_t3_by_stage),
            )
            cand["time_key"] = tk
            candidate_rows.append(cand)

    if candidate_rows:
        pd.DataFrame(candidate_rows).to_csv(outdir / "T3_time_candidate_metrics.csv", index=False)

    pd.DataFrame([t3]).to_csv(outdir / "T3_metrics_direct.csv", index=False)
    obs_df.to_csv(outdir / "T3_eval_cells.csv", index=False)

    return {
        "t3_pairacc": t3.get("PairAcc", np.nan),
        "t3_pairauc": t3.get("PairAUC", np.nan),
        "t3_poa": t3.get("POA", np.nan),
        "t3_poauc": t3.get("POAUC", np.nan),
        "t3_n_eval": t3.get("n_eval", 0),
        "t3_status": t3.get("status", "unknown"),
        "t3_time_flipped": t3.get("time_flipped", False),
        "t3_label_key": t3.get("label_key", label_key),
        "t3_reference": t3.get("reference", None),
    }


def evaluate_native_t1_t2(coral, meta, infer_res, alignment_pairs, prediction_pairs, outdir: Path):
    """
    Also run the model module's original evaluate_t1/evaluate_t2 if available.
    """
    row = {}

    if hasattr(coral, "evaluate_t1"):
        try:
            t1_native = coral.evaluate_t1(meta, infer_res, alignment_pairs, outdir)
            row["foscttm_native"] = get_scalar_from_metric_df(t1_native, ["FOSCTTM"])
            row["ped_native"] = get_scalar_from_metric_df(t1_native, ["PED"])
        except Exception as e:
            warnings.warn(f"Native coral.evaluate_t1 failed: {repr(e)}")
            row["foscttm_native"] = np.nan
            row["ped_native"] = np.nan

    if hasattr(coral, "evaluate_t2"):
        try:
            t2_native = coral.evaluate_t2(meta, infer_res, prediction_pairs, outdir)
            row["pearson_cell_mean_native"] = get_scalar_from_metric_df(
                t2_native, ["Pearson_cell", "pearson_cell_mean", "cellwise_pearson"]
            )
            row["pearson_gene_mean_native"] = get_scalar_from_metric_df(
                t2_native, ["Pearson_gene", "pearson_gene_mean", "genewise_pearson"]
            )
            row["rmse_native"] = get_scalar_from_metric_df(t2_native, ["RMSE", "rmse"])
        except Exception as e:
            warnings.warn(f"Native coral.evaluate_t2 failed: {repr(e)}")
            row["pearson_cell_mean_native"] = np.nan
            row["pearson_gene_mean_native"] = np.nan
            row["rmse_native"] = np.nan

    return row


# =========================================================
# 6. stability summary
# =========================================================

def make_stability_summary(df: pd.DataFrame, out_dir: Path):
    metric_cols = [
        "foscttm_scmrdr",
        "ped_scmrdr",
        "top1_acc_scmrdr",
        "top5_acc_scmrdr",
        "top10_acc_scmrdr",
        "pearson_cell_mean_scmrdr",
        "pearson_gene_mean_scmrdr",
        "rmse_scmrdr",
        "t3_poa",
        "t3_poauc",
        "foscttm_native",
        "ped_native",
        "pearson_cell_mean_native",
        "pearson_gene_mean_native",
        "rmse_native",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    rows = []
    for c in metric_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue

        # Direction: lower is better for FOSCTTM/PED/RMSE; higher is better otherwise.
        lower_better = any(x in c.lower() for x in ["foscttm", "ped", "rmse"])
        if lower_better:
            best_idx = vals.idxmin()
        else:
            best_idx = vals.idxmax()

        mean = float(finite.mean())
        std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
        cv = float(std / (abs(mean) + 1e-8))

        rows.append({
            "metric": c,
            "direction": "lower_better" if lower_better else "higher_better",
            "n_valid": int(len(finite)),
            "mean": mean,
            "std": std,
            "cv": cv,
            "min": float(finite.min()),
            "max": float(finite.max()),
            "best_checkpoint": df.loc[best_idx, "checkpoint"],
            "best_epoch": df.loc[best_idx, "epoch"],
            "best_phase": df.loc[best_idx, "phase"],
            "best_value": float(vals.loc[best_idx]),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "stability_summary.csv", index=False)

    # A compact recommendation table.
    key_metrics = [m for m in ["rmse_scmrdr", "pearson_cell_mean_scmrdr", "t3_poa", "t3_poauc", "foscttm_scmrdr"] if m in df.columns]
    rec_rows = []
    for m in key_metrics:
        vals = pd.to_numeric(df[m], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        lower_better = any(x in m.lower() for x in ["foscttm", "ped", "rmse"])
        idx = vals.idxmin() if lower_better else vals.idxmax()
        rec_rows.append({
            "metric": m,
            "best_checkpoint": df.loc[idx, "checkpoint"],
            "best_epoch": df.loc[idx, "epoch"],
            "best_phase": df.loc[idx, "phase"],
            "best_value": vals.loc[idx],
        })

    pd.DataFrame(rec_rows).to_csv(out_dir / "best_checkpoints_by_metric.csv", index=False)


# =========================================================
# 7. main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--coral-module", type=str, required=True)
    parser.add_argument("--helper-module", type=str, required=True,
                        help="Path to coral_ratio_loop_T1T2T3_direct.py")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--label-key", type=str, default=None)
    parser.add_argument("--reference-time-key", type=str, default=None)
    parser.add_argument("--stage-order-json", type=str, default=None)

    parser.add_argument("--label-csv", type=str, default=None)
    parser.add_argument("--label-csv-cell-col", type=str, default="cell_id")
    parser.add_argument("--label-csv-label-col", type=str, default="cell_type")

    parser.add_argument("--t3-cell-set", type=str, default="val_all",
                        choices=["val_all", "val_atac", "val_paired"])
    parser.add_argument("--t3-time-mode", type=str, default="head",
                        choices=["head", "dpt", "graph_auto", "graph_root"])
    parser.add_argument("--root-labels", nargs="*", default=None)
    parser.add_argument("--graph-k", type=int, default=30)
    parser.add_argument("--eval-all-time-candidates", action="store_true")
    parser.add_argument("--no-orient-t3-by-stage", action="store_true")

    parser.add_argument("--use-val-lag-banks", action="store_true",
                        help="Run a second validation inference with validation lag banks if dynamic modules exist.")

    parser.add_argument("--include-final", action="store_true",
                        help="Include final .pt files, e.g. CORAL_optimized_batchfix_final.pt.")

    # Fallback config values if train_config.json is missing.
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
    parser.add_argument("--training-time-source", type=str, default="head",
                        choices=["head", "dpt", "blend"])

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    split_dir = Path(args.split_dir)
    manifest_path = Path(args.manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_checkpoint").mkdir(parents=True, exist_ok=True)

    helper = load_py_module(args.helper_module, "coral_eval_helper")
    coral = helper.load_coral_module(args.coral_module)

    if hasattr(coral, "set_seed"):
        coral.set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = make_cfg(coral, args, checkpoint_dir, out_dir)

    # Parse manifest.
    (
        modalities,
        alignment_pairs,
        prediction_pairs,
        lag_edges,
        label_key,
        future_label_key,
        reference_time_key,
        stage_order_map,
    ) = coral.parse_manifest(manifest_path)

    if args.label_key is not None:
        label_key = args.label_key
    if args.reference_time_key is not None:
        reference_time_key = args.reference_time_key
    if args.stage_order_json is not None:
        stage_order_map = load_json_maybe(args.stage_order_json)

    # Build validation dataset once.
    ds_train, meta, val_adatas = coral.build_multimodal_ratio_dataset(
        split_dir,
        modalities,
        label_key=label_key,
        standardize=True,
    )

    # Optional external labels.
    if args.label_csv is not None:
        helper.merge_label_csv_into_meta(
            meta,
            label_csv=args.label_csv,
            cell_col=args.label_csv_cell_col,
            label_col=args.label_csv_label_col,
            output_label_key=label_key or "cell_type",
        )
        label_key = label_key or "cell_type"

    ckpts = list_checkpoints(checkpoint_dir, include_final=args.include_final)

    print("\n==============================")
    print("Checkpoint directory:", checkpoint_dir)
    print("Number of checkpoints:", len(ckpts))
    print("Output directory:", out_dir)
    print("==============================\n")

    all_rows = []

    for i, ckpt_path in enumerate(ckpts, start=1):
        ckpt_stem = ckpt_path.stem
        ckpt_out = out_dir / "by_checkpoint" / ckpt_stem
        ckpt_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(ckpts)}] Evaluating {ckpt_path.name}")

        model = build_model(
            coral=coral,
            ds_train=ds_train,
            meta=meta,
            prediction_pairs=prediction_pairs,
            lag_edges=lag_edges,
            cfg=cfg,
            device=device,
        )

        load_info = safe_load_checkpoint(model, ckpt_path, device)

        infer_res = infer_checkpoint(
            coral=coral,
            model=model,
            meta=meta,
            cfg=cfg,
            device=device,
            args=args,
        )

        # Save basic arrays for this checkpoint.
        np.save(ckpt_out / "val_h.npy", infer_res["h"])
        np.save(ckpt_out / "val_time.npy", infer_res["t"])

        if "t_head" in infer_res:
            np.save(ckpt_out / "val_time_head.npy", infer_res["t_head"])
        if "t_dpt" in infer_res:
            np.save(ckpt_out / "val_time_dpt.npy", infer_res["t_dpt"])

        if hasattr(coral, "save_npz_dict"):
            coral.save_npz_dict(infer_res["shared"], ckpt_out / "val_shared_latents.npz")
            coral.save_npz_dict(infer_res["private"], ckpt_out / "val_private_latents.npz")

        row = {
            "checkpoint": ckpt_path.name,
            "checkpoint_path": str(ckpt_path),
            "epoch": load_info.get("epoch"),
            "phase": load_info.get("phase"),
            "load_status": load_info.get("load_status"),
            "t3_time_mode": args.t3_time_mode,
            "use_val_lag_banks": bool(args.use_val_lag_banks),
        }

        # Native T1/T2.
        row.update(
            evaluate_native_t1_t2(
                coral=coral,
                meta=meta,
                infer_res=infer_res,
                alignment_pairs=alignment_pairs,
                prediction_pairs=prediction_pairs,
                outdir=ckpt_out,
            )
        )

        # scMRDR-style T1/T2.
        try:
            row.update(
                evaluate_static_scmrdr_style(
                    coral=coral,
                    helper=helper,
                    meta=meta,
                    infer_res=infer_res,
                    prediction_pairs=prediction_pairs,
                    outdir=ckpt_out,
                    seed=args.seed,
                )
            )
        except Exception as e:
            warnings.warn(f"scMRDR-style T1/T2 failed for {ckpt_path.name}: {repr(e)}")
            row.update({
                "prediction_pair": None,
                "query_cells": 0,
                "common_features": len(meta.get("common_features", [])),
                "foscttm_scmrdr": np.nan,
                "ped_scmrdr": np.nan,
                "top1_acc_scmrdr": np.nan,
                "top5_acc_scmrdr": np.nan,
                "top10_acc_scmrdr": np.nan,
                "pearson_cell_mean_scmrdr": np.nan,
                "pearson_gene_mean_scmrdr": np.nan,
                "rmse_scmrdr": np.nan,
            })

        # Direct T3.
        try:
            row.update(
                evaluate_t3_direct(
                    coral=coral,
                    helper=helper,
                    meta=meta,
                    infer_res=infer_res,
                    val_adatas=val_adatas,
                    label_key=label_key,
                    future_label_key=future_label_key,
                    reference_time_key=reference_time_key,
                    stage_order_map=stage_order_map,
                    outdir=ckpt_out,
                    args=args,
                )
            )
        except Exception as e:
            warnings.warn(f"T3 evaluation failed for {ckpt_path.name}: {repr(e)}")
            row.update({
                "t3_pairacc": np.nan,
                "t3_pairauc": np.nan,
                "t3_poa": np.nan,
                "t3_poauc": np.nan,
                "t3_n_eval": 0,
                "t3_status": f"failed: {repr(e)}",
            })

        # Save per-checkpoint summary.
        with open(ckpt_out / "summary_one_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

        all_rows.append(row)

        df_tmp = pd.DataFrame(all_rows)
        df_tmp.to_csv(out_dir / "all_checkpoint_metrics.partial.csv", index=False)

        print(
            f"Done {ckpt_path.name}: "
            f"FOS={row.get('foscttm_scmrdr', np.nan):.4f}, "
            f"RMSE={row.get('rmse_scmrdr', np.nan):.4f}, "
            f"POA={row.get('t3_poa', np.nan):.4f}, "
            f"POAUC={row.get('t3_poauc', np.nan):.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "all_checkpoint_metrics.csv", index=False)

    make_stability_summary(df, out_dir)

    # Also save config used.
    used = vars(args).copy()
    used["device_resolved"] = str(device)
    used["n_checkpoints"] = len(ckpts)
    with open(out_dir / "reeval_config.json", "w", encoding="utf-8") as f:
        json.dump(used, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("Finished checkpoint re-evaluation.")
    print("Saved:")
    print("  ", out_dir / "all_checkpoint_metrics.csv")
    print("  ", out_dir / "stability_summary.csv")
    print("  ", out_dir / "best_checkpoints_by_metric.csv")
    print("==============================\n")


if __name__ == "__main__":
    main()