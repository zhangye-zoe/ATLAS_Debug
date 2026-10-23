#!/usr/bin/env python3
"""
Re-infer T3 metrics for every saved CORAL checkpoint under each ratio folder.

This script does NOT retrain the model. It rebuilds each ratio's validation data,
loads every checkpoint (*.pt by default), runs inference, constructs T3 time
candidates, and recomputes POA / POAUC.

Typical use:
python reevaluate_coral_checkpoints_graphtime.py \
  --coral-module /data5/zhangye/CORAL/coral_v1_graphtime.py \
  --manifest /data5/zhangye/CORAL/pbmc_rna_atac_manifest.json \
  --split-root /data5/zhangye/scMRDR/input/PBMC/preprocessed_input/results_ratio_loop \
  --out-root /data5/zhangye/CORAL/output/PBMC_graphtime \
  --label-key cell_type \
  --stage-order-json /data5/zhangye/CORAL/pbmc_stage_order.json \
  --label-csv /data5/zhangye/scMRDR/input/PBMC/raw_input/pbmc_azimuth_annotations.csv \
  --label-csv-cell-col cell_id \
  --label-csv-label-col predicted.celltype.l2 \
  --t3-time-mode graph_root \
  --root-labels HSPC "B naive" "CD4 Naive" "CD8 Naive" \
  --graph-k 30
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def load_coral_module(module_path: str):
    module_path = str(module_path)
    spec = importlib.util.spec_from_file_location("coral_user_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json_maybe(x: Optional[str]):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    if x.startswith("{") or x.startswith("["):
        return json.loads(x)
    p = Path(x)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(x)


def read_id_file(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(str(s))
    return ids


def save_h5ad_safe(adata: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(path))


def resolve_ratio_labels(split_root: Path, requested: Optional[Sequence[str]]) -> List[str]:
    if requested:
        return list(requested)
    return sorted([p.name for p in split_root.iterdir() if p.is_dir() and p.name.startswith("single_")])


def parse_checkpoint_name(path: Path) -> Dict[str, object]:
    name = path.name
    if name in {"CORAL_T1T2T3_final.pt", "CORAL_final.pt", "final.pt"}:
        return {"ckpt_name": name, "ckpt_phase": "final", "ckpt_epoch": np.nan, "ckpt_sort": 10**9}
    m = re.search(r"(phase\d+)_epoch_(\d+)\.pt$", name)
    if m:
        phase = m.group(1)
        epoch = int(m.group(2))
        phase_num = int(phase.replace("phase", ""))
        return {"ckpt_name": name, "ckpt_phase": phase, "ckpt_epoch": epoch, "ckpt_sort": phase_num * 100000 + epoch}
    return {"ckpt_name": name, "ckpt_phase": "unknown", "ckpt_epoch": np.nan, "ckpt_sort": 10**8}


# -----------------------------------------------------------------------------
# Label merging
# -----------------------------------------------------------------------------

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

    csv_col = f"{output_label_key}_from_csv"
    if csv_col in obs.columns:
        if output_label_key in obs.columns:
            obs[output_label_key] = obs[output_label_key].where(obs[output_label_key].notna(), obs[csv_col])
        else:
            obs[output_label_key] = obs[csv_col]
        obs = obs.drop(columns=[csv_col])

    meta["val_obs_union"] = obs
    val_ids = np.asarray(meta["val_cell_ids"]).astype(str)
    if output_label_key in obs.columns:
        meta["labels"] = obs.loc[val_ids, output_label_key].astype(str).to_numpy()
    else:
        meta["labels"] = None
    meta["label_key"] = output_label_key


# -----------------------------------------------------------------------------
# T3 metrics
# -----------------------------------------------------------------------------

def _find_first_existing_key(container, candidates):
    for k in candidates:
        if k in container:
            return k
    return None


def resolve_label_key(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["celltype", "cell_type", "annotation", "annot", "labels", "leiden"])
    return _find_first_existing_key(df.columns, candidates)


def interclass_pairwise_accuracy(pred, ref, weighted: bool = True):
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


def interclass_pairwise_auc(pred, ref, weighted: bool = True):
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
    pred = np.asarray(pred, dtype=float).copy()
    ref = np.asarray(ref, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() < 3:
        return pred, False
    classes = np.sort(np.unique(ref[valid]))
    if len(classes) < 2:
        return pred, False
    early, late = classes[0], classes[-1]
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
    final_label_key = None
    missing_stage_labels: List[str] = []

    if reference_time_key is not None and reference_time_key in obs_df.columns:
        ref = pd.to_numeric(obs_df[reference_time_key], errors="coerce").to_numpy(dtype=float)
        reference_name = reference_time_key
    else:
        final_label_key = resolve_label_key(obs_df, label_key)
        if final_label_key is None:
            return {"PairAcc": np.nan, "PairAUC": np.nan, "POA": np.nan, "POAUC": np.nan, "n_eval": 0, "status": "failed_no_label_key"}
        if stage_order_map is None:
            return {"PairAcc": np.nan, "PairAUC": np.nan, "POA": np.nan, "POAUC": np.nan, "n_eval": 0, "label_key": final_label_key, "status": "failed_no_stage_order_map"}
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


# -----------------------------------------------------------------------------
# Model / checkpoint helpers
# -----------------------------------------------------------------------------

def make_config(coral, ratio_outdir: Path, args) -> object:
    train_config_path = ratio_outdir / "train_config.json"
    if train_config_path.exists():
        cfg_raw = json.loads(train_config_path.read_text())
    else:
        cfg_raw = {}

    # CLI fallback / override for runtime-only options.
    cfg_raw.setdefault("output_dir", str(ratio_outdir))
    cfg_raw.setdefault("device", args.device)
    cfg_raw["output_dir"] = str(ratio_outdir)
    cfg_raw["device"] = args.device

    fields = set(getattr(coral.TrainConfig, "__dataclass_fields__", {}).keys())
    if fields:
        cfg_raw = {k: v for k, v in cfg_raw.items() if k in fields}
    return coral.TrainConfig(**cfg_raw)


def instantiate_model(coral, ds_train, meta, manifest_path: Path, cfg):
    modalities, alignment_pairs, prediction_pairs, lag_edges, label_key, future_label_key, reference_time_key, stage_order_map = coral.parse_manifest(manifest_path)
    input_dims = {m: ds_train.modalities[m].shape[1] for m in ds_train.modality_names}
    n_batches = len(meta.get("batch_categories", []))
    if n_batches <= 0:
        n_batches = 1

    if hasattr(coral, "CORALOptimizedBatchFix"):
        model = coral.CORALOptimizedBatchFix(
            input_dims=input_dims,
            prediction_pairs=prediction_pairs,
            lag_edges=lag_edges,
            n_batches=n_batches,
            shared_dim=cfg.shared_dim,
            private_dim=cfg.private_dim,
            dropout=cfg.dropout,
        )
    elif hasattr(coral, "CORALOptimized"):
        model = coral.CORALOptimized(
            input_dims=input_dims,
            prediction_pairs=prediction_pairs,
            lag_edges=lag_edges,
            shared_dim=cfg.shared_dim,
            private_dim=cfg.private_dim,
            dropout=cfg.dropout,
        )
    elif hasattr(coral, "CORAL"):
        model = coral.CORAL(
            input_dims=input_dims,
            prediction_pairs=prediction_pairs,
            lag_edges=lag_edges,
            shared_dim=cfg.shared_dim,
            private_dim=cfg.private_dim,
            dropout=cfg.dropout,
        )
    else:
        raise AttributeError("Cannot find a supported CORAL model class in the imported module.")

    return model, modalities, alignment_pairs, prediction_pairs, lag_edges, label_key, future_label_key, reference_time_key, stage_order_map


def load_checkpoint_into_model(model, ckpt_path: Path, device: torch.device) -> Dict[str, object]:
    obj = torch.load(str(ckpt_path), map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    missing, unexpected = model.load_state_dict(state, strict=False)
    info = parse_checkpoint_name(ckpt_path)
    if isinstance(obj, dict):
        if "epoch" in obj:
            info["ckpt_epoch_in_file"] = obj["epoch"]
        if "phase" in obj:
            info["ckpt_phase_in_file"] = obj["phase"]
    info["n_missing_keys"] = len(missing)
    info["n_unexpected_keys"] = len(unexpected)
    if len(missing) > 0:
        info["missing_keys_preview"] = ";".join(list(missing)[:8])
    if len(unexpected) > 0:
        info["unexpected_keys_preview"] = ";".join(list(unexpected)[:8])
    return info


def build_val_lag_banks_if_possible(coral, infer_res: dict, meta: dict, cfg, model) -> Optional[Dict[str, np.ndarray]]:
    if not hasattr(model, "lag_modules") or len(model.lag_modules) == 0:
        return None
    if not hasattr(coral, "build_lag_bank"):
        return None
    h = np.asarray(infer_res["h"], dtype=np.float32)
    t = np.asarray(infer_res.get("t", np.zeros(h.shape[0])), dtype=np.float32)
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


def run_inference_for_checkpoint(coral, model, meta, val_adatas, cfg, device, label_key, future_label_key, args):
    infer_res = coral.infer_on_arrays(model, meta["val_modalities"], meta["val_masks"], device)

    # Optional second pass for velocity diagnostics if lag modules exist.
    val_lag_banks = build_val_lag_banks_if_possible(coral, infer_res, meta, cfg, model)
    if val_lag_banks is not None:
        try:
            infer_res = coral.infer_on_arrays(model, meta["val_modalities"], meta["val_masks"], device, lag_banks_np=val_lag_banks)
        except TypeError:
            pass

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
            )
            obs_df = adata_eval.obs.copy()
            obs_df["cell_id"] = val_ids
        except TypeError as e:
            warnings.warn(f"build_eval_anndata does not support graph-time arguments, fallback to infer_res['t']: {repr(e)}")

    if obs_df is None:
        obs_df = obs_union.loc[val_ids].copy()
        obs_df["CORAL_time"] = np.asarray(infer_res.get("t", np.zeros(len(val_ids))), dtype=float)
        obs_df["cell_id"] = val_ids

    return infer_res, obs_df, adata_eval


# -----------------------------------------------------------------------------
# Main per-ratio evaluator
# -----------------------------------------------------------------------------

def evaluate_one_ratio(coral, ratio_label: str, split_root: Path, out_root: Path, manifest_path: Path, stage_order_map: dict, fixed_cell_ids: Optional[List[str]], args) -> pd.DataFrame:
    split_dir = split_root / ratio_label
    ratio_outdir = out_root / ratio_label
    eval_outdir = ratio_outdir / args.eval_subdir
    eval_outdir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")
    if not ratio_outdir.exists():
        raise FileNotFoundError(f"Missing ratio output dir: {ratio_outdir}")

    cfg = make_config(coral, ratio_outdir, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    modalities, alignment_pairs, prediction_pairs, lag_edges, manifest_label_key, future_label_key, reference_time_key, manifest_stage_order_map = coral.parse_manifest(manifest_path)
    label_key = args.label_key or manifest_label_key
    if args.reference_time_key is not None:
        reference_time_key = args.reference_time_key
    if stage_order_map is None:
        stage_order_map = manifest_stage_order_map

    ds_train, meta, val_adatas = coral.build_multimodal_ratio_dataset(
        split_dir,
        modalities,
        label_key=label_key,
        standardize=True,
    )
    if args.label_csv is not None:
        merge_label_csv_into_meta(
            meta,
            label_csv=args.label_csv,
            cell_col=args.label_csv_cell_col,
            label_col=args.label_csv_label_col,
            output_label_key=label_key or "cell_type",
        )
        label_key = label_key or "cell_type"

    model, *_ = instantiate_model(coral, ds_train, meta, manifest_path, cfg)
    model.to(device)

    ckpt_paths = sorted(ratio_outdir.glob(args.checkpoint_glob), key=lambda p: parse_checkpoint_name(p)["ckpt_sort"])
    if args.include_final and (ratio_outdir / "CORAL_T1T2T3_final.pt").exists() and (ratio_outdir / "CORAL_T1T2T3_final.pt") not in ckpt_paths:
        ckpt_paths.append(ratio_outdir / "CORAL_T1T2T3_final.pt")
    if args.exclude_final:
        ckpt_paths = [p for p in ckpt_paths if p.name != "CORAL_T1T2T3_final.pt"]

    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {ratio_outdir} with glob {args.checkpoint_glob!r}")

    rows: List[dict] = []
    t3_eval_mask = get_t3_eval_mask(meta, t3_cell_set=args.t3_cell_set, fixed_cell_ids=fixed_cell_ids)

    for ckpt_path in ckpt_paths:
        print(f"[{ratio_label}] Re-infer checkpoint: {ckpt_path.name}")
        model_cpu, *_ = instantiate_model(coral, ds_train, meta, manifest_path, cfg)
        model_cpu.to(device)
        ckpt_info = load_checkpoint_into_model(model_cpu, ckpt_path, device)
        model_cpu.eval()

        with torch.no_grad():
            infer_res, obs_df, adata_eval = run_inference_for_checkpoint(
                coral, model_cpu, meta, val_adatas, cfg, device, label_key, future_label_key, args
            )

        candidate_keys = [
            "CORAL_time",
            "CORAL_time_head",
            "CORAL_time_dpt",
            "CORAL_time_graph_auto",
            "CORAL_time_graph_root",
        ]
        found_any = False
        for tk in candidate_keys:
            if tk not in obs_df.columns:
                continue
            found_any = True
            tt = pd.to_numeric(obs_df[tk], errors="coerce").to_numpy(dtype=float)
            metric = compute_t3_direct_metrics(
                obs_df=obs_df,
                pred_time=tt,
                label_key=label_key,
                stage_order_map=stage_order_map,
                reference_time_key=reference_time_key,
                eval_mask=t3_eval_mask,
                orient_by_stage=(not args.no_orient_t3_by_stage),
            )
            row = {
                "ratio_label": ratio_label,
                "time_key": tk,
                "selected_t3_time_mode": args.t3_time_mode,
                "root_labels": ";".join(args.root_labels) if args.root_labels else None,
                "graph_k": args.graph_k,
                **ckpt_info,
                **metric,
                "time_mean": float(np.nanmean(tt)),
                "time_std": float(np.nanstd(tt)),
                "time_min": float(np.nanmin(tt)),
                "time_median": float(np.nanmedian(tt)),
                "time_max": float(np.nanmax(tt)),
            }
            rows.append(row)

        if not found_any:
            warnings.warn(f"No time columns found for {ckpt_path}")

        if args.save_h5ad:
            ckpt_stem = ckpt_path.name.replace(".pt", "")
            if adata_eval is not None:
                save_h5ad_safe(adata_eval, eval_outdir / f"{ckpt_stem}_eval.h5ad")
            else:
                adata_simple = ad.AnnData(X=np.asarray(infer_res["h"], dtype=np.float32), obs=obs_df.copy())
                adata_simple.obsm["X_CORAL"] = np.asarray(infer_res["h"], dtype=np.float32)
                save_h5ad_safe(adata_simple, eval_outdir / f"{ckpt_stem}_eval.h5ad")

        del model_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv = eval_outdir / "T3_checkpoint_metrics.csv"
    df.to_csv(out_csv, index=False)

    # Save a compact view for the selected final time only.
    selected = df[df["time_key"] == "CORAL_time"].copy()
    selected.to_csv(eval_outdir / "T3_checkpoint_metrics_selected_time.csv", index=False)

    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-infer validation T3 metrics for every CORAL checkpoint across ratio folders.")
    parser.add_argument("--coral-module", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True, help="Existing training output root, e.g. /data5/zhangye/CORAL/output/PBMC_graphtime")
    parser.add_argument("--ratio-labels", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-subdir", type=str, default="checkpoint_reinfer")

    parser.add_argument("--checkpoint-glob", type=str, default="*.pt")
    parser.add_argument("--include-final", action="store_true", help="Ensure CORAL_T1T2T3_final.pt is included even if the glob misses it.")
    parser.add_argument("--exclude-final", action="store_true", help="Exclude CORAL_T1T2T3_final.pt from checkpoint evaluation.")
    parser.add_argument("--save-h5ad", action="store_true", help="Optionally save an eval h5ad for each checkpoint. This can use a lot of disk space.")

    # T3 / label arguments.
    parser.add_argument("--label-key", type=str, default=None)
    parser.add_argument("--reference-time-key", type=str, default=None)
    parser.add_argument("--stage-order-json", type=str, default=None)
    parser.add_argument("--t3-cell-set", type=str, default="val_all", choices=["val_all", "val_atac", "val_paired"])
    parser.add_argument("--fixed-cell-ids", type=str, default=None)
    parser.add_argument("--no-orient-t3-by-stage", action="store_true")

    # Graph-time arguments, used if coral_v1_graphtime.py is imported.
    parser.add_argument("--t3-time-mode", type=str, default="graph_root", choices=["dpt", "head", "graph_auto", "graph_root"])
    parser.add_argument("--root-labels", nargs="*", default=None)
    parser.add_argument("--graph-k", type=int, default=30)

    # External labels. Aliases included for compatibility with prepare_monocle3_input_from_coral.py style.
    parser.add_argument("--label-csv", "--azimuth-csv", dest="label_csv", type=str, default=None)
    parser.add_argument("--label-csv-cell-col", dest="label_csv_cell_col", type=str, default="cell_id")
    parser.add_argument("--label-csv-label-col", "--azimuth-label-col", dest="label_csv_label_col", type=str, default="cell_type")

    args = parser.parse_args()

    coral = load_coral_module(args.coral_module)
    if hasattr(coral, "set_seed"):
        coral.set_seed(1234)

    split_root = Path(args.split_root)
    out_root = Path(args.out_root)
    manifest_path = Path(args.manifest)
    stage_order_map = load_json_maybe(args.stage_order_json)
    fixed_cell_ids = read_id_file(args.fixed_cell_ids)
    ratio_labels = resolve_ratio_labels(split_root, args.ratio_labels)

    all_rows: List[pd.DataFrame] = []
    for ratio_label in ratio_labels:
        try:
            print("\n" + "=" * 100)
            print(f"Re-inferring ratio: {ratio_label}")
            print("=" * 100)
            df = evaluate_one_ratio(
                coral=coral,
                ratio_label=ratio_label,
                split_root=split_root,
                out_root=out_root,
                manifest_path=manifest_path,
                stage_order_map=stage_order_map,
                fixed_cell_ids=fixed_cell_ids,
                args=args,
            )
            all_rows.append(df)
        except Exception as e:
            warnings.warn(f"Failed ratio {ratio_label}: {repr(e)}")
            err_dir = out_root / ratio_label / args.eval_subdir
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(repr(e) + "\n")

    if all_rows:
        all_df = pd.concat(all_rows, axis=0, ignore_index=True)
        out_path = out_root / "T3_checkpoint_metrics_all_ratios.csv"
        all_df.to_csv(out_path, index=False)

        selected = all_df[all_df["time_key"] == "CORAL_time"].copy()
        selected.to_csv(out_root / "T3_checkpoint_metrics_selected_time_all_ratios.csv", index=False)

        print("\nAll checkpoint re-inference finished.")
        print("Saved:")
        print(out_path)
        print(out_root / "T3_checkpoint_metrics_selected_time_all_ratios.csv")
        print("\nCompact selected-time preview:")
        cols = ["ratio_label", "ckpt_name", "ckpt_phase", "ckpt_epoch", "time_key", "POA", "POAUC", "n_eval", "status"]
        print(selected[[c for c in cols if c in selected.columns]].to_string(index=False))
    else:
        print("No successful checkpoint evaluations.")


if __name__ == "__main__":
    main()
