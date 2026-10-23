#!/usr/bin/env python3
"""
Run a comprehensive scib-metrics evaluation on modality-specific integrated embeddings.

Designed for the ATLAS/CORAL BMMC RNA-ATAC outputs used by the UMAP notebooks.
It builds one observation per observed modality-cell pair (RNA::cell_id or ATAC::cell_id),
uses modality as the batch key, and cell_type as the biological label.

Baseline methods can be added with a JSON configuration. Supported baseline formats:
  1) combined_h5ad: one h5ad containing both RNA and ATAC observations.
  2) two_h5ad: separate RNA and ATAC h5ad files.

Example:
python run_scib_metrics_full.py \
  --atlas-root /data5/zhangye/CORAL/output/BMMC_graphtime \
  --split-root /data5/zhangye/scMRDR/input/BMMC/preprocessed_input/RNA_ATAC/results_ratio_loop_rna_atac \
  --ratios single_000 single_020 single_040 single_060 single_080 single_100 \
  --baseline-config /data5/zhangye/CORAL/scib_baselines_bmmc.json \
  --out-root /data5/zhangye/CORAL/output/BMMC_graphtime/scib_full \
  --n-jobs 8
"""

from __future__ import annotations

import argparse
import inspect
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from scib_metrics import silhouette_batch
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation


def find_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def to_numpy(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def infer_modality_keys(keys: Iterable[str]) -> Tuple[str, str]:
    rna_key = None
    atac_key = None
    keys = list(keys)
    for key in keys:
        low = key.lower()
        if rna_key is None and ("rna" in low or low in {"s", "spliced", "gex"}):
            rna_key = key
        if atac_key is None and ("atac" in low or "chrom" in low or "activity" in low):
            atac_key = key
    if rna_key is None or atac_key is None:
        raise ValueError(
            f"Cannot infer RNA/ATAC keys from {keys}. "
            "Set explicit keys in the code or rename the NPZ arrays."
        )
    return rna_key, atac_key


def canonical_id(modality: str, cell_id: str) -> str:
    return f"{str(modality).upper()}::{str(cell_id)}"


def load_atlas_ratio(atlas_dir: Path, split_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load the exact modality-specific shared embeddings used by the UMAP workflow."""
    t3_csv = atlas_dir / "T3_eval_cells.csv"
    eval_h5ad = find_existing(
        [atlas_dir / "CORAL_eval_embeddings_T3.h5ad", atlas_dir / "CORAL_eval_embeddings.h5ad"]
    )

    if t3_csv.exists():
        obs_df = pd.read_csv(t3_csv)
        cell_col = "cell_id" if "cell_id" in obs_df.columns else obs_df.columns[0]
        obs_df["cell_id"] = obs_df[cell_col].astype(str)
    elif eval_h5ad is not None:
        tmp = sc.read_h5ad(eval_h5ad)
        obs_df = tmp.obs.copy()
        obs_df["cell_id"] = tmp.obs_names.astype(str)
    else:
        raise FileNotFoundError(
            f"Missing T3_eval_cells.csv and CORAL_eval_embeddings*.h5ad under {atlas_dir}"
        )

    if "cell_type" not in obs_df.columns:
        raise KeyError(f"No cell_type column found in metadata under {atlas_dir}")

    union_cells = obs_df["cell_id"].astype(str).to_numpy()
    cell_type_map = obs_df.drop_duplicates("cell_id").set_index("cell_id")["cell_type"].astype(str)

    latent_path = find_existing(
        [
            atlas_dir / "val_shared_latents.npz",
            atlas_dir / "shared_latents_val.npz",
            atlas_dir / "shared_latents.npz",
        ]
    )
    if latent_path is None:
        raise FileNotFoundError(f"No shared latent NPZ found under {atlas_dir}")

    shared = np.load(latent_path, allow_pickle=False)
    rna_key, atac_key = infer_modality_keys(shared.files)
    z_rna = np.asarray(shared[rna_key], dtype=np.float32)
    z_atac = np.asarray(shared[atac_key], dtype=np.float32)

    if z_rna.shape[0] != len(union_cells) or z_atac.shape[0] != len(union_cells):
        raise ValueError(
            f"Latent rows do not match metadata rows in {atlas_dir}: "
            f"RNA={z_rna.shape[0]}, ATAC={z_atac.shape[0]}, metadata={len(union_cells)}"
        )

    rna_path = split_dir / "val_true_rna.h5ad"
    atac_path = find_existing(
        [split_dir / "val_query_atac.h5ad", split_dir / "val_atac_activity.h5ad"]
    )
    if not rna_path.exists() or atac_path is None:
        raise FileNotFoundError(f"Missing validation RNA/ATAC h5ad files under {split_dir}")

    rna_ids = pd.Index(sc.read_h5ad(rna_path, backed="r").obs_names.astype(str))
    atac_ids = pd.Index(sc.read_h5ad(atac_path, backed="r").obs_names.astype(str))
    union_index = pd.Index(union_cells)
    rna_mask = union_index.isin(rna_ids)
    atac_mask = union_index.isin(atac_ids)

    parts = []
    embeddings = []
    for modality, mask, z in [("RNA", rna_mask, z_rna), ("ATAC", atac_mask, z_atac)]:
        ids = union_index[mask].astype(str)
        frame = pd.DataFrame(
            {
                "cell_id": ids,
                "modality": modality,
                "cell_type": [cell_type_map.get(x, "Unknown") for x in ids],
            }
        )
        frame.index = [canonical_id(modality, x) for x in ids]
        parts.append(frame)
        embeddings.append(z[mask])

    obs = pd.concat(parts, axis=0)
    X = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    if not obs.index.is_unique:
        raise ValueError("ATLAS canonical observation IDs are not unique")
    return obs, X


def _read_embedding(adata: ad.AnnData, key: Optional[str]) -> np.ndarray:
    if key in (None, "X", ".X"):
        return to_numpy(adata.X).astype(np.float32, copy=False)
    if key not in adata.obsm:
        raise KeyError(f"Embedding key {key!r} not found. Available obsm keys: {list(adata.obsm.keys())}")
    return np.asarray(adata.obsm[key], dtype=np.float32)


def load_combined_h5ad(spec: Dict[str, Any], ratio: str) -> Tuple[pd.DataFrame, np.ndarray]:
    path = Path(spec["path"].format(ratio=ratio))
    if not path.exists():
        raise FileNotFoundError(path)
    a = sc.read_h5ad(path)
    emb = _read_embedding(a, spec.get("embedding_key", "X"))
    cell_key = spec.get("cell_id_key")
    modality_key = spec.get("modality_key", "modality")
    if modality_key not in a.obs.columns:
        raise KeyError(f"{path}: missing obs[{modality_key!r}]")
    cell_ids = a.obs[cell_key].astype(str).to_numpy() if cell_key else a.obs_names.astype(str)
    modalities = a.obs[modality_key].astype(str).str.upper().to_numpy()
    obs = pd.DataFrame({"cell_id": cell_ids, "modality": modalities})
    obs.index = [canonical_id(m, c) for m, c in zip(modalities, cell_ids)]
    return obs, emb


def load_two_h5ad(spec: Dict[str, Any], ratio: str) -> Tuple[pd.DataFrame, np.ndarray]:
    frames = []
    arrays = []
    common_key = spec.get("embedding_key", "X")
    for modality, path_key, key_override in [
        ("RNA", "rna_path", "rna_embedding_key"),
        ("ATAC", "atac_path", "atac_embedding_key"),
    ]:
        path = Path(spec[path_key].format(ratio=ratio))
        if not path.exists():
            raise FileNotFoundError(path)
        a = sc.read_h5ad(path)
        emb = _read_embedding(a, spec.get(key_override, common_key))
        cell_key = spec.get("cell_id_key")
        cell_ids = a.obs[cell_key].astype(str).to_numpy() if cell_key else a.obs_names.astype(str)
        frame = pd.DataFrame({"cell_id": cell_ids, "modality": modality})
        frame.index = [canonical_id(modality, c) for c in cell_ids]
        frames.append(frame)
        arrays.append(emb)
    return pd.concat(frames, axis=0), np.concatenate(arrays, axis=0).astype(np.float32)


def load_baseline(spec: Dict[str, Any], ratio: str) -> Tuple[pd.DataFrame, np.ndarray]:
    fmt = spec.get("format", "combined_h5ad")
    if fmt == "combined_h5ad":
        return load_combined_h5ad(spec, ratio)
    if fmt == "two_h5ad":
        return load_two_h5ad(spec, ratio)
    raise ValueError(f"Unsupported baseline format: {fmt}")


def align_embeddings(
    canonical_obs: pd.DataFrame,
    method_data: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """Use the exact same observations for every method."""
    common = set(canonical_obs.index)
    for obs, _ in method_data.values():
        common &= set(obs.index)
    ordered_ids = [x for x in canonical_obs.index if x in common]
    if not ordered_ids:
        raise ValueError("No common modality-cell observations across methods")

    aligned = {}
    audit_rows = []
    for method, (obs, X) in method_data.items():
        if not obs.index.is_unique:
            raise ValueError(f"{method}: duplicated canonical observation IDs")
        pos = pd.Series(np.arange(len(obs)), index=obs.index)
        idx = pos.loc[ordered_ids].to_numpy(dtype=int)
        aligned[method] = np.asarray(X[idx], dtype=np.float32)
        audit_rows.append(
            {
                "method": method,
                "available_observations": len(obs),
                "evaluated_observations": len(ordered_ids),
                "dropped_observations": len(obs) - len(ordered_ids),
                "embedding_dim": X.shape[1],
            }
        )
    return canonical_obs.loc[ordered_ids].copy(), aligned, pd.DataFrame(audit_rows)


def make_batch_metrics(enable_pcr: bool) -> BatchCorrection:
    kwargs = {
        "bras": True,
        "ilisi_knn": True,
        "kbet_per_label": True,
        "graph_connectivity": True,
        "pcr_comparison": enable_pcr,
    }
    # Newer development versions expose SBEE; released versions may not.
    if "sbee" in inspect.signature(BatchCorrection).parameters:
        kwargs["sbee"] = True
    return BatchCorrection(**kwargs)


def run_one_ratio(
    ratio: str,
    atlas_root: Path,
    split_root: Path,
    baseline_specs: Dict[str, Dict[str, Any]],
    out_root: Path,
    n_jobs: int,
    enable_pcr: bool,
) -> pd.DataFrame:
    ratio_out = out_root / ratio
    ratio_out.mkdir(parents=True, exist_ok=True)

    atlas_obs, atlas_X = load_atlas_ratio(atlas_root / ratio, split_root / ratio)
    method_data: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {"ATLAS": (atlas_obs, atlas_X)}

    for method, spec in baseline_specs.items():
        try:
            method_data[method] = load_baseline(spec, ratio)
        except FileNotFoundError as exc:
            warnings.warn(f"Skipping {method} for {ratio}; missing file: {exc}")
        except Exception as exc:
            warnings.warn(f"Skipping {method} for {ratio}; failed to load: {exc!r}")

    obs, embeddings, audit = align_embeddings(atlas_obs, method_data)
    audit.insert(0, "ratio", ratio)
    audit.to_csv(ratio_out / "alignment_audit.csv", index=False)

    obs["modality"] = obs["modality"].astype("category")
    obs["cell_type"] = obs["cell_type"].astype("category")
    obs["paired_percent"] = 100 - int(ratio.split("_")[-1])
    obs.to_csv(ratio_out / "evaluated_observations.csv")

    first = next(iter(embeddings.values()))
    # X is not used for reported metrics when PCR is disabled. Supplying a valid matrix
    # prevents prepare() from failing in versions that always initialize a PCA object.
    a = ad.AnnData(X=first.copy(), obs=obs.copy())
    for method, X in embeddings.items():
        a.obsm[method] = X

    if enable_pcr:
        raise NotImplementedError(
            "PCR requires a defensible common pre-integration representation. "
            "For RNA vs ATAC activity, build a PCA from the common normalized gene space first, "
            "store it as adata.obsm['X_pre'], and pass it to Benchmarker."
        )

    bio = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_kmeans=True,
        nmi_ari_cluster_labels_leiden=True,
        silhouette_label=True,
        clisi_knn=True,
    )
    batch = make_batch_metrics(enable_pcr=False)

    # Build Benchmarker arguments in a version-compatible way.
    # The `solver` argument was added in newer scib-metrics releases,
    # so only pass it when the installed version supports it.
    bm_kwargs = {
        "adata": a,
        "batch_key": "modality",
        "label_key": "cell_type",
        "embedding_obsm_keys": list(embeddings.keys()),
        "bio_conservation_metrics": bio,
        "batch_correction_metrics": batch,
        "pre_integrated_embedding_obsm_key": None,
        "n_jobs": n_jobs,
        "progress_bar": True,
    }
    if "solver" in inspect.signature(Benchmarker).parameters:
        bm_kwargs["solver"] = "arpack"

    bm = Benchmarker(**bm_kwargs)
    if hasattr(bm, "prepare"):
        bm.prepare()
    bm.benchmark()
    results = bm.get_results(min_max_scale=False, clean_names=True)

    # Add traditional batch ASW explicitly. Recent scib-metrics uses BRAS by default.
    label_codes = obs["cell_type"].cat.codes.to_numpy(dtype=np.int32)
    batch_codes = obs["modality"].cat.codes.to_numpy(dtype=np.int32)
    for method, X in embeddings.items():
        results.loc[method, "Batch ASW"] = float(
            silhouette_batch(
                X=X,
                labels=label_codes,
                batch=batch_codes,
                rescale=True,
                chunk_size=256,
                metric="euclidean",
                between_cluster_distances="nearest",
            )
        )
    results.loc["Metric Type", "Batch ASW"] = "Batch correction"

    results.to_csv(ratio_out / "scib_results_native.csv")

    metric_types = results.loc["Metric Type"].copy()
    numeric = results.drop(index="Metric Type").apply(pd.to_numeric, errors="coerce")
    wide = numeric.T
    wide.insert(0, "Category", metric_types.reindex(wide.index))
    wide.index.name = "Metric"
    wide.to_csv(ratio_out / "scib_results_wide.csv", float_format="%.8f")

    long = numeric.reset_index(names="Method").melt(
        id_vars="Method", var_name="Metric", value_name="Score"
    )
    long.insert(0, "Ratio", ratio)
    long["PairedPercent"] = 100 - int(ratio.split("_")[-1])
    long["Category"] = long["Metric"].map(metric_types)
    long.to_csv(ratio_out / "scib_results_long.csv", index=False, float_format="%.8f")
    return long


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", default=["single_000", "single_020", "single_040", "single_060", "single_080", "single_100"])
    parser.add_argument("--baseline-config", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--enable-pcr", action="store_true")
    args = parser.parse_args()

    baseline_specs: Dict[str, Dict[str, Any]] = {}
    if args.baseline_config is not None:
        with open(args.baseline_config, "r", encoding="utf-8") as f:
            baseline_specs = json.load(f)

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_long = []
    for ratio in args.ratios:
        print(f"\n{'=' * 80}\nEvaluating {ratio}\n{'=' * 80}")
        all_long.append(
            run_one_ratio(
                ratio=ratio,
                atlas_root=args.atlas_root,
                split_root=args.split_root,
                baseline_specs=baseline_specs,
                out_root=args.out_root,
                n_jobs=args.n_jobs,
                enable_pcr=args.enable_pcr,
            )
        )

    combined = pd.concat(all_long, ignore_index=True)
    combined.to_csv(args.out_root / "scib_all_ratios_long.csv", index=False, float_format="%.8f")

    summary = (
        combined.groupby(["Method", "Metric", "Category"], dropna=False)["Score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(args.out_root / "scib_all_ratios_mean_std.csv", index=False, float_format="%.8f")

    print("\nSaved results under:", args.out_root)
    print("Main summary:", args.out_root / "scib_all_ratios_mean_std.csv")


if __name__ == "__main__":
    main()
