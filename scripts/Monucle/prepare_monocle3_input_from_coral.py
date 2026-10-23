#!/usr/bin/env python3
"""
Prepare Monocle3 input files from CORAL ratio-loop outputs.

This script mirrors the scMRDR -> Monocle3 preparation logic, but uses CORAL
outputs such as:
  - val_h.npy                         fused latent for validation cells
  - val_shared_latents.npz            optional modality-specific shared latents
  - pred_rna_val.h5ad                 predicted RNA for ATAC query cells
  - true_rna_val.h5ad                 true RNA saved by CORAL, if available
and the original ratio split directory containing:
  - split_info.json
  - val_true_rna.h5ad
  - val_atac_activity.h5ad
  - val_query_atac.h5ad               optional

Outputs per ratio:
  - expr_gene_by_cell.mtx
  - cell_metadata.csv
  - gene_metadata.csv
  - latent_embedding.csv
  - umap_embedding.csv
  - monocle3_input_cells.h5ad
  - split_info_used.json
  - prepare_summary.json

Typical use, one ratio:
python prepare_monocle3_input_from_coral.py \
  --coral-dir /data5/zhangye/CORAL/output/PBMC/single_000 \
  --split-dir /data5/zhangye/scMRDR/input/PBMC/preprocessed_input/results_ratio_loop/single_000 \
  --outdir /data5/zhangye/CORAL/Monocle_prepare/PBMC/single_000 \
  --cell-set val_all \
  --latent-source fused \
  --azimuth-csv /data5/zhangye/scMRDR/input/PBMC/raw_input/pbmc_azimuth_annotations.csv \
  --azimuth-label-col predicted.celltype.l2

Typical use, loop all ratios:
python prepare_monocle3_input_from_coral.py \
  --coral-root /data5/zhangye/CORAL/output/PBMC \
  --split-root /data5/zhangye/scMRDR/input/PBMC/preprocessed_input/results_ratio_loop \
  --out-root /data5/zhangye/CORAL/Monocle_prepare/PBMC \
  --cell-set val_all \
  --latent-source fused
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, io as spio


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def pick_count_layer(adata: ad.AnnData, preferred: Sequence[str] = ("count", "counts", "data")) -> np.ndarray:
    for key in preferred:
        if key in adata.layers:
            return to_dense(adata.layers[key])
    return to_dense(adata.X)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_fixed_cell_ids(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                ids.append(str(x))
    return ids


def unique_preserve_order(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def pick_existing_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def get_common_names(*arrays: Sequence[str]) -> List[str]:
    if len(arrays) == 0:
        return []
    common = set(map(str, arrays[0]))
    for arr in arrays[1:]:
        common &= set(map(str, arr))
    return sorted(common)


# -----------------------------------------------------------------------------
# Label helpers
# -----------------------------------------------------------------------------

def load_azimuth_label_df(csv_path: Optional[str], label_col: str = "predicted.celltype.l2") -> Optional[pd.DataFrame]:
    """Read Azimuth annotation csv and return a label_df indexed by cell_id."""
    if csv_path is None or str(csv_path).strip() == "":
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Azimuth annotation file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "cell_id" not in df.columns:
        raise KeyError("Azimuth csv must contain column 'cell_id'.")

    keep_cols = ["cell_id"]
    pred_cols = [
        "predicted.celltype.l1.score",
        "predicted.celltype.l1",
        "predicted.celltype.l2.score",
        "predicted.celltype.l2",
        "predicted.celltype.l3.score",
        "predicted.celltype.l3",
    ]
    keep_cols += [c for c in pred_cols if c in df.columns]
    df = df[keep_cols].copy()
    df["cell_id"] = df["cell_id"].astype(str)
    df = df.drop_duplicates(subset="cell_id").set_index("cell_id")

    if label_col not in df.columns:
        raise KeyError(
            f"Requested azimuth label column {label_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    df["cell_type"] = df[label_col].astype(str)
    return df


def build_label_df_from_obs(obs_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    obs_df = obs_df.copy()
    if "cell_id" not in obs_df.columns:
        obs_df["cell_id"] = obs_df.index.astype(str)

    label_candidates = [
        "celltype", "cell_type", "annotation", "annot", "labels", "leiden",
        "future_label", "batch", "predicted.celltype.l1", "predicted.celltype.l2", "predicted.celltype.l3",
    ]
    label_cols = pick_existing_columns(obs_df, label_candidates)
    if len(label_cols) == 0:
        return None

    label_df = obs_df[["cell_id"] + label_cols].copy()
    label_df["cell_id"] = label_df["cell_id"].astype(str)
    label_df = label_df.drop_duplicates(subset="cell_id").set_index("cell_id")
    return label_df


def merge_label_dfs(*dfs: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    out: Optional[pd.DataFrame] = None
    for df in dfs:
        if df is None:
            continue
        df = df.copy()
        df.index = df.index.astype(str)
        out = df if out is None else out.combine_first(df)
        for col in df.columns:
            if col not in out.columns:
                out[col] = df[col]
    return out


# -----------------------------------------------------------------------------
# CORAL output readers
# -----------------------------------------------------------------------------

def existing_val_modality_files(split_dir: Path) -> List[Path]:
    """
    Files used to reconstruct the exact sorted validation union in CORAL's
    build_multimodal_ratio_dataset. This should match meta['val_cell_ids'] order.
    """
    candidates = [
        "val_true_rna.h5ad",
        "val_rna.h5ad",
        "val_atac_activity.h5ad",
        "val_atac_full.h5ad",
        "val_query_atac.h5ad",
        "val_protein.h5ad",
        "val_adt.h5ad",
        "val_s.h5ad",
        "val_u.h5ad",
    ]
    return [split_dir / x for x in candidates if (split_dir / x).exists()]


def reconstruct_coral_val_order(split_dir: Path) -> List[str]:
    files = existing_val_modality_files(split_dir)
    if not files:
        split_info = read_json(split_dir / "split_info.json")
        fallback = split_info.get("val_cells", split_info.get("val_query_cells", split_info.get("val_query_atac_cells", [])))
        if not fallback:
            raise ValueError(f"Cannot reconstruct validation cell order for {split_dir}; no val h5ad or split val list found.")
        return sorted(map(str, fallback))

    union = set()
    for p in files:
        a = sc.read_h5ad(str(p), backed="r")
        union |= set(map(str, a.obs_names.tolist()))
        a.file.close() if getattr(a, "isbacked", False) else None
    return sorted(union)


def load_coral_latent(
    coral_dir: Path,
    split_dir: Path,
    latent_source: str = "fused",
    shared_key: Optional[str] = None,
) -> pd.DataFrame:
    """Return latent dataframe indexed by validation cell_id."""
    val_order = reconstruct_coral_val_order(split_dir)

    if latent_source == "fused":
        latent_path = coral_dir / "val_h.npy"
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing fused latent file: {latent_path}")
        latent = np.load(latent_path)
        source_name = "val_h.npy"

    elif latent_source == "shared":
        npz_path = coral_dir / "val_shared_latents.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing shared latent file: {npz_path}")
        npz = np.load(npz_path)
        keys = list(npz.keys())
        if shared_key is None:
            if len(keys) == 1:
                shared_key = keys[0]
            elif "atac" in keys:
                shared_key = "atac"
            elif "rna" in keys:
                shared_key = "rna"
            else:
                raise ValueError(f"Please provide --shared-key. Available keys: {keys}")
        if shared_key not in npz:
            raise KeyError(f"shared_key {shared_key!r} not found in {npz_path}. Available keys: {keys}")
        latent = npz[shared_key]
        source_name = f"val_shared_latents.npz:{shared_key}"

    else:
        raise ValueError(f"Unknown latent_source={latent_source!r}. Use 'fused' or 'shared'.")

    if latent.shape[0] != len(val_order):
        raise ValueError(
            f"Latent row count does not match reconstructed validation order. "
            f"latent={latent.shape[0]}, val_order={len(val_order)}. "
            f"Check whether the CORAL output and split directory match. Source={source_name}"
        )

    latent = np.asarray(latent, dtype=np.float32)
    latent_df = pd.DataFrame(
        latent,
        index=pd.Index(val_order, name="cell_id"),
        columns=[f"latent_{i+1}" for i in range(latent.shape[1])],
    )
    latent_df.attrs["source_name"] = source_name
    return latent_df


def load_true_rna_for_val(split_dir: Path, coral_dir: Optional[Path] = None) -> ad.AnnData:
    """Prefer split val_true_rna; fall back to CORAL true_rna_val."""
    candidates = [split_dir / "val_true_rna.h5ad"]
    if coral_dir is not None:
        candidates.append(coral_dir / "true_rna_val.h5ad")
    for p in candidates:
        if p.exists():
            a = sc.read_h5ad(str(p))
            a.obs["cell_id"] = a.obs_names.astype(str)
            return a
    raise FileNotFoundError(f"Could not find val_true_rna.h5ad in {split_dir} or true_rna_val.h5ad in {coral_dir}")


def load_pred_rna_for_val_atac(coral_dir: Path) -> ad.AnnData:
    p = coral_dir / "pred_rna_val.h5ad"
    if not p.exists():
        raise FileNotFoundError(f"Missing predicted RNA file for val_atac: {p}")
    a = sc.read_h5ad(str(p))
    a.obs["cell_id"] = a.obs_names.astype(str)
    return a


def subset_and_align_by_ids(adata: ad.AnnData, target_ids: Sequence[str], strict: bool, name: str) -> ad.AnnData:
    target_ids = unique_preserve_order(target_ids)
    adata = adata.copy()
    adata.obs["cell_id"] = adata.obs.get("cell_id", adata.obs_names.astype(str)).astype(str)
    adata = adata[~adata.obs["cell_id"].duplicated().to_numpy()].copy()
    adata.obs = adata.obs.set_index("cell_id", drop=False)

    present = set(adata.obs.index.astype(str))
    missing = [x for x in target_ids if x not in present]
    if strict and missing:
        raise ValueError(
            f"{name}: missing {len(missing)} / {len(target_ids)} target cells. First few: {missing[:10]}"
        )
    keep_ids = [x for x in target_ids if x in present]
    if len(keep_ids) == 0:
        raise ValueError(f"{name}: no requested cells are present.")
    return adata[keep_ids].copy()


def build_meta_from_expr_and_latent(
    expr_cells: ad.AnnData,
    latent_df: pd.DataFrame,
    label_df: Optional[pd.DataFrame],
    target_ids: Sequence[str],
    strict: bool,
) -> Tuple[ad.AnnData, pd.DataFrame, np.ndarray, List[str]]:
    target_ids = unique_preserve_order(target_ids)

    expr_cells = subset_and_align_by_ids(expr_cells, target_ids, strict=strict, name="expr_cells")

    latent_ids = set(latent_df.index.astype(str))
    missing_latent = [x for x in target_ids if x not in latent_ids]
    if strict and missing_latent:
        raise ValueError(
            f"latent_df: missing {len(missing_latent)} / {len(target_ids)} target cells. "
            f"First few: {missing_latent[:10]}"
        )

    keep_ids = [x for x in expr_cells.obs_names.astype(str).tolist() if x in latent_ids]
    if len(keep_ids) == 0:
        raise ValueError("No overlap between expression cells and CORAL latent cells.")

    expr_cells = expr_cells[keep_ids].copy()
    obs = expr_cells.obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    obs["cell_id"] = obs.index.astype(str)

    if label_df is not None:
        label_df = label_df.copy()
        label_df.index = label_df.index.astype(str)
        # Add label columns not already present; fill missing values when possible.
        for col in label_df.columns:
            if col not in obs.columns:
                obs[col] = label_df[col]
            else:
                obs[col] = obs[col].where(obs[col].notna(), label_df[col])

    latent = latent_df.loc[keep_ids].to_numpy(dtype=np.float32)
    return expr_cells, obs, latent, keep_ids


# -----------------------------------------------------------------------------
# Core preparation
# -----------------------------------------------------------------------------

def choose_target_ids(split_info: dict, cell_set: str, fixed_cell_ids: Optional[List[str]], strict: bool) -> List[str]:
    if cell_set == "val_all":
        base = split_info.get("val_cells", split_info.get("val_query_cells", split_info.get("val_query_atac_cells", [])))
        field = "val_cells"
    elif cell_set == "val_atac":
        base = split_info.get("val_query_atac_cells", split_info.get("val_query_cells", split_info.get("val_cells", [])))
        field = "val_query_atac_cells"
    else:
        raise ValueError("This CORAL preparation script currently supports --cell-set val_all or val_atac.")

    base = unique_preserve_order(map(str, base))
    if not base:
        raise ValueError(f"split_info has no cells for cell_set={cell_set} field={field}")

    if fixed_cell_ids is None:
        return base

    fixed = unique_preserve_order(map(str, fixed_cell_ids))
    invalid = [x for x in fixed if x not in set(base)]
    if strict and invalid:
        raise ValueError(
            f"--fixed-cell-ids contains cells not in split_info[{field!r}] for this ratio. "
            f"Invalid={len(invalid)}. First few: {invalid[:10]}"
        )
    return [x for x in fixed if x in set(base)]


def prepare_one_ratio(
    coral_dir: Path,
    split_dir: Path,
    outdir: Path,
    cell_set: str = "val_all",
    latent_source: str = "fused",
    shared_key: Optional[str] = None,
    fixed_cell_ids: Optional[List[str]] = None,
    strict_fixed_set: bool = False,
    azimuth_csv: Optional[str] = None,
    azimuth_label_col: str = "predicted.celltype.l2",
    neighbors_k: int = 30,
    seed: int = 1234,
    require_cell_type: bool = True,
) -> dict:
    ensure_dir(outdir)

    split_info_path = split_dir / "split_info.json"
    if not split_info_path.exists():
        raise FileNotFoundError(f"Missing split_info.json: {split_info_path}")
    split_info = read_json(split_info_path)

    latent_df = load_coral_latent(coral_dir, split_dir, latent_source=latent_source, shared_key=shared_key)
    target_ids = choose_target_ids(split_info, cell_set, fixed_cell_ids, strict=strict_fixed_set or fixed_cell_ids is not None)

    # Expression matrix source:
    #   val_all  -> true RNA expression from split val_true_rna.h5ad
    #   val_atac -> CORAL-imputed RNA from pred_rna_val.h5ad
    true_rna = load_true_rna_for_val(split_dir, coral_dir=coral_dir)
    existing_label_df = build_label_df_from_obs(true_rna.obs)
    azimuth_label_df = load_azimuth_label_df(azimuth_csv, label_col=azimuth_label_col)
    label_df = merge_label_dfs(existing_label_df, azimuth_label_df)

    if cell_set == "val_all":
        expr_cells = true_rna.copy()
        expr_source = str(split_dir / "val_true_rna.h5ad")
    elif cell_set == "val_atac":
        expr_cells = load_pred_rna_for_val_atac(coral_dir)
        # If predicted h5ad contains label columns, include them as highest priority labels.
        pred_label_df = build_label_df_from_obs(expr_cells.obs)
        label_df = merge_label_dfs(pred_label_df, existing_label_df, azimuth_label_df)
        expr_source = str(coral_dir / "pred_rna_val.h5ad")
    else:
        raise ValueError(f"Unsupported cell_set={cell_set!r}")

    expr_cells, obs, latent, keep_ids = build_meta_from_expr_and_latent(
        expr_cells=expr_cells,
        latent_df=latent_df,
        label_df=label_df,
        target_ids=target_ids,
        strict=strict_fixed_set or fixed_cell_ids is not None,
    )

    obs["monocle_input_source"] = cell_set
    obs["coral_ratio_dir"] = str(coral_dir)
    obs["split_ratio_dir"] = str(split_dir)
    obs["latent_source"] = latent_df.attrs.get("source_name", latent_source)
    obs["expr_source"] = expr_source

    if require_cell_type and "cell_type" not in obs.columns:
        raise ValueError(
            "No 'cell_type' column found after merging annotations. "
            "Provide --azimuth-csv or set --no-require-cell-type."
        )

    var = expr_cells.var.copy()
    var.index = var.index.astype(str)
    var.index.name = "gene_short_name"
    if "gene_short_name" not in var.columns:
        var["gene_short_name"] = var.index.astype(str)

    expr = pick_count_layer(expr_cells)
    expr = np.asarray(expr, dtype=np.float32)

    if expr.shape[0] != obs.shape[0]:
        raise ValueError(f"Expression cells ({expr.shape[0]}) and metadata cells ({obs.shape[0]}) do not match.")
    if expr.shape[1] != var.shape[0]:
        raise ValueError(f"Expression genes ({expr.shape[1]}) and gene metadata ({var.shape[0]}) do not match.")
    if latent.shape[0] != obs.shape[0]:
        raise ValueError(f"Latent cells ({latent.shape[0]}) and metadata cells ({obs.shape[0]}) do not match.")

    # Compute UMAP from CORAL latent, exactly as scMRDR preparation does for latent_embedding.
    emb_adata = ad.AnnData(
        X=np.zeros((obs.shape[0], 1), dtype=np.float32),
        obs=obs.copy(),
    )
    emb_adata.obsm["X_latent"] = latent
    sc.pp.neighbors(
        emb_adata,
        use_rep="X_latent",
        n_neighbors=min(neighbors_k, max(2, obs.shape[0] - 1)),
    )
    sc.tl.umap(emb_adata, random_state=seed)
    umap = np.asarray(emb_adata.obsm["X_umap"], dtype=np.float32)

    # Save Monocle3-compatible files.
    expr_gene_by_cell = sparse.csr_matrix(expr.T)
    spio.mmwrite(str(outdir / "expr_gene_by_cell.mtx"), expr_gene_by_cell)
    obs.to_csv(outdir / "cell_metadata.csv")
    var.to_csv(outdir / "gene_metadata.csv")

    pd.DataFrame(
        latent,
        index=obs.index,
        columns=[f"latent_{i+1}" for i in range(latent.shape[1])],
    ).to_csv(outdir / "latent_embedding.csv")

    pd.DataFrame(
        umap,
        index=obs.index,
        columns=["UMAP_1", "UMAP_2"],
    ).to_csv(outdir / "umap_embedding.csv")

    out_adata = ad.AnnData(X=expr, obs=obs.copy(), var=var.copy())
    out_adata.obsm["X_latent"] = latent
    out_adata.obsm["X_umap"] = umap
    out_adata.write_h5ad(str(outdir / "monocle3_input_cells.h5ad"))

    write_json(split_info, outdir / "split_info_used.json")

    summary = {
        "ratio": coral_dir.name,
        "coral_dir": str(coral_dir),
        "split_dir": str(split_dir),
        "outdir": str(outdir),
        "cell_set": cell_set,
        "target_ids": len(target_ids),
        "kept_cells": len(keep_ids),
        "n_genes": int(expr.shape[1]),
        "latent_dim": int(latent.shape[1]),
        "expr_source": expr_source,
        "latent_source": latent_df.attrs.get("source_name", latent_source),
        "has_cell_type": "cell_type" in obs.columns,
        "cell_type_counts": obs["cell_type"].value_counts().to_dict() if "cell_type" in obs.columns else {},
    }
    write_json(summary, outdir / "prepare_summary.json")

    print("\nSaved Monocle3 input to:", outdir)
    print("Final number of cells:", obs.shape[0])
    print("Final number of genes:", var.shape[0])
    print("Latent dim:", latent.shape[1])
    print("Expression source:", expr_source)
    print("Latent source:", summary["latent_source"])
    if "cell_type" in obs.columns:
        print("Top cell types:")
        print(obs["cell_type"].value_counts().head(20))

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def resolve_ratio_labels(root: Path, requested: Optional[Sequence[str]]) -> List[str]:
    if requested:
        return list(requested)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("single_")])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare Monocle3 input files from CORAL ratio-loop outputs.")

    # One-ratio mode
    p.add_argument("--coral-dir", type=str, default=None, help="One CORAL ratio output directory, e.g. .../single_000")
    p.add_argument("--split-dir", type=str, default=None, help="Matching split ratio directory, e.g. .../results_ratio_loop/single_000")
    p.add_argument("--outdir", type=str, default=None, help="Output directory for one ratio")

    # Loop mode
    p.add_argument("--coral-root", type=str, default=None, help="CORAL output root containing single_XXX dirs")
    p.add_argument("--split-root", type=str, default=None, help="Split root containing single_XXX dirs")
    p.add_argument("--out-root", type=str, default=None, help="Output root for prepared Monocle3 inputs")
    p.add_argument("--ratio-labels", nargs="*", default=None, help="Optional ratio labels for loop mode")

    p.add_argument("--cell-set", type=str, default="val_all", choices=["val_all", "val_atac"],
                   help="val_all uses true RNA for all validation cells; val_atac uses imputed RNA for ATAC query cells.")
    p.add_argument("--latent-source", type=str, default="fused", choices=["fused", "shared"],
                   help="fused uses val_h.npy; shared uses val_shared_latents.npz[--shared-key].")
    p.add_argument("--shared-key", type=str, default=None, help="Modality key if --latent-source shared, e.g. atac or rna")
    p.add_argument("--neighbors-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--fixed-cell-ids", type=str, default=None,
                   help="Optional one-cell-id-per-line file. Forces the same target cells across ratios.")
    p.add_argument("--strict-fixed-set", action="store_true",
                   help="Raise error if requested fixed cells are missing from the selected ratio/cell-set.")

    p.add_argument("--azimuth-csv", type=str, default=None,
                   help="Optional Azimuth annotation CSV with cell_id and predicted labels.")
    p.add_argument("--azimuth-label-col", type=str, default="predicted.celltype.l2")
    p.add_argument("--no-require-cell-type", action="store_true",
                   help="Do not fail if cell_type is unavailable. Monocle3 can run, but T3 label metrics may fail.")

    return p


def main() -> None:
    args = build_parser().parse_args()
    fixed_ids = read_fixed_cell_ids(args.fixed_cell_ids)

    summaries: List[dict] = []

    # One-ratio mode
    if args.coral_dir is not None:
        if args.split_dir is None or args.outdir is None:
            raise ValueError("One-ratio mode requires --coral-dir, --split-dir, and --outdir.")
        summary = prepare_one_ratio(
            coral_dir=Path(args.coral_dir),
            split_dir=Path(args.split_dir),
            outdir=Path(args.outdir),
            cell_set=args.cell_set,
            latent_source=args.latent_source,
            shared_key=args.shared_key,
            fixed_cell_ids=fixed_ids,
            strict_fixed_set=args.strict_fixed_set,
            azimuth_csv=args.azimuth_csv,
            azimuth_label_col=args.azimuth_label_col,
            neighbors_k=args.neighbors_k,
            seed=args.seed,
            require_cell_type=not args.no_require_cell_type,
        )
        summaries.append(summary)

    # Loop mode
    else:
        if args.coral_root is None or args.split_root is None or args.out_root is None:
            raise ValueError("Loop mode requires --coral-root, --split-root, and --out-root.")

        coral_root = Path(args.coral_root)
        split_root = Path(args.split_root)
        out_root = Path(args.out_root)
        ensure_dir(out_root)

        ratio_labels = resolve_ratio_labels(coral_root, args.ratio_labels)
        if not ratio_labels:
            raise ValueError(f"No ratio folders found under {coral_root}")

        for ratio in ratio_labels:
            coral_dir = coral_root / ratio
            split_dir = split_root / ratio
            outdir = out_root / ratio
            print("\n" + "=" * 100)
            print(f"Preparing ratio: {ratio}")
            print("CORAL dir:", coral_dir)
            print("Split dir:", split_dir)
            print("Out dir:", outdir)
            print("=" * 100)

            try:
                summary = prepare_one_ratio(
                    coral_dir=coral_dir,
                    split_dir=split_dir,
                    outdir=outdir,
                    cell_set=args.cell_set,
                    latent_source=args.latent_source,
                    shared_key=args.shared_key,
                    fixed_cell_ids=fixed_ids,
                    strict_fixed_set=args.strict_fixed_set,
                    azimuth_csv=args.azimuth_csv,
                    azimuth_label_col=args.azimuth_label_col,
                    neighbors_k=args.neighbors_k,
                    seed=args.seed,
                    require_cell_type=not args.no_require_cell_type,
                )
                summaries.append(summary)
            except Exception as e:
                print(f"[FAILED] {ratio}: {repr(e)}")
                summaries.append({"ratio": ratio, "status": "failed", "error": repr(e)})

        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(out_root / "prepare_monocle3_from_coral_summary.csv", index=False)
        write_json(summaries, out_root / "prepare_monocle3_from_coral_summary.json")
        print("\nSaved loop summary to:", out_root / "prepare_monocle3_from_coral_summary.csv")


if __name__ == "__main__":
    main()
