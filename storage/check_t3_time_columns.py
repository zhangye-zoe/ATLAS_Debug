#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PBMC_STAGE_ORDER = {
    "HSPC": 0,
    "B naive": 1,
    "CD4 Naive": 1,
    "CD8 Naive": 1,
    "Eryth": 1,
    "B intermediate": 2,
    "CD4 TCM": 2,
    "CD8 TCM": 2,
    "CD14 Mono": 2,
    "CD16 Mono": 2,
    "cDC1": 2,
    "cDC2": 2,
    "pDC": 2,
    "CD4 Proliferating": 2,
    "NK Proliferating": 2,
    "B memory": 3,
    "Plasmablast": 3,
    "CD4 TEM": 3,
    "CD8 TEM": 3,
    "Treg": 3,
    "MAIT": 3,
    "NK": 3,
    "dnT": 3,
    "gdT": 3,
}


def interclass_pairwise_accuracy(pred, ref, weighted=True):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)

    valid = np.isfinite(pred) & np.isfinite(ref)
    pred = pred[valid]
    ref = ref[valid]

    correct = 0.0
    total = 0.0
    n = len(pred)

    for i in range(n):
        for j in range(i + 1, n):
            if ref[i] == ref[j]:
                continue

            w = abs(ref[i] - ref[j]) if weighted else 1.0

            if ref[i] < ref[j]:
                is_correct = pred[i] < pred[j]
            else:
                is_correct = pred[j] < pred[i]

            correct += w * float(is_correct)
            total += w

    return correct / total if total > 0 else np.nan


def interclass_pairwise_auc(pred, ref, weighted=True):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)

    valid = np.isfinite(pred) & np.isfinite(ref)
    pred = pred[valid]
    ref = ref[valid]

    classes = np.sort(np.unique(ref))
    aucs = []
    weights = []

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


def per_stage_pair_auc(df, time_key, stage_key="stage"):
    pred = pd.to_numeric(df[time_key], errors="coerce").to_numpy()
    ref = pd.to_numeric(df[stage_key], errors="coerce").to_numpy()

    valid = np.isfinite(pred) & np.isfinite(ref)
    pred = pred[valid]
    ref = ref[valid]

    rows = []
    classes = np.sort(np.unique(ref))

    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            mask = (ref == a) | (ref == b)
            y = (ref[mask] == b).astype(int)
            s = pred[mask]

            if len(np.unique(y)) < 2:
                continue

            auc = roc_auc_score(y, s)
            rows.append({
                "time_key": time_key,
                "stage_pair": f"{int(a)}->{int(b)}",
                "stage_low": int(a),
                "stage_high": int(b),
                "auc": float(auc),
                "auc_flip": float(1.0 - auc),
                "stage_gap": float(abs(b - a)),
                "n_low": int(np.sum(ref == a)),
                "n_high": int(np.sum(ref == b)),
                "mean_low": float(np.mean(pred[ref == a])),
                "mean_high": float(np.mean(pred[ref == b])),
                "median_low": float(np.median(pred[ref == a])),
                "median_high": float(np.median(pred[ref == b])),
            })

    return pd.DataFrame(rows)


def compute_metrics_for_time(df, time_key):
    pred = pd.to_numeric(df[time_key], errors="coerce").to_numpy()
    ref = pd.to_numeric(df["stage"], errors="coerce").to_numpy()

    valid = np.isfinite(pred) & np.isfinite(ref)
    pred = pred[valid]
    ref = ref[valid]

    poa = interclass_pairwise_accuracy(pred, ref, weighted=True)
    poauc = interclass_pairwise_auc(pred, ref, weighted=True)

    pred_flip = 1.0 - pred
    poa_flip = interclass_pairwise_accuracy(pred_flip, ref, weighted=True)
    poauc_flip = interclass_pairwise_auc(pred_flip, ref, weighted=True)

    return {
        "time_key": time_key,
        "n_eval": int(valid.sum()),
        "POA": poa,
        "POAUC": poauc,
        "POA_flip": poa_flip,
        "POAUC_flip": poauc_flip,
        "POAUC_best": max(poauc, poauc_flip),
        "time_mean": float(np.nanmean(pred)),
        "time_std": float(np.nanstd(pred)),
        "time_min": float(np.nanmin(pred)),
        "time_q25": float(np.nanquantile(pred, 0.25)),
        "time_median": float(np.nanmedian(pred)),
        "time_q75": float(np.nanquantile(pred, 0.75)),
        "time_max": float(np.nanmax(pred)),
    }


def inspect_one_ratio(ratio_dir: Path, label_key="cell_type", stage_order=None):
    stage_order = stage_order or PBMC_STAGE_ORDER

    t3_csv = ratio_dir / "T3_metrics_direct.csv"
    t3_json = ratio_dir / "summary_t3.json"
    eval_csv = ratio_dir / "T3_eval_cells.csv"

    print("\n" + "=" * 100)
    print(f"Ratio: {ratio_dir.name}")
    print("=" * 100)

    if t3_csv.exists():
        print("\n[T3_metrics_direct.csv]")
        print(pd.read_csv(t3_csv))
    else:
        print("\n[T3_metrics_direct.csv] NOT FOUND")

    if t3_json.exists():
        print("\n[summary_t3.json]")
        try:
            obj = json.loads(t3_json.read_text())
            print(json.dumps(obj, indent=2, ensure_ascii=False)[:3000])
        except Exception as e:
            print("Failed to read json:", repr(e))
    else:
        print("\n[summary_t3.json] NOT FOUND")

    if not eval_csv.exists():
        print("\n[T3_eval_cells.csv] NOT FOUND")
        return None, None

    df = pd.read_csv(eval_csv)

    if "stage" not in df.columns:
        if label_key not in df.columns:
            raise KeyError(
                f"{eval_csv} has no 'stage' column and no label_key {label_key!r}. "
                f"Available columns: {list(df.columns)}"
            )
        df["stage"] = df[label_key].astype(str).map(stage_order)

    print("\n[Eval cell composition]")
    print("n_cells:", len(df))
    print("valid_stage:", int(df["stage"].notna().sum()))
    if label_key in df.columns:
        print("\nTop cell types:")
        print(df[label_key].value_counts().head(20))
    print("\nStage counts:")
    print(df["stage"].value_counts(dropna=False).sort_index())

    time_keys = [
        k for k in ["CORAL_time", "CORAL_time_head", "CORAL_time_dpt"]
        if k in df.columns
    ]

    print("\nTime columns found:", time_keys)

    metric_rows = []
    pair_rows = []

    for key in time_keys:
        m = compute_metrics_for_time(df, key)
        metric_rows.append(m)

        p = per_stage_pair_auc(df, key)
        if len(p) > 0:
            pair_rows.append(p)

        print(f"\n[Metrics from {key}]")
        print(pd.DataFrame([m]).T)

        print(f"\n[Stage time summary: {key}]")
        print(df.groupby("stage")[key].agg(["count", "mean", "std", "min", "max"]))

        print(f"\n[Per-stage-pair AUC: {key}]")
        print(p)

    metric_df = pd.DataFrame(metric_rows)
    pair_df = pd.concat(pair_rows, axis=0, ignore_index=True) if pair_rows else pd.DataFrame()

    metric_out = ratio_dir / "T3_time_column_check_metrics.csv"
    pair_out = ratio_dir / "T3_time_column_check_pair_auc.csv"

    metric_df.to_csv(metric_out, index=False)
    pair_df.to_csv(pair_out, index=False)

    print("\nSaved:")
    print(metric_out)
    print(pair_out)

    return metric_df, pair_df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Output root, e.g. /data5/zhangye/CORAL/output/PBMC_dynamic_unsup")
    parser.add_argument("--ratios", nargs="*", default=None,
                        help="Optional ratio labels, e.g. single_000 single_020")
    parser.add_argument("--label-key", type=str, default="cell_type")
    args = parser.parse_args()

    root = Path(args.root)

    if args.ratios:
        ratios = args.ratios
    else:
        ratios = sorted([
            p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith("single_")
        ])

    all_metrics = []

    for r in ratios:
        ratio_dir = root / r
        metric_df, pair_df = inspect_one_ratio(
            ratio_dir,
            label_key=args.label_key,
            stage_order=PBMC_STAGE_ORDER,
        )
        if metric_df is not None and len(metric_df) > 0:
            metric_df.insert(0, "ratio", r)
            all_metrics.append(metric_df)

    if all_metrics:
        out = pd.concat(all_metrics, axis=0, ignore_index=True)
        out_path = root / "T3_time_column_check_all_ratios.csv"
        out.to_csv(out_path, index=False)
        print("\n" + "=" * 100)
        print("ALL RATIO SUMMARY")
        print("=" * 100)
        print(out)
        print("\nSaved:", out_path)


if __name__ == "__main__":
    main()