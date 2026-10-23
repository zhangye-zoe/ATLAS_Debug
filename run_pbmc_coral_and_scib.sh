#!/usr/bin/env bash
set -euo pipefail

CORAL_ROOT="${CORAL_ROOT:-/data5/zhangye/CORAL}"
SPLIT_ROOT="${PBMC_SPLIT_ROOT:-/data5/zhangye/scMRDR/input/PBMC/preprocessed_input/results_ratio_loop}"
MANIFEST="${PBMC_MANIFEST:-${CORAL_ROOT}/configs/datasets/pbmc/pbmc_rna_atac_manifest.json}"
STAGE_ORDER="${PBMC_STAGE_ORDER:-${CORAL_ROOT}/configs/datasets/pbmc/pbmc_stage_order.json}"
LABEL_CSV="${PBMC_LABEL_CSV:-/data5/zhangye/scMRDR/input/PBMC/raw_input/pbmc_azimuth_annotations.csv}"
OUT_ROOT="${CORAL_PBMC_OUT_ROOT:-${CORAL_ROOT}/output/PBMC_graphtime_scib_rerun}"

LOOP_SCRIPT="${CORAL_LOOP_SCRIPT:-${CORAL_ROOT}/storage/coral_ratio_loop_T1T2T3_graphtime.py}"
MODEL_SCRIPT="${CORAL_MODEL_SCRIPT:-${CORAL_ROOT}/storage/coral_v1_graphtime.py}"
SCIB_SCRIPT="${CORAL_SCIB_SCRIPT:-${CORAL_ROOT}/run_scib_metrics_full_compatible.py}"

RATIOS=(single_000 single_020 single_040 single_060 single_080 single_100)

for f in "$LOOP_SCRIPT" "$MODEL_SCRIPT" "$MANIFEST" "$STAGE_ORDER" "$LABEL_CSV" "$SCIB_SCRIPT"; do
  [[ -e "$f" ]] || { echo "[MISSING] $f"; exit 1; }
done

for ratio in "${RATIOS[@]}"; do
  d="${SPLIT_ROOT}/${ratio}"
  [[ -d "$d" ]] || { echo "[MISSING] $d"; exit 1; }
  for f in train_rna_ref.h5ad train_atac_full.h5ad train_atac_activity.h5ad \
           val_query_atac.h5ad val_true_rna.h5ad val_atac_activity.h5ad split_info.json; do
    [[ -f "${d}/${f}" ]] || { echo "[MISSING] ${d}/${f}"; exit 1; }
  done
done

mkdir -p "$OUT_ROOT"

echo "Training CORAL on PBMC..."
python "$LOOP_SCRIPT" \
  --coral-module "$MODEL_SCRIPT" \
  --manifest "$MANIFEST" \
  --split-root "$SPLIT_ROOT" \
  --out-root "$OUT_ROOT" \
  --ratio-labels "${RATIOS[@]}" \
  --seed 1234 \
  --device cuda \
  --epochs-phase1 80 \
  --epochs-phase2 60 \
  --epochs-phase3 80 \
  --batch-size 256 \
  --run-phase3 \
  --t3-time-mode graph_root \
  --root-labels HSPC \
  --graph-k 30 \
  --label-csv "$LABEL_CSV" \
  --label-csv-cell-col cell_id \
  --label-csv-label-col predicted.celltype.l2 \
  --label-key cell_type \
  --stage-order-json "$STAGE_ORDER" \
  2>&1 | tee "${OUT_ROOT}/train_pbmc_all_ratios.log"

mkdir -p "${OUT_ROOT}/scib_full"

echo "Computing CORAL scIB metrics..."
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python "$SCIB_SCRIPT" \
  --atlas-root "$OUT_ROOT" \
  --split-root "$SPLIT_ROOT" \
  --ratios "${RATIOS[@]}" \
  --out-root "${OUT_ROOT}/scib_full" \
  --n-jobs 8 \
  2>&1 | tee "${OUT_ROOT}/scib_full/run_scib_pbmc.log"

echo "CORAL long table:"
echo "${OUT_ROOT}/scib_full/scib_all_ratios_long.csv"
