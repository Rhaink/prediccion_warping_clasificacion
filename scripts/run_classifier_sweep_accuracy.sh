#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/classifier_warped_lung_best/sweeps_2026-01-12}"
CONFIG="${CONFIG:-configs/classifier_warped_base.json}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 123 321}"
LRS="${LRS:-5e-5 2e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-50}"
CLASS_WEIGHTS_MODES="${CLASS_WEIGHTS_MODES:-on off}"
FORCE="${FORCE:-false}"

mkdir -p "$OUTPUT_ROOT"

for lr in $LRS; do
  for seed in $SEEDS; do
    for cw_mode in $CLASS_WEIGHTS_MODES; do
      tag="lr${lr}_seed${seed}_${cw_mode}"
      out_dir="${OUTPUT_ROOT}/${tag}"
      log_file="${out_dir}/train.log"

      if [[ -f "${out_dir}/best_classifier.pt" && "${FORCE}" != "true" ]]; then
        echo "Skipping ${tag} (best_classifier.pt exists)."
        continue
      fi

      mkdir -p "$out_dir"

      cw_flag=()
      if [[ "${cw_mode}" == "off" ]]; then
        cw_flag=(--no-class-weights)
      fi

      echo "=== Run ${tag} ==="
      echo "Output: ${out_dir}"
      python -m src_v2 train-classifier \
        --config "$CONFIG" \
        --output-dir "$out_dir" \
        --lr "$lr" \
        --seed "$seed" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        "${cw_flag[@]}" | tee "$log_file"
    done
  done
done
