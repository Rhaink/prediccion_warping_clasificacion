#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SEED [SEED...]" >&2
  exit 1
fi

SESSION="${SESSION:-session15}"
SEED_TAG=$(printf '%s_' "$@")
SEED_TAG=${SEED_TAG%_}

append_if_exists() {
  local path="$1"
  local -n arr_ref="$2"
  if [[ -f "$path" ]]; then
    arr_ref+=("$path")
  fi
}

BASE_MODELS=(
  checkpoints/session10/ensemble/seed123/final_model.pt
  checkpoints/session10/ensemble/seed456/final_model.pt
  checkpoints/session13/seed321/final_model.pt
  checkpoints/session13/seed789/final_model.pt
)

EXISTING_MODELS=()
append_if_exists checkpoints/repro_split111/session14/seed111/final_model.pt EXISTING_MODELS
append_if_exists checkpoints/repro_split222/session14/seed222/final_model.pt EXISTING_MODELS

TRAINED_MODELS=()
for seed in "$@"; do
  save_dir="checkpoints/repro_split${seed}/${SESSION}/seed${seed}"
  output_dir="outputs/repro_split${seed}/${SESSION}/seed${seed}"

  if [[ -n "${TRAIN_CONFIG:-}" ]]; then
    python scripts/train.py --config "$TRAIN_CONFIG" \
      --seed "$seed" --split-seed "$seed" \
      --save-dir "$save_dir" \
      --output-dir "$output_dir"
  else
    python scripts/train.py --seed "$seed" --split-seed "$seed" \
      --save-dir "$save_dir" \
      --output-dir "$output_dir" \
      --batch-size 16 \
      --phase1-epochs 15 --phase2-epochs 100 \
      --phase1-lr 1e-3 --phase2-backbone-lr 2e-5 --phase2-head-lr 2e-4 \
      --phase1-patience 5 --phase2-patience 15 \
      --coord-attention --deep-head --hidden-dim 768 --dropout 0.3 \
      --clahe --clahe-clip 2.0 --clahe-tile 4 \
      --loss wing --tta
  fi

  TRAINED_MODELS+=("$save_dir/final_model.pt")
done

SWEEP_MODELS=("${BASE_MODELS[@]}" "${EXISTING_MODELS[@]}" "${TRAINED_MODELS[@]}")
python scripts/sweep_ensemble_combos.py --tta --clahe \
  --out "outputs/ensemble_combo_sweep_${SEED_TAG}.txt" \
  "${SWEEP_MODELS[@]}"

OPTION2_MODELS=("${SWEEP_MODELS[@]}")
append_if_exists checkpoints/repro_split456_rerun/session10/ensemble/seed456/final_model.pt OPTION2_MODELS
append_if_exists checkpoints/repro_split321/session13/seed321/final_model.pt OPTION2_MODELS
append_if_exists checkpoints/repro_split789/session13/seed789/final_model.pt OPTION2_MODELS

python scripts/sweep_ensemble_combos.py --tta --clahe \
  --out "outputs/ensemble_combo_sweep_option2_${SEED_TAG}.txt" \
  "${OPTION2_MODELS[@]}"
