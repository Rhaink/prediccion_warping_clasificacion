#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_seed() {
  local seed="$1"
  local save_dir="$2"
  local output_dir="$3"

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
}

run_seed 321 checkpoints/repro_split321/session13/seed321 outputs/repro_split321/session13/seed321
run_seed 789 checkpoints/repro_split789/session13/seed789 outputs/repro_split789/session13/seed789

for ckpt in \
  checkpoints/repro_split123/session10/ensemble/seed123/final_model.pt \
  checkpoints/repro_split456/session10/ensemble/seed456/final_model.pt \
  checkpoints/repro_split321/session13/seed321/final_model.pt \
  checkpoints/repro_split789/session13/seed789/final_model.pt; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    exit 1
  fi
done

python -m src_v2 evaluate-ensemble \
  checkpoints/repro_split123/session10/ensemble/seed123/final_model.pt \
  checkpoints/repro_split456/session10/ensemble/seed456/final_model.pt \
  checkpoints/repro_split321/session13/seed321/final_model.pt \
  checkpoints/repro_split789/session13/seed789/final_model.pt \
  --tta --clahe
