#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_seed() {
  local seed="$1"
  local save_dir="$2"
  local output_dir="$3"

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
}

run_seed 111 checkpoints/repro_split111/session14/seed111 outputs/repro_split111/session14/seed111
run_seed 222 checkpoints/repro_split222/session14/seed222 outputs/repro_split222/session14/seed222

python scripts/sweep_ensemble_combos.py --tta --clahe \
  --out outputs/ensemble_combo_sweep_111_222.txt \
  checkpoints/session10/ensemble/seed123/final_model.pt \
  checkpoints/session10/ensemble/seed456/final_model.pt \
  checkpoints/session13/seed321/final_model.pt \
  checkpoints/session13/seed789/final_model.pt \
  checkpoints/repro_split111/session14/seed111/final_model.pt \
  checkpoints/repro_split222/session14/seed222/final_model.pt
