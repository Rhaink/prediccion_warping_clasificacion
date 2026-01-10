#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m src_v2 evaluate-ensemble \
  checkpoints/session10/ensemble/seed123/final_model.pt \
  checkpoints/session10/ensemble/seed456/final_model.pt \
  checkpoints/session13/seed321/final_model.pt \
  checkpoints/repro_split111/session14/seed111/final_model.pt \
  --tta --clahe
