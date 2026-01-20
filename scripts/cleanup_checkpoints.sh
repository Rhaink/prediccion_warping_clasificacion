#!/usr/bin/env bash
# Script para limpieza de checkpoints
# Libera ~120 GB de espacio preservando modelos críticos

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Configuración
BACKUP_DIR="${BACKUP_DIR:-/tmp/checkpoints_backup_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-true}"

echo "=============================================="
echo "  Checkpoint Cleanup Script - MODERADO"
echo "=============================================="
echo "Working directory: $ROOT"
echo "Backup directory: $BACKUP_DIR"
echo "Dry run: $DRY_RUN"
echo ""
echo "Expected cleanup: ~120 GB"
echo "Preserved models: 6 (4 critical + seed456 + seed789)"
echo "=============================================="
echo ""

# Función para eliminar de forma segura
safe_delete() {
    local path="$1"
    local desc="$2"

    if [[ ! -e "$path" ]]; then
        echo "[$desc] $path - NOT FOUND, skipping"
        return
    fi

    local size=$(du -sh "$path" 2>/dev/null | cut -f1 || echo "unknown")

    echo "[$desc] $path ($size)"

    if [[ "$DRY_RUN" == "false" ]]; then
        rm -rf "$path"
        echo "  ✓ Deleted"
    else
        echo "  [DRY RUN] Would delete"
    fi
}

# Paso 1: Backup de modelos críticos
echo "=== STEP 1: Creating backup of critical models ==="
mkdir -p "$BACKUP_DIR"

CRITICAL_MODELS=(
    "checkpoints/session10/ensemble/seed123/final_model.pt"
    "checkpoints/session13/seed321/final_model.pt"
    "checkpoints/repro_split111/session14/seed111/final_model.pt"
    "checkpoints/repro_split666/session16/seed666/final_model.pt"
    "checkpoints/session10/ensemble/seed456/final_model.pt"  # Best individual
    "checkpoints/session13/seed789/final_model.pt"  # Historical
)

for model in "${CRITICAL_MODELS[@]}"; do
    if [[ -f "$model" ]]; then
        cp "$model" "$BACKUP_DIR/$(basename $(dirname $model))_final.pt"
        echo "  ✓ Backed up: $model"
    else
        echo "  ⚠ WARNING: $model not found!"
    fi
done

echo ""
echo "Backup created at: $BACKUP_DIR"
echo "Creating tarball..."
tar -czf "checkpoints_backup_$(date +%Y%m%d).tar.gz" -C "$BACKUP_DIR" .
BACKUP_SIZE=$(du -sh "checkpoints_backup_$(date +%Y%m%d).tar.gz" | cut -f1)
echo "  ✓ Tarball: checkpoints_backup_$(date +%Y%m%d).tar.gz ($BACKUP_SIZE)"
echo ""

# Paso 2: Checkpoints intermedios
echo "=== STEP 2: Deleting intermediate checkpoints ==="
INTERMEDIATE_COUNT=$(find checkpoints/ -name "checkpoint_epoch*.pt" -type f 2>/dev/null | wc -l)
echo "Found $INTERMEDIATE_COUNT checkpoint files (~43 GB estimated)"

if [[ "$DRY_RUN" == "false" ]]; then
    find checkpoints/ -name "checkpoint_epoch*.pt" -type f -delete
    echo "  ✓ Deleted all intermediate checkpoints"
else
    echo "  [DRY RUN] Would delete $INTERMEDIATE_COUNT files"
fi
echo ""

# Paso 3: repro_quickstart
echo "=== STEP 3: Deleting repro_quickstart ==="
safe_delete "checkpoints/repro_quickstart" "repro_quickstart"
echo ""

# Paso 4: Experimentos no críticos
echo "=== STEP 4: Deleting non-critical experiments ==="
safe_delete "checkpoints/repro_split456" "repro_split456"
safe_delete "checkpoints/repro_split123" "repro_split123"
safe_delete "checkpoints/repro_split222" "repro_split222"
safe_delete "checkpoints/repro_split333" "repro_split333"
safe_delete "checkpoints/repro_split444" "repro_split444"
safe_delete "checkpoints/repro_split555" "repro_split555"
safe_delete "checkpoints/repro_split789" "repro_split789"
safe_delete "checkpoints/repro_split321" "repro_split321"
safe_delete "checkpoints/repro_split456_rerun" "repro_split456_rerun"
safe_delete "checkpoints/repro_tuned" "repro_tuned"
safe_delete "checkpoints/repro_tuned2" "repro_tuned2"
safe_delete "checkpoints/repro_exact" "repro_exact"
safe_delete "checkpoints/repro_exact_longpat" "repro_exact_longpat"
safe_delete "checkpoints/repro" "repro"
echo ""

# Paso 5: Experimentos de ablation
echo "=== STEP 5: Deleting ablation experiments ==="
for exp in exp1_dropout02 exp2_hidden1024 exp3_hidden512 exp4_epochs100 exp5_lr1e5; do
    if [[ -d "checkpoints/session10/$exp" ]]; then
        safe_delete "checkpoints/session10/$exp" "$exp"
    fi
done
echo ""

# Paso 6: Debug runs
echo "=== STEP 6: Deleting debug runs ==="
safe_delete "checkpoints/debug_runs" "debug_runs"
if [[ -f "checkpoints/final_model.pt" ]]; then
    safe_delete "checkpoints/final_model.pt" "orphan final_model.pt"
fi
echo ""

# Resumen
echo "=============================================="
echo "  SUMMARY"
echo "=============================================="
TOTAL_BEFORE="134G"
TOTAL_AFTER=$(du -sh checkpoints/ 2>/dev/null | cut -f1 || echo "unknown")
echo "Before: $TOTAL_BEFORE"
echo "After:  $TOTAL_AFTER"
echo "Backup: $BACKUP_DIR"
echo "Tarball: checkpoints_backup_$(date +%Y%m%d).tar.gz ($BACKUP_SIZE)"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=============================================="
    echo "  THIS WAS A DRY RUN - NO FILES DELETED"
    echo "=============================================="
    echo ""
    echo "To execute cleanup for real:"
    echo "  DRY_RUN=false bash scripts/cleanup_checkpoints.sh"
    echo ""
    echo "To preserve backup in a specific location:"
    echo "  BACKUP_DIR=/path/to/backup DRY_RUN=false bash scripts/cleanup_checkpoints.sh"
else
    echo "=============================================="
    echo "  CLEANUP COMPLETED"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "1. Verify critical models exist:"
    echo "   ls -lh checkpoints/session10/ensemble/seed123/final_model.pt"
    echo "   ls -lh checkpoints/session13/seed321/final_model.pt"
    echo "   ls -lh checkpoints/repro_split111/session14/seed111/final_model.pt"
    echo "   ls -lh checkpoints/repro_split666/session16/seed666/final_model.pt"
    echo ""
    echo "2. Test ensemble (expected: ~3.61 px):"
    echo "   python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json"
fi
