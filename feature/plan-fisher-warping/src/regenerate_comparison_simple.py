#!/usr/bin/env python3
"""
Script para regenerar solo la imagen comparativa simplificada.
Usa los datos ya calculados en summary.json de Phase 7.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Agregar directorio src al path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Directorios
BASE_DIR = Path(__file__).parent.parent
PHASE6_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase6_classification"
PHASE7_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase7_comparison"
OUTPUT_FIGURES_DIR = BASE_DIR / "results" / "figures" / "phase7_comparison"


def load_phase7_summary():
    """Carga resumen de Phase 7."""
    summary_path = PHASE7_METRICS_DIR / "summary.json"
    with open(summary_path, 'r') as f:
        return json.load(f)


def load_phase6_summary():
    """Carga resumen de Phase 6."""
    summary_path = PHASE6_METRICS_DIR / "summary.json"
    with open(summary_path, 'r') as f:
        return json.load(f)


def generate_comparison_simple():
    """
    Genera figura comparativa simplificada (sin 2C Comparable).
    Solo incluye: 2C (12K imgs) y 3C (6K imgs).
    """
    # Cargar datos
    phase6_data = load_phase6_summary()
    phase7_data = load_phase7_summary()

    results_2c_original = phase6_data.get("datasets", {})
    results_3c = phase7_data.get("experiments_3class", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Comparacion: 2 Clases vs 3 Clases | Warped vs Original",
                 fontsize=14, fontweight='bold')

    # Recolectar datos para graficar
    conditions = []
    accuracies = []
    colors = []

    # 2C Original (12,402 imgs)
    if "full_original" in results_2c_original:
        conditions.append("2C-12K\nOriginal")
        accuracies.append(results_2c_original["full_original"].get("test_metrics", {}).get("accuracy", 0))
        colors.append('#3498DB')

    if "full_warped" in results_2c_original:
        conditions.append("2C-12K\nWarped")
        accuracies.append(results_2c_original["full_warped"].get("test_metrics", {}).get("accuracy", 0))
        colors.append('#2ECC71')

    # 3C (6,725 imgs)
    for name in sorted(results_3c.keys()):
        result = results_3c[name]
        if "original" in name:
            conditions.append("3C-6K\nOriginal")
            colors.append('#E74C3C')
        else:
            conditions.append("3C-6K\nWarped")
            colors.append('#F39C12')
        accuracies.append(result["test_accuracy"])

    # Panel izquierdo: Accuracy
    ax_acc = axes[0]
    bars = ax_acc.bar(range(len(conditions)), accuracies, color=colors, alpha=0.8, edgecolor='black')

    for bar, val in zip(bars, accuracies):
        ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_acc.set_xticks(range(len(conditions)))
    ax_acc.set_xticklabels(conditions, fontsize=9)
    ax_acc.set_ylabel('Accuracy', fontsize=11)
    ax_acc.set_title('Accuracy por Experimento', fontsize=12)
    ax_acc.set_ylim(0, 1.0)
    ax_acc.grid(True, alpha=0.3, axis='y')

    # Panel derecho: Mejora por warping
    ax_improve = axes[1]

    improvements = []
    labels = []
    imp_colors = []

    # 2C Original (12K)
    if "full_original" in results_2c_original and "full_warped" in results_2c_original:
        acc_o = results_2c_original["full_original"].get("test_metrics", {}).get("accuracy", 0)
        acc_w = results_2c_original["full_warped"].get("test_metrics", {}).get("accuracy", 0)
        improvements.append((acc_w - acc_o) * 100)
        labels.append("2C (12K imgs)")
        imp_colors.append('#2ECC71')

    # 3C (6K)
    orig_3c = results_3c.get("3class_original")
    warp_3c = results_3c.get("3class_warped")
    if orig_3c and warp_3c:
        acc_o = orig_3c["test_accuracy"]
        acc_w = warp_3c["test_accuracy"]
        improvements.append((acc_w - acc_o) * 100)
        labels.append("3C (6K imgs)")
        imp_colors.append('#F39C12')

    bars = ax_improve.bar(labels, improvements, color=imp_colors, alpha=0.8, edgecolor='black')

    for bar, val in zip(bars, improvements):
        sign = "+" if val >= 0 else ""
        ax_improve.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{sign}{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_improve.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_improve.set_ylabel('Mejora por Warping (puntos %)', fontsize=11)
    ax_improve.set_title('Mejora: Warped vs Original', fontsize=12)
    ax_improve.set_ylim(0, max(improvements) * 1.15)  # 15% más espacio arriba
    ax_improve.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Guardar
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FIGURES_DIR / "comparacion_simple.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_path


def main():
    print("=" * 70)
    print("REGENERANDO IMAGEN COMPARATIVA SIMPLIFICADA")
    print("=" * 70)
    print()
    print("Esta imagen incluye solo:")
    print("  - 2C (12K imgs): Original vs Warped")
    print("  - 3C (6K imgs): Original vs Warped")
    print()
    print("(Excluye: 2C Comparable)")
    print()

    output_path = generate_comparison_simple()

    print(f"✓ Imagen guardada en: {output_path}")
    print()


if __name__ == "__main__":
    main()
