#!/usr/bin/env python3
"""
Script para análisis a nivel de caso (Case-Level Impact Analysis) — Fase 10.

Compara predicciones de elastic+curriculum vs v1.0 baseline y genera:
- Clasificación por caso: mejorado / regresado / neutral
- Grillas de imágenes de rayos X con predicciones y confianza
- impact_summary.json con todos los detalles a nivel de caso

También cruza con el análisis forense de la Fase 6 para verificar si
los casos regresados estaban en el conjunto original de 33 errores.

Uso:
    python scripts/generate_case_analysis_phase10.py \
        --predictions-dir outputs/phase10/predictions \
        --data-dir outputs/warped_cleaned/session_warping
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_MAP = {0: "COVID-19", 1: "Normal", 2: "Neumonía Viral"}
CLASS_KEY_MAP = {0: "COVID", 1: "Normal", 2: "Viral_Pneumonia"}

MODEL_PAIRS = [
    ("elastic_curriculum", "v1_baseline", "elastic_curriculum_vs_v1"),
    ("cleaned_baseline", "v1_baseline", "cleaned_vs_v1"),
    ("curriculum", "v1_baseline", "curriculum_vs_v1"),
]

FORENSICS_PATH = Path("outputs/error_forensics/error_analysis_results.json")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_predictions(predictions_dir: Path) -> dict:
    """
    Carga los 4 CSVs de predicciones y los combina en un diccionario.

    Returns:
        Dict {model_name: DataFrame}
    """
    import csv

    def read_csv(path: Path) -> list:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    return {
        "v1_baseline": read_csv(predictions_dir / "v1_baseline_predictions.csv"),
        "cleaned_baseline": read_csv(predictions_dir / "cleaned_baseline_predictions.csv"),
        "curriculum": read_csv(predictions_dir / "curriculum_predictions.csv"),
        "elastic_curriculum": read_csv(predictions_dir / "elastic_curriculum_predictions.csv"),
    }


def load_forensics_errors(base_dir: Path) -> dict:
    """
    Carga los 33 errores identificados en la Fase 6 (análisis forense).

    Returns:
        Set de nombres de imagen que estaban en el conjunto de errores original.
    """
    forensics_path = base_dir / FORENSICS_PATH
    if not forensics_path.exists():
        return {}

    with open(forensics_path) as f:
        data = json.load(f)

    errors = {}
    for e in data.get("errors", []):
        img_name = e.get("image_name", "")
        if img_name:
            errors[img_name] = {
                "true_class": e.get("true_class"),
                "predicted_class": e.get("predicted_class"),
                "confidence": e.get("confidence"),
                "category": e.get("category"),
                "recoverability": e.get("recoverability"),
                "failure_origin": e.get("failure_origin"),
            }
    return errors


# ---------------------------------------------------------------------------
# Case Categorization
# ---------------------------------------------------------------------------

def categorize_cases(model_new: list, model_ref: list, label: str) -> dict:
    """
    Categoriza cada caso comparando el modelo nuevo vs el de referencia.

    Args:
        model_new: Lista de filas del modelo nuevo (e.g. elastic+curriculum)
        model_ref: Lista de filas del modelo de referencia (v1.0)
        label: Nombre de la comparación para logging

    Returns:
        Dict con listas de casos mejorados, regresados y neutrales
    """
    assert len(model_new) == len(model_ref), "Desalineación de predicciones"

    improved = []
    regressed = []
    neutral = []

    for i, (row_new, row_ref) in enumerate(zip(model_new, model_ref)):
        true_label = int(row_new["true_label"])
        pred_new = int(row_new["predicted_label"])
        pred_ref = int(row_ref["predicted_label"])
        conf_new = float(row_new["confidence"])
        conf_ref = float(row_ref["confidence"])
        image_path = row_new["image_path"]

        correct_new = (pred_new == true_label)
        correct_ref = (pred_ref == true_label)

        case = {
            "sample_idx": i,
            "image_path": image_path,
            "true_label": true_label,
            "true_label_str": CLASS_MAP[true_label],
            "pred_ref": pred_ref,
            "pred_ref_str": CLASS_MAP[pred_ref],
            "conf_ref": conf_ref,
            "pred_new": pred_new,
            "pred_new_str": CLASS_MAP[pred_new],
            "conf_new": conf_new,
        }

        if not correct_ref and correct_new:
            improved.append(case)
        elif correct_ref and not correct_new:
            regressed.append(case)
        else:
            neutral.append(case)

    print(f"  {label}: {len(improved)} mejorados, {len(regressed)} regresados, "
          f"{len(neutral)} neutrales (total={len(improved)+len(regressed)+len(neutral)})")
    return {
        "improved": improved,
        "regressed": regressed,
        "neutral": neutral,
    }


# ---------------------------------------------------------------------------
# Image Loading
# ---------------------------------------------------------------------------

def load_xray_image(image_path_str: str, data_dir: Path, base_dir: Path) -> np.ndarray | None:
    """
    Carga una imagen de rayos X.

    Intenta primero la ruta tal como está en el CSV (relativa al base_dir),
    luego busca en data_dir por clase y nombre de archivo.

    Args:
        image_path_str: Ruta de la imagen desde el CSV
        data_dir: Directorio de datos warp (warped_cleaned/session_warping)
        base_dir: Directorio raíz del proyecto

    Returns:
        Array numpy de la imagen en escala de grises, o None si no se encontró.
    """
    # Intentar ruta relativa desde base_dir
    candidate = base_dir / image_path_str
    if candidate.exists():
        return _read_image(candidate)

    # Extraer solo el nombre del archivo y buscar en data_dir/test/
    filename = Path(image_path_str).name
    for split in ["test", "val", "train"]:
        for class_dir in (data_dir / split).iterdir() if (data_dir / split).exists() else []:
            candidate = class_dir / filename
            if candidate.exists():
                return _read_image(candidate)

    return None


def _read_image(path: Path) -> np.ndarray | None:
    """Lee una imagen usando PIL o cv2."""
    if PIL_AVAILABLE:
        try:
            img = Image.open(path).convert('L')
            return np.array(img)
        except Exception:
            pass

    if CV2_AVAILABLE:
        try:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            return img
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Image Grid Generation
# ---------------------------------------------------------------------------

def generate_image_grid(
    cases: list,
    title: str,
    border_color: str,
    model_ref_label: str,
    model_new_label: str,
    output_path: Path,
    max_images: int = 16,
    grid_cols: int = 4,
    data_dir: Path = None,
    base_dir: Path = None,
):
    """
    Genera una grilla de imágenes de rayos X con predicciones y confianza.

    Args:
        cases: Lista de casos a mostrar
        title: Título de la figura
        border_color: Color del borde ('green' para mejorado, 'red' para regresado)
        model_ref_label: Nombre corto del modelo de referencia
        model_new_label: Nombre corto del modelo nuevo
        output_path: Ruta de salida PNG
        max_images: Máximo de imágenes a mostrar
        grid_cols: Número de columnas en la grilla
        data_dir: Directorio de datos warped
        base_dir: Directorio raíz del proyecto
    """
    if not cases:
        print(f"  Sin casos para: {output_path.name} (0 casos)")
        return

    n_cases = min(len(cases), max_images)
    subset = cases[:n_cases]
    grid_rows = (n_cases + grid_cols - 1) // grid_cols

    fig_width = grid_cols * 3.2
    fig_height = grid_rows * 3.8 + 1.0

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))

    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = axes[np.newaxis, :]
    elif grid_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (ax, case) in enumerate(zip(axes.flatten(), subset)):
        img = None
        if data_dir is not None and base_dir is not None:
            img = load_xray_image(case["image_path"], data_dir, base_dir)

        if img is not None:
            ax.imshow(img, cmap='gray', aspect='auto')
        else:
            # Placeholder si no se puede cargar la imagen
            ax.set_facecolor('#222222')
            ax.text(
                0.5, 0.5, 'Imagen\nno\ndisponible',
                ha='center', va='center', color='white',
                fontsize=9, transform=ax.transAxes,
            )

        # Borde de color
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        ax.set_xticks([])
        ax.set_yticks([])

        # Título con real y predicciones
        true_str = case["true_label_str"]
        ref_pred = case["pred_ref_str"]
        new_pred = case["pred_new_str"]
        conf_ref = case["conf_ref"]
        conf_new = case["conf_new"]

        title_line1 = f'Real: {true_str}'
        title_line2 = f'{model_ref_label}: {ref_pred} ({conf_ref:.0%})'
        title_line3 = f'{model_new_label}: {new_pred} ({conf_new:.0%})'

        ax.set_title(
            f'{title_line1}\n{title_line2}\n{title_line3}',
            fontsize=7.5, pad=4, loc='center',
            color='black',
        )

    # Ocultar ejes vacíos
    for ax in axes.flatten()[n_cases:]:
        ax.set_visible(False)

    # Leyenda del borde
    border_patch = patches.Patch(
        facecolor='none', edgecolor=border_color, linewidth=3,
        label=f'Borde {border_color}: {title.split("—")[0].strip()}'
    )
    fig.legend(handles=[border_patch], loc='lower center', ncol=1,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f'{title}\n({n_cases} de {len(cases)} casos)',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    size_kb = output_path.stat().st_size / 1024
    print(f"  Guardada: {output_path.name} ({size_kb:.0f} KB, {n_cases} imágenes)")
    plt.close()


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------

def run_analysis(predictions_dir: Path, data_dir: Path, output_dir: Path, base_dir: Path):
    """
    Ejecuta el análisis a nivel de caso completo.

    Args:
        predictions_dir: Directorio con los 4 CSVs de predicciones
        data_dir: Directorio warped (para cargar imágenes)
        output_dir: Directorio de salida para JSON y PNGs
        base_dir: Raíz del proyecto
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("ANÁLISIS A NIVEL DE CASO — FASE 10")
    print("=" * 70)

    # Cargar predicciones
    print("\n[1/4] Cargando predicciones...")
    preds = load_predictions(predictions_dir)
    n_samples = len(preds["v1_baseline"])
    print(f"  Muestras: {n_samples}")
    for model_name in preds:
        assert len(preds[model_name]) == n_samples, f"Conteo incorrecto para {model_name}"

    # Cargar datos forenses de Fase 6
    print("\n[2/4] Cargando errores de la Fase 6 (análisis forense)...")
    forensics_errors = load_forensics_errors(base_dir)
    print(f"  Errores forenses: {len(forensics_errors)} casos")

    # Análisis por comparación
    print("\n[3/4] Categorizando casos...")
    all_analyses = {}
    for model_new_name, model_ref_name, comp_label in MODEL_PAIRS:
        all_analyses[comp_label] = categorize_cases(
            model_new=preds[model_new_name],
            model_ref=preds[model_ref_name],
            label=comp_label,
        )

    # Análisis principal: elastic+curriculum vs v1.0
    main_analysis = all_analyses["elastic_curriculum_vs_v1"]
    n_improved = len(main_analysis["improved"])
    n_regressed = len(main_analysis["regressed"])
    n_neutral = len(main_analysis["neutral"])
    n_total = n_improved + n_regressed + n_neutral
    assert n_total == n_samples, f"Total mismatch: {n_total} != {n_samples}"

    print(f"\nElastic+Curriculum vs v1.0: "
          f"{n_improved} mejorados, {n_regressed} regresados, {n_neutral} neutrales")

    # Cruce con análisis forense
    def enrich_with_forensics(cases: list, forensics: dict) -> list:
        """Agrega info forense de la Fase 6 a cada caso si está disponible."""
        for case in cases:
            filename = Path(case["image_path"]).stem.replace("_warped", "")
            if filename in forensics:
                case["in_phase6_errors"] = True
                case["phase6_info"] = forensics[filename]
            else:
                case["in_phase6_errors"] = False
        return cases

    main_analysis["improved"] = enrich_with_forensics(main_analysis["improved"], forensics_errors)
    main_analysis["regressed"] = enrich_with_forensics(main_analysis["regressed"], forensics_errors)

    # Estadísticas de cruce forense
    in_forensics_regressed = sum(1 for c in main_analysis["regressed"] if c.get("in_phase6_errors"))
    in_forensics_improved = sum(1 for c in main_analysis["improved"] if c.get("in_phase6_errors"))

    # Generar grillas de imágenes
    print("\n[4/4] Generando grillas de imágenes...")

    # Grilla de casos mejorados: elastic+curriculum vs v1.0
    generate_image_grid(
        cases=main_analysis["improved"],
        title="Casos Mejorados — Elástico+Curriculum vs v1.0",
        border_color='green',
        model_ref_label="v1.0",
        model_new_label="Elást.+Curr.",
        output_path=output_dir / "case_grid_improved.png",
        max_images=16,
        grid_cols=4,
        data_dir=data_dir,
        base_dir=base_dir,
    )

    # Grilla de casos regresados: elastic+curriculum vs v1.0
    generate_image_grid(
        cases=main_analysis["regressed"],
        title="Casos Regresados — Elástico+Curriculum vs v1.0",
        border_color='red',
        model_ref_label="v1.0",
        model_new_label="Elást.+Curr.",
        output_path=output_dir / "case_grid_regressed.png",
        max_images=16,
        grid_cols=4,
        data_dir=data_dir,
        base_dir=base_dir,
    )

    # Grilla adicional: cleaned vs v1.0
    cleaned_analysis = all_analyses["cleaned_vs_v1"]
    if len(cleaned_analysis["improved"]) > 0 or len(cleaned_analysis["regressed"]) > 0:
        generate_image_grid(
            cases=cleaned_analysis["improved"],
            title="Casos Mejorados — Datos Limpios vs v1.0",
            border_color='green',
            model_ref_label="v1.0",
            model_new_label="Limpios",
            output_path=output_dir / "case_grid_cleaned_improved.png",
            max_images=16,
            grid_cols=4,
            data_dir=data_dir,
            base_dir=base_dir,
        )
        generate_image_grid(
            cases=cleaned_analysis["regressed"],
            title="Casos Regresados — Datos Limpios vs v1.0",
            border_color='red',
            model_ref_label="v1.0",
            model_new_label="Limpios",
            output_path=output_dir / "case_grid_cleaned_regressed.png",
            max_images=16,
            grid_cols=4,
            data_dir=data_dir,
            base_dir=base_dir,
        )

    # Grilla adicional: curriculum vs v1.0
    curriculum_analysis = all_analyses["curriculum_vs_v1"]
    if len(curriculum_analysis["improved"]) > 0 or len(curriculum_analysis["regressed"]) > 0:
        generate_image_grid(
            cases=curriculum_analysis["improved"],
            title="Casos Mejorados — Curriculum vs v1.0",
            border_color='green',
            model_ref_label="v1.0",
            model_new_label="Curriculum",
            output_path=output_dir / "case_grid_curriculum_improved.png",
            max_images=16,
            grid_cols=4,
            data_dir=data_dir,
            base_dir=base_dir,
        )
        generate_image_grid(
            cases=curriculum_analysis["regressed"],
            title="Casos Regresados — Curriculum vs v1.0",
            border_color='red',
            model_ref_label="v1.0",
            model_new_label="Curriculum",
            output_path=output_dir / "case_grid_curriculum_regressed.png",
            max_images=16,
            grid_cols=4,
            data_dir=data_dir,
            base_dir=base_dir,
        )

    # Construir impact_summary.json
    def build_summary_for(analysis: dict, model_new: str, model_ref: str) -> dict:
        """Construye el resumen estructurado para una comparación."""
        improved = analysis["improved"]
        regressed = analysis["regressed"]
        neutral = analysis["neutral"]

        # Per-class breakdown
        def per_class_count(cases: list) -> dict:
            counts = {0: 0, 1: 0, 2: 0}
            for c in cases:
                counts[c["true_label"]] += 1
            return {CLASS_KEY_MAP[k]: v for k, v in counts.items()}

        return {
            "model_new": model_new,
            "model_ref": model_ref,
            "improved_count": len(improved),
            "regressed_count": len(regressed),
            "neutral_count": len(neutral),
            "total": len(improved) + len(regressed) + len(neutral),
            "improved_per_class": per_class_count(improved),
            "regressed_per_class": per_class_count(regressed),
            "improved_cases": [
                {k: v for k, v in c.items() if k != 'phase6_info'}
                for c in improved
            ],
            "regressed_cases": [
                {k: v for k, v in c.items() if k != 'phase6_info'}
                for c in regressed
            ],
        }

    impact_summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "improved_count": n_improved,
        "regressed_count": n_regressed,
        "neutral_count": n_neutral,
        "total": n_samples,
        "forensics_cross_reference": {
            "phase6_error_set_size": len(forensics_errors),
            "regressed_in_phase6_errors": in_forensics_regressed,
            "improved_in_phase6_errors": in_forensics_improved,
        },
        "comparisons": {
            "elastic_curriculum_vs_v1": build_summary_for(
                all_analyses["elastic_curriculum_vs_v1"],
                "elastic_curriculum",
                "v1_baseline",
            ),
            "cleaned_vs_v1": build_summary_for(
                all_analyses["cleaned_vs_v1"],
                "cleaned_baseline",
                "v1_baseline",
            ),
            "curriculum_vs_v1": build_summary_for(
                all_analyses["curriculum_vs_v1"],
                "curriculum",
                "v1_baseline",
            ),
        },
    }

    # Guardar JSON
    summary_path = output_dir / "impact_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(impact_summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Guardado: {summary_path.name}")

    # Resumen en consola
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"\nElastic+Curriculum vs v1.0:")
    print(f"  Mejorados:  {n_improved}")
    print(f"  Regresados: {n_regressed}")
    print(f"  Neutrales:  {n_neutral}")
    print(f"  Total:      {n_samples}")

    if forensics_errors:
        print(f"\nCruce con Fase 6 (forense, {len(forensics_errors)} errores originales):")
        print(f"  Casos regresados en errores forenses: {in_forensics_regressed}")
        print(f"  Casos mejorados en errores forenses:  {in_forensics_improved}")

    print(f"\nComparaciones adicionales:")
    for comp_label, display in [
        ("cleaned_vs_v1", "Datos Limpios vs v1.0"),
        ("curriculum_vs_v1", "Curriculum vs v1.0"),
    ]:
        a = all_analyses[comp_label]
        print(f"  {display}: +{len(a['improved'])} mejorados, -{len(a['regressed'])} regresados")

    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)

    return impact_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis a nivel de caso para Fase 10 (impact analysis con grillas de imágenes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/phase10/predictions"),
        help="Directorio con los 4 CSVs de predicciones.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/warped_cleaned/session_warping"),
        help="Directorio de datos warped (para cargar imágenes).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase10/case_analysis"),
        help="Directorio de salida para JSON y grillas de imágenes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).parent.parent
    predictions_dir = (
        base_dir / args.predictions_dir
        if not args.predictions_dir.is_absolute()
        else args.predictions_dir
    )
    data_dir = (
        base_dir / args.data_dir
        if not args.data_dir.is_absolute()
        else args.data_dir
    )
    output_dir = (
        base_dir / args.output_dir
        if not args.output_dir.is_absolute()
        else args.output_dir
    )

    if not predictions_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de predicciones: {predictions_dir}")

    if not data_dir.exists():
        print(f"ADVERTENCIA: Directorio de datos no encontrado: {data_dir}")
        print("  Las grillas se generarán sin imágenes (placeholders).")

    run_analysis(predictions_dir, data_dir, output_dir, base_dir)


if __name__ == "__main__":
    main()
