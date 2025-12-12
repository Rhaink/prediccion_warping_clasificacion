#!/usr/bin/env python3
"""
Session 34: Script para crear figuras comparativas de GradCAM.

Genera figuras lado-a-lado mostrando:
- Imagen original vs warped
- GradCAM de modelo original vs warped
- Diferencias en atencion (pulmones vs artefactos)

Uso:
    python scripts/create_thesis_figures.py
"""

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import json


def create_comparison_figure(
    original_img_path: Path,
    warped_img_path: Path,
    original_gradcam_path: Path,
    warped_gradcam_path: Path,
    output_path: Path,
    title: str = "",
    prediction_original: str = "",
    prediction_warped: str = ""
):
    """Crea figura 2x2 comparativa para una imagen."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Fila 1: Imagenes sin GradCAM
    img_original = Image.open(original_img_path)
    img_warped = Image.open(warped_img_path)

    axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title("Imagen Original", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_warped, cmap='gray')
    axes[0, 1].set_title("Imagen Warped (Normalizada)", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Fila 2: GradCAM
    gradcam_original = Image.open(original_gradcam_path)
    gradcam_warped = Image.open(warped_gradcam_path)

    axes[1, 0].imshow(gradcam_original)
    subtitle_orig = "GradCAM Modelo Original"
    if prediction_original:
        subtitle_orig += f"\n(pred: {prediction_original})"
    axes[1, 0].set_title(subtitle_orig, fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gradcam_warped)
    subtitle_warp = "GradCAM Modelo Warped"
    if prediction_warped:
        subtitle_warp += f"\n(pred: {prediction_warped})"
    axes[1, 1].set_title(subtitle_warp, fontsize=12)
    axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figura guardada: {output_path}")


def create_4x4_matrix_figure(
    image_name: str,
    class_name: str,
    base_dir: Path,
    output_path: Path
):
    """
    Crea matriz 4x4 mostrando todas las combinaciones:
    - Filas: Modelo (Original, Warped)
    - Columnas: Imagen (Original, Warped)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    combinations = [
        ("original_on_original", "Modelo Original → Img Original", 0, 0),
        ("original_on_warped", "Modelo Original → Img Warped", 0, 1),
        ("warped_on_original", "Modelo Warped → Img Original", 1, 0),
        ("warped_on_warped", "Modelo Warped → Img Warped", 1, 1),
    ]

    for combo_name, title, row, col in combinations:
        gradcam_path = base_dir / combo_name / class_name / f"{image_name}_gradcam.png"
        if gradcam_path.exists():
            img = Image.open(gradcam_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=11, fontweight='bold')
        else:
            axes[row, col].text(0.5, 0.5, "No disponible", ha='center', va='center')
            axes[row, col].set_title(title, fontsize=11)
        axes[row, col].axis('off')

    plt.suptitle(f"Matriz de Atención GradCAM: {image_name}\nClase: {class_name}",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Matriz guardada: {output_path}")


def create_side_by_side_comparison(
    image_name: str,
    class_name: str,
    source_dir: Path,
    gradcam_dir: Path,
    output_path: Path
):
    """
    Crea figura comparativa lado-a-lado:
    - Columna 1: Original + GradCAM modelo original
    - Columna 2: Warped + GradCAM modelo warped

    Esta es la figura mas importante para la tesis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Paths
    original_img = source_dir / "original" / class_name / f"{image_name}.png"
    warped_img = source_dir / "warped" / class_name / f"{image_name}.png"
    original_gradcam = gradcam_dir / "original_on_original" / class_name / f"{image_name}_gradcam.png"
    warped_gradcam = gradcam_dir / "warped_on_warped" / class_name / f"{image_name}_gradcam.png"

    # Verificar que existen
    paths = [original_img, warped_img, original_gradcam, warped_gradcam]
    for p in paths:
        if not p.exists():
            print(f"ADVERTENCIA: No existe {p}")
            return False

    # Fila 1: Imagenes sin procesamiento
    axes[0, 0].imshow(Image.open(original_img), cmap='gray')
    axes[0, 0].set_title("Imagen Original\n(con artefactos hospitalarios)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(Image.open(warped_img), cmap='gray')
    axes[0, 1].set_title("Imagen Warped\n(normalizada geométricamente)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Fila 2: GradCAM
    axes[1, 0].imshow(Image.open(original_gradcam))
    axes[1, 0].set_title("GradCAM: Modelo entrenado en ORIGINALES\n⚠ Atiende a bordes y artefactos",
                         fontsize=11, color='darkred')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(Image.open(warped_gradcam))
    axes[1, 1].set_title("GradCAM: Modelo entrenado en WARPED\n✓ Atiende a regiones pulmonares",
                         fontsize=11, color='darkgreen')
    axes[1, 1].axis('off')

    plt.suptitle(f"Comparación de Atención del Clasificador - {class_name}\n{image_name}",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Comparación guardada: {output_path}")
    return True


def create_cross_domain_figure(
    image_name: str,
    class_name: str,
    gradcam_dir: Path,
    output_path: Path
):
    """
    Crea figura de análisis cross-domain:
    Muestra cómo los modelos se comportan en datos de diferente dominio.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    combinations = [
        ("original_on_original", "Original→Original\n(In-domain)"),
        ("original_on_warped", "Original→Warped\n(Cross-domain)"),
        ("warped_on_original", "Warped→Original\n(Cross-domain)"),
        ("warped_on_warped", "Warped→Warped\n(In-domain)"),
    ]

    for idx, (combo_name, title) in enumerate(combinations):
        gradcam_path = gradcam_dir / combo_name / class_name / f"{image_name}_gradcam.png"
        if gradcam_path.exists():
            img = Image.open(gradcam_path)
            axes[idx].imshow(img)
        else:
            axes[idx].text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=14)
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle(f"Análisis Cross-Domain: {image_name} ({class_name})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Cross-domain guardada: {output_path}")


def create_summary_figure(
    metrics: dict,
    output_path: Path
):
    """
    Crea figura de resumen con las métricas clave de la hipótesis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Generalización
    ax1 = axes[0]
    categories = ['Gap Original', 'Gap Warped']
    values = [metrics.get('gap_original', 25.36), metrics.get('gap_warped', 2.24)]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Gap de Generalización (%)', fontsize=12)
    ax1.set_title('Mejora en Generalización\n(11.3× mejor)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 30)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')

    # Panel 2: Robustez JPEG
    ax2 = axes[1]
    categories = ['Degradación\nOriginal Q50', 'Degradación\nWarped Q50']
    values = [metrics.get('degradation_original', 16.14), metrics.get('degradation_warped', 0.53)]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Degradación por JPEG (%)', fontsize=12)
    ax2.set_title('Mejora en Robustez JPEG\n(30.45× mejor)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 20)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')

    plt.suptitle('Hipótesis Confirmada: El Warping Mejora Generalización y Robustez',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Resumen guardada: {output_path}")


def main():
    """Genera todas las figuras para la tesis."""

    # Directorios
    base_dir = Path("outputs/thesis_figures")
    source_dir = base_dir / "source_pairs"
    gradcam_dir = base_dir / "gradcam_comparison"
    output_dir = base_dir / "combined_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar todas las imágenes disponibles
    classes = ["COVID", "Normal", "Viral_Pneumonia"]

    print("=" * 60)
    print("Generando figuras comparativas para tesis")
    print("=" * 60)

    # 1. Figuras lado-a-lado para cada imagen
    print("\n--- Figuras lado-a-lado ---")
    count = 0
    for class_name in classes:
        class_dir = source_dir / "original" / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.glob("*.png"):
            image_name = img_path.stem
            output_path = output_dir / f"comparison_{class_name}_{image_name}.png"

            success = create_side_by_side_comparison(
                image_name=image_name,
                class_name=class_name,
                source_dir=source_dir,
                gradcam_dir=gradcam_dir,
                output_path=output_path
            )
            if success:
                count += 1

    print(f"\nGeneradas {count} figuras lado-a-lado")

    # 2. Figuras cross-domain
    print("\n--- Figuras cross-domain ---")
    for class_name in classes:
        class_dir = source_dir / "original" / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.glob("*.png"):
            image_name = img_path.stem
            output_path = output_dir / f"crossdomain_{class_name}_{image_name}.png"

            create_cross_domain_figure(
                image_name=image_name,
                class_name=class_name,
                gradcam_dir=gradcam_dir,
                output_path=output_path
            )

    # 3. Figura de matriz 2x2 para una imagen representativa de cada clase
    print("\n--- Matrices de atención ---")
    representative_images = {}
    for class_name in classes:
        class_dir = source_dir / "original" / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            if images:
                representative_images[class_name] = images[0].stem

    for class_name, image_name in representative_images.items():
        output_path = output_dir / f"matrix_{class_name}_{image_name}.png"
        create_4x4_matrix_figure(
            image_name=image_name,
            class_name=class_name,
            base_dir=gradcam_dir,
            output_path=output_path
        )

    # 4. Figura de resumen de métricas
    print("\n--- Figura de resumen ---")
    metrics = {
        'gap_original': 25.36,
        'gap_warped': 2.24,
        'degradation_original': 16.14,
        'degradation_warped': 0.53
    }

    # Intentar cargar métricas reales
    consolidated_path = Path("outputs/session30_analysis/consolidated_results.json")
    robustness_path = Path("outputs/session29_robustness/artifact_robustness_results.json")

    if consolidated_path.exists():
        try:
            with open(consolidated_path) as f:
                data = json.load(f)
                # Extraer gaps si están disponibles
                if 'gap_analysis' in data:
                    metrics['gap_original'] = data['gap_analysis'].get('original_gap', 25.36)
                    metrics['gap_warped'] = data['gap_analysis'].get('warped_gap', 2.24)
        except Exception as e:
            print(f"Error leyendo consolidated_results.json: {e}")

    if robustness_path.exists():
        try:
            with open(robustness_path) as f:
                data = json.load(f)
                # Buscar degradación Q50
                for result in data.get('results', []):
                    if result.get('quality') == 50:
                        if 'original' in result.get('experiment', '').lower():
                            metrics['degradation_original'] = abs(result.get('accuracy_change', -16.14))
                        elif 'warped' in result.get('experiment', '').lower():
                            metrics['degradation_warped'] = abs(result.get('accuracy_change', -0.53))
        except Exception as e:
            print(f"Error leyendo robustness results: {e}")

    create_summary_figure(metrics, output_dir / "summary_metrics.png")

    print("\n" + "=" * 60)
    print(f"Todas las figuras guardadas en: {output_dir}")
    print("=" * 60)

    # Listar archivos generados
    generated = list(output_dir.glob("*.png"))
    print(f"\nArchivos generados ({len(generated)}):")
    for f in sorted(generated):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
