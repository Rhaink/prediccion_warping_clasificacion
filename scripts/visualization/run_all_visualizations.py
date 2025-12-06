#!/usr/bin/env python3
"""
Script maestro para generar todas las visualizaciones del pipeline.
Sesion 17: Visualizaciones Detalladas del Pipeline.

Este script ejecuta todos los generadores de visualizaciones:
1. Pipeline de preprocesamiento y augmentation
2. Diagramas detallados de arquitectura
3. Animaciones GIF
4. Mapas de atencion

Genera un resumen completo de todas las figuras para la tesis.
"""

import os
import sys
import time
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))


def run_pipeline_visualizations():
    """Ejecuta generate_pipeline_visualizations.py"""
    print("\n" + "="*70)
    print(" PARTE 1: VISUALIZACIONES DEL PIPELINE")
    print("="*70)

    from generate_pipeline_visualizations import main as pipeline_main
    return pipeline_main()


def run_detailed_diagrams():
    """Ejecuta generate_detailed_diagrams.py"""
    print("\n" + "="*70)
    print(" PARTE 2: DIAGRAMAS DETALLADOS DE ARQUITECTURA")
    print("="*70)

    from generate_detailed_diagrams import main as diagrams_main
    return diagrams_main()


def run_animations():
    """Ejecuta generate_animations.py"""
    print("\n" + "="*70)
    print(" PARTE 3: ANIMACIONES GIF")
    print("="*70)

    from generate_animations import main as animations_main
    return animations_main()


def run_attention_maps():
    """Ejecuta generate_attention_maps.py"""
    print("\n" + "="*70)
    print(" PARTE 4: MAPAS DE ATENCION")
    print("="*70)

    from generate_attention_maps import main as attention_main
    return attention_main()


def count_files_in_directory(directory):
    """Cuenta archivos en un directorio y subdirectorios."""
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len([f for f in files if f.endswith(('.png', '.gif'))])
    return total


def generate_summary_report(output_dir, all_figures, elapsed_time):
    """Genera un reporte resumen de todas las visualizaciones."""

    report_path = os.path.join(output_dir, 'RESUMEN_VISUALIZACIONES.md')

    with open(report_path, 'w') as f:
        f.write("# Resumen de Visualizaciones del Pipeline\n\n")
        f.write("**Sesion 17: Visualizaciones Detalladas del Pipeline**\n\n")
        f.write(f"**Tiempo total de generacion:** {elapsed_time:.1f} segundos\n\n")
        f.write("---\n\n")

        f.write("## Estructura de Directorios\n\n")
        f.write("```\n")
        f.write("outputs/pipeline_viz/\n")

        subdirs = ['preprocessing', 'augmentation', 'inference', 'training',
                   'categories', 'diagrams', 'animations', 'attention_maps']

        for subdir in subdirs:
            subdir_path = os.path.join(output_dir, subdir)
            if os.path.exists(subdir_path):
                files = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.gif'))]
                f.write(f"├── {subdir}/\n")
                for file in sorted(files)[:5]:  # Mostrar max 5 por carpeta
                    f.write(f"│   ├── {file}\n")
                if len(files) > 5:
                    f.write(f"│   └── ... ({len(files) - 5} mas)\n")

        f.write("```\n\n")

        f.write("## Visualizaciones por Categoria\n\n")

        categories = [
            ("Preprocesamiento", "preprocessing", "Pasos del pipeline de preprocesamiento"),
            ("Augmentation", "augmentation", "Data augmentation con flip y rotacion"),
            ("Inferencia", "inference", "Pipeline de inferencia con ensemble"),
            ("Categorias", "categories", "Comparacion entre Normal/COVID/Viral"),
            ("Diagramas", "diagrams", "Arquitectura detallada del modelo"),
            ("Animaciones", "animations", "GIFs animados para presentacion"),
            ("Atencion", "attention_maps", "Mapas de atencion y Grad-CAM"),
        ]

        total_files = 0
        for name, subdir, desc in categories:
            subdir_path = os.path.join(output_dir, subdir)
            if os.path.exists(subdir_path):
                files = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.gif'))]
                total_files += len(files)
                f.write(f"### {name}\n")
                f.write(f"- **Descripcion:** {desc}\n")
                f.write(f"- **Archivos:** {len(files)}\n")
                f.write(f"- **Ubicacion:** `outputs/pipeline_viz/{subdir}/`\n\n")

        f.write("---\n\n")
        f.write(f"**Total de visualizaciones generadas:** {total_files}\n\n")

        f.write("## Uso en la Tesis\n\n")
        f.write("Las visualizaciones estan listas para incluir en:\n")
        f.write("- Capitulo de Metodologia (preprocesamiento, arquitectura)\n")
        f.write("- Capitulo de Resultados (comparaciones, ejemplos)\n")
        f.write("- Presentacion de defensa (animaciones)\n")
        f.write("- Apendices (diagramas detallados)\n")

    print(f"\nReporte guardado en: {report_path}")
    return report_path


def main():
    """Ejecuta todos los generadores de visualizaciones."""

    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "     GENERADOR MAESTRO DE VISUALIZACIONES".center(68) + "#")
    print("#" + "     Sesion 17 - Tesis de Maestria".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    output_dir = 'outputs/pipeline_viz'
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    all_figures = []

    # 1. Pipeline visualizations
    try:
        figures = run_pipeline_visualizations()
        if figures:
            all_figures.extend(figures)
    except Exception as e:
        print(f"Error en pipeline visualizations: {e}")

    # 2. Detailed diagrams
    try:
        figures = run_detailed_diagrams()
        if figures:
            all_figures.extend(figures)
    except Exception as e:
        print(f"Error en detailed diagrams: {e}")

    # 3. Animations
    try:
        figures = run_animations()
        if figures:
            all_figures.extend(figures)
    except Exception as e:
        print(f"Error en animations: {e}")

    # 4. Attention maps
    try:
        figures = run_attention_maps()
        if figures:
            all_figures.extend(figures)
    except Exception as e:
        print(f"Error en attention maps: {e}")

    elapsed_time = time.time() - start_time

    # Generar reporte
    generate_summary_report(output_dir, all_figures, elapsed_time)

    # Resumen final
    total_files = count_files_in_directory(output_dir)

    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "     RESUMEN FINAL".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    print(f"\n  Total de visualizaciones generadas: {total_files}")
    print(f"  Tiempo total: {elapsed_time:.1f} segundos")
    print(f"  Directorio de salida: {output_dir}/")

    print("\n  Contenido por subdirectorio:")
    subdirs = ['preprocessing', 'augmentation', 'inference', 'training',
               'categories', 'diagrams', 'animations', 'attention_maps']

    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.exists(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.gif'))]
            print(f"    - {subdir}/: {len(files)} archivos")

    print("\n  Las visualizaciones estan listas para la tesis!")
    print("#"*70)

    return all_figures


if __name__ == '__main__':
    main()
