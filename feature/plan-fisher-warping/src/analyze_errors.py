#!/usr/bin/env python3
"""
Analisis de errores de clasificacion.

Este script analiza los casos mal clasificados para entender:
1. Que tipo de errores son mas comunes (falsos positivos vs falsos negativos)
2. Si hay imagenes consistentemente mal clasificadas en multiples escenarios
3. Diferencias de errores entre warped y original
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_predictions(phase6_dir: Path, phase7_dir: Path) -> dict:
    """Carga todos los archivos de predicciones."""
    predictions = {}

    # Fase 6: 2 clases
    for name in ['full_warped', 'full_original', 'manual_warped', 'manual_original']:
        filepath = phase6_dir / f'{name}_predictions.csv'
        if filepath.exists():
            predictions[f'{name}_2class'] = pd.read_csv(filepath)

    # Fase 7: 3 clases
    for name in ['full_warped', 'full_original', 'manual_warped', 'manual_original']:
        filepath = phase7_dir / f'{name}_3class_predictions.csv'
        if filepath.exists():
            predictions[f'{name}_3class'] = pd.read_csv(filepath)

    return predictions


def analyze_errors(df: pd.DataFrame, scenario_name: str) -> dict:
    """Analiza errores para un escenario especifico."""
    errors = df[df['correct'] == False].copy()
    correct = df[df['correct'] == True]

    total = len(df)
    n_errors = len(errors)
    accuracy = len(correct) / total * 100

    # Analisis por tipo de error
    error_types = errors.groupby(['true_class', 'pred_class']).size().to_dict()

    # Lista de IDs mal clasificados
    error_ids = errors['image_id'].tolist()

    return {
        'scenario': scenario_name,
        'total': total,
        'n_errors': n_errors,
        'accuracy': accuracy,
        'error_types': error_types,
        'error_ids': error_ids,
        'errors_df': errors
    }


def find_consistent_errors(all_analyses: dict) -> dict:
    """Encuentra imagenes mal clasificadas en multiples escenarios."""
    # Agrupar por dataset (full vs manual) y numero de clases
    groups = {
        'full_2class': ['full_warped_2class', 'full_original_2class'],
        'full_3class': ['full_warped_3class', 'full_original_3class'],
        'manual_2class': ['manual_warped_2class', 'manual_original_2class'],
        'manual_3class': ['manual_warped_3class', 'manual_original_3class'],
    }

    consistent_errors = {}

    for group_name, scenarios in groups.items():
        available = [s for s in scenarios if s in all_analyses]
        if len(available) < 2:
            continue

        # Encontrar IDs que fallan en AMBOS escenarios (warped Y original)
        error_sets = [set(all_analyses[s]['error_ids']) for s in available]
        common_errors = error_sets[0].intersection(*error_sets[1:])

        # Encontrar IDs que fallan SOLO en original (warping corrigio)
        warped_scenario = [s for s in available if 'warped' in s][0]
        original_scenario = [s for s in available if 'original' in s][0]

        warped_errors = set(all_analyses[warped_scenario]['error_ids'])
        original_errors = set(all_analyses[original_scenario]['error_ids'])

        fixed_by_warping = original_errors - warped_errors
        broken_by_warping = warped_errors - original_errors

        consistent_errors[group_name] = {
            'common_errors': list(common_errors),
            'fixed_by_warping': list(fixed_by_warping),
            'broken_by_warping': list(broken_by_warping),
            'n_common': len(common_errors),
            'n_fixed': len(fixed_by_warping),
            'n_broken': len(broken_by_warping),
        }

    return consistent_errors


def generate_report(all_analyses: dict, consistent_errors: dict, output_path: Path):
    """Genera el informe de analisis de errores."""

    lines = []
    lines.append("=" * 70)
    lines.append("ANALISIS DE ERRORES DE CLASIFICACION")
    lines.append("Pipeline: Eigenfaces + Fisher + KNN")
    lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)
    lines.append("")

    # Seccion 1: Resumen por escenario
    lines.append("-" * 70)
    lines.append("1. RESUMEN POR ESCENARIO")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"{'Escenario':<30} {'Total':>8} {'Errores':>8} {'Accuracy':>10}")
    lines.append("-" * 56)

    for name, analysis in sorted(all_analyses.items()):
        lines.append(f"{name:<30} {analysis['total']:>8} {analysis['n_errors']:>8} {analysis['accuracy']:>9.2f}%")

    lines.append("")

    # Seccion 2: Tipos de errores por escenario
    lines.append("-" * 70)
    lines.append("2. TIPOS DE ERRORES (Matriz de Confusion de Errores)")
    lines.append("-" * 70)
    lines.append("")

    for name, analysis in sorted(all_analyses.items()):
        lines.append(f"### {name}")
        if analysis['error_types']:
            for (true_cls, pred_cls), count in sorted(analysis['error_types'].items()):
                lines.append(f"    {true_cls} -> {pred_cls}: {count} casos")
        else:
            lines.append("    Sin errores")
        lines.append("")

    # Seccion 3: Analisis warped vs original
    lines.append("-" * 70)
    lines.append("3. IMPACTO DEL WARPING EN ERRORES")
    lines.append("-" * 70)
    lines.append("")

    for group_name, data in sorted(consistent_errors.items()):
        lines.append(f"### {group_name}")
        lines.append(f"    Errores en comun (ambos fallan): {data['n_common']}")
        lines.append(f"    Corregidos por warping:          {data['n_fixed']}")
        lines.append(f"    Introducidos por warping:        {data['n_broken']}")
        lines.append(f"    Balance neto:                    +{data['n_fixed'] - data['n_broken']} corregidos")
        lines.append("")

    # Seccion 4: Imagenes problematicas (fallan en todos los escenarios)
    lines.append("-" * 70)
    lines.append("4. IMAGENES CONSISTENTEMENTE MAL CLASIFICADAS")
    lines.append("-" * 70)
    lines.append("")
    lines.append("Estas imagenes fallan en AMBOS escenarios (warped y original),")
    lines.append("sugiriendo que son inherentemente dificiles de clasificar.")
    lines.append("")

    for group_name, data in sorted(consistent_errors.items()):
        if data['n_common'] > 0:
            lines.append(f"### {group_name} ({data['n_common']} imagenes)")
            # Mostrar primeros 20
            for img_id in sorted(data['common_errors'])[:20]:
                lines.append(f"    - {img_id}")
            if data['n_common'] > 20:
                lines.append(f"    ... y {data['n_common'] - 20} mas")
            lines.append("")

    # Seccion 5: Imagenes corregidas por warping
    lines.append("-" * 70)
    lines.append("5. IMAGENES CORREGIDAS POR WARPING")
    lines.append("-" * 70)
    lines.append("")
    lines.append("Estas imagenes se clasifican MAL sin warping pero BIEN con warping.")
    lines.append("Esto demuestra el valor del preprocesamiento de alineacion.")
    lines.append("")

    for group_name, data in sorted(consistent_errors.items()):
        if data['n_fixed'] > 0:
            lines.append(f"### {group_name} ({data['n_fixed']} imagenes)")
            for img_id in sorted(data['fixed_by_warping'])[:20]:
                lines.append(f"    - {img_id}")
            if data['n_fixed'] > 20:
                lines.append(f"    ... y {data['n_fixed'] - 20} mas")
            lines.append("")

    # Seccion 6: Analisis detallado de errores del mejor modelo
    lines.append("-" * 70)
    lines.append("6. DETALLE DE ERRORES - MEJOR MODELO (full_warped_2class)")
    lines.append("-" * 70)
    lines.append("")

    if 'full_warped_2class' in all_analyses:
        analysis = all_analyses['full_warped_2class']
        errors_df = analysis['errors_df']

        # Falsos negativos (Enfermo clasificado como Normal)
        fn = errors_df[errors_df['true_class'] == 'Enfermo']
        lines.append(f"### Falsos Negativos (Enfermo -> Normal): {len(fn)} casos")
        lines.append("    (Pacientes enfermos clasificados incorrectamente como sanos)")
        for _, row in fn.head(10).iterrows():
            lines.append(f"    - {row['image_id']}")
        if len(fn) > 10:
            lines.append(f"    ... y {len(fn) - 10} mas")
        lines.append("")

        # Falsos positivos (Normal clasificado como Enfermo)
        fp = errors_df[errors_df['true_class'] == 'Normal']
        lines.append(f"### Falsos Positivos (Normal -> Enfermo): {len(fp)} casos")
        lines.append("    (Pacientes sanos clasificados incorrectamente como enfermos)")
        for _, row in fp.head(10).iterrows():
            lines.append(f"    - {row['image_id']}")
        if len(fp) > 10:
            lines.append(f"    ... y {len(fp) - 10} mas")
        lines.append("")

    # Seccion 7: Conclusiones
    lines.append("-" * 70)
    lines.append("7. CONCLUSIONES")
    lines.append("-" * 70)
    lines.append("")

    # Calcular estadisticas globales
    total_fixed = sum(d['n_fixed'] for d in consistent_errors.values())
    total_broken = sum(d['n_broken'] for d in consistent_errors.values())
    total_common = sum(d['n_common'] for d in consistent_errors.values())

    lines.append(f"1. El warping corrige {total_fixed} clasificaciones erroneas en total.")
    lines.append(f"2. El warping introduce {total_broken} nuevos errores.")
    lines.append(f"3. Balance neto: +{total_fixed - total_broken} clasificaciones correctas.")
    lines.append(f"4. {total_common} imagenes son inherentemente dificiles (fallan siempre).")
    lines.append("")
    lines.append("El warping mejora la clasificacion al normalizar la posicion de las")
    lines.append("estructuras anatomicas, facilitando la comparacion entre imagenes.")
    lines.append("Las imagenes que siguen fallando pueden tener:")
    lines.append("  - Patologia atipica o sutil")
    lines.append("  - Problemas de calidad de imagen")
    lines.append("  - Caracteristicas ambiguas entre clases")
    lines.append("")
    lines.append("=" * 70)
    lines.append("FIN DEL INFORME")
    lines.append("=" * 70)

    # Escribir archivo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Informe guardado en: {output_path}")
    return '\n'.join(lines)


def main():
    """Funcion principal."""
    # Rutas
    base_dir = Path(__file__).parent.parent
    phase6_dir = base_dir / 'results' / 'metrics' / 'phase6_classification'
    phase7_dir = base_dir / 'results' / 'metrics' / 'phase7_comparison'
    output_path = base_dir / 'results' / 'logs' / 'analisis_errores.txt'

    print("=" * 60)
    print("ANALISIS DE ERRORES DE CLASIFICACION")
    print("=" * 60)

    # Cargar predicciones
    print("\n1. Cargando predicciones...")
    predictions = load_predictions(phase6_dir, phase7_dir)
    print(f"   Escenarios cargados: {len(predictions)}")
    for name in predictions:
        print(f"   - {name}")

    # Analizar errores por escenario
    print("\n2. Analizando errores por escenario...")
    all_analyses = {}
    for name, df in predictions.items():
        analysis = analyze_errors(df, name)
        all_analyses[name] = analysis
        print(f"   {name}: {analysis['n_errors']}/{analysis['total']} errores ({100-analysis['accuracy']:.2f}%)")

    # Encontrar errores consistentes
    print("\n3. Analizando impacto del warping...")
    consistent_errors = find_consistent_errors(all_analyses)
    for group, data in consistent_errors.items():
        print(f"   {group}:")
        print(f"     - Corregidos por warping: {data['n_fixed']}")
        print(f"     - Introducidos por warping: {data['n_broken']}")

    # Generar informe
    print("\n4. Generando informe...")
    report = generate_report(all_analyses, consistent_errors, output_path)

    print("\n" + "=" * 60)
    print("ANALISIS COMPLETADO")
    print("=" * 60)


if __name__ == '__main__':
    main()
