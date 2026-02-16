#!/usr/bin/env python3
"""Generate Spanish-language error forensics report from analysis results.

Reads structured JSON results from error analysis, duplicate detection,
and quality assessment to produce a thesis-ready markdown report.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_json(path: Path) -> dict | None:
    """Load JSON file, return None if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_report(
    error_results: dict,
    duplicate_results: dict | None,
    quality_results: dict | None,
) -> str:
    """Generate the full Spanish forensics report."""
    meta = error_results['metadata']
    errors = error_results['errors']
    cat_summary = error_results['category_summary']
    recov_summary = error_results['recoverability_summary']
    origin_summary = error_results['failure_origin_summary']
    confusion = error_results['confusion_pairs']

    # Count fixable errors
    fixable = recov_summary.get('fixable', 0) + recov_summary.get('partially_fixable', 0)

    # Duplicate stats
    if duplicate_results:
        dup_orig = duplicate_results['original']
        dup_warp = duplicate_results['warped']
        dup_comp = duplicate_results['comparison']
        total_dup_pairs = dup_orig['total_pairs']
        cross_split_pairs = dup_warp['cross_split']
    else:
        total_dup_pairs = 0
        cross_split_pairs = 0

    # Quality stats
    if quality_results:
        q_overall = quality_results['overall']
        q_outliers = quality_results['outliers']
    else:
        q_overall = None
        q_outliers = None

    lines = []
    lines.append("# Informe de Forensics de Errores y Auditoria de Calidad de Datos")
    lines.append("")
    lines.append("## Resumen Ejecutivo")
    lines.append("")
    lines.append(f"- **Modelo evaluado:** {meta['model']} (5 folds, soft voting, horizontal flip TTA)")
    lines.append(
        f"- **Precision en test:** {meta['accuracy']}% "
        f"({meta['total_test'] - meta['total_errors']}/{meta['total_test']} correctas, "
        f"{meta['total_errors']} errores)"
    )
    lines.append(f"- **Errores potencialmente recuperables:** {fixable} de {meta['total_errors']}")
    if duplicate_results:
        lines.append(
            f"- **Pares de duplicados detectados:** {total_dup_pairs} en original, "
            f"{dup_warp['total_pairs']} en warped ({cross_split_pairs} cross-split criticos)"
        )
    else:
        lines.append("- **Duplicados:** Datos no disponibles")
    if q_overall:
        lines.append(
            f"- **Calidad promedio BRISQUE:** {q_overall['mean']:.2f} "
            f"(rango: {q_overall['min']:.2f} - {q_overall['max']:.2f})"
        )
    else:
        lines.append("- **Calidad BRISQUE:** Datos no disponibles")
    lines.append("")

    # Section 1: Error Analysis
    lines.append("## 1. Analisis de Errores de Clasificacion")
    lines.append("")

    # 1.1 Confusion pairs
    lines.append("### 1.1 Distribucion por Par de Confusion")
    lines.append("")
    lines.append("| Verdadero | Predicho | Cantidad | Porcentaje |")
    lines.append("|-----------|----------|----------|------------|")
    for pair, count in sorted(confusion.items(), key=lambda x: -x[1]):
        true_c, pred_c = pair.split('_to_')
        pct = 100 * count / meta['total_errors']
        lines.append(f"| {true_c} | {pred_c} | {count} | {pct:.1f}% |")
    lines.append("")
    lines.append(
        "**Observacion principal:** El par Viral_Pneumonia -> Normal es el error dominante "
        f"({confusion.get('Viral_Pneumonia_to_Normal', 0)} casos, "
        f"{100 * confusion.get('Viral_Pneumonia_to_Normal', 0) / meta['total_errors']:.1f}%). "
        "La confusion bidireccional Normal <-> Viral_Pneumonia sugiere solapamiento entre clases."
    )
    lines.append("")

    # 1.2 Error categories
    lines.append("### 1.2 Categorizacion de Errores")
    lines.append("")
    cat_descriptions = {
        'UNANIMOUS_HIGH_CONF': (
            'Todos los folds coinciden con alta confianza incorrecta',
            'Probable ruido de etiqueta'
        ),
        'UNANIMOUS_LOW_CONF': (
            'Todos los folds coinciden con baja confianza',
            'Ejemplo dificil'
        ),
        'HIGH_CONF_ERROR': (
            'Mayoria de folds coinciden con alta confianza incorrecta',
            'Sesgo sistematico'
        ),
        'SPLIT_DECISION': (
            'Los folds no coinciden (acuerdo < 60%)',
            'Cerca del limite de decision'
        ),
        'MODERATE': (
            'Confianza y acuerdo moderados',
            'Limitacion del modelo'
        ),
    }
    lines.append("| Categoria | Cantidad | Interpretacion | Causa probable |")
    lines.append("|-----------|----------|----------------|----------------|")
    for cat, count in cat_summary.items():
        desc, cause = cat_descriptions.get(cat, ('Otro', 'No clasificado'))
        lines.append(f"| {cat} | {count} | {desc} | {cause} |")
    lines.append("")

    # 1.3 Failure origins
    lines.append("### 1.3 Origen de Fallos en el Pipeline")
    lines.append("")
    origin_descriptions = {
        'bad_landmarks': 'Error de landmarks > 2x promedio del dataset (>7.22 px)',
        'bad_warp': 'Tasa de llenado < 85% o landmarks marginales',
        'ambiguous_image': 'Buenos landmarks y warp pero imagen ambigua',
        'suspect_label_noise': 'Buenos landmarks y warp, alta confianza incorrecta',
    }
    lines.append("| Origen de fallo | Cantidad | Descripcion |")
    lines.append("|-----------------|----------|-------------|")
    for origin, count in origin_summary.items():
        desc = origin_descriptions.get(origin, 'Sin descripcion')
        lines.append(f"| {origin} | {count} | {desc} |")
    lines.append("")

    # 1.4 Recoverability
    lines.append("### 1.4 Evaluacion de Recuperabilidad")
    lines.append("")
    recov_descriptions = {
        'fixable': 'Revisar etiquetas manualmente, corregir landmarks',
        'partially_fixable': 'Mejorar entrenamiento, augmentacion mas agresiva',
        'inherent': 'Limitacion del modelo, imagenes genuinamente ambiguas',
    }
    lines.append("| Recuperabilidad | Cantidad | Accion recomendada |")
    lines.append("|-----------------|----------|--------------------|")
    for recov, count in recov_summary.items():
        action = recov_descriptions.get(recov, 'Sin accion definida')
        lines.append(f"| {recov} | {count} | {action} |")
    lines.append("")

    # 1.5 Representative examples
    lines.append("### 1.5 Ejemplos Representativos")
    lines.append("")
    # Group by confusion pair and show top example per pair
    by_pair: dict[str, list] = {}
    for e in errors:
        key = f"{e['true_class']} -> {e['predicted_class']}"
        by_pair.setdefault(key, []).append(e)

    for pair_name, pair_errors in sorted(by_pair.items(), key=lambda x: -len(x[1])):
        example = pair_errors[0]
        lines.append(f"**{pair_name}** ({len(pair_errors)} errores)")
        lines.append("")
        lines.append(f"- Ejemplo: `{example['image_name']}` (idx: {example['sample_idx']})")
        lines.append(
            f"- Confianza: {example['confidence']:.3f} | "
            f"Categoria: {example['category']} | "
            f"Origen: {example['failure_origin']}"
        )
        lines.append(f"- Pipeline trace: `{example['pipeline_trace_path']}`")
        lines.append("")

    # Section 2: Duplicates
    lines.append("## 2. Deteccion de Duplicados")
    lines.append("")

    if duplicate_results:
        lines.append("### 2.1 Resultados en Dataset Original")
        lines.append("")
        lines.append(f"- Total de imagenes analizadas: {dup_orig['total_images']}")
        lines.append(f"- Total de pares duplicados: {dup_orig['total_pairs']}")
        lines.append(f"- Cross-split (fuga de datos): {dup_orig['cross_split']} pares")
        lines.append(
            f"- Cross-class (posibles errores de etiqueta): {dup_orig['cross_class']} pares"
        )
        lines.append(f"- Within-split (redundancia): {dup_orig['within_split']} pares")
        lines.append("")

        lines.append("### 2.2 Resultados en Dataset Warpeado")
        lines.append("")
        lines.append(f"- Total de imagenes analizadas: {dup_warp['total_images']}")
        lines.append(f"- Total de pares duplicados: {dup_warp['total_pairs']}")
        lines.append(f"- Cross-split: {dup_warp['cross_split']} pares")
        lines.append(f"- Cross-class: {dup_warp['cross_class']} pares")
        lines.append(f"- Within-split: {dup_warp['within_split']} pares")
        lines.append("")
        lines.append("Comparacion con originales:")
        lines.append(f"- Divergieron (warping diferencio): {dup_comp['diverged']} pares")
        lines.append(f"- Convergieron (warping unifico): {dup_comp['converged']} pares")
        lines.append(f"- Persistieron (duplicados inherentes): {dup_comp['persistent']} pares")
        lines.append("")

        lines.append("### 2.3 Hallazgos Criticos")
        lines.append("")
        if duplicate_results.get('critical_findings'):
            for finding in duplicate_results['critical_findings']:
                lines.append(f"- **{finding}**")
        else:
            lines.append("No se encontraron hallazgos criticos.")
        lines.append("")

        if dup_warp['cross_split'] > 0:
            lines.append(
                f"> **CRITICO:** Se detectaron {dup_warp['cross_split']} pares de duplicados "
                "que cruzan la particion train/test en el dataset warpeado. Esto constituye fuga "
                "de datos y puede inflar las metricas de evaluacion."
            )
            lines.append("")
        if dup_warp['cross_class'] > 0:
            lines.append(
                f"> **ALERTA:** Se detectaron {dup_warp['cross_class']} pares de imagenes "
                "similares con etiquetas diferentes. Estos podrian ser errores de etiquetado "
                "que contribuyen a los errores de clasificacion."
            )
            lines.append("")
    else:
        lines.append("Datos de duplicados no disponibles.")
        lines.append("")

    # Section 3: Quality Assessment
    lines.append("## 3. Evaluacion de Calidad de Imagen")
    lines.append("")

    if quality_results and q_overall:
        lines.append("### 3.1 Distribucion de Puntajes BRISQUE")
        lines.append("")
        lines.append("| Estadistica | Valor |")
        lines.append("|-------------|-------|")
        lines.append(f"| Media | {q_overall['mean']:.2f} |")
        lines.append(f"| Desviacion estandar | {q_overall['std']:.2f} |")
        lines.append(f"| Mediana | {q_overall['median']:.2f} |")
        lines.append(f"| Percentil 10 | {q_overall['p10']:.2f} |")
        lines.append(f"| Percentil 90 | {q_overall['p90']:.2f} |")
        lines.append(f"| Minimo | {q_overall['min']:.2f} |")
        lines.append(f"| Maximo | {q_overall['max']:.2f} |")
        lines.append("")
        lines.append(
            f"Referencia visual: `quality_scores/quality_distribution.png`"
        )
        lines.append("")

        if quality_results.get('per_class') and quality_results['per_class']:
            lines.append("**Estadisticas por clase:**")
            lines.append("")
            lines.append("| Clase | Media | Std | Mediana | P10 | P90 |")
            lines.append("|-------|-------|-----|---------|-----|-----|")
            for cls, stats in quality_results['per_class'].items():
                lines.append(
                    f"| {cls} | {stats.get('mean', 0):.2f} | "
                    f"{stats.get('std', 0):.2f} | {stats.get('median', 0):.2f} | "
                    f"{stats.get('p10', 0):.2f} | {stats.get('p90', 0):.2f} |"
                )
            lines.append("")

        lines.append("### 3.2 Imagenes de Baja Calidad")
        lines.append("")
        lines.append(
            f"- {q_outliers['count']} imagenes por encima del percentil 90 "
            f"(BRISQUE > {q_outliers['threshold']:.2f})"
        )
        lines.append("")
        if q_outliers.get('top_10_worst'):
            lines.append("**Top 10 peores imagenes:**")
            lines.append("")
            lines.append("| Imagen | BRISQUE |")
            lines.append("|--------|---------|")
            for img in q_outliers['top_10_worst'][:10]:
                lines.append(f"| {img['filename']} | {img['brisque_score']:.2f} |")
            lines.append("")
    else:
        lines.append("Datos de calidad no disponibles.")
        lines.append("")

    # Section 4: Recommendations
    lines.append("## 4. Recomendaciones para Fase 7 (Limpieza de Datos)")
    lines.append("")

    lines.append("### 4.1 Acciones Inmediatas (Alta Prioridad)")
    lines.append("")
    rec_num = 1
    if duplicate_results and dup_warp['cross_split'] > 0:
        lines.append(
            f"{rec_num}. **Eliminar duplicados cross-split:** {dup_warp['cross_split']} pares "
            "de duplicados cruzan la particion train/test en el dataset warpeado, "
            "comprometiendo la validez de la evaluacion."
        )
        rec_num += 1
    if 'UNANIMOUS_HIGH_CONF' in cat_summary and cat_summary['UNANIMOUS_HIGH_CONF'] > 0:
        lines.append(
            f"{rec_num}. **Revision manual de etiquetas:** "
            f"{cat_summary['UNANIMOUS_HIGH_CONF']} muestras con alta confianza incorrecta "
            "unanime sugieren etiquetas erroneas."
        )
        rec_num += 1
    if duplicate_results and dup_warp['cross_class'] > 0:
        lines.append(
            f"{rec_num}. **Investigar duplicados cross-class:** {dup_warp['cross_class']} pares "
            "de imagenes similares con etiquetas diferentes."
        )
        rec_num += 1
    if rec_num == 1:
        lines.append(
            "1. **Revision manual de errores:** Revisar las 33 muestras mal clasificadas, "
            "priorizando el par Viral_Pneumonia -> Normal."
        )
    lines.append("")

    lines.append("### 4.2 Acciones de Mejora (Media Prioridad)")
    lines.append("")
    rec_num = 1
    bad_lm = origin_summary.get('bad_landmarks', 0)
    if bad_lm > 0:
        lines.append(
            f"{rec_num}. **Mejorar deteccion de landmarks:** {bad_lm} errores correlacionados "
            "con landmarks erroneos (> 7.22 px)."
        )
        rec_num += 1
    if q_outliers and q_outliers['count'] > 0:
        lines.append(
            f"{rec_num}. **Considerar exclusion de imagenes de baja calidad:** "
            f"{q_outliers['count']} imagenes con BRISQUE > {q_outliers['threshold']:.2f}."
        )
        rec_num += 1
    if duplicate_results and dup_comp['converged'] > 0:
        lines.append(
            f"{rec_num}. **Investigar convergencia por warping:** {dup_comp['converged']} pares "
            "de imagenes que se volvieron similares tras el warping. Esto podria indicar "
            "sobre-normalizacion."
        )
        rec_num += 1
    if rec_num == 1:
        lines.append("1. No se identificaron acciones de mejora adicionales.")
    lines.append("")

    lines.append("### 4.3 Sin Accion Requerida (Informativo)")
    lines.append("")
    inherent = recov_summary.get('inherent', 0)
    lines.append(
        f"1. **{inherent} errores inherentes:** Limitacion del modelo en imagenes "
        "genuinamente ambiguas. No son defectos del dato."
    )
    if duplicate_results and dup_orig['within_split'] > 0:
        lines.append(
            f"2. **{dup_orig['within_split']} duplicados within-split en originales:** "
            "Redundancia pero no fuga de datos."
        )
    lines.append("")

    # Section 5: Figure index
    lines.append("## 5. Apendice: Indice de Figuras")
    lines.append("")
    lines.append(
        "- `error_visualizations/overview_grid_all_33.png`: "
        "Vista general de los 33 errores"
    )
    lines.append(
        "- `error_visualizations/per_sample_detailed/`: "
        "33 trazas de pipeline individuales"
    )
    lines.append(
        "- `error_visualizations/by_confusion_pair/`: "
        "Errores agrupados por par de confusion"
    )
    if quality_results:
        lines.append(
            "- `quality_scores/quality_distribution.png`: "
            "Distribucion de calidad BRISQUE"
        )
    lines.append("")

    # Metadata
    lines.append("## Metadatos")
    lines.append("")
    lines.append(f"- **Fecha de generacion:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Modelo:** {meta['model']} ({meta['accuracy']}% accuracy)")
    total_images = quality_results['metadata']['total_images'] if quality_results else 'N/A'
    lines.append(f"- **Dataset:** COVID-19 Radiography Dataset ({total_images} imagenes)")
    lines.append("- **Herramientas:** imagededup, pyiqa (BRISQUE), matplotlib")
    lines.append("")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Spanish forensics report from analysis results'
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('outputs/error_forensics'),
        help='Output directory containing analysis results'
    )
    parser.add_argument(
        '--error-results', type=Path, default=None,
        help='Path to error_analysis_results.json'
    )
    parser.add_argument(
        '--duplicate-results', type=Path, default=None,
        help='Path to duplicate_analysis_summary.json'
    )
    parser.add_argument(
        '--quality-results', type=Path, default=None,
        help='Path to quality_analysis_summary.json'
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    # Resolve input paths
    error_path = args.error_results or output_dir / 'error_analysis_results.json'
    dup_path = args.duplicate_results or output_dir / 'duplicates' / 'duplicate_analysis_summary.json'
    quality_path = args.quality_results or output_dir / 'quality_scores' / 'quality_analysis_summary.json'

    # Load data
    error_results = load_json(error_path)
    if error_results is None:
        print(f"ERROR: No se encontro {error_path}")
        return 1

    duplicate_results = load_json(dup_path)
    if duplicate_results is None:
        print(f"AVISO: No se encontro {dup_path}, seccion de duplicados omitida")

    quality_results = load_json(quality_path)
    if quality_results is None:
        print(f"AVISO: No se encontro {quality_path}, seccion de calidad omitida")

    # Generate report
    report = generate_report(error_results, duplicate_results, quality_results)

    # Write report
    report_path = output_dir / 'report_error_forensics.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
    print(f"Reporte generado: {report_path}")
    print(f"Longitud: {len(report)} caracteres")

    # Print summary
    print("\n" + "=" * 60)
    print("RESUMEN DEL REPORTE")
    print("=" * 60)
    print(f"  Errores analizados: {error_results['metadata']['total_errors']}")
    print(f"  Categorias: {list(error_results['category_summary'].keys())}")
    if duplicate_results:
        print(f"  Duplicados originales: {duplicate_results['original']['total_pairs']} pares")
        print(f"  Cross-split criticos: {duplicate_results['warped']['cross_split']} pares")
    if quality_results:
        print(f"  Calidad media BRISQUE: {quality_results['overall']['mean']:.2f}")
        print(f"  Outliers de calidad: {quality_results['outliers']['count']}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
