#!/usr/bin/env python3
"""
Script de verificación automática post-corrección del error crítico.

Verifica que:
1. Se usó el CSV correcto (02_full_balanced_2class_*.csv)
2. Test size = 1,245 (no 680)
3. Distribución de clases correcta
4. Todos los archivos de resultados existen
5. Números son consistentes a través de todas las fases

Fecha: 2026-01-07
Autor: Claude (corrección post-mortem)
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Colores para output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_check(passed: bool, message: str):
    """Imprime resultado de verificación."""
    symbol = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    print(f"  {symbol} {message}")
    return passed

def verify_csv_correct():
    """Verifica que se usó el CSV correcto."""
    print(f"\n{BOLD}[1/6] VERIFICANDO CSV CORRECTO{RESET}")
    print("="*70)

    base_dir = Path(__file__).parent
    csv_path = base_dir / "results/metrics/02_full_balanced_2class_warped.csv"

    all_passed = True

    # Verificar que existe
    passed = print_check(csv_path.exists(), f"CSV correcto existe: {csv_path.name}")
    all_passed &= passed

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # Verificar tamaño total
        expected_total = 12402
        actual_total = len(df)
        passed = print_check(
            actual_total == expected_total,
            f"Total imágenes: {actual_total} (esperado: {expected_total})"
        )
        all_passed &= passed

        # Verificar tamaño test
        test_df = df[df['split'] == 'test']
        expected_test = 1245
        actual_test = len(test_df)
        passed = print_check(
            actual_test == expected_test,
            f"Test size: {actual_test} (esperado: {expected_test})"
        )
        all_passed &= passed

        # Verificar distribución de clases en test
        test_dist = test_df['class'].value_counts().to_dict()
        expected_enfermo = 498
        expected_normal = 747

        enfermo_count = test_dist.get('Enfermo', 0)
        normal_count = test_dist.get('Normal', 0)

        passed = print_check(
            enfermo_count == expected_enfermo,
            f"Test Enfermo: {enfermo_count} (esperado: {expected_enfermo})"
        )
        all_passed &= passed

        passed = print_check(
            normal_count == expected_normal,
            f"Test Normal: {normal_count} (esperado: {expected_normal})"
        )
        all_passed &= passed

        # Verificar ratio
        ratio = normal_count / enfermo_count if enfermo_count > 0 else 0
        expected_ratio = 1.5
        passed = print_check(
            abs(ratio - expected_ratio) < 0.01,
            f"Ratio Normal/Enfermo: {ratio:.2f} (esperado: ~{expected_ratio})"
        )
        all_passed &= passed

    return all_passed

def verify_phase4():
    """Verifica Phase 4 features."""
    print(f"\n{BOLD}[2/6] VERIFICANDO PHASE 4 (Features){RESET}")
    print("="*70)

    base_dir = Path(__file__).parent
    phase4_dir = base_dir / "results/metrics/phase4_features"

    all_passed = True

    # Verificar archivos
    for dataset in ["full_warped", "full_original"]:
        for split in ["train", "val", "test"]:
            file_path = phase4_dir / f"{dataset}_{split}_features.csv"
            passed = print_check(file_path.exists(), f"{dataset}_{split}_features.csv existe")
            all_passed &= passed

            if file_path.exists() and split == "test":
                df = pd.read_csv(file_path)
                expected = 1245
                actual = len(df)
                passed = print_check(
                    actual == expected,
                    f"{dataset} test: {actual} muestras (esperado: {expected})"
                )
                all_passed &= passed

    return all_passed

def verify_phase5():
    """Verifica Phase 5 Fisher."""
    print(f"\n{BOLD}[3/6] VERIFICANDO PHASE 5 (Fisher){RESET}")
    print("="*70)

    base_dir = Path(__file__).parent
    phase5_dir = base_dir / "results/metrics/phase5_fisher"

    all_passed = True

    for dataset in ["full_warped", "full_original"]:
        for split in ["train", "val", "test"]:
            file_path = phase5_dir / f"{dataset}_{split}_amplified.csv"
            passed = print_check(file_path.exists(), f"{dataset}_{split}_amplified.csv existe")
            all_passed &= passed

            if file_path.exists() and split == "test":
                df = pd.read_csv(file_path)
                expected = 1245
                actual = len(df)  # pandas ya maneja el header
                passed = print_check(
                    actual == expected,
                    f"{dataset} test amplified: {actual} muestras (esperado: {expected})"
                )
                all_passed &= passed

    return all_passed

def verify_phase6():
    """Verifica Phase 6 Classification."""
    print(f"\n{BOLD}[4/6] VERIFICANDO PHASE 6 (Classification){RESET}")
    print("="*70)

    base_dir = Path(__file__).parent
    summary_path = base_dir / "results/metrics/phase6_classification/summary.json"

    all_passed = True

    # Verificar summary.json existe
    passed = print_check(summary_path.exists(), "summary.json existe")
    all_passed &= passed

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        # Verificar test size en resultados
        for dataset in ["full_warped", "full_original"]:
            if dataset in summary["datasets"]:
                n_samples = summary["datasets"][dataset]["test_metrics"]["n_samples"]
                expected = 1245
                passed = print_check(
                    n_samples == expected,
                    f"{dataset} test samples: {n_samples} (esperado: {expected})"
                )
                all_passed &= passed

                # Verificar accuracy
                acc = summary["datasets"][dataset]["test_metrics"]["accuracy"]
                print(f"      {dataset} accuracy: {acc*100:.2f}%")

        # Verificar mejora warped vs original
        if "full_warped" in summary["datasets"] and "full_original" in summary["datasets"]:
            warped_acc = summary["datasets"]["full_warped"]["test_metrics"]["accuracy"]
            original_acc = summary["datasets"]["full_original"]["test_metrics"]["accuracy"]
            diff = warped_acc - original_acc

            passed = print_check(
                diff > 0,
                f"Mejora warped vs original: +{diff*100:.2f}% ({warped_acc*100:.2f}% vs {original_acc*100:.2f}%)"
            )
            all_passed &= passed

    return all_passed

def verify_phase7():
    """Verifica Phase 7 Comparison."""
    print(f"\n{BOLD}[5/6] VERIFICANDO PHASE 7 (Comparison){RESET}")
    print("="*70)

    base_dir = Path(__file__).parent
    summary_path = base_dir / "results/metrics/phase7_comparison/summary.json"

    all_passed = True

    passed = print_check(summary_path.exists(), "summary.json existe")
    all_passed &= passed

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        # Verificar que tiene 2-class y 3-class para full datasets
        for dataset in ["full_warped", "full_original"]:
            for scenario in ["2class", "3class"]:
                key = f"{dataset}_{scenario}"
                if key in summary:
                    n_samples = summary[key]["test_metrics"]["n_samples"]
                    expected = 1245
                    passed = print_check(
                        n_samples == expected,
                        f"{key} test samples: {n_samples} (esperado: {expected})"
                    )
                    all_passed &= passed

    return all_passed

def verify_coherence():
    """Verifica coherencia global entre fases."""
    print(f"\n{BOLD}[6/6] VERIFICANDO COHERENCIA GLOBAL{RESET}")
    print("="*70)

    base_dir = Path(__file__).parent

    all_passed = True

    # Cargar CSV original
    csv_path = base_dir / "results/metrics/02_full_balanced_2class_warped.csv"
    if not csv_path.exists():
        print_check(False, "CSV original no encontrado")
        return False

    df_csv = pd.read_csv(csv_path)
    test_csv = df_csv[df_csv['split'] == 'test']
    csv_test_size = len(test_csv)

    # Verificar Phase 6 summary
    summary_path = base_dir / "results/metrics/phase6_classification/summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        phase6_test_size = summary["datasets"]["full_warped"]["test_metrics"]["n_samples"]

        passed = print_check(
            csv_test_size == phase6_test_size,
            f"CSV test size ({csv_test_size}) = Phase 6 test size ({phase6_test_size})"
        )
        all_passed &= passed

        # Verificar distribución de clases coincide
        csv_dist = test_csv['class'].value_counts().to_dict()
        cm = summary["datasets"]["full_warped"]["test_metrics"]["confusion_matrix"]

        enfermo_csv = csv_dist.get('Enfermo', 0)
        normal_csv = csv_dist.get('Normal', 0)

        # La suma de cada fila de la matriz de confusión debe coincidir
        # Matriz: [[TP_enfermo, FP_normal], [FN_enfermo, TN_normal]]
        enfermo_cm = cm[0][0] + cm[0][1]  # Fila 0: total enfermo
        normal_cm = cm[1][0] + cm[1][1]   # Fila 1: total normal

        passed = print_check(
            enfermo_csv == enfermo_cm,
            f"CSV Enfermo ({enfermo_csv}) = CM Enfermo ({enfermo_cm})"
        )
        all_passed &= passed

        passed = print_check(
            normal_csv == normal_cm,
            f"CSV Normal ({normal_csv}) = CM Normal ({normal_cm})"
        )
        all_passed &= passed

    return all_passed

def main():
    """Ejecuta todas las verificaciones."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}VERIFICACIÓN POST-CORRECCIÓN - ERROR CRÍTICO 2026-01-07{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    results = []

    results.append(("CSV Correcto", verify_csv_correct()))
    results.append(("Phase 4", verify_phase4()))
    results.append(("Phase 5", verify_phase5()))
    results.append(("Phase 6", verify_phase6()))
    results.append(("Phase 7", verify_phase7()))
    results.append(("Coherencia Global", verify_coherence()))

    # Resumen
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}RESUMEN DE VERIFICACIÓN{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    all_passed = True
    for name, passed in results:
        symbol = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{symbol} {name:<30} {status}")
        all_passed &= passed

    print(f"\n{BOLD}{'='*70}{RESET}")
    if all_passed:
        print(f"{GREEN}{BOLD}✓ TODAS LAS VERIFICACIONES PASARON{RESET}")
        print(f"{GREEN}{BOLD}✓ CORRECCIÓN EXITOSA{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}✗ ALGUNAS VERIFICACIONES FALLARON{RESET}")
        print(f"{RED}{BOLD}✗ REVISAR ERRORES ARRIBA{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
