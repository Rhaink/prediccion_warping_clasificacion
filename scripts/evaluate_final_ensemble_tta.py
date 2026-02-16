#!/usr/bin/env python3
"""
Script de evaluación final del ensemble+TTA sobre el conjunto de prueba completo.

Este script representa la validación culminante del proyecto completo, produciendo
resultados defendibles para tesis con pruebas programáticas de integridad metodológica,
reproducibilidad determinística y manejo transparente de duplicados.

Características:
- Verificación de aislamiento del conjunto de prueba (múltiples métodos)
- Evaluación en conjunto original (1,895 muestras) y limpio (1,894, sin duplicado)
- Verificación de reproducibilidad mediante doble ejecución y comparación de hash
- Actualización de GROUND_TRUTH.json con resultados canónicos finales
- Salida en español para integración directa en tesis
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.evaluation.ensemble import (
    load_ensemble_models,
    ensemble_inference_with_tta,
    weighted_soft_voting
)
from src_v2.models.classifier import get_classifier_transforms


def verify_test_set_isolation(cv_dir: Path) -> Dict[str, bool]:
    """
    Verificación de aislamiento del conjunto de prueba mediante múltiples métodos.

    Realiza las siguientes verificaciones:
    1. Historiales de entrenamiento no contienen métricas de prueba
    2. Archivos de configuración tienen eval_test=false
    3. Separación temporal: checkpoints anteriores a resultados de prueba

    Args:
        cv_dir: Directorio outputs/classifier_cv/

    Returns:
        Dict con resultados de cada verificación
    """
    print("\n" + "="*70)
    print("VERIFICACIÓN DE AISLAMIENTO DEL CONJUNTO DE PRUEBA")
    print("="*70)

    checks = {}

    # Check 1: No test metrics in training history
    print("\n[1] Verificando historiales de entrenamiento...")
    all_clean = True
    for fold in range(1, 6):
        history_path = cv_dir / f"fold_{fold:02d}" / "training_history.json"
        if not history_path.exists():
            print(f"  ⚠ Fold {fold}: Historial no encontrado en {history_path}")
            all_clean = False
            continue

        with open(history_path) as f:
            history = json.load(f)

        # Check if any metric key contains "test"
        if history.get("history"):
            first_epoch = history["history"][0]
            has_test = any("test" in key.lower() for key in first_epoch.keys())
            if has_test:
                print(f"  ✗ Fold {fold}: ADVERTENCIA - Métricas de prueba encontradas")
                all_clean = False
            else:
                print(f"  ✓ Fold {fold}: Solo métricas train/val presentes")

    checks["no_test_in_history"] = all_clean

    # Check 2: Config files have eval_test=false
    print("\n[2] Verificando archivos de configuración...")
    config_clean = True
    for fold in range(1, 6):
        config_path = cv_dir / f"fold_{fold:02d}" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            eval_test = config.get("eval_test", True)
            if eval_test:
                print(f"  ✗ Fold {fold}: eval_test={eval_test}")
                config_clean = False
            else:
                print(f"  ✓ Fold {fold}: eval_test=false confirmado")

    checks["eval_test_false"] = config_clean

    # Check 3: Temporal separation
    print("\n[3] Verificando separación temporal...")
    checkpoint_path = cv_dir / "fold_01" / "best_classifier.pt"
    if checkpoint_path.exists():
        checkpoint_mtime = checkpoint_path.stat().st_mtime
        checkpoint_date = datetime.fromtimestamp(checkpoint_mtime).strftime("%Y-%m-%d")
        print(f"  Fecha de checkpoints: {checkpoint_date}")

        # Models should be dated 2026-01-16 or earlier
        temporal_ok = checkpoint_date <= "2026-01-16"
        checks["temporal_separation"] = temporal_ok
        if temporal_ok:
            print(f"  ✓ Checkpoints anteriores a evaluación final (esperado: 2026-01-16 o anterior)")
        else:
            print(f"  ⚠ ADVERTENCIA: Checkpoints más recientes de lo esperado")
    else:
        checks["temporal_separation"] = False
        print(f"  ✗ Checkpoint no encontrado: {checkpoint_path}")

    # Summary
    print("\n" + "-"*70)
    passed = sum(checks.values())
    total = len(checks)
    print(f"RESUMEN: {passed}/{total} verificaciones pasadas")

    if not all(checks.values()):
        print("⚠ ADVERTENCIA: Algunas verificaciones fallaron, pero continuando...")
    else:
        print("✓ AISLAMIENTO DEL CONJUNTO DE PRUEBA VERIFICADO")

    print("="*70 + "\n")

    return checks


def load_test_dataset(
    data_dir: Path,
    filter_duplicates: bool = False,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Carga el conjunto de prueba con filtrado opcional de duplicados.

    Args:
        data_dir: Directorio base del dataset (e.g., outputs/warped_lung_best/session_warping)
        filter_duplicates: Si True, excluye el duplicado conocido Normal-817
        batch_size: Tamaño de batch
        num_workers: Workers del DataLoader (usar 0 para determinismo)

    Returns:
        Tuple de (dataloader, class_counts)
    """
    # Load transforms
    transform = get_classifier_transforms(train=False)

    # Load dataset
    test_dir = data_dir / "test"
    dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Filter duplicate if requested
    KNOWN_DUPLICATE = "Normal-817_warped.png"
    if filter_duplicates:
        original_count = len(dataset.samples)
        filtered_samples = [
            sample for sample in dataset.samples
            if KNOWN_DUPLICATE not in str(sample[0])
        ]
        dataset.samples = filtered_samples
        dataset.targets = [s[1] for s in filtered_samples]
        removed_count = original_count - len(dataset.samples)
        print(f"  Duplicados removidos: {removed_count} (quedan {len(dataset.samples)} muestras)")

    # Count samples per class
    class_counts = {}
    for class_name in dataset.classes:
        class_idx = dataset.class_to_idx[class_name]
        count = sum(1 for target in dataset.targets if target == class_idx)
        class_counts[class_name] = count

    # Create dataloader with deterministic settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Critical: deterministic order
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader, class_counts


def verify_class_counts(
    class_counts: Dict[str, int],
    expected_counts: Dict[str, int],
    total_expected: int
) -> None:
    """
    Verificación estricta de conteos de clases (hard assertion).

    Cualquier desviación indica corrupción de datos o error de carga.
    """
    total_actual = sum(class_counts.values())

    print(f"\nVerificando conteos de clases:")
    print(f"  Total esperado: {total_expected}, Total real: {total_actual}")

    all_match = True
    for class_name, expected in expected_counts.items():
        actual = class_counts.get(class_name, 0)
        match = "✓" if actual == expected else "✗"
        print(f"  {match} {class_name}: {actual} (esperado: {expected})")
        if actual != expected:
            all_match = False

    assert total_actual == total_expected, \
        f"Conteo total no coincide: {total_actual} != {total_expected}"
    assert all_match, \
        f"Conteos por clase no coinciden con valores esperados"

    print("✓ Conteos verificados correctamente\n")


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calcula métricas de clasificación completas.

    Convierte todos los tipos numpy a tipos nativos de Python para serialización JSON.
    """
    # Overall accuracy
    accuracy = float(accuracy_score(labels, predictions))

    # Per-class metrics via classification_report
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Build metrics dict with native Python types
    metrics = {
        "test_set_size": int(len(labels)),
        "accuracy": float(accuracy),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),  # Convert numpy array to list
        "per_class": {}
    }

    # Per-class metrics
    for class_name in class_names:
        metrics["per_class"][class_name] = {
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1-score": float(report[class_name]["f1-score"]),
            "support": int(report[class_name]["support"])
        }

    return metrics


def compute_metrics_hash(metrics: Dict[str, Any]) -> str:
    """
    Calcula hash SHA-256 determinístico de métricas para verificación de reproducibilidad.
    """
    # Sort keys for deterministic serialization
    metrics_str = json.dumps(metrics, sort_keys=True)
    return hashlib.sha256(metrics_str.encode()).hexdigest()


def evaluate_ensemble_tta(
    models: List[torch.nn.Module],
    weights: List[float],
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    description: str
) -> Dict[str, Any]:
    """
    Evalúa ensemble+TTA y calcula métricas.
    """
    print(f"\n{description}")
    print("-" * 70)

    # Run inference with TTA
    model_preds, model_probs, labels, tta_details = ensemble_inference_with_tta(
        models=models,
        dataloader=dataloader,
        device=device,
        use_tta=True
    )

    # Weighted soft voting
    ensemble_preds, ensemble_probs = weighted_soft_voting(model_probs, weights)

    # Convert to numpy
    predictions = ensemble_preds.cpu().numpy()

    # Compute metrics
    metrics = compute_metrics(predictions, labels, class_names)

    print(f"✓ Evaluación completada: {len(labels)} muestras")
    print(f"  Exactitud: {metrics['accuracy']:.4f}")
    print(f"  F1-macro: {metrics['f1_macro']:.4f}")
    print(f"  F1-weighted: {metrics['f1_weighted']:.4f}")

    return metrics


def check_reproducibility(
    models: List[torch.nn.Module],
    weights: List[float],
    dataloader_original: DataLoader,
    dataloader_cleaned: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Verifica reproducibilidad determinística mediante doble ejecución y comparación de hash.

    Ejecuta ambas evaluaciones (original y limpia) dos veces y verifica que los
    resultados sean bit-idénticos.
    """
    print("\n" + "="*70)
    print("VERIFICACIÓN DE REPRODUCIBILIDAD DETERMINÍSTICA")
    print("="*70)

    print("\n[Ejecución 1]")
    # First run - original
    model_preds_1, model_probs_1, labels_1, _ = ensemble_inference_with_tta(
        models, dataloader_original, device, use_tta=True
    )
    ensemble_preds_1, _ = weighted_soft_voting(model_probs_1, weights)
    metrics_orig_1 = compute_metrics(ensemble_preds_1.cpu().numpy(), labels_1, class_names)
    hash_orig_1 = compute_metrics_hash(metrics_orig_1)

    # First run - cleaned
    model_preds_1c, model_probs_1c, labels_1c, _ = ensemble_inference_with_tta(
        models, dataloader_cleaned, device, use_tta=True
    )
    ensemble_preds_1c, _ = weighted_soft_voting(model_probs_1c, weights)
    metrics_clean_1 = compute_metrics(ensemble_preds_1c.cpu().numpy(), labels_1c, class_names)
    hash_clean_1 = compute_metrics_hash(metrics_clean_1)

    print(f"  Original: hash={hash_orig_1[:16]}..., acc={metrics_orig_1['accuracy']:.4f}")
    print(f"  Limpio:   hash={hash_clean_1[:16]}..., acc={metrics_clean_1['accuracy']:.4f}")

    print("\n[Ejecución 2]")
    # Second run - original
    model_preds_2, model_probs_2, labels_2, _ = ensemble_inference_with_tta(
        models, dataloader_original, device, use_tta=True
    )
    ensemble_preds_2, _ = weighted_soft_voting(model_probs_2, weights)
    metrics_orig_2 = compute_metrics(ensemble_preds_2.cpu().numpy(), labels_2, class_names)
    hash_orig_2 = compute_metrics_hash(metrics_orig_2)

    # Second run - cleaned
    model_preds_2c, model_probs_2c, labels_2c, _ = ensemble_inference_with_tta(
        models, dataloader_cleaned, device, use_tta=True
    )
    ensemble_preds_2c, _ = weighted_soft_voting(model_probs_2c, weights)
    metrics_clean_2 = compute_metrics(ensemble_preds_2c.cpu().numpy(), labels_2c, class_names)
    hash_clean_2 = compute_metrics_hash(metrics_clean_2)

    print(f"  Original: hash={hash_orig_2[:16]}..., acc={metrics_orig_2['accuracy']:.4f}")
    print(f"  Limpio:   hash={hash_clean_2[:16]}..., acc={metrics_clean_2['accuracy']:.4f}")

    # Verify hashes match
    orig_match = hash_orig_1 == hash_orig_2
    clean_match = hash_clean_1 == hash_clean_2
    reproducible = orig_match and clean_match

    print("\n" + "-"*70)
    print("RESULTADO:")
    print(f"  Original: {'✓ IDÉNTICO' if orig_match else '✗ DIFIERE'}")
    print(f"  Limpio:   {'✓ IDÉNTICO' if clean_match else '✗ DIFIERE'}")
    print(f"\n{'✓ REPRODUCIBILIDAD VERIFICADA' if reproducible else '✗ REPRODUCIBILIDAD FALLIDA'}")
    print("="*70 + "\n")

    return {
        "reproducibility_verified": reproducible,
        "original_hash_run1": hash_orig_1,
        "original_hash_run2": hash_orig_2,
        "cleaned_hash_run1": hash_clean_1,
        "cleaned_hash_run2": hash_clean_2,
        "run1_timestamp": datetime.now().astimezone().isoformat(),
        "run2_timestamp": datetime.now().astimezone().isoformat()
    }


def check_improvement_range(
    accuracy: float,
    baseline: float,
    expected_range: Tuple[float, float] = (0.005, 0.010)
) -> None:
    """
    Verifica si la mejora está dentro del rango esperado.

    Imprime advertencia si está fuera del rango, pero continúa (el número real es lo que importa).
    """
    improvement = accuracy - baseline
    improvement_pp = improvement * 100  # Convert to percentage points

    min_pp, max_pp = expected_range[0] * 100, expected_range[1] * 100

    print(f"\nVerificación de rango de mejora:")
    print(f"  Baseline:  {baseline:.4f} ({baseline*100:.2f}%)")
    print(f"  Actual:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Mejora:    {improvement:.4f} ({improvement_pp:+.2f} puntos porcentuales)")
    print(f"  Esperado:  +{min_pp:.1f} a +{max_pp:.1f} pp")

    if improvement_pp < min_pp or improvement_pp > max_pp:
        print(f"\n⚠ ADVERTENCIA: Mejora fuera del rango esperado")
        print(f"  El número real ({improvement_pp:+.2f} pp) es lo que importa para la tesis")
    else:
        print(f"\n✓ Mejora dentro del rango esperado")


def print_summary_table(
    metrics_original: Dict[str, Any],
    metrics_cleaned: Dict[str, Any],
    baseline_accuracy: float
) -> None:
    """
    Imprime tabla resumen comparativa en español.
    """
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS FINALES")
    print("="*70)

    print(f"\n{'Métrica':<25} {'Original':<15} {'Limpio':<15} {'Diferencia':<15}")
    print("-"*70)

    # Accuracy
    acc_orig = metrics_original["accuracy"]
    acc_clean = metrics_cleaned["accuracy"]
    acc_diff = acc_clean - acc_orig
    print(f"{'Exactitud':<25} {acc_orig:.4f}{'':<9} {acc_clean:.4f}{'':<9} {acc_diff:+.4f}")

    # F1-macro
    f1m_orig = metrics_original["f1_macro"]
    f1m_clean = metrics_cleaned["f1_macro"]
    f1m_diff = f1m_clean - f1m_orig
    print(f"{'F1-macro':<25} {f1m_orig:.4f}{'':<9} {f1m_clean:.4f}{'':<9} {f1m_diff:+.4f}")

    # F1-weighted
    f1w_orig = metrics_original["f1_weighted"]
    f1w_clean = metrics_cleaned["f1_weighted"]
    f1w_diff = f1w_clean - f1w_orig
    print(f"{'F1-weighted':<25} {f1w_orig:.4f}{'':<9} {f1w_clean:.4f}{'':<9} {f1w_diff:+.4f}")

    # Sample counts
    n_orig = metrics_original["test_set_size"]
    n_clean = metrics_cleaned["test_set_size"]
    print(f"{'Tamaño conjunto':<25} {n_orig:<15} {n_clean:<15} {n_clean - n_orig}")

    # Improvement over baseline
    imp_orig = (acc_orig - baseline_accuracy) * 100
    imp_clean = (acc_clean - baseline_accuracy) * 100
    print(f"\n{'Mejora sobre baseline':<25} {imp_orig:+.2f} pp{'':<7} {imp_clean:+.2f} pp")

    print("\n" + "="*70)

    # Per-class metrics
    print("\nMÉTRICAS POR CLASE (Conjunto Original):")
    print(f"{'Clase':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Soporte':<12}")
    print("-"*70)
    for class_name in ["COVID", "Normal", "Viral_Pneumonia"]:
        pc = metrics_original["per_class"][class_name]
        print(f"{class_name:<20} {pc['precision']:.4f}{'':<6} {pc['recall']:.4f}{'':<6} "
              f"{pc['f1-score']:.4f}{'':<6} {pc['support']}")

    print("\nMÉTRICAS POR CLASE (Conjunto Limpio):")
    print(f"{'Clase':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Soporte':<12}")
    print("-"*70)
    for class_name in ["COVID", "Normal", "Viral_Pneumonia"]:
        pc = metrics_cleaned["per_class"][class_name]
        print(f"{class_name:<20} {pc['precision']:.4f}{'':<6} {pc['recall']:.4f}{'':<6} "
              f"{pc['f1-score']:.4f}{'':<6} {pc['support']}")

    print("="*70 + "\n")


def update_ground_truth(
    gt_path: Path,
    metrics_original: Dict[str, Any],
    metrics_cleaned: Dict[str, Any],
    reproducibility_verified: bool
) -> None:
    """
    Actualiza GROUND_TRUTH.json con sección final_evaluation.
    """
    print(f"Actualizando {gt_path}...")

    # Read existing
    with open(gt_path) as f:
        gt = json.load(f)

    # Compute baseline improvement
    baseline_accuracy = gt["classification"]["classifier_ensemble_cv"]["baseline_individual_mean"]
    baseline_improvement_pp = (metrics_original["accuracy"] - baseline_accuracy) * 100

    # Add final_evaluation section
    gt["classification"]["final_evaluation"] = {
        "description": "Final test set evaluation with ensemble+TTA (Phase 5)",
        "timestamp": datetime.now().astimezone().isoformat(),
        "validated": datetime.now().date().isoformat(),
        "original_test_set": {
            "test_set_size": metrics_original["test_set_size"],
            "duplicates_present": 1,
            "accuracy": metrics_original["accuracy"],
            "f1_macro": metrics_original["f1_macro"],
            "f1_weighted": metrics_original["f1_weighted"],
            "confusion_matrix": metrics_original["confusion_matrix"],
            "per_class": metrics_original["per_class"]
        },
        "cleaned_test_set": {
            "test_set_size": metrics_cleaned["test_set_size"],
            "duplicates_removed": 1,
            "accuracy": metrics_cleaned["accuracy"],
            "f1_macro": metrics_cleaned["f1_macro"],
            "f1_weighted": metrics_cleaned["f1_weighted"],
            "confusion_matrix": metrics_cleaned["confusion_matrix"],
            "per_class": metrics_cleaned["per_class"]
        },
        "reproducibility_verified": reproducibility_verified,
        "baseline_improvement_pp": baseline_improvement_pp,
        "expected_range_pp": [0.5, 1.0],
        "note": "Both results reported equally; duplicate noted for transparency"
    }

    # Update metadata
    gt["_metadata"]["last_updated"] = datetime.now().date().isoformat()
    gt["_metadata"]["version"] = "2.2.0"

    # Write back
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    print(f"✓ GROUND_TRUTH.json actualizado a versión 2.2.0\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación final del ensemble+TTA con verificaciones de integridad metodológica"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ensemble_classifier.json",
        help="Ruta al archivo de configuración del ensemble"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/classifier_cv",
        help="Directorio de salida para resultados"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detecta si no se especifica)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch para evaluación"
    )

    args = parser.parse_args()

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsando device: {device}\n")

    # Set deterministic mode
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    output_dir = Path(args.output_dir)
    data_dir = Path(config["data_dir"])
    class_names = config["class_names"]
    expected_samples = config["expected_samples"]
    baseline_accuracy = config["baseline_accuracy"]

    # === STEP 1: Test Set Isolation Verification ===
    isolation_checks = verify_test_set_isolation(output_dir)

    # === STEP 2: Load Ensemble Models ===
    print("Cargando modelos del ensemble...")
    models, weights = load_ensemble_models(config["checkpoint_paths"], device)
    print(f"✓ {len(models)} modelos cargados")
    print(f"  Pesos de validación F1-macro: {[f'{w:.4f}' for w in weights]}\n")

    # === STEP 3: Load Test Data - Original (1,895) ===
    print("Cargando conjunto de prueba ORIGINAL (1,895 muestras)...")
    dataloader_original, class_counts_original = load_test_dataset(
        data_dir,
        filter_duplicates=False,
        batch_size=args.batch_size,
        num_workers=0  # Deterministic
    )

    # Verify class counts
    verify_class_counts(
        class_counts_original,
        expected_samples,
        expected_samples["total"]
    )

    # === STEP 4: Evaluate Original Test Set ===
    metrics_original = evaluate_ensemble_tta(
        models=models,
        weights=weights,
        dataloader=dataloader_original,
        device=device,
        class_names=class_names,
        description="EVALUACIÓN EN CONJUNTO ORIGINAL (1,895 muestras con duplicado)"
    )

    # Save original results
    results_original_path = output_dir / "final_evaluation_original.json"
    with open(results_original_path, "w") as f:
        json.dump(metrics_original, f, indent=2, ensure_ascii=False)
    print(f"✓ Resultados guardados en {results_original_path}")

    # === STEP 5: Load Test Data - Cleaned (1,894) ===
    print("\nCargando conjunto de prueba LIMPIO (1,894 muestras, duplicado removido)...")
    dataloader_cleaned, class_counts_cleaned = load_test_dataset(
        data_dir,
        filter_duplicates=True,
        batch_size=args.batch_size,
        num_workers=0  # Deterministic
    )

    expected_cleaned = {k: v for k, v in expected_samples.items()}
    expected_cleaned["total"] = expected_samples["total"] - 1
    expected_cleaned["Normal"] = expected_samples["Normal"] - 1

    verify_class_counts(
        class_counts_cleaned,
        expected_cleaned,
        expected_cleaned["total"]
    )

    # === STEP 6: Evaluate Cleaned Test Set ===
    metrics_cleaned = evaluate_ensemble_tta(
        models=models,
        weights=weights,
        dataloader=dataloader_cleaned,
        device=device,
        class_names=class_names,
        description="EVALUACIÓN EN CONJUNTO LIMPIO (1,894 muestras sin duplicado)"
    )

    # Save cleaned results
    results_cleaned_path = output_dir / "final_evaluation_cleaned.json"
    with open(results_cleaned_path, "w") as f:
        json.dump(metrics_cleaned, f, indent=2, ensure_ascii=False)
    print(f"✓ Resultados guardados en {results_cleaned_path}")

    # === STEP 7: Reproducibility Check ===
    reproducibility_result = check_reproducibility(
        models=models,
        weights=weights,
        dataloader_original=dataloader_original,
        dataloader_cleaned=dataloader_cleaned,
        device=device,
        class_names=class_names
    )

    # Save reproducibility results
    repro_path = output_dir / "final_evaluation_reproducibility.json"
    with open(repro_path, "w") as f:
        json.dump(reproducibility_result, f, indent=2, ensure_ascii=False)
    print(f"Resultados de reproducibilidad guardados en {repro_path}")

    # === STEP 8: Range Check ===
    check_improvement_range(
        accuracy=metrics_original["accuracy"],
        baseline=baseline_accuracy,
        expected_range=(0.005, 0.010)
    )

    # === STEP 9: Print Summary Table ===
    print_summary_table(metrics_original, metrics_cleaned, baseline_accuracy)

    # === STEP 10: Update GROUND_TRUTH.json ===
    gt_path = Path("GROUND_TRUTH.json")
    update_ground_truth(
        gt_path=gt_path,
        metrics_original=metrics_original,
        metrics_cleaned=metrics_cleaned,
        reproducibility_verified=reproducibility_result["reproducibility_verified"]
    )

    print("="*70)
    print("EVALUACIÓN FINAL COMPLETADA")
    print("="*70)
    print(f"\nResultados guardados en:")
    print(f"  - {results_original_path}")
    print(f"  - {results_cleaned_path}")
    print(f"  - {repro_path}")
    print(f"  - {gt_path} (actualizado)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
