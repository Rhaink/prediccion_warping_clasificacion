#!/usr/bin/env python3
"""
Script maestro para generar todas las figuras de validación cruzada.

Ejecuta en orden:
1. F5.7: Matriz de confusión CV agregada
2. F5.8: Comparación mixta (CV + test set)
3. F5.9: Casos mal clasificados del mejor fold

Uso:
    python scripts/generate_cv_figures_master.py [--lang es|en]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, args: list = None) -> bool:
    """
    Ejecuta un script de Python.

    Args:
        script_name: Nombre del script (sin path)
        args: Lista de argumentos adicionales

    Returns:
        True si el script se ejecutó exitosamente, False en caso contrario
    """
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)]

    if args:
        cmd.extend(args)

    print(f"\n{'='*70}")
    print(f"Ejecutando: {script_name}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {script_name} completado exitosamente")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error al ejecutar {script_name}")
        print(f"Código de error: {e.returncode}")
        return False

    except Exception as e:
        print(f"\n✗ Error inesperado al ejecutar {script_name}: {e}")
        return False


def verify_outputs(output_dir: Path) -> dict:
    """
    Verifica que todas las figuras hayan sido generadas.

    Args:
        output_dir: Directorio de salida

    Returns:
        Dict con status de cada figura
    """
    expected_figures = {
        "F5.7": output_dir / "F5.7_matriz_confusion_cv.png",
        "F5.8": output_dir / "F5.8_comparacion_cv.png",
        "F5.9": output_dir / "F5.9_casos_mal_clasificados_cv.png",
    }

    status = {}
    for name, path in expected_figures.items():
        exists = path.exists()
        status[name] = exists
        if exists:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {name}: No generado")

    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera todas las figuras de validación cruzada para el capítulo 5.3."
    )
    parser.add_argument(
        "--lang",
        choices=["es", "en"],
        default="es",
        help="Idioma de los textos en las figuras (default: es).",
    )
    parser.add_argument(
        "--skip-f5-7",
        action="store_true",
        help="Omitir generación de F5.7 (matriz de confusión).",
    )
    parser.add_argument(
        "--skip-f5-8",
        action="store_true",
        help="Omitir generación de F5.8 (comparación).",
    )
    parser.add_argument(
        "--skip-f5-9",
        action="store_true",
        help="Omitir generación de F5.9 (mal clasificados).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/Tesis/Figures"),
        help="Directorio de salida para las figuras.",
    )
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()
    base_dir = Path(__file__).parent.parent

    print("=" * 70)
    print("GENERACIÓN MAESTRO DE FIGURAS CV - CAPÍTULO 5.3")
    print("=" * 70)
    print(f"\nIdioma: {args.lang}")
    print(f"Directorio de salida: {args.output_dir}")

    # Lista de scripts a ejecutar
    scripts = []

    if not args.skip_f5_7:
        scripts.append(("generate_confusion_matrix_cv.py", ["--lang", args.lang]))

    if not args.skip_f5_8:
        scripts.append(("generate_F5_8_comparison_cv.py", []))

    if not args.skip_f5_9:
        scripts.append(("generate_F5_9_misclassified_cv.py", []))

    # Ejecutar scripts
    results = {}
    for script_name, script_args in scripts:
        success = run_script(script_name, script_args)
        results[script_name] = success

        if not success:
            print(f"\n⚠ Advertencia: {script_name} falló. Continuando con los demás...")

    # Verificar outputs
    print("\n" + "=" * 70)
    print("VERIFICACIÓN DE OUTPUTS")
    print("=" * 70)

    output_dir = base_dir / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    status = verify_outputs(output_dir)

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    all_success = all(results.values())
    all_generated = all(status.values())

    if all_success and all_generated:
        print("\n✓ Todas las figuras fueron generadas exitosamente")
        print("\nFiguras generadas:")
        for name, path in [
            ("F5.7", output_dir / "F5.7_matriz_confusion_cv.png"),
            ("F5.8", output_dir / "F5.8_comparacion_cv.png"),
            ("F5.9", output_dir / "F5.9_casos_mal_clasificados_cv.png"),
        ]:
            if path.exists():
                print(f"  - {path}")

        print("\nPróximos pasos:")
        print("  1. Revisar las figuras generadas")
        print("  2. Compilar el documento LaTeX 5_3_resultados_clasificacion_CV.tex")
        print("  3. Verificar que las figuras se muestren correctamente")

        return 0

    else:
        print("\n⚠ Algunos scripts fallaron o figuras no fueron generadas")
        print("\nEstado de scripts:")
        for script_name, success in results.items():
            status_str = "✓ OK" if success else "✗ FALLÓ"
            print(f"  {status_str}: {script_name}")

        print("\nEstado de figuras:")
        for name, exists in status.items():
            status_str = "✓ Generado" if exists else "✗ Falta"
            print(f"  {status_str}: {name}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
