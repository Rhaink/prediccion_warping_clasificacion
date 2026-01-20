#!/usr/bin/env python3
"""
Build script for creating Windows standalone executable.

This script automates the entire build process:
1. Creates a clean virtual environment for building
2. Installs CPU-only PyTorch and dependencies
3. Verifies all models and assets exist
4. Runs PyInstaller with the spec file
5. Validates the output executable

Usage:
    python scripts/build_windows_exe.py --prepare  # Setup build environment
    python scripts/build_windows_exe.py --build    # Build executable
    python scripts/build_windows_exe.py --clean    # Clean build artifacts
    python scripts/build_windows_exe.py --all      # Do everything (prepare + build)
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message):
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_info(message):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def verify_models_exist():
    """Verify that all required models and assets exist."""
    project_root = get_project_root()

    required_files = [
        # Landmark ensemble models (4 models) - using actual development paths
        'checkpoints/session10/ensemble/seed123/final_model.pt',
        'checkpoints/session13/seed321/final_model.pt',
        'checkpoints/repro_split111/session14/seed111/final_model.pt',
        'checkpoints/repro_split666/session16/seed666/final_model.pt',

        # Classifier model - using actual development path
        'outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt',

        # Geometric data
        'outputs/shape_analysis/canonical_shape_gpa.json',
        'outputs/shape_analysis/canonical_delaunay_triangles.json',

        # Ground truth metrics
        'GROUND_TRUTH.json',
    ]

    missing_files = []
    total_size = 0

    print_info("Verificando archivos requeridos...")

    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print_error(f"Faltante: {file_path}")
        else:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print_success(f"Encontrado: {file_path} ({size_mb:.1f} MB)")

    if missing_files:
        print_error(f"\n{len(missing_files)} archivo(s) faltante(s):")
        for f in missing_files:
            print(f"  - {f}")
        print_warning("\nGenera los modelos antes de continuar:")
        print("  1. Entrena modelos de landmarks (4 seeds)")
        print("  2. Entrena clasificador")
        print("  3. Ejecuta compute-canonical para generar shape_analysis")
        return False

    print_success(f"\n✓ Todos los archivos encontrados (Total: {total_size:.1f} MB)")
    return True


def create_build_environment():
    """Create a clean virtual environment for building."""
    project_root = get_project_root()
    venv_path = project_root / '.venv_build'

    print_header("Creando Entorno Virtual de Build")

    # Remove existing venv if present
    if venv_path.exists():
        print_warning(f"Removiendo entorno virtual existente: {venv_path}")
        shutil.rmtree(venv_path)

    # Create new venv
    print_info(f"Creando nuevo entorno virtual en: {venv_path}")
    subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
    print_success("Entorno virtual creado")

    # Determine pip executable
    if platform.system() == 'Windows':
        pip_exe = venv_path / 'Scripts' / 'pip.exe'
    else:
        pip_exe = venv_path / 'bin' / 'pip'

    # Upgrade pip
    print_info("Actualizando pip...")
    subprocess.run([str(pip_exe), 'install', '--upgrade', 'pip'], check=True)
    print_success("pip actualizado")

    # Install requirements
    requirements_file = project_root / 'scripts' / 'requirements_windows_cpu.txt'
    print_info(f"Instalando dependencias desde: {requirements_file}")
    print_warning("Esto puede tomar 5-10 minutos...")

    subprocess.run(
        [str(pip_exe), 'install', '-r', str(requirements_file)],
        check=True
    )
    print_success("Dependencias instaladas")

    return venv_path


def build_executable():
    """Build the executable using PyInstaller."""
    project_root = get_project_root()
    spec_file = project_root / 'scripts' / 'covid_demo.spec'

    print_header("Compilando Ejecutable con PyInstaller")

    # Verify spec file exists
    if not spec_file.exists():
        print_error(f"Archivo .spec no encontrado: {spec_file}")
        print_warning("Crea el archivo covid_demo.spec primero")
        return False

    # Clean previous builds FIRST (before preparing models)
    build_dir = project_root / 'build'
    dist_dir = project_root / 'dist'

    if build_dir.exists():
        print_info("Limpiando directorio build/")
        shutil.rmtree(build_dir)

    if dist_dir.exists():
        print_info("Limpiando directorio dist/")
        shutil.rmtree(dist_dir)

    # Prepare models (copy and rename to staging directory)
    print_info("Preparando modelos para empaquetado...")
    prepare_script = project_root / 'scripts' / 'prepare_models_for_build.py'
    try:
        subprocess.run([sys.executable, str(prepare_script)], check=True)
        print_success("Modelos preparados en staging directory")
    except subprocess.CalledProcessError as e:
        print_error(f"Falló la preparación de modelos: {e}")
        return False

    # Determine PyInstaller executable
    venv_path = project_root / '.venv_build'
    if platform.system() == 'Windows':
        pyinstaller_exe = venv_path / 'Scripts' / 'pyinstaller.exe'
    else:
        pyinstaller_exe = venv_path / 'bin' / 'pyinstaller'

    if not pyinstaller_exe.exists():
        print_error("PyInstaller no encontrado en el entorno virtual")
        print_warning("Ejecuta primero: python scripts/build_windows_exe.py --prepare")
        return False

    # Run PyInstaller
    print_info(f"Ejecutando PyInstaller con: {spec_file}")
    print_warning("Este proceso puede tomar 30-40 minutos...")

    try:
        subprocess.run(
            [str(pyinstaller_exe), str(spec_file), '--clean', '--noconfirm'],
            cwd=str(project_root),
            check=True
        )
        print_success("PyInstaller completado")
    except subprocess.CalledProcessError as e:
        print_error(f"PyInstaller falló con código {e.returncode}")
        return False

    # Verify output (platform-aware)
    exe_name = 'COVID19_Demo.exe' if platform.system() == 'Windows' else 'COVID19_Demo'
    exe_path = dist_dir / exe_name

    if not exe_path.exists():
        print_error(f"Ejecutable no encontrado en: {exe_path}")

        # Provide helpful guidance if building on wrong platform
        if platform.system() != 'Windows':
            print_warning("Estás en Linux/macOS - PyInstaller creó un ejecutable nativo (no .exe)")
            print_warning("Para crear un .exe de Windows, ejecuta el build en Windows")
            print_info("Opciones disponibles:")
            print_info("  1. Ejecutar build en Windows (VM, dual boot, o máquina física)")
            print_info("  2. Usar GitHub Actions con runner Windows")
            print_info("  3. Usar ejecutable Linux para demo (si aplica)")
            print_info("\nVer docs/BUILD_WINDOWS_STANDALONE.md para detalles completos")

        return False

    # Get file size
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print_success(f"Ejecutable creado: {exe_path}")
    print_info(f"Tamaño: {size_mb:.1f} MB")

    # Generate checksum
    print_info("Generando checksum SHA256...")
    sha256_hash = hashlib.sha256()
    with open(exe_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    checksum = sha256_hash.hexdigest()
    checksum_file = dist_dir / f'{exe_name}.sha256'
    with open(checksum_file, 'w') as f:
        f.write(f"{checksum}  {exe_name}\n")

    print_success(f"Checksum guardado: {checksum_file}")
    print_info(f"SHA256: {checksum}")

    return True


def clean_build_artifacts():
    """Clean all build artifacts."""
    project_root = get_project_root()

    print_header("Limpiando Artefactos de Build")

    dirs_to_clean = [
        project_root / 'build',
        project_root / 'dist',
        project_root / '.venv_build',
    ]

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print_info(f"Removiendo: {dir_path}")
            shutil.rmtree(dir_path)
            print_success(f"Removido: {dir_path}")
        else:
            print_warning(f"No existe: {dir_path}")

    print_success("Limpieza completada")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build Windows standalone executable for COVID-19 detection system'
    )
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='Create build environment and install dependencies'
    )
    parser.add_argument(
        '--build',
        action='store_true',
        help='Build executable using PyInstaller'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean build artifacts'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Do everything: prepare + verify + build'
    )

    args = parser.parse_args()

    # If no arguments, show help
    if not any([args.prepare, args.build, args.clean, args.all]):
        parser.print_help()
        return 0

    try:
        # Clean
        if args.clean:
            clean_build_artifacts()
            return 0

        # Prepare
        if args.prepare or args.all:
            # Verify models exist
            if not verify_models_exist():
                print_error("\nVerificación de modelos falló. Abortando.")
                return 1

            # Create build environment
            create_build_environment()
            print_success("\n✓ Preparación completada")

            if not args.all:
                print_info("\nAhora ejecuta: python scripts/build_windows_exe.py --build")

        # Build
        if args.build or args.all:
            if not verify_models_exist():
                print_error("\nVerificación de modelos falló. Abortando.")
                return 1

            if not build_executable():
                print_error("\nBuild falló. Revisa los errores arriba.")
                return 1

            print_success("\n✓ Build completado exitosamente")

            exe_name = 'COVID19_Demo.exe' if platform.system() == 'Windows' else 'COVID19_Demo'
            print_info(f"\nEjecutable disponible en: dist/{exe_name}")
            print_info(f"Checksum disponible en: dist/{exe_name}.sha256")

            if platform.system() != 'Windows':
                print_warning("\nNOTA: Este es un ejecutable Linux/macOS, no Windows .exe")
                print_info("Para crear .exe de Windows, ejecuta este script en Windows")

        return 0

    except KeyboardInterrupt:
        print_warning("\n\nInterrumpido por el usuario")
        return 130
    except Exception as e:
        print_error(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
