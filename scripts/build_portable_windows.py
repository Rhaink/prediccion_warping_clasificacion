#!/usr/bin/env python3
"""
Build script for creating a portable Windows package of the COVID-19 Detection System.

This script creates a standalone Windows distribution using Python Embeddable Package
that can be run on Windows 10/11 without requiring Python installation.

Usage:
    python scripts/build_portable_windows.py --version 1.0.0 --output build/releases

Features:
    - Downloads Python embeddable package (3.12.8)
    - Installs all dependencies (PyTorch CPU, Gradio, OpenCV, etc.)
    - Copies models and source code
    - Creates batch scripts for Windows
    - Generates user documentation
    - Packages everything into a ZIP file (~800 MB)
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
PYTHON_VERSION = "3.12.8"
PYTHON_EMBED_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
REQUIREMENTS_FILE = PROJECT_ROOT / "scripts" / "requirements_windows_portable.txt"

# Model mappings (source -> destination)
MODEL_MAPPINGS = {
    # Landmark models (4 models)
    "checkpoints/session10/ensemble/seed123/final_model.pt": "models/landmarks/resnet18_seed123_best.pt",
    "checkpoints/session13/seed321/final_model.pt": "models/landmarks/resnet18_seed321_best.pt",
    "checkpoints/repro_split111/session14/seed111/final_model.pt": "models/landmarks/resnet18_seed111_best.pt",
    "checkpoints/repro_split666/session16/seed666/final_model.pt": "models/landmarks/resnet18_seed666_best.pt",
    # Classifier model
    "outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt": "models/classifier/best_classifier.pt",
    # Shape analysis files
    "outputs/shape_analysis/canonical_shape_gpa.json": "models/shape_analysis/canonical_shape_gpa.json",
    "outputs/shape_analysis/canonical_delaunay_triangles.json": "models/shape_analysis/canonical_delaunay_triangles.json",
}


class PortableBuilder:
    """Builder for creating portable Windows package."""

    def __init__(self, version: str, output_dir: Path, verbose: bool = True):
        self.version = version
        self.output_dir = output_dir
        self.verbose = verbose
        self.staging_dir = output_dir / f"covid19-demo-v{version}-portable-windows"
        self.checksums: Dict[str, str] = {}

    def log(self, message: str, level: str = "INFO"):
        """Print log message."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def validate_environment(self) -> bool:
        """Validate that all required files exist."""
        self.log("Step 1/11: Validating environment...")

        # Check project structure
        required_paths = [
            PROJECT_ROOT / "src_v2",
            PROJECT_ROOT / "configs",
            PROJECT_ROOT / "scripts" / "run_demo.py",
            PROJECT_ROOT / "GROUND_TRUTH.json",
            REQUIREMENTS_FILE,
        ]

        missing = [p for p in required_paths if not p.exists()]
        if missing:
            self.log(f"Missing required paths: {missing}", "ERROR")
            return False

        # Check models exist
        missing_models = []
        for src, _ in MODEL_MAPPINGS.items():
            src_path = PROJECT_ROOT / src
            if not src_path.exists():
                missing_models.append(src)

        if missing_models:
            self.log(f"Missing model files: {missing_models}", "ERROR")
            return False

        # Check disk space (need ~2 GB for staging)
        try:
            stat = shutil.disk_usage(self.output_dir.parent)
            free_gb = stat.free / (1024**3)
            if free_gb < 3:
                self.log(f"Low disk space: {free_gb:.1f} GB (need 3 GB)", "WARNING")
        except Exception as e:
            self.log(f"Could not check disk space: {e}", "WARNING")

        self.log("Environment validation passed", "INFO")
        return True

    def download_embeddable_python(self) -> bool:
        """Download and extract Python embeddable package."""
        self.log("Step 2/11: Downloading Python embeddable package...")

        python_dir = self.staging_dir / "python"
        python_dir.mkdir(parents=True, exist_ok=True)

        zip_path = self.output_dir / f"python-{PYTHON_VERSION}-embed-amd64.zip"

        # Download if not cached
        if not zip_path.exists():
            self.log(f"Downloading from {PYTHON_EMBED_URL}...")
            try:
                with urllib.request.urlopen(PYTHON_EMBED_URL) as response:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    chunk_size = 8192

                    with open(zip_path, 'wb') as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0 and self.verbose:
                                pct = (downloaded / total_size) * 100
                                print(f"\rDownload progress: {pct:.1f}%", end='')
                if self.verbose:
                    print()  # New line after progress
            except Exception as e:
                self.log(f"Download failed: {e}", "ERROR")
                return False
        else:
            self.log("Using cached Python download")

        # Extract
        self.log("Extracting Python embeddable package...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(python_dir)
        except Exception as e:
            self.log(f"Extraction failed: {e}", "ERROR")
            return False

        # Verify python.exe exists
        if not (python_dir / "python.exe").exists():
            self.log("python.exe not found after extraction", "ERROR")
            return False

        self.log(f"Python embeddable extracted to {python_dir}")
        return True

    def configure_python_pth(self) -> bool:
        """Configure Python._pth to enable site-packages."""
        self.log("Step 3/11: Configuring Python path...")

        python_dir = self.staging_dir / "python"
        # Python 3.12.8 uses python312._pth (major.minor only)
        major_minor = '.'.join(PYTHON_VERSION.split('.')[:2])
        pth_file = python_dir / f"python{major_minor.replace('.', '')}._pth"

        if not pth_file.exists():
            self.log(f"PTH file not found: {pth_file}", "ERROR")
            return False

        # Read existing content
        original_content = pth_file.read_text()

        # Add site-packages and import site
        new_content = original_content.rstrip() + "\nLib/site-packages\nimport site\n"

        pth_file.write_text(new_content)
        self.log(f"Configured {pth_file.name} to enable site-packages")
        return True

    def bootstrap_pip(self) -> bool:
        """Download get-pip.py and pip wheel for Windows."""
        self.log("Step 4/11: Preparing pip for Windows...")

        # Download get-pip.py
        get_pip_path = self.staging_dir / "get-pip.py"
        if not get_pip_path.exists():
            self.log("Downloading get-pip.py...")
            try:
                urllib.request.urlretrieve(GET_PIP_URL, get_pip_path)
            except Exception as e:
                self.log(f"Failed to download get-pip.py: {e}", "ERROR")
                return False

        self.log("get-pip.py ready for first-run installation")
        return True

    def install_dependencies(self) -> bool:
        """Download Python dependencies and copy to package for first-run installation."""
        self.log("Step 5/11: Downloading dependencies (this takes ~10 minutes)...")

        if not REQUIREMENTS_FILE.exists():
            self.log(f"Requirements file not found: {REQUIREMENTS_FILE}", "ERROR")
            return False

        # Create wheels directory in staging (will be included in package)
        wheels_dir_staging = self.staging_dir / "wheels"
        wheels_dir_staging.mkdir(parents=True, exist_ok=True)

        # Temporary download location
        wheels_dir_temp = self.output_dir / "wheels_temp"
        wheels_dir_temp.mkdir(parents=True, exist_ok=True)

        # Download PyTorch separately (CPU version for Windows)
        self.log("Downloading PyTorch CPU for Windows...")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "torch==2.4.1",
                    "torchvision==0.19.1",
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu",
                    "--platform", "win_amd64",
                    "--python-version", "312",
                    "--only-binary", ":all:",
                    "--no-deps",  # No dependencies to avoid CUDA issues
                    "--dest", str(wheels_dir_temp),
                ],
                capture_output=not self.verbose,
                text=True,
            )
            if result.returncode != 0:
                self.log(f"PyTorch download failed: {result.stderr}", "ERROR")
                return False
        except Exception as e:
            self.log(f"Failed to download PyTorch: {e}", "ERROR")
            return False

        # Download other dependencies (without PyTorch)
        self.log("Downloading other packages...")
        other_packages = [
            "gradio>=6.0.0,<7.0.0",
            "opencv-python-headless>=4.8.0",
            "Pillow>=11.0.0",
            "numpy>=2.1.0,<3.0.0",
            "scipy>=1.14.0",
            "scikit-learn>=1.5.0",
            "matplotlib>=3.10.0",
            "pandas>=2.2.0",
            "reportlab>=4.2.0",
            "tqdm>=4.66.0",
        ]

        for package in other_packages:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        package,
                        "--platform", "win_amd64",
                        "--python-version", "312",
                        "--only-binary", ":all:",
                        "--dest", str(wheels_dir_temp),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    self.log(f"Warning: Failed to download {package}: {result.stderr}", "WARNING")
            except Exception as e:
                self.log(f"Warning: Failed to download {package}: {e}", "WARNING")

        # Download Windows-specific dependencies explicitly
        # These are packages with platform_system == "Windows" markers
        # that pip download doesn't capture when running on Linux
        self.log("Downloading Windows-specific dependencies...")
        windows_only_packages = [
            "colorama>=0.4.0",  # Required by: click, tqdm, uvicorn on Windows
        ]

        for package in windows_only_packages:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        package,
                        "--platform", "win_amd64",
                        "--python-version", "312",
                        "--only-binary", ":all:",
                        "--dest", str(wheels_dir_temp),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    self.log(f"  Downloaded {package}")
                else:
                    self.log(f"  Warning: Failed to download {package}", "WARNING")
            except Exception as e:
                self.log(f"  Warning: Failed to download {package}: {e}", "WARNING")

        # Copy wheels to staging directory
        wheel_files = list(wheels_dir_temp.glob("*.whl"))
        if not wheel_files:
            self.log("No wheel files found after download", "ERROR")
            return False

        self.log(f"Copying {len(wheel_files)} wheel packages to staging...")
        for wheel_file in wheel_files:
            shutil.copy2(wheel_file, wheels_dir_staging)

        # Create a simpler requirements file for Windows pip install
        win_requirements = self.staging_dir / "requirements.txt"
        win_requirements.write_text("# All dependencies will be installed from wheels/ directory\n")

        self.log(f"Dependencies prepared ({len(wheel_files)} packages, will install on first run)")
        return True

    def cleanup_unnecessary_files(self) -> bool:
        """Remove unnecessary files to reduce package size."""
        self.log("Cleaning up temporary files...")

        # Clean up temporary wheels directory
        wheels_temp = self.output_dir / "wheels_temp"
        if wheels_temp.exists():
            shutil.rmtree(wheels_temp, ignore_errors=True)
            self.log("Removed temporary wheels directory")

        return True

    def copy_models(self) -> bool:
        """Copy model files with simplified structure."""
        self.log("Step 6/11: Copying models...")

        total_size = 0
        for src, dst in MODEL_MAPPINGS.items():
            src_path = PROJECT_ROOT / src
            dst_path = self.staging_dir / dst

            # Create parent directory
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(src_path, dst_path)
            size_mb = dst_path.stat().st_size / (1024**2)
            total_size += size_mb

            # Compute checksum
            sha256 = hashlib.sha256()
            with open(dst_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            self.checksums[dst] = sha256.hexdigest()

            self.log(f"  Copied {dst_path.name} ({size_mb:.1f} MB)")

        self.log(f"Models copied successfully (total: {total_size:.1f} MB)")
        return True

    def copy_source_code(self) -> bool:
        """Copy source code and configuration files."""
        self.log("Step 7/11: Copying source code...")

        # Directories to copy
        dirs_to_copy = [
            "src_v2",
            "configs",
        ]

        for dirname in dirs_to_copy:
            src = PROJECT_ROOT / dirname
            dst = self.staging_dir / dirname
            if src.exists():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
                self.log(f"  Copied {dirname}/")

        # Individual files
        files_to_copy = [
            "GROUND_TRUTH.json",
            "scripts/run_demo.py",
        ]

        for filepath in files_to_copy:
            src = PROJECT_ROOT / filepath
            dst = self.staging_dir / filepath
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                self.log(f"  Copied {filepath}")

        self.log("Source code copied successfully")
        return True

    def create_install_script(self) -> bool:
        """Create Python script to install dependencies from wheels."""
        self.log("Step 8/11: Creating install_deps.py...")

        install_script = self.staging_dir / "install_deps.py"
        install_script.write_text('''#!/usr/bin/env python3
"""
Dependency installer for Windows portable package.
Installs all wheel packages from the wheels/ directory.
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Install all wheels from wheels/ directory."""
    script_dir = Path(__file__).parent
    wheels_dir = script_dir / "wheels"

    if not wheels_dir.exists():
        print("ERROR: wheels/ directory not found")
        return False

    wheels = sorted(wheels_dir.glob("*.whl"))
    if not wheels:
        print("ERROR: No .whl files found in wheels/")
        return False

    print(f"Installing {len(wheels)} packages...")
    print("=" * 70)
    print("This may take 2-3 minutes. Please wait...")
    print("=" * 70)
    print()

    # Install ALL wheels in a single pip call
    # This allows pip to properly resolve dependencies
    wheel_paths = [str(wheel) for wheel in wheels]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links=wheels",
            *wheel_paths,  # Pass all wheels at once
            "--no-warn-script-location",
        ],
        capture_output=False,  # Show pip output in real-time
        text=True,
    )

    print()
    print("=" * 70)

    if result.returncode != 0:
        print("ERROR: Installation failed")
        print("=" * 70)
        return False

    # Verify critical packages
    print("Verifying installation...")
    critical_packages = ["torch", "torchvision", "gradio", "numpy", "cv2"]
    missing = []

    for package in critical_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package}")
            missing.append(package)

    print("=" * 70)

    if missing:
        print(f"ERROR: {len(missing)} critical packages failed:")
        for pkg in missing:
            print(f"  - {pkg}")
        return False

    print(f"SUCCESS: All {len(wheels)} packages installed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
''', encoding='utf-8')

        self.log("install_deps.py created successfully")
        return True

    def create_batch_files(self) -> bool:
        """Create Windows batch files."""
        self.log("Step 9/11: Creating batch files...")

        # RUN_DEMO.bat
        run_demo_bat = self.staging_dir / "RUN_DEMO.bat"
        run_demo_bat.write_text(self._get_run_demo_template(), encoding='utf-8')
        self.log("  Created RUN_DEMO.bat")

        # INSTALL.bat
        install_bat = self.staging_dir / "INSTALL.bat"
        install_bat.write_text(self._get_install_template(), encoding='utf-8')
        self.log("  Created INSTALL.bat")

        self.log("Batch files created successfully")
        return True

    def create_documentation(self) -> bool:
        """Create user documentation files."""
        self.log("Step 10/11: Creating documentation...")

        # README.txt
        readme = self.staging_dir / "README.txt"
        readme.write_text(self._get_readme_template(), encoding='utf-8')
        self.log("  Created README.txt")

        # VERSION.txt
        version_info = {
            "version": self.version,
            "build_date": datetime.now().isoformat(),
            "python_version": PYTHON_VERSION,
            "model_checksums": self.checksums,
        }
        version_file = self.staging_dir / "VERSION.txt"
        version_file.write_text(json.dumps(version_info, indent=2), encoding='utf-8')
        self.log("  Created VERSION.txt")

        self.log("Documentation created successfully")
        return True

    def create_zip_package(self) -> Path:
        """Create final ZIP package."""
        self.log("Step 10/11: Creating ZIP package...")

        zip_path = self.output_dir / f"covid19-demo-v{self.version}-portable-windows.zip"

        # Remove existing ZIP if present
        if zip_path.exists():
            zip_path.unlink()

        self.log("Compressing files (this may take a few minutes)...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for file_path in self.staging_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.staging_dir.parent)
                    zf.write(file_path, arcname)

        size_mb = zip_path.stat().st_size / (1024**2)
        self.log(f"ZIP package created: {zip_path.name} ({size_mb:.1f} MB)")
        return zip_path

    def verify_integrity(self, zip_path: Path) -> bool:
        """Verify package integrity."""
        self.log("Step 11/11: Verifying package integrity...")

        # Check ZIP is valid
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = zf.namelist()
                if len(file_list) == 0:
                    self.log("ZIP file is empty", "ERROR")
                    return False
        except Exception as e:
            self.log(f"ZIP verification failed: {e}", "ERROR")
            return False

        # Check critical files exist in ZIP
        critical_files = [
            f"covid19-demo-v{self.version}-portable-windows/python/python.exe",
            f"covid19-demo-v{self.version}-portable-windows/RUN_DEMO.bat",
            f"covid19-demo-v{self.version}-portable-windows/models/classifier/best_classifier.pt",
        ]

        missing = [f for f in critical_files if f not in file_list]
        if missing:
            self.log(f"Missing critical files in ZIP: {missing}", "ERROR")
            return False

        self.log(f"Package integrity verified ({len(file_list)} files in ZIP)")
        return True

    def build(self) -> bool:
        """Execute full build process."""
        self.log(f"Starting build for version {self.version}")
        self.log(f"Output directory: {self.output_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Execute build steps
        steps = [
            self.validate_environment,
            self.download_embeddable_python,
            self.configure_python_pth,
            self.bootstrap_pip,
            self.install_dependencies,
            self.cleanup_unnecessary_files,
            self.copy_models,
            self.copy_source_code,
            self.create_install_script,
            self.create_batch_files,
            self.create_documentation,
        ]

        for step in steps:
            if not step():
                self.log(f"Build failed at step: {step.__name__}", "ERROR")
                return False

        # Create ZIP and verify
        zip_path = self.create_zip_package()
        if not self.verify_integrity(zip_path):
            return False

        self.log("=" * 70)
        self.log("BUILD SUCCESSFUL!", "INFO")
        self.log(f"Package: {zip_path}")
        self.log(f"Size: {zip_path.stat().st_size / (1024**2):.1f} MB")
        self.log("=" * 70)
        return True

    def _get_run_demo_template(self) -> str:
        """Get RUN_DEMO.bat template."""
        return f"""@echo off
title COVID-19 Detection System - Demo
color 0A

echo ================================================================
echo   COVID-19 Detection System v{self.version}
echo   Portable Windows Edition
echo ================================================================
echo.

REM Verify Python exists
if not exist python\\python.exe (
    echo ERROR: Python not found. Please extract the complete ZIP file.
    echo.
    pause
    exit /b 1
)

REM Check if dependencies are installed
if not exist python\\Lib\\site-packages\\gradio (
    echo ================================================================
    echo   First-time setup: Installing dependencies...
    echo   This takes 2-3 minutes and only happens once.
    echo   Please wait while pip resolves dependencies...
    echo ================================================================
    echo.

    REM Install pip first
    echo [1/2] Installing pip...
    python\\python.exe get-pip.py --no-warn-script-location >nul 2>&1

    REM Install dependencies using Python script
    echo [2/2] Installing packages from wheels...
    python\\python.exe install_deps.py

    if errorlevel 1 (
        echo.
        echo ERROR: Dependency installation failed.
        echo Please run INSTALL.bat for diagnostics.
        pause
        exit /b 1
    )

    echo.
    echo Installation complete!
    echo.
    timeout /t 2 >nul
)

REM Configure environment
set PYTHONPATH=%cd%
set COVID_DEMO_MODELS_DIR=%cd%\\models
set COVID_DEMO_FROZEN=0

REM Start application
echo Starting Gradio server...
echo Browser will open automatically at http://localhost:7860
echo.
echo Press Ctrl+C to stop the application.
echo.

python\\python.exe scripts\\run_demo.py

REM Handle errors
if errorlevel 1 (
    echo.
    echo ERROR: The application terminated with errors.
    echo See messages above for diagnostics.
    echo.
    pause
)
"""

    def _get_install_template(self) -> str:
        """Get INSTALL.bat template."""
        return f"""@echo off
echo ================================================================
echo   COVID-19 Detection System - Installation Verification
echo ================================================================
echo.

REM Verify Python
echo [1/4] Checking Python embeddable...
if not exist python\\python.exe (
    echo   [FAIL] Python embeddable not found
    echo   Solution: Extract the complete ZIP file
    pause
    exit /b 1
)
echo   [OK] Python found

REM Verify wheels directory
echo [2/4] Checking installation files...
if not exist wheels (
    echo   [FAIL] Wheels directory not found
    echo   Solution: Re-extract the complete ZIP file
    pause
    exit /b 1
)
echo   [OK] Installation files present

REM Check if dependencies are installed
echo [3/4] Checking dependencies...
python\\python.exe -c "import sys; import gradio; import torch; import cv2; print('All packages OK')" >nul 2>&1
if errorlevel 1 (
    echo   [WARN] Dependencies not installed yet
    echo   Solution: Run RUN_DEMO.bat - it will install automatically
    echo.
    echo   Installing now? (This takes 2-3 minutes, pip will resolve dependencies)
    pause

    echo   Installing pip...
    python\\python.exe get-pip.py --no-warn-script-location >nul 2>&1

    echo   Installing packages (please wait)...
    python\\python.exe install_deps.py

    if errorlevel 1 (
        echo   [FAIL] Installation failed
        echo   Try running RUN_DEMO.bat instead
        pause
        exit /b 1
    )
    echo   [OK] Dependencies installed successfully
) else (
    echo   [OK] All dependencies installed
)

REM Verify models
echo [4/4] Checking model files...
python\\python.exe -c "from pathlib import Path; models=['models/landmarks/resnet18_seed123_best.pt','models/classifier/best_classifier.pt']; exit(1 if any(not Path(m).exists() for m in models) else 0)" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Model files missing
    echo   Solution: Re-extract the ZIP file
    pause
    exit /b 1
)
echo   [OK] All models present

echo.
echo ================================================================
echo   Verification complete! Installation is ready.
echo.
echo   To run the application: Double-click RUN_DEMO.bat
echo ================================================================
pause
"""

    def _get_readme_template(self) -> str:
        """Get README.txt template."""
        return f"""=================================================================
  COVID-19 Detection System - Portable Windows Package
=================================================================

VERSION: {self.version}
BUILD DATE: {datetime.now().strftime('%Y-%m-%d')}
PYTHON VERSION: {PYTHON_VERSION}

VALIDATED RESULTS
==================
- Landmark Detection Error: 3.61 ± 2.48 px (ensemble of 4 models)
- Classification Accuracy: 98.05%
- F1-Score: 97.12% (macro-averaged)

SYSTEM REQUIREMENTS
===================
- Windows 10/11 (64-bit)
- 4 GB RAM minimum, 8 GB recommended
- 2 GB disk space
- NO Python installation required
- NO administrator privileges required
- NO internet connection required

QUICK START
===========
1. Extract this ZIP file to a folder (e.g., C:\\covid19-demo)
2. (Optional) Run INSTALL.bat to verify installation
3. Double-click RUN_DEMO.bat
4. The web interface will open at http://localhost:7860

USING THE INTERFACE
===================
1. Load Image: Drag and drop a chest X-ray or use example images
2. Process: Click "Procesar Imagen" button
3. Results: View the 4-stage analysis
   - Original Image
   - Landmarks (15 anatomical points)
   - Warped (geometrically normalized)
   - GradCAM (explainability visualization)
   - Classification (COVID/Normal/Viral Pneumonia)

4. Export: Click "Exportar PDF" to save results

TROUBLESHOOTING
===============

"Port 7860 is already in use"
→ Edit RUN_DEMO.bat and add at the end: --server-port 8080

"Error loading models"
→ Run INSTALL.bat for diagnostics
→ Verify models/ folder is 227 MB

"Browser does not open automatically"
→ Open manually: http://localhost:7860

"Processing is slow"
→ This package uses CPU (1-2 seconds/image is normal)
→ For GPU: requires native PyTorch installation

"Access denied / Antivirus blocking"
→ Add folder to antivirus exceptions
→ This is a false positive (source code is visible)

ARCHITECTURE
============
Three-stage pipeline:
1. Landmark Detection: Ensemble of 4× ResNet-18 models
2. Geometric Normalization: Piecewise affine warping (18 Delaunay triangles)
3. Classification: ResNet-18 + GradCAM explainability

Technical Details:
- 15 lung contour landmarks (5 symmetric pairs + 5 central points)
- Generalized Procrustes Analysis for canonical shape
- Test-Time Augmentation (horizontal flip)
- CLAHE preprocessing (tile_size=4, clip_limit=2.0)
- Wing Loss for landmark regression

DATASET
=======
Trained on COVID-19 Radiography Database:
- 3,616 COVID-19 cases
- 10,192 Normal cases
- 1,345 Viral Pneumonia cases
- Source: Kaggle (CC BY 4.0)

CITATION
========
If you use this system in research, please cite:
[Your thesis/publication information]

CONTACT
=======
For issues or questions:
- GitHub: [Your repository URL]
- Email: [Your email]

=================================================================
"""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build portable Windows package for COVID-19 Detection System"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Version string (e.g., 1.0.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "build" / "releases",
        help="Output directory for build artifacts",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Create builder and run
    builder = PortableBuilder(
        version=args.version,
        output_dir=args.output,
        verbose=args.verbose,
    )

    success = builder.build()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
