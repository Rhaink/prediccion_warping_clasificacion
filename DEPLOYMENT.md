# Deployment Guide - COVID-19 Detection System

This guide explains how to create and deploy the portable Windows package for the COVID-19 Detection System.

## Table of Contents

1. [Overview](#overview)
2. [Building the Package](#building-the-package)
3. [Package Structure](#package-structure)
4. [Deployment for End Users](#deployment-for-end-users)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [Version History](#version-history)

---

## Overview

### What is the Portable Package?

The portable Windows package is a self-contained distribution that includes:
- Python 3.12.8 embeddable (no installation required)
- All dependencies (PyTorch CPU, Gradio, OpenCV, etc.)
- Trained models (landmarks + classifier)
- Source code and configuration
- Batch scripts for easy execution

**Key Features:**
- ✅ Built entirely from Linux (no Windows VM required)
- ✅ No Python installation needed on target Windows machine
- ✅ No administrator privileges required
- ✅ No internet connection required for execution
- ✅ Fast startup (2-3 seconds)
- ✅ Portable (~800 MB compressed)

### Why Not PyInstaller?

The portable Python embeddable approach was chosen over PyInstaller because:

| Feature | Python Embeddable | PyInstaller |
|---------|------------------|-------------|
| Cross-compilation from Linux | ✅ Yes | ❌ No (requires Windows) |
| Build time | ~15 minutes | 1+ hour |
| Package size | ~800 MB | ~1.8 GB |
| Startup time | 2-3 seconds | 10-30 seconds |
| Debugging | ✅ Source visible | ❌ Binary opaque |
| Antivirus issues | ✅ Rare | ⚠️ Common false positives |
| Updates | ✅ Replace files | ❌ Rebuild required |

---

## Building the Package

### Prerequisites

**On your Linux development machine:**
- Python 3.8+ installed
- Internet connection (to download Python embeddable)
- ~3 GB free disk space
- All trained models in place (checkpoints/, outputs/)

**Verify models exist:**
```bash
python -c "
from pathlib import Path

models = [
    'checkpoints/session10/ensemble/seed123/final_model.pt',
    'checkpoints/session13/seed321/final_model.pt',
    'checkpoints/repro_split111/session14/seed111/final_model.pt',
    'checkpoints/repro_split666/session16/seed666/final_model.pt',
    'outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt',
    'outputs/shape_analysis/canonical_shape_gpa.json',
    'outputs/shape_analysis/canonical_delaunay_triangles.json',
]

missing = [m for m in models if not Path(m).exists()]
if missing:
    print('❌ Missing models:', missing)
else:
    print('✅ All models present')
"
```

### Build Steps

**1. Navigate to project root:**
```bash
cd /home/donrobot/Projects/prediccion_warping_clasificacion
```

**2. Run build script:**
```bash
python scripts/build_portable_windows.py --version 1.0.0 --output build/releases
```

**Build process overview:**
```
Step 1/11: Validate environment (files, models, disk space)
Step 2/11: Download Python 3.12.8 embeddable (~25 MB)
Step 3/11: Configure Python path (enable site-packages)
Step 4/11: Bootstrap pip
Step 5/11: Install dependencies (~10 minutes, ~350 MB)
Step 6/11: Cleanup unnecessary files
Step 7/11: Copy models (227 MB)
Step 8/11: Copy source code
Step 9/11: Create batch files (RUN_DEMO.bat, INSTALL.bat)
Step 10/11: Create documentation (README.txt, VERSION.txt)
Step 11/11: Create ZIP package (~800 MB compressed)
```

**Total time:** ~15 minutes

**Output:**
```
build/releases/covid19-demo-v1.0.0-portable-windows.zip (~800 MB)
```

### Build Options

```bash
# Custom version
python scripts/build_portable_windows.py --version 1.2.0

# Custom output directory
python scripts/build_portable_windows.py --output /path/to/releases

# Quiet mode (errors only)
python scripts/build_portable_windows.py --no-verbose
```

### Build Artifacts

The build process creates:
```
build/
├── releases/
│   └── covid19-demo-v1.0.0-portable-windows.zip  # Final package
│
└── covid19-demo-v1.0.0-portable-windows/         # Staging directory
    ├── python/                                    # Python embeddable
    ├── models/                                    # ML models
    ├── src_v2/                                    # Source code
    ├── configs/                                   # Configuration files
    ├── scripts/                                   # Helper scripts
    ├── RUN_DEMO.bat                              # Main launcher
    ├── INSTALL.bat                                # Verification script
    ├── README.txt                                 # User manual
    ├── VERSION.txt                                # Build metadata
    └── GROUND_TRUTH.json                         # Validated metrics
```

---

## Package Structure

### Final Package Contents

```
covid19-demo-v1.0.0-portable-windows/
│
├── python/                          # Python 3.12.8 embeddable (~375 MB)
│   ├── python.exe                   # Python interpreter
│   ├── python312.dll                # Core runtime
│   ├── python312._pth               # Path configuration
│   └── Lib/
│       └── site-packages/           # All dependencies
│           ├── torch/               # PyTorch CPU (~150 MB)
│           ├── gradio/              # Web UI (~50 MB)
│           ├── cv2/                 # OpenCV (~40 MB)
│           └── ...                  # Other packages
│
├── models/                          # Trained models (~227 MB)
│   ├── landmarks/                   # Landmark detection models
│   │   ├── resnet18_seed123_best.pt (45 MB)
│   │   ├── resnet18_seed321_best.pt (45 MB)
│   │   ├── resnet18_seed111_best.pt (45 MB)
│   │   └── resnet18_seed666_best.pt (45 MB)
│   │
│   ├── classifier/                  # Classification model
│   │   └── best_classifier.pt       (43 MB)
│   │
│   └── shape_analysis/              # Canonical shape data
│       ├── canonical_shape_gpa.json
│       └── canonical_delaunay_triangles.json
│
├── src_v2/                          # Source code (~1.5 MB)
│   ├── models/                      # Model architectures
│   ├── processing/                  # GPA, warping
│   ├── evaluation/                  # Metrics, GradCAM
│   ├── data/                        # Datasets, transforms
│   └── gui/                         # Gradio interface
│
├── configs/                         # Configuration files
│   ├── ensemble_best.json
│   └── warping_best.json
│
├── scripts/
│   └── run_demo.py                  # Demo launcher script
│
├── RUN_DEMO.bat                     # ⭐ Main launcher (double-click)
├── INSTALL.bat                      # Verification script
├── README.txt                       # User manual (English/Spanish)
├── VERSION.txt                      # Build metadata + checksums
└── GROUND_TRUTH.json               # Validated metrics
```

### Total Package Size

- **Uncompressed:** ~1.2 GB
- **Compressed (ZIP):** ~800 MB

---

## Deployment for End Users

### Distribution

**1. Upload the ZIP file:**
- Google Drive / Dropbox / OneDrive
- University file server
- GitHub Releases (if < 2 GB)
- FTP/HTTP server

**2. Share the link with instructions:**

```
# INSTRUCCIONES DE USO - Sistema de Detección COVID-19

1. Descargar el archivo:
   covid19-demo-v1.0.0-portable-windows.zip (~800 MB)

2. Extraer el ZIP completo a una carpeta:
   Ejemplo: C:\covid19-demo

3. (Opcional) Ejecutar INSTALL.bat para verificar la instalación

4. Doble clic en RUN_DEMO.bat para iniciar

5. El navegador se abrirá automáticamente en:
   http://localhost:7860

6. Para detener: Presionar Ctrl+C en la ventana de comandos
```

### End User Workflow

**Step 1: Extract**
```powershell
# On Windows, right-click the ZIP and select "Extract All..."
# Or use PowerShell:
Expand-Archive -Path covid19-demo-v1.0.0-portable-windows.zip -DestinationPath C:\covid19-demo
```

**Step 2: Verify (Optional)**
```batch
# Double-click INSTALL.bat
# Expected output:
[1/3] Checking Python embeddable... [OK]
[2/3] Checking dependencies... [OK]
[3/3] Checking model files... [OK]

Verification complete! Installation is ready.
```

**Step 3: Run**
```batch
# Double-click RUN_DEMO.bat
# Browser opens automatically at http://localhost:7860
```

**Step 4: Use Interface**
1. Upload a chest X-ray image (PNG/JPG)
2. Click "Procesar Imagen"
3. View results:
   - Original image
   - Landmarks (15 points)
   - Warped (normalized)
   - GradCAM (explainability)
   - Classification (COVID/Normal/Viral)
4. Export to PDF if needed

**Step 5: Stop**
- Press Ctrl+C in the command window
- Or simply close the window

---

## Testing

### Pre-Distribution Testing Checklist

**On Linux (build verification):**
- [ ] Build completes without errors
- [ ] ZIP file is created (~800 MB)
- [ ] ZIP contains all expected files
- [ ] Model checksums match VERSION.txt

```bash
# Verify ZIP structure
unzip -l build/releases/covid19-demo-v1.0.0-portable-windows.zip | head -30

# Verify size
du -sh build/releases/covid19-demo-v1.0.0-portable-windows.zip

# Verify checksums
cat build/releases/covid19-demo-v1.0.0-portable-windows/VERSION.txt
```

**On Windows (functional testing):**
- [ ] Extract ZIP without errors
- [ ] INSTALL.bat passes all checks
- [ ] RUN_DEMO.bat starts without errors
- [ ] Browser opens automatically
- [ ] Interface loads correctly
- [ ] Upload image works
- [ ] Processing completes in < 3 seconds
- [ ] All 4 visualizations appear
- [ ] Export PDF works
- [ ] Ctrl+C stops cleanly

### Testing Without Windows Machine

**Option 1: Wine (Linux)**
```bash
# Install Wine
sudo apt install wine64

# Extract and test
cd build/releases
unzip covid19-demo-v1.0.0-portable-windows.zip
cd covid19-demo-v1.0.0-portable-windows
wine RUN_DEMO.bat
```

**Option 2: Windows VM (temporary)**
- Use VirtualBox + Windows 10 evaluation ISO (free, 90 days)
- Test installation and basic functionality
- Delete VM after testing

**Option 3: Remote testing**
- Send ZIP to colleague with Windows machine
- Video call for 15-minute test session
- Use checklist above

---

## Troubleshooting

### Build Issues

**Problem: "Missing model files"**
```
Solution: Verify models exist with verification script above
```

**Problem: "Download failed: HTTP Error 404"**
```
Solution: Check Python version URL is correct
         Verify internet connection
         Try manual download: https://www.python.org/downloads/
```

**Problem: "Low disk space"**
```
Solution: Need 3 GB free for staging + final package
         Clean up old builds: rm -rf build/releases/*
```

**Problem: "pip installation failed"**
```
Solution: Check internet connection
         Check firewall settings
         Try manual pip bootstrap
```

### Runtime Issues (Windows)

**Problem: "Port 7860 already in use"**
```batch
Solution: Edit RUN_DEMO.bat, add at the end:
         python\python.exe scripts\run_demo.py --server-port 8080
```

**Problem: "Python embeddable not found"**
```
Solution: Extract the complete ZIP file
         Do not move files individually
```

**Problem: "Model files missing"**
```
Solution: Re-extract the ZIP (may be corrupted)
         Verify models/ folder is 227 MB
```

**Problem: "Access denied / Antivirus blocking"**
```
Solution: Add folder to antivirus exceptions
         Windows Defender may flag python.exe (false positive)
```

**Problem: "Browser doesn't open automatically"**
```
Solution: Open manually: http://localhost:7860
         Check firewall isn't blocking localhost
```

### Performance Issues

**Problem: "Processing is slow (> 5 seconds)"**
```
Status: Normal for CPU-only package
        1-2 seconds/image is expected
        For faster processing, use GPU version (requires PyTorch GPU install)
```

**Problem: "High memory usage"**
```
Status: Normal (models load ~1.5 GB RAM)
        Ensure 4 GB+ RAM available
        Close other applications if needed
```

---

## Version History

### v1.0.0 (2026-01-19)

**Initial release**

**Features:**
- Python 3.12.8 embeddable
- PyTorch 2.4.1+cpu
- Gradio 6.x web interface
- 4 landmark detection models (ensemble)
- 1 classification model
- GradCAM explainability

**Validated Metrics:**
- Landmark error: 3.61 ± 2.48 px
- Classification accuracy: 98.05%
- F1-Score: 97.12% (macro)

**Models:**
- resnet18_seed123_best.pt (45 MB)
- resnet18_seed321_best.pt (45 MB)
- resnet18_seed111_best.pt (45 MB)
- resnet18_seed666_best.pt (45 MB)
- best_classifier.pt (43 MB)

**Package Size:** ~800 MB compressed

**Known Limitations:**
- CPU-only (no GPU support)
- Windows 10/11 only
- English/Spanish UI only

---

## Technical Details: Dependency Management

### The --no-deps Problem and Solution (v1.0.5+)

#### Background

When building a Windows portable package from Linux, we face a critical challenge with PyTorch dependencies:

**The Problem:**
- PyTorch CPU for Windows has a dependency on `pytorch-triton-rocm`
- This package is Linux x86_64 only (500+ MB)
- It cannot be installed on Windows, causing build failures

**Initial Solution (v1.0.3-1.0.4):**
- Use `pip download torch --no-deps` to avoid triton
- **Side Effect:** This also removes CRITICAL cross-platform dependencies:
  - `sympy` (torch.fx symbolic computation)
  - `mpmath` (sympy backend)
  - `networkx` (torch graph operations)
  - `filelock`, `fsspec`, `setuptools` (torch utilities)

**Result:** Package failed on Windows with `ModuleNotFoundError: No module named 'sympy'`

#### The Hybrid Strategy (v1.0.5+)

**File:** `scripts/requirements_windows_full.txt`

This file is the **single source of truth** for all Windows dependencies. It includes:

1. **Torch dependencies lost by --no-deps:**
   ```txt
   sympy>=1.12              # CRITICAL: torch.fx symbolic computation
   mpmath>=1.1.0,<1.4       # CRITICAL: sympy backend
   networkx>=2.8            # CRITICAL: torch graph operations
   filelock>=3.0
   fsspec>=2022.1.0
   setuptools>=60.0
   typing-extensions>=4.8
   jinja2>=3.0
   ```

2. **Windows-specific packages (environment marker workaround):**
   ```txt
   colorama>=0.4.0          # platform_system == 'Windows' not evaluated on Linux
   python-dateutil>=2.8
   pytz>=2023.3
   ```

3. **Full Gradio stack:**
   - CLI framework (typer, click, rich)
   - HTTP server (uvicorn, starlette, httpx)
   - WebSocket support (websockets)
   - Utilities (watchfiles, aiofiles)

4. **Scientific computing:**
   - numpy, scipy, scikit-learn, pandas
   - opencv-python-headless, Pillow
   - matplotlib, reportlab

**Build Process:**

```python
# Step 1: Download torch/torchvision with --no-deps (avoid triton)
pip download torch==2.4.1 torchvision==0.19.1 --no-deps --platform win_amd64 ...

# Step 2: Download ALL other deps from requirements_windows_full.txt
pip download -r requirements_windows_full.txt --platform win_amd64 ...

# Step 3: Remove triton if accidentally downloaded
rm *triton*rocm*.whl

# Step 4: Generate MANIFEST.txt for validation
ls *.whl > MANIFEST.txt
```

**Validation (install_deps.py):**

The package includes automated validation of critical imports:

```python
critical_packages = [
    "torch", "torchvision", "gradio", "numpy", "cv2",
    "sympy",      # torch.fx (would fail without this)
    "networkx",   # torch graphs (would fail without this)
    "click",      # CLI framework
]
```

#### Why This Approach Works

1. **Avoids platform-specific packages:** `--no-deps` only on torch prevents Linux-only triton
2. **Includes all cross-platform deps:** requirements.txt pulls everything else
3. **Handles environment markers:** Explicit inclusion of Windows-specific packages
4. **Automated validation:** install_deps.py catches missing packages immediately
5. **Maintainable:** Single requirements file instead of hardcoded lists

#### Package Size Impact

| Version | Strategy | Wheels Count | Total Size | Status |
|---------|----------|--------------|------------|--------|
| v1.0.3 | Hardcoded list | 66 | 562 MB | ❌ Failed (colorama) |
| v1.0.4 | List + colorama | 67 | 562 MB | ❌ Failed (sympy) |
| **v1.0.5** | **requirements.txt** | **73+** | **~600 MB** | ✅ **Functional** |

**Size increase:** +40 MB (~7% increase)
**Value:** Complete functionality, no runtime errors

#### Common Issues

**Q: Why not just use `pip download -r requirements.txt` for everything?**
A: Because it would download pytorch-triton-rocm (Linux-only, 500+ MB), making the package unusable on Windows.

**Q: Why not use `--platform win_amd64` to filter Linux packages?**
A: pip doesn't filter by platform when using `--platform` flag with dependencies that have platform-specific markers.

**Q: How do I add a new dependency?**
A: Add it to `scripts/requirements_windows_full.txt` and rebuild. The build process will automatically include it.

**Q: How do I verify all dependencies are included?**
A: Check `MANIFEST.txt` in the package or run `INSTALL.bat` on Windows to validate.

---

## Advanced Topics

### Customizing the Build

**Change Python version:**
```python
# Edit scripts/build_portable_windows.py
PYTHON_VERSION = "3.11.9"  # Use any available version
```

**Change dependencies:**
```bash
# Edit scripts/requirements_windows_portable.txt
# Add/remove packages as needed
# Rebuild package
```

**Reduce package size:**
```python
# In build_portable_windows.py, enhance cleanup_unnecessary_files():
def cleanup_unnecessary_files(self):
    # Remove more test files, docs, examples
    patterns_to_remove += [
        'Lib/site-packages/*/tests',
        'Lib/site-packages/*/docs',
        'Lib/site-packages/matplotlib/mpl-data/sample_data',
    ]
```

**Compress with 7z (better compression):**
```bash
# After build completes:
cd build/releases
7z a -t7z -m0=lzma2 -mx=9 covid19-demo-v1.0.0.7z covid19-demo-v1.0.0-portable-windows/
# Result: ~30% smaller than ZIP
```

### GPU Support (Advanced)

To create a GPU-enabled package (requires Windows):

1. Replace CPU PyTorch wheels with CUDA wheels in requirements
2. Build on Windows machine with CUDA installed
3. Package size will increase to ~2.5 GB (CUDA libraries)

**Not recommended for portable distribution** (too large).

### Automated CI/CD

**GitHub Actions example:**
```yaml
name: Build Portable Package

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Build package
        run: |
          python scripts/build_portable_windows.py --version ${{ github.ref_name }}

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: portable-windows
          path: build/releases/*.zip

      - name: Create release
        uses: softprops/action-gh-release@v1
        with:
          files: build/releases/*.zip
```

---

## Support

For issues or questions:
- GitHub: [Your repository URL]
- Email: [Your email]
- Documentation: See docs/ folder

---

## License

[Add your license information here]

---

**Last updated:** 2026-01-20
**Build script version:** 1.0.5 (dependency management fix)
**Tested on:** Windows 10 (21H2), Windows 11 (23H2)
