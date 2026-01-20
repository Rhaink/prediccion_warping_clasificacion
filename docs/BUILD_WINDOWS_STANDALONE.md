# Building Windows Standalone Executable

**Target**: Create a single-file Windows executable (`.exe`) for thesis defense demonstrations.

**Goal**: Enable non-technical users (thesis committee members) to run the COVID-19 detection system on any Windows laptop without Python or dependencies installed.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Build Process](#detailed-build-process)
- [Architecture](#architecture)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Distribution](#distribution)
- [Technical Details](#technical-details)

---

## Overview

### What This Build Creates

- **Executable**: `COVID19_Demo.exe` (~1.8 GB)
- **Platform**: Windows 10/11 (64-bit)
- **Mode**: CPU-only (no GPU required)
- **Dependencies**: All bundled (PyTorch, Gradio, OpenCV, models)
- **Internet**: Not required (fully offline)
- **Installation**: None (double-click to run)

### Build Method

- **Tool**: PyInstaller 6.3.0
- **Mode**: `--onefile` (single executable)
- **Compression**: UPX enabled
- **Python**: 3.11 (embedded in executable)
- **PyTorch**: CPU-only build (reduces size from 3GB to 1.8GB)

### Build Time

- **Preparation**: ~10 minutes (one-time setup)
- **Build**: ~30-40 minutes (CPU-dependent)
- **Total**: ~1 hour (first time)

---

## Prerequisites

### Required Files

Before building, ensure all models and assets exist:

```bash
# Landmark models (4 models, ~180 MB total)
checkpoints/landmarks/resnet18_seed123_best.pt
checkpoints/landmarks/resnet18_seed321_best.pt
checkpoints/landmarks/resnet18_seed111_best.pt
checkpoints/landmarks/resnet18_seed666_best.pt

# Classifier model (~45 MB)
outputs/classifier_warped_lung_best/best_classifier.pt

# Geometric data (~2 MB)
outputs/shape_analysis/canonical_shape_gpa.json
outputs/shape_analysis/canonical_delaunay_triangles.json

# Ground truth metrics
GROUND_TRUTH.json
```

**Total models size**: ~227 MB

If models are missing, train them first:
```bash
# See docs/REPRO_FULL_PIPELINE.md for training instructions
python -m src_v2 train-landmark --config configs/landmarks_train_base.json
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv
```

### System Requirements (Build Machine)

- **OS**: Windows 10/11 or Linux (cross-compilation not tested)
- **Python**: 3.8+ (3.11 recommended)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk**: 10 GB free space
- **Internet**: Required for downloading dependencies during build

---

## Quick Start

### One-Command Build

```bash
# Install build dependencies and create executable
python scripts/build_windows_exe.py --all
```

This will:
1. Create clean virtual environment (`.venv_build`)
2. Install CPU-only PyTorch and dependencies
3. Verify all models exist
4. Run PyInstaller with optimized settings
5. Generate `dist/COVID19_Demo.exe`

**Output**:
```
dist/COVID19_Demo.exe           (~1.8 GB)
dist/COVID19_Demo.exe.sha256    (checksum)
```

### Manual Build (Step-by-Step)

```bash
# 1. Prepare build environment
python scripts/build_windows_exe.py --prepare

# 2. Build executable
python scripts/build_windows_exe.py --build

# 3. Verify output
ls -lh dist/COVID19_Demo.exe

# 4. Test locally
./dist/COVID19_Demo.exe
```

---

## Detailed Build Process

### Phase 1: Environment Preparation

**Script**: `scripts/build_windows_exe.py --prepare`

**Actions**:
1. Removes old build environment (if exists)
2. Creates fresh virtual environment: `.venv_build/`
3. Installs dependencies from `scripts/requirements_windows_cpu.txt`

**Key dependencies**:
```txt
torch==2.4.1+cpu         (~500 MB)
torchvision==0.19.1+cpu  (~100 MB)
gradio==6.3.0            (~300 MB)
opencv-python-headless   (~50 MB)
pyinstaller==6.3.0       (~5 MB)
```

**Duration**: ~5-10 minutes (internet speed dependent)

### Phase 2: PyInstaller Configuration

**Spec file**: `scripts/covid_demo.spec`

**Key configuration**:

```python
# Entry point
a = Analysis(
    ['scripts/run_demo.py'],
    ...
)

# Data files to bundle
datas = [
    ('checkpoints/landmarks/*.pt', 'models/landmarks'),
    ('outputs/classifier_warped_lung_best/best_classifier.pt', 'models/classifier'),
    ('outputs/shape_analysis/*.json', 'models/shape_analysis'),
    ('src_v2/', 'src_v2'),  # Source code
    ('configs/', 'configs'),
    ('GROUND_TRUTH.json', '.'),
]

# Hidden imports (not auto-detected)
hiddenimports = [
    'gradio', 'torch', 'torchvision', 'cv2',
    'matplotlib.backends.backend_agg',
    'scipy.spatial.transform', 'sklearn.metrics',
    ...
]

# Excluded modules (reduce size)
excludes = [
    'pytest', 'jupyter', 'IPython', 'tkinter',
    'PyQt5', 'matplotlib.tests',
    ...
]

# Single-file executable with UPX compression
exe = EXE(..., upx=True, console=True, ...)
```

### Phase 3: PyInstaller Execution

**Command** (automated by build script):
```bash
pyinstaller scripts/covid_demo.spec --clean --noconfirm
```

**Process**:
1. **Analysis** (~2 min): PyInstaller analyzes dependencies
2. **Collection** (~10 min): Gathers all Python modules and binaries
3. **Bundling** (~15 min): Packages everything into executable
4. **Compression** (~5 min): UPX compresses binaries
5. **Output** (~5 min): Creates final `.exe`

**Output directories**:
```
build/           # Temporary build files (can delete)
dist/            # Final executable
  COVID19_Demo.exe
```

### Phase 4: Post-Build

**Automatic actions** (by build script):
1. Compute SHA256 checksum
2. Save to `dist/COVID19_Demo.exe.sha256`
3. Display file size and checksum
4. Verify executable exists

---

## Architecture

### How PyInstaller Packages Work

When the user runs `COVID19_Demo.exe`:

1. **Extraction** (first run only, ~10-20s):
   - Extracts embedded files to temp directory: `%TEMP%\_MEI******`
   - Contains Python runtime, libraries, models (~1.8 GB uncompressed)

2. **Execution**:
   - Sets `sys.frozen = True`
   - Sets `sys._MEIPASS` to temp extraction directory
   - Runs `scripts/run_demo.py` as entry point

3. **Resource Loading**:
   - `run_demo.py` detects frozen mode via `getattr(sys, 'frozen', False)`
   - Sets `COVID_DEMO_MODELS_DIR` environment variable
   - Models loaded from `sys._MEIPASS/models/`

4. **Cleanup** (on exit):
   - Temp directory is deleted
   - Next run extracts again (but faster if cached)

### Code Modifications for PyInstaller

**File**: `scripts/run_demo.py`

```python
def get_base_path():
    """Get base path for resources."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        return Path(sys._MEIPASS)
    else:
        # Running as Python script
        return Path(__file__).parent.parent

# Configure models directory
if getattr(sys, 'frozen', False):
    os.environ['COVID_DEMO_MODELS_DIR'] = str(get_base_path() / 'models')
    os.environ['COVID_DEMO_FROZEN'] = '1'
```

**File**: `src_v2/gui/config.py`

```python
# Detect deployment mode
MODELS_DIR = os.environ.get('COVID_DEMO_MODELS_DIR')
IS_FROZEN = os.environ.get('COVID_DEMO_FROZEN', '0') == '1'

if MODELS_DIR or IS_FROZEN:
    # Use simplified paths for bundled models
    MODELS_BASE = Path(MODELS_DIR) if MODELS_DIR else Path(sys._MEIPASS) / 'models'
    LANDMARK_MODELS = [
        MODELS_BASE / "landmarks/resnet18_seed123_best.pt",
        ...
    ]
```

### Size Breakdown

| Component | Size (Compressed) | Size (Uncompressed) |
|-----------|-------------------|---------------------|
| PyTorch CPU | ~400 MB | ~500 MB |
| Gradio + deps | ~250 MB | ~300 MB |
| OpenCV, NumPy, SciPy | ~300 MB | ~400 MB |
| Models (5 files) | ~227 MB | ~227 MB |
| Python runtime | ~50 MB | ~100 MB |
| Source code (src_v2) | ~5 MB | ~5 MB |
| PyInstaller bootloader | ~10 MB | ~10 MB |
| **Total** | **~1.8 GB** | **~2.5 GB** |

---

## Testing

### Local Testing (Development Machine)

```bash
# After build completes
cd dist

# Run executable
./COVID19_Demo.exe

# Expected:
# - Console window appears with status messages
# - "Loading models (this may take 10-30 seconds)..."
# - Browser opens to http://localhost:7860
# - Gradio interface loads
```

**Basic functionality test**:
1. Upload a chest X-ray image
2. Click "Procesar Imagen"
3. Verify 4 visualizations appear:
   - Original with landmarks
   - Warped image
   - GradCAM heatmap
   - Classification results
4. Expand "Métricas del Sistema"
5. Click "Exportar a PDF"
6. Close browser and Ctrl+C in console

### Testing on Clean Windows VM

**Recommended**: Test on a Windows VM without Python installed.

```bash
# 1. Copy COVID19_Demo.exe to VM
# 2. Double-click executable
# 3. If SmartScreen appears: "More info" → "Run anyway"
# 4. Wait for browser to open (~20-30 seconds)
# 5. Run full functionality test (see checklist below)
```

**Full Test Checklist**: See `CHECKLIST_DEFENSA.txt` section "3 DÍAS ANTES"

### Automated Testing Script

```bash
# Run automated validation tests
python scripts/test_exe_startup.py --exe dist/COVID19_Demo.exe
```

This script (see Phase 6) validates:
- Executable exists and is correct size
- SHA256 checksum matches
- Can launch (smoke test)
- Models load without errors

---

## Troubleshooting

### Build Failures

#### "Model files not found"

**Cause**: Missing trained models

**Solution**:
```bash
# Verify models exist
ls checkpoints/landmarks/*.pt
ls outputs/classifier_warped_lung_best/best_classifier.pt
ls outputs/shape_analysis/*.json

# If missing, train models first
# See docs/REPRO_FULL_PIPELINE.md
```

#### "ModuleNotFoundError" during build

**Cause**: Missing hidden import in spec file

**Solution**:
1. Identify missing module from error traceback
2. Add to `hiddenimports` in `scripts/covid_demo.spec`
3. Rebuild

#### "UPX is not available"

**Cause**: UPX not installed (optional compression)

**Solution** (Option 1 - Install UPX):
```bash
# Windows (with Chocolatey)
choco install upx

# Linux
sudo apt install upx
```

**Solution** (Option 2 - Disable UPX):
Edit `scripts/covid_demo.spec`:
```python
exe = EXE(..., upx=False, ...)  # Disable UPX
```

Note: Executable will be larger (~2.2 GB vs 1.8 GB)

#### Build crashes / out of memory

**Cause**: Insufficient RAM during build

**Solution**:
- Close other applications
- Increase swap space
- Use machine with more RAM (16 GB recommended)

### Runtime Failures

#### "Windows protected your PC" (SmartScreen)

**Cause**: Executable is not digitally signed

**Solution for users**:
1. Click "More info"
2. Click "Run anyway"

**Solution for developers** (optional):
- Purchase code signing certificate (~$100-500/year)
- Sign executable:
  ```bash
  signtool sign /f cert.pfx /p password /t http://timestamp.digicert.com COVID19_Demo.exe
  ```

#### "Missing VCRUNTIME140.dll"

**Cause**: Visual C++ Redistributable not installed

**Solution**:
1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install
3. Retry running executable

**Prevention**:
- Include `vc_redist.x64.exe` on USB drive
- Documented in `README_USUARIO.txt`

#### "Failed to execute script" on startup

**Cause**: Corrupted extraction or antivirus blocking

**Solution**:
1. Run as Administrator
2. Add exception in antivirus
3. Verify checksum matches (not corrupted)
4. Try on different machine

#### Slow startup (~1 minute)

**Cause**: First-time extraction (normal behavior)

**Explanation**: PyInstaller extracts ~1.8 GB to temp directory on first run.

**Mitigation**:
- Warn users in documentation (✓ done in README_USUARIO.txt)
- SSD instead of HDD helps
- Subsequent runs are faster (~10-15s)

#### GPU errors even though CPU-only

**Cause**: Code trying to use CUDA

**Solution**:
Ensure `src_v2/gui/config.py` has:
```python
DEVICE_PREFERENCE = "cuda"  # Will auto-fallback to CPU
```

And `src_v2/gui/model_manager.py` handles fallback:
```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Distribution

### Packaging for Thesis Defense

**Recommended structure on USB drive** (4 GB minimum):

```
USB_Drive/
├── COVID19_Demo.exe                    # Main executable
├── COVID19_Demo.exe.sha256             # Integrity check
├── README_USUARIO.txt                  # User instructions
├── Ejemplos/                           # Example X-rays
│   ├── covid_ejemplo_1.png
│   ├── normal_ejemplo_1.png
│   └── viral_ejemplo_1.png
├── Resultados_Ejemplo/                 # Pre-generated PDFs
│   ├── resultado_covid.pdf
│   ├── resultado_normal.pdf
│   └── resultado_viral.pdf
├── Visual_C++_Redistributable/         # Dependency installer
│   └── vc_redist.x64.exe
└── Video_Demo_Backup/                  # Plan B
    └── demo_completo.mp4
```

**Total USB size**: ~2.5-3 GB

### Checksum Verification

**Generate** (automatic during build):
```bash
sha256sum dist/COVID19_Demo.exe > dist/COVID19_Demo.exe.sha256
```

**Verify** (on target machine):
```bash
# Windows (PowerShell)
Get-FileHash COVID19_Demo.exe -Algorithm SHA256

# Compare with .sha256 file
```

### Distribution Checklist

- [ ] Build completed successfully
- [ ] Tested on clean Windows VM
- [ ] Checksum generated and verified
- [ ] README_USUARIO.txt prepared
- [ ] Example images selected (3-5 cases)
- [ ] PDFs pre-generated
- [ ] USB formatted and tested
- [ ] Backup USB created (exact copy)
- [ ] VC++ Redistributable included
- [ ] Backup video recorded (optional)

---

## Technical Details

### PyInstaller Hooks

If custom imports fail, create hook files in `hooks/`:

**Example**: `hooks/hook-gradio.py`
```python
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('gradio')
```

Then in spec file:
```python
a = Analysis(..., hookspath=['hooks'], ...)
```

### Reducing Executable Size

Current size is ~1.8 GB. To reduce further:

1. **Remove unused models** (if not using all seeds):
   ```python
   # In spec file, comment out unused models
   # datas = [
   #     ('checkpoints/landmarks/resnet18_seed111_best.pt', ...),  # Skip this
   # ]
   ```

2. **Remove matplotlib** (if not exporting PDFs):
   ```python
   excludes = [..., 'matplotlib', 'reportlab']
   ```

3. **Use lighter deep learning framework**:
   - ONNX Runtime (~100 MB vs PyTorch ~500 MB)
   - Requires converting models to ONNX format

4. **Use `--onedir` instead of `--onefile`**:
   - Creates folder with files instead of single exe
   - Faster startup (no extraction)
   - But less convenient for distribution

### Build Reproducibility

To ensure reproducible builds:

```bash
# Pin exact versions
pip freeze > requirements_exact.txt

# Use deterministic build flags
export PYTHONHASHSEED=0
pyinstaller ... --seed=42
```

### Cross-Compilation

**Windows executable from Linux**: Not officially supported by PyInstaller.

**Workarounds**:
1. Use Wine + PyInstaller (experimental, many issues)
2. Use Windows VM for building (recommended)
3. Use Docker with Windows base image (requires Windows licenses)

**Recommendation**: Build on Windows native or Windows VM.

### Performance Profiling

To analyze startup time:

```python
# In run_demo.py, add timing:
import time
start = time.time()
# ... code ...
print(f"Startup took {time.time() - start:.2f} seconds")
```

Expected times (on modern laptop, CPU-only):
- Extraction: 10-20s (first run)
- Python import: 2-3s
- Model loading: 5-10s
- Gradio launch: 2-3s
- **Total**: 20-30s (first run), 10-15s (subsequent)

### Debugging Frozen Mode

Enable debug console output:

```python
# In spec file
exe = EXE(..., console=True, debug=True)
```

This shows:
- Extraction progress
- Import errors
- File access attempts
- Exception tracebacks

### Comparison with Alternatives

| Method | Size | Startup | Pros | Cons |
|--------|------|---------|------|------|
| **PyInstaller** | 1.8 GB | 20-30s | Single file, easy distribution | Large size, slow startup |
| Docker | 2-3 GB | 10-20s | Reproducible, isolated | Requires Docker installed |
| WinPython Portable | 2 GB | 5-10s | Faster, editable | Folder (not single file) |
| Conda Package | 1.5 GB | 10-15s | Familiar to researchers | Requires Miniconda |
| ONNX Runtime | 800 MB | 5-10s | Smaller, faster | Requires model conversion |

**Our choice**: PyInstaller for maximum simplicity for non-technical users.

---

## Advanced: Customization

### Custom Icon

1. Create icon file:
   ```bash
   python scripts/generate_icon.py
   # Generates: assets/covid_icon.ico
   ```

2. Update spec file:
   ```python
   exe = EXE(..., icon='assets/covid_icon.ico')
   ```

3. Rebuild

### Version Information (Windows)

Create `version.txt`:
```
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    ...
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'Your University'),
        StringStruct('FileDescription', 'COVID-19 Detection System'),
        StringStruct('FileVersion', '1.0.0'),
        StringStruct('ProductName', 'COVID-19 Detector'),
      ])
    ])
  ]
)
```

In spec file:
```python
exe = EXE(..., version='version.txt')
```

### Multi-Language Support

To add English interface:

1. Create `src_v2/gui/config_en.py` with English text
2. Detect system language in `run_demo.py`
3. Load appropriate config

### Splash Screen

Add loading splash screen:

```python
# In spec file
splash = Splash('assets/splash.png', ...)
exe = EXE(..., splash=splash)
```

Displays image while unpacking (improves perceived startup time).

---

## FAQ for Developers

**Q: Can I distribute the executable commercially?**
A: Check licenses of all dependencies (PyTorch, Gradio, etc.). Most are permissive (Apache, MIT) but read carefully.

**Q: How do I update models without rebuilding?**
A: Use `--onedir` mode and replace model files in the folder. Or use external model directory (not bundled).

**Q: Why CPU-only instead of GPU?**
A: GPU version adds 1-2 GB (CUDA libraries) and requires NVIDIA drivers. CPU works on any laptop.

**Q: Can I reduce startup time?**
A: Yes, use `--onedir` mode (no extraction needed), or use lighter framework (ONNX Runtime).

**Q: Is code secure/encrypted in the executable?**
A: No, Python bytecode can be extracted. Don't bundle secrets. Use server API for sensitive deployments.

**Q: How to auto-update the executable?**
A: Not supported in single-file mode. Alternatives: distribute via GitHub Releases, use web interface instead.

**Q: Can I run this on macOS/Linux?**
A: Rebuild on those platforms. PyInstaller supports macOS/Linux, but spec file may need adjustments.

---

## References

- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [PyInstaller Spec Files](https://pyinstaller.org/en/stable/spec-files.html)
- [UPX Compression](https://upx.github.io/)
- [Code Signing on Windows](https://learn.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)

---

## Changelog

### Version 1.0.0 (January 2026)
- Initial standalone build system
- CPU-only PyTorch for maximum compatibility
- Single-file executable with UPX compression
- Automated build script with verification
- Comprehensive user and developer documentation

---

## Support

For build issues or questions:
1. Check [Troubleshooting](#troubleshooting) section above
2. Review PyInstaller logs in `build/COVID19_Demo/warn-COVID19_Demo.txt`
3. Open issue on GitHub repository
4. Contact project maintainer: [Add contact info]

---

**Last updated**: January 2026
**Maintainer**: [Your name]
**License**: [Specify license]
