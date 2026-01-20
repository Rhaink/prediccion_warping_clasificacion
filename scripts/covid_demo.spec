# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for COVID-19 Detection System standalone executable.

This creates a single-file Windows executable with all dependencies bundled.
Target size: ~1.8GB (includes PyTorch CPU, models, and all dependencies)

Build command:
    pyinstaller scripts/covid_demo.spec --clean --noconfirm

Output:
    dist/COVID19_Demo.exe
"""

import os
import sys
from pathlib import Path

# Get project root directory
project_root = Path(SPECPATH).parent

# Define data files to include
# Models are prepared in build/models_staging by prepare_models_for_build.py
staging_dir = project_root / 'build' / 'models_staging'

datas = [
    # Landmark ensemble models (4 models, ~180MB)
    # Using renamed models from staging directory
    (str(staging_dir / 'landmarks' / 'resnet18_seed123_best.pt'), 'models/landmarks'),
    (str(staging_dir / 'landmarks' / 'resnet18_seed321_best.pt'), 'models/landmarks'),
    (str(staging_dir / 'landmarks' / 'resnet18_seed111_best.pt'), 'models/landmarks'),
    (str(staging_dir / 'landmarks' / 'resnet18_seed666_best.pt'), 'models/landmarks'),

    # Classifier model (~45MB)
    (str(staging_dir / 'classifier' / 'best_classifier.pt'), 'models/classifier'),

    # Geometric analysis data (~2MB)
    (str(staging_dir / 'shape_analysis' / 'canonical_shape_gpa.json'), 'models/shape_analysis'),
    (str(staging_dir / 'shape_analysis' / 'canonical_delaunay_triangles.json'), 'models/shape_analysis'),

    # Ground truth metrics
    (str(project_root / 'GROUND_TRUTH.json'), '.'),

    # Source code (entire src_v2 package)
    (str(project_root / 'src_v2'), 'src_v2'),

    # Configuration files
    (str(project_root / 'configs'), 'configs'),
]

# Add example images if they exist
examples_dir = project_root / 'data' / 'examples'
if examples_dir.exists():
    datas.append((str(examples_dir), 'examples'))

# Hidden imports (modules not detected by PyInstaller's analysis)
hiddenimports = [
    # Gradio dependencies
    'gradio',
    'gradio.blocks',
    'gradio.components',
    'gradio.routes',
    'gradio.utils',
    'fastapi',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',

    # PyTorch
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchvision.models',
    'torchvision.transforms',

    # OpenCV
    'cv2',
    'cv2.aruco',

    # Matplotlib backends
    'matplotlib',
    'matplotlib.backends',
    'matplotlib.backends.backend_agg',
    'matplotlib.pyplot',

    # Scientific computing
    'numpy',
    'scipy',
    'scipy.spatial',
    'scipy.spatial.transform',
    'scipy.ndimage',
    'sklearn',
    'sklearn.metrics',

    # Data processing
    'pandas',
    'PIL',
    'PIL.Image',

    # PDF generation
    'reportlab',
    'reportlab.pdfgen',
    'reportlab.lib',

    # Standard library modules sometimes missed
    'queue',
    'json',
    'csv',
    'hashlib',
    'base64',
]

# Modules to exclude (reduce size)
excludes = [
    # Testing frameworks
    'pytest',
    'unittest',
    '_pytest',

    # Documentation tools
    'sphinx',
    'docutils',

    # Jupyter/IPython
    'jupyter',
    'IPython',
    'notebook',

    # GUI frameworks we don't use
    'tkinter',
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',

    # Development tools
    'pip',
    'setuptools',
    'wheel',

    # Large optional dependencies
    'matplotlib.tests',
    'numpy.tests',
    'scipy.tests',
]

# Analysis phase: collect all Python files and dependencies
a = Analysis(
    [str(project_root / 'scripts' / 'run_demo.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ archive: compress Python bytecode
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None
)

# EXE: create single-file executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='COVID19_Demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Enable UPX compression to reduce size
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Show console window with status messages
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # TODO: Add icon file when available
    version_file=None,  # TODO: Add version info when available
)
