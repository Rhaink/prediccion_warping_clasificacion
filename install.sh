#!/bin/bash
set -e

echo "=========================================="
echo "COVID-19 Detection Demo - Installer"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies (this may take 2-3 minutes)..."
pip install -r requirements.txt

# Verify models
echo ""
echo "Verifying models..."
python3 -c "
import sys
from pathlib import Path

models = [
    'models/landmarks/seed123_final.pt',
    'models/landmarks/seed321_final.pt',
    'models/landmarks/seed111_final.pt',
    'models/landmarks/seed666_final.pt',
    'models/classifier/best_classifier.pt',
    'models/shape_analysis/canonical_shape_gpa.json',
    'models/shape_analysis/canonical_delaunay_triangles.json',
]

missing = []
for model in models:
    if not Path(model).exists():
        missing.append(model)

if missing:
    print('❌ Missing models:')
    for m in missing:
        print(f'  - {m}')
    sys.exit(1)
else:
    print('✓ All models verified')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Installation incomplete. Please check errors above."
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "To run the demo:"
echo "  bash run_demo.sh"
echo ""
