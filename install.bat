@echo off
echo ==========================================
echo COVID-19 Detection Demo - Installer
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Verifying models...
python -c "import sys; from pathlib import Path; models=['models/landmarks/seed123_final.pt','models/landmarks/seed321_final.pt','models/landmarks/seed111_final.pt','models/landmarks/seed666_final.pt','models/classifier/best_classifier.pt','models/shape_analysis/canonical_shape_gpa.json','models/shape_analysis/canonical_delaunay_triangles.json']; missing=[m for m in models if not Path(m).exists()]; sys.exit(1) if missing else print('All models verified')"

if errorlevel 1 (
    echo Installation incomplete. Please check errors above.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Installation complete!
echo ==========================================
echo.
echo To run the demo, execute: run_demo.bat
pause
