@echo off
call .venv\Scripts\activate.bat
set COVID_DEMO_MODELS_DIR=%cd%\models
python scripts\run_demo.py %*
