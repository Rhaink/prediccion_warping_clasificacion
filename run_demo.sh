#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set environment variable to use models from local directory
export COVID_DEMO_MODELS_DIR=$(pwd)/models

# Run demo
python3 scripts/run_demo.py "$@"
