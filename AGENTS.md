# Repository Guidelines

## Project Structure & Module Organization
- `src_v2/`: core package (CLI in `cli.py`, data transforms, models, training, evaluation, warping/processing, visualization).
- `configs/`: JSON configs for training and ensembles.
- `scripts/`: helper and legacy automation scripts.
- `tests/`: pytest suite.
- `docs/`: experiment notes and reports.
- `data/`, `checkpoints/`, `outputs/`, `results/`: local artifacts; not committed to Git.

## Build, Test, and Development Commands
- Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Install dev dependencies: `pip install -e ".[dev]"` (runtime only: `pip install -r requirements.txt`).
- Install PyTorch for your hardware (CUDA/ROCm/CPU) as documented in `README.md`.
- CLI entrypoint: `python -m src_v2 --help` or `covid-landmarks --help`.
- Explore training and evaluation flags: `python -m src_v2 train --help`, `python -m src_v2 evaluate --help`.

## Coding Style & Naming Conventions
- Python 3.9+, PEP 8, 4-space indentation, max line length 100.
- Use type hints and Google-style docstrings for public functions.
- Import order: standard library, third-party, local; alphabetical within each group.
- English, descriptive names; `snake_case` for functions/variables, `PascalCase` for classes.

## Testing Guidelines
- Frameworks: `pytest` and `pytest-cov`.
- Run all tests: `python -m pytest tests/ -v`.
- Coverage report: `python -m pytest tests/ -v --cov=src_v2 --cov-report=html`.
- Naming: files `test_*.py`, functions `test_*`; prefer fixtures for shared setup.

## Commit & Pull Request Guidelines
- Branch from `main` as `feature/your-feature-name`.
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `style:`, `chore:`.
- PRs should include a concise summary, linked issues, and test results; attach sample outputs if metrics or plots change.

## Data, Checkpoints, and Outputs
- Keep large datasets and model artifacts out of Git.
- Store local runs in `data/`, `checkpoints/`, and `outputs/`; document new paths in `docs/` or `configs/`.
