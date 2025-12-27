# Repository Guidelines

## Project Structure & Module Organization
- `src_v2/`: Core package (`cli.py` for Typer commands, plus `data/`, `models/`, `training/`, `evaluation/`, `processing/`, `visualization/`). Key constants live in `src_v2/constants.py`.
- `scripts/`: Legacy training/evaluation helpers (e.g., `train.py`, `evaluate_ensemble.py`); prefer the CLI equivalents in `src_v2`.
- `configs/`: Experiment configuration files; keep new configs small and documented.
- `tests/`: Pytest suite (`test_*.py`) for data loaders, processing, and model utilities.
- Local assets expected but not tracked: `data/` (datasets), `checkpoints/`, `outputs/`, `audit/` artifacts. Use relative paths in scripts to keep runs portable.

## Build, Test, and Development Commands
- Install (dev mode): `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"` plus the correct Torch wheel for your hardware (CUDA/ROCm/CPU).
- Quick sanity run: `python -m src_v2 --help` to list all CLI tasks.
- Train landmarks (canonical recipe): `python -m src_v2 train --data-root data/ --csv-path data/coordenadas/coordenadas_maestro.csv --checkpoint-dir checkpoints_v2 --coord-attention --deep-head --hidden-dim 768 --clahe --loss wing --seed 123`.
- Evaluate a checkpoint: `python -m src_v2 evaluate checkpoints_v2/final_model.pt --data-root data/ --csv-path data/coordenadas/coordenadas_maestro.csv --tta --split test --clahe`.
- Legacy scripts (only if needed for reproduction): `python scripts/train.py`, `python scripts/train_classifier.py`.

## Coding Style & Naming Conventions
- Python: follow PEP 8, 100-char line limit, and English names; prefer type hints for all public functions.
- Imports: stdlib, third-party, local (in that order), alphabetical within groups.
- Docstrings: Google style for public functions/classes; note tensor shapes and units where relevant.
- Naming: modules and functions in `snake_case`, classes in `CapWords`, CLI options in `kebab-case` (Typer handles conversion).

## Testing Guidelines
- Run all tests: `python -m pytest -v`. With coverage: `python -m pytest -v --cov=src_v2 --cov-report=html`.
- Add tests in `tests/` using `test_*.py` and `test_*` functions; use fixtures for repeated setups and include edge cases (e.g., missing landmarks, invalid image sizes).
- When adding new CLI flows, include at least one test that exercises argument parsing and a minimal data sample.

## Commit & Pull Request Guidelines
- Commit messages use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `style:`, `chore:`). Example: `feat: add combined loss option for landmark training`.
- Branch from `main` with `feature/<topic>` or `fix/<topic>`. Keep commits scoped and reviewable.
- PRs should include: succinct summary, linked issues, key metrics or sample outputs (e.g., landmark error, accuracy), and test command outputs. Include data path assumptions or config changes if applicable.

## Data, Security, and Configuration Tips
- Datasets, checkpoints, and outputs stay local; avoid committing PHI or large artifacts. Use `.gitignore` entries already present.
- Prefer relative paths in configs and scripts so experiments remain reproducible across machines.
- Validate hardware-specific installs (CUDA vs ROCm) before long runs; mismatched wheels are a common failure mode.
