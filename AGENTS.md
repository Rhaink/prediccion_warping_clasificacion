# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src_v2/`, organized by domain (`data/`, `models/`, `training/`,
`processing/`, `evaluation/`, `visualization/`). Configuration templates are in `configs/`
(JSON). Tests are in `tests/`. Repro notes and experiment docs live in `docs/`. Utility
scripts are under `scripts/`, but prefer the CLI (`python -m src_v2`) unless a script is
explicitly required. Local artifacts are kept in `data/`, `checkpoints/`, and `outputs/`
and are not versioned.

## Build, Test, and Development Commands
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Install dev extras when needed:
```bash
pip install -e ".[dev]"
```
Run the CLI (examples):
```bash
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv --output-dir outputs/shape_analysis
python -m src_v2 generate-dataset --config configs/warping_best.json
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
```

## Coding Style & Naming Conventions
Follow PEP 8 with type hints and a 100-character line limit. Use Google-style docstrings.
Group imports as stdlib, third-party, local, and alphabetize within each group. Use clear
English identifiers in code, but preserve existing Spanish file names and labels tied to
datasets or documentation.

## Testing Guidelines
Pytest is the test runner. Test files use `test_*.py`, and test functions use `test_*`.
Run the suite with:
```bash
python -m pytest tests/ -v
```
Optional coverage:
```bash
python -m pytest tests/ -v --cov=src_v2 --cov-report=html
```
Add tests for warping, transforms, and CLI flows when touching those areas.

## Commit & Pull Request Guidelines
Commit history mixes descriptive Spanish messages and Conventional Commits
(`feat:`, `fix:`, `docs:`). Prefer concise Conventional Commits for new work. PRs should
include a short summary, linked issues if any, and test results. Add screenshots only for
visualization changes.

## Data, Artifacts, and Source of Truth
`GROUND_TRUTH.json` is the source for validated metrics. Do not commit data or model
checkpoints. Keep outputs under `outputs/` and document reproducible runs in `docs/` when
results change.
