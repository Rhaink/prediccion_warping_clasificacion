# Repository Guidelines

## Project Structure & Module Organization
`src_v2/` contains the main CLI and modules (`data/`, `models/`, `training/`, `processing/`,
`evaluation/`, `visualization/`). Configuration lives in `configs/` (JSON templates). Tests
live in `tests/`. Repro and experiment notes are in `docs/`. Utility scripts are in
`scripts/`, but prefer the CLI (`python -m src_v2`) unless a script is explicitly required.
Local artifacts (`data/`, `checkpoints/`, `outputs/`) are not versioned.

## Build, Test, and Development Commands
Create a venv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
For dev extras:
```bash
pip install -e ".[dev]"
```
Run the main CLI (examples from the current pipeline):
```bash
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv --output-dir outputs/shape_analysis
python -m src_v2 generate-dataset --config configs/warping_best.json
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
```

## Coding Style & Naming Conventions
Follow PEP 8 with type hints, 100-character max lines, and Google-style docstrings.
Keep imports grouped (stdlib, third-party, local) and alphabetized. Test files are
`test_*.py` and functions are `test_*`. Use clear English identifiers in code, but
preserve existing Spanish file names and labels where they are part of the dataset
or documentation.

## Testing Guidelines
Pytest is the test runner. Run the full suite with:
```bash
python -m pytest tests/ -v
```
Coverage (optional):
```bash
python -m pytest tests/ -v --cov=src_v2 --cov-report=html
```
Add tests under `tests/`, and target edge cases for warping, transforms, and CLI
flows.

## Commit & Pull Request Guidelines
Recent history mixes descriptive Spanish messages and Conventional Commits
(`feat:`, `docs:`, `fix:`). Prefer Conventional Commits for new work and keep
messages short and imperative. PRs should include a concise summary, linked issues
if any, and test results; include screenshots only when visualizations or figures
change.

## Data, Artifacts, and Source of Truth
`GROUND_TRUTH.json` is the source for validated metrics. Do not commit data or
checkpoints; keep outputs under `outputs/` and document reproducible runs in
`docs/` when results change.
