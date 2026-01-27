# Testing Patterns

**Analysis Date:** 2026-01-27

## Test Framework

**Runner:**
- pytest 7.0.0+
- Config: `pyproject.toml` ([project section](file:///home/donrobot/Projects/prediccion_warping_clasificacion/pyproject.toml))
- No separate pytest.ini (configuration in pyproject.toml)

**Assertion Library:**
- Standard Python `assert` statements
- No specialized assertion library (assert statements with meaningful messages)

**Run Commands:**
```bash
# Run all tests (not yet organized in tests/ directory)
python -m pytest tests/ -v

# Watch mode (requires pytest-watch)
pytest-watch tests/

# Coverage report
python -m pytest tests/ -v --cov=src_v2 --cov-report=html

# Run specific test file
python -m pytest tests/test_processing.py -v

# Run tests matching pattern
python -m pytest tests/ -k "test_warp" -v
```

**pytest Configuration (pyproject.toml):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
env = ["FORCE_NUM_WORKERS_ZERO=1"]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::UserWarning",
]

[tool.coverage.run]
source = ["src_v2"]
omit = ["tests/*", "scripts/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Test File Organization

**Location:**
- Tests directory: `/home/donrobot/Projects/prediccion_warping_clasificacion/tests/` (standard pytest location)
- Currently MISSING - no active test suite in main codebase
- Archive tests in `scripts/archive/`: `test_forward_pass.py`, `test_dataset.py`, `test_hierarchical_forward.py` (legacy)

**Naming:**
- Pattern: `test_*.py` files (configured in pytest.ini_options)
- Test functions: `test_*` prefix
- Test classes: `Test*` prefix (not yet used in archive tests)

**Structure:**
```
tests/
├── test_processing.py       # GPA, warping, geometry tests
├── test_models.py           # Model creation, forward pass
├── test_data.py             # Dataset loading, transforms
├── test_training.py         # Trainer, callbacks
└── test_integration.py      # End-to-end pipeline tests
```

## Test Structure

**Suite Organization:**
```python
# Pattern from scripts/archive/test_forward_pass.py
import sys
from pathlib import Path

# Agregar paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src_v2.models.resnet_landmark import ResNet18Landmarks
from src_v2.data.dataset import create_dataloaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_creation():
    """Prueba creación del modelo."""
    print("=" * 60)
    print("TEST 1: Creación del modelo")
    print("=" * 60)

    model = create_model(num_landmarks=15, pretrained=True, device=DEVICE)
    total, trainable = count_parameters(model)

    # Verificar output shape
    dummy_input = torch.rand(4, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 30), f"Expected (4, 30), got {output.shape}"
    assert output.min() >= 0 and output.max() <= 1, "Output should be in [0, 1]"

    print("✓ Test 1 PASADO\n")
    return model

def test_unfreeze_backbone():
    """Prueba descongelar backbone (Phase 2)."""
    model = create_model(freeze_backbone=True, device=DEVICE)
    _, trainable_frozen = count_parameters(model)

    model.unfreeze_backbone()
    _, trainable_unfrozen = count_parameters(model)

    assert trainable_unfrozen > trainable_frozen
    print("✓ Test 2 PASADO\n")
```

**Patterns:**
- Setup via direct function calls (no pytest fixtures configured yet)
- Teardown: Not explicit (torch.cuda.empty_cache() when needed)
- Assertions: `assert` statements with meaningful error messages
- Output: Print statements for test progress (verbose mode)

## Mocking

**Framework:** None explicitly configured

**Patterns:**
- Mocking patterns not used in archive tests
- Data loading uses real CSV files with limited datasets
- Mock approach: Use `torch.rand()` for dummy tensors when testing model forward pass (see `test_model_creation()`)

**What to Mock:**
- External file I/O (optional, currently tested with real data)
- GPU availability (handled via device selection: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`)

**What NOT to Mock:**
- Model forward pass (test with real models and data)
- Dataset loading (test with real CSV and image files)
- Loss functions (test with actual landmark tensors)
- PyTorch operations (let PyTorch handle its own testing)

## Fixtures and Factories

**Test Data:**
```python
# Pattern from scripts/archive/test_forward_pass.py
def test_forward_pass_real_data(model, val_loader):
    """Test forward pass with real data from validation set."""
    model.eval()
    with torch.no_grad():
        for images, landmarks, meta in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            assert outputs.shape[0] == images.shape[0]
            assert outputs.shape[1] == 30
            assert (outputs >= 0).all() and (outputs <= 1).all()
```

**Location:**
- Fixtures not yet formalized in pytest format
- Test data loading: Direct calls to `create_dataloaders()` from `src_v2.data.dataset`
- Sample test CSV: `data/coordenadas/coordenadas_maestro.csv` used in archive tests

## Coverage

**Requirements:**
- No explicit coverage target enforced in CI
- Coverage reports generated via pytest-cov: `pytest --cov=src_v2 --cov-report=html`
- Target: Implied high coverage on core modules (src_v2)

**View Coverage:**
```bash
# Generate HTML report
python -m pytest tests/ --cov=src_v2 --cov-report=html

# View in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Exclusions (from pyproject.toml):**
- `pragma: no cover` decorator for uncoverable code
- `__repr__` methods
- `raise NotImplementedError` stubs
- `if __name__ == '__main__':` blocks

## Test Types

**Unit Tests:**
- Scope: Individual functions and model components
- Approach: Test model creation, forward pass shape, loss computation
- Example: `test_model_creation()` in `scripts/archive/test_forward_pass.py`
- Isolated testing of: Landmark predictions, loss functions, metric computation

**Integration Tests:**
- Scope: Data pipeline, training loop, evaluation
- Approach: Load real dataset, run training epoch, check metrics
- Example: `test_training_step()` in `scripts/archive/test_forward_pass.py`
- End-to-end: Dataset -> Transform -> Model -> Loss -> Backward

**E2E Tests:**
- Framework: Not formally established
- Manual validation in scripts (e.g., `QUICKSTART_WARPING.md`)
- Pipeline testing: Via bash scripts in `scripts/`

## Common Patterns

**Async Testing:**
- Not applicable (synchronous PyTorch code, no async operations)

**Error Testing:**
```python
# Pattern from scripts/archive/test_forward_pass.py
def test_dataset_loading():
    """Prueba carga de dataset."""
    csv_path = "data/coordenadas/coordenadas_maestro.csv"

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            csv_path=csv_path,
            data_root="data",
            batch_size=16,
            num_workers=0,
            split_seed=42
        )

        # Verify loader yields correct shape
        images, landmarks, meta = next(iter(train_loader))
        assert images.shape == (16, 3, 224, 224)
        assert landmarks.shape == (16, 30)

        print("✓ Dataset loading test PASSED")
    except Exception as e:
        print(f"✗ Dataset loading test FAILED: {e}")
        raise
```

## Device/Environment Setup

**Special Environment Variables:**
- `FORCE_NUM_WORKERS_ZERO=1`: Forces num_workers=0 in DataLoader for deterministic testing
- Set in `pyproject.toml` pytest config: `env = ["FORCE_NUM_WORKERS_ZERO=1"]`
- Purpose: Ensures reproducible data loading during tests

**Device Handling Pattern:**
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All model operations move to device
model = model.to(DEVICE)
images = images.to(DEVICE)
landmarks = landmarks.to(DEVICE)

# Use torch.no_grad() for inference testing
with torch.no_grad():
    output = model(dummy_input)
```

## Historical Testing Notes

**Archive Location:** `scripts/archive/`

**Key Archive Tests:**
- `test_forward_pass.py`: 10+ test functions for model creation, backbone freezing, dataset loading, forward pass, backward pass, training steps
- `test_dataset.py`: Dataset loading, horizontal flip validation, dataloader integration
- `test_hierarchical_forward.py`: Hierarchical model forward pass testing
- `test_reconstruct.py`: Warping reconstruction validation
- `test_robustness_geometric.py`: Robustness to geometric transformations
- `test_robustness_artifacts.py`: Robustness to medical artifacts

**Why Archived:**
- Tests written as standalone scripts with manual assertions
- No formal pytest structure in main codebase
- Tests now integrated into pipeline scripts (e.g., `scripts/evaluate_ensemble_from_config.py`)
- Manual testing preferred for complex ML workflows (validated in CLAUDE.md GROUND_TRUTH.json)

## To Organize Tests

**Next Steps (for future implementation):**
1. Create `tests/` directory with proper pytest structure
2. Migrate archive test functions to `tests/test_*.py` with pytest fixtures
3. Add parametrized tests for model architectures and loss functions
4. Create fixtures for: dataloaders, pre-trained models, test datasets
5. Set coverage target: 80%+ for src_v2 core modules
6. Add CI/CD integration (GitHub Actions) to run tests on PR

---

*Testing analysis: 2026-01-27*
