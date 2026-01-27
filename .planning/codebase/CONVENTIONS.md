# Coding Conventions

**Analysis Date:** 2026-01-27

## Naming Patterns

**Files:**
- Module files use snake_case: `resnet_landmark.py`, `dataset.py`, `warp.py`
- Spanish names acceptable for domain-specific files: `gpa.py` (Generalized Procrustes Analysis referenced in Spanish context)
- Preserved dataset categories use Spanish names: `Viral_Pneumonia` (not translated in code)

**Functions:**
- snake_case for all functions: `compute_pixel_error()`, `create_dataloaders()`, `piecewise_affine_warp()`
- Private helper functions prefixed with underscore: `_prepare_image_size_tensor()`, `_triangle_area_2x()`
- Method names following Python conventions: `__init__()`, `__getitem__()`, `forward()`

**Variables:**
- snake_case for variables: `image_size`, `num_landmarks`, `total_loss`
- Constants in UPPERCASE: `NUM_LANDMARKS`, `DEFAULT_IMAGE_SIZE`, `SYMMETRIC_PAIRS`
- Short names acceptable in loops and tensor operations: `B` (batch size), `C` (channels), `H` (height), `W` (width)
- Tensor shape variables: `pred.shape[0]` for batch, `.shape[1]` for landmarks
- Meaningful abbreviations in scientific code: `loss_dict`, `error_px` (pixel error), `lm` (landmarks when space-constrained)

**Types:**
- Use Type hints throughout: `Optional[Tensor]`, `Tuple[int, int]`, `Dict[str, float]`
- Import from `typing` module: `from typing import Optional, List, Tuple, Dict, Callable`
- Generic types for collections: `List[str]`, `Dict[str, float]`, `Tuple[float, float]`
- PyTorch types explicit: `torch.Tensor`, `torch.device`

## Code Style

**Formatting:**
- PEP 8 with 100-character line limit (inferred from code organization)
- 4-space indentation
- Black-compatible style (no explicit formatter configured, but code follows compatible patterns)
- Spacing: 2 blank lines between top-level definitions, 1 blank line between methods

**Linting:**
- No strict linting tool configured (no `.pylintrc`, `.flake8`, or `setup.cfg`)
- Convention: Type hints required for function parameters and return values
- Convention: Docstrings required for public functions and classes (Google-style preferred)

## Import Organization

**Order:**
1. Standard library imports (logging, sys, pathlib, etc.)
2. Third-party imports (torch, torchvision, numpy, pandas, scipy, cv2, PIL, matplotlib)
3. Local imports (src_v2.* modules)

**Path Aliases:**
- No path aliases configured (no absolute imports used)
- Relative imports from src_v2: `from src_v2.constants import NUM_LANDMARKS`
- All imports from root: `from src_v2.data.dataset import LandmarkDataset`

**Example Pattern:**
```python
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict

from src_v2.constants import NUM_LANDMARKS, DEFAULT_IMAGE_SIZE
from src_v2.data.utils import load_coordinates_csv
```

## Error Handling

**Patterns:**
- Use standard Python exceptions: `FileNotFoundError`, `ValueError`, `IOError`, `NotImplementedError`
- Chain exceptions with `from e`: `raise ValueError(...) from e`
- Log errors before raising: `logger.error("message: %s", details)`
- Context managers for resource management: `with torch.no_grad():`
- Warnings via `logger.warning()`: `logger.warning("Image size mismatch...")`
- Try-except in data loading with informative messages (see `src_v2/data/dataset.py::LandmarkDataset.__getitem__()`)

**Error Location Context:**
- Use `logger.error()` with formatting for reproducibility
- Include indices and shapes when debugging tensor operations
- Store warnings with flags to prevent spam: `_size_warning_emitted` in `LandmarkDataset`

## Logging

**Framework:** Python's standard `logging` module

**Initialization Pattern:**
```python
import logging
logger = logging.getLogger(__name__)
```

**Patterns:**
- All modules define `logger = logging.getLogger(__name__)` at module level
- Info level for pipeline progress: `logger.info("Loading coordinates from %s", csv_path)`
- Warning level for non-fatal issues: `logger.warning("Image size mismatch...")`
- Error level for exceptions: `logger.error("Image not found: %s (idx=%d)", path, idx)`
- No explicit log level configuration in code (delegated to application setup)

**Usage Examples from codebase:**
- `src_v2/data/utils.py`: `logger.info("Loaded %d samples: %s", len(df), df['category'].value_counts().to_dict())`
- `src_v2/data/dataset.py`: `logger.warning("Image size mismatch (expected %s, got %s)...", ...)`
- `src_v2/models/losses.py`: State logging of weight calculations in `__init__`

## Comments

**When to Comment:**
- Docstrings required for all public functions and classes
- Inline comments for non-obvious algorithmic steps
- Comments explaining parameter scaling or transformations
- Comments marking TODO items or known limitations
- Section headers with "=" dividers in module docstrings

**JSDoc/TSDoc:**
- Google-style docstrings for Python functions
- Required sections: `Args:`, `Returns:`, `Raises:` where applicable
- Type information in docstrings matches function signatures

**Example Pattern:**
```python
def compute_pixel_error(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Calcula error euclidiano promedio en pixeles.

    Args:
        pred: Predicciones (B, 30) en [0, 1]
        target: Ground truth (B, 30) en [0, 1]

    Returns:
        Error promedio en pixeles
    """
```

## Function Design

**Size:**
- Functions typically 20-80 lines
- Helper functions for repeated logic: `_prepare_image_size_tensor()` extracts 40-line tensor preparation
- Larger functions have section comments to mark logic blocks

**Parameters:**
- Use explicit keyword arguments over *args
- Type hints on all parameters
- Default values for optional parameters: `normalized: bool = True`
- Related parameters grouped: image_size/padding together
- Consistent ordering: required params first, then optional with defaults

**Return Values:**
- Return dict for multiple related values: `{'loss': ..., 'error_px': ...}`
- Return tuple for heterogeneous types: `Tuple[torch.Tensor, torch.Tensor, dict]`
- Return None explicitly only when side-effect focused
- Always type-hint return type

**Example from `src_v2/training/trainer.py::train_epoch()`:**
```python
def train_epoch(
    self,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    scheduler_callback: Optional[LRSchedulerCallback] = None
) -> Dict[str, float]:
    """Entrena una epoca. Returns: Dict con loss y error_px promedio"""
```

## Module Design

**Exports:**
- No `__all__` explicitly defined (module exports all public functions/classes)
- Private functions/classes prefixed with underscore
- Public API clear from imports in main `__init__.py` files

**Barrel Files:**
- Minimal barrel files used
- `src_v2/__init__.py`: Only version and module docstring
- `src_v2/models/__init__.py`: Minimal imports
- Main pattern: Direct imports from submodules (`from src_v2.data.dataset import LandmarkDataset`)

## Configuration & Constants

**Location:**
- All domain constants centralized in `src_v2/constants.py`
- Configuration for experiments in JSON files in `configs/` directory
- Landmark definitions, image sizes, learning rates, loss parameters all in `constants.py`

**Pattern:**
```python
# From src_v2/constants.py
NUM_LANDMARKS: int = 15
NUM_COORDINATES: int = NUM_LANDMARKS * 2
DEFAULT_IMAGE_SIZE: int = 224
SYMMETRIC_PAIRS: List[Tuple[int, int]] = [(2, 3), (4, 5), ...]
DEFAULT_WING_OMEGA: float = 10.0
OPTIMAL_MARGIN_SCALE: float = 1.05
```

---

*Convention analysis: 2026-01-27*
