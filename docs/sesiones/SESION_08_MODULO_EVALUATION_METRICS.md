# Sesion 08: Modulo Evaluation - Metrics

## Fecha: 2025-12-07

## Objetivo
Refactorizar `src_v2/evaluation/metrics.py` para eliminar constantes duplicadas, corregir type hints y reemplazar magic numbers.

## Archivos Modificados
- `src_v2/evaluation/metrics.py`

## Cambios Realizados

### 1. Eliminar constantes duplicadas e importar desde constants.py

**Antes:**
```python
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Nombres de landmarks
LANDMARK_NAMES = [
    'L1',  'L2',  'L3',  'L4',  'L5',  'L6',  'L7',  'L8',
    'L9',  'L10', 'L11', 'L12', 'L13', 'L14', 'L15'
]
...
# Pares simetricos para TTA (indices 0-based)
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
```

**Despues:**
```python
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src_v2.constants import (
    LANDMARK_NAMES,
    SYMMETRIC_PAIRS,
    DEFAULT_IMAGE_SIZE,
)


logger = logging.getLogger(__name__)
```

### 2. Corregir type hints: any -> Any

**Antes:**
```python
def evaluate_model(...) -> Dict[str, any]:
def evaluate_model_with_tta(...) -> Dict[str, any]:
```

**Despues:**
```python
def evaluate_model(...) -> Dict[str, Any]:
def evaluate_model_with_tta(...) -> Dict[str, Any]:
```

**Razon:** `any` no es un tipo valido en Python, debe ser `Any` de typing.

### 3. Reemplazar magic number 224 por DEFAULT_IMAGE_SIZE

Aplicado en 5 funciones:

- `compute_pixel_error()`
- `compute_error_per_landmark()`
- `evaluate_model()`
- `compute_error_per_category()`
- `evaluate_model_with_tta()`

## Verificaciones Ejecutadas

```bash
# Sin constantes duplicadas
grep -n "LANDMARK_NAMES = \[" src_v2/evaluation/metrics.py
# Resultado: (vacio)

grep -n "SYMMETRIC_PAIRS = \[" src_v2/evaluation/metrics.py
# Resultado: (vacio)

# Sin magic number 224
grep -n "= 224" src_v2/evaluation/metrics.py
# Resultado: (vacio)

# Tests
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed

# Imports funcionan
python -c "from src_v2.evaluation.metrics import evaluate_model; print('OK')"
# Resultado: Imports OK
```

## Estado Final
- [x] LANDMARK_NAMES eliminado, importado desde constants.py
- [x] SYMMETRIC_PAIRS eliminado, importado desde constants.py
- [x] Type hints corregidos: `any` -> `Any`
- [x] Magic number 224 reemplazado por DEFAULT_IMAGE_SIZE (5 lugares)
- [x] Logging agregado
- [x] Tests pasan (50/50)
- [x] Documento de sesion creado

## Notas para Proxima Sesion
- **Modulo 9: CLI con Typer** - Crear interfaz CLI mejorada
- **Modulo 10: Tests de Integracion** - Crear tests adicionales
