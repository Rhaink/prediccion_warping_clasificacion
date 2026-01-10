# Sesion 04: Modulo Models - Losses

## Fecha: 2025-12-07

## Objetivo
Refactorizar `src_v2/models/losses.py` para eliminar constantes duplicadas, reemplazar magic numbers y agregar logging.

## Archivos Modificados
- `src_v2/models/losses.py`
- `tests/test_losses.py`

## Cambios Realizados

### 1. Eliminar constantes duplicadas y agregar imports

**Antes:**
```python
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict


# Pares simetricos (indices 0-based)
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]

# Landmarks centrales
CENTRAL_LANDMARKS = [8, 9, 10]  # L9, L10, L11
```

**Despues:**
```python
import logging

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict

from src_v2.constants import (
    SYMMETRIC_PAIRS,
    CENTRAL_LANDMARKS,
    DEFAULT_IMAGE_SIZE,
)


logger = logging.getLogger(__name__)
```

**Razon:** Centralizar constantes en constants.py (DRY) y habilitar logging.

### 2. Reemplazar magic number 224 por DEFAULT_IMAGE_SIZE

Aplicado en 5 lugares:

- `WingLoss.__init__`: `image_size: int = 224` -> `image_size: int = DEFAULT_IMAGE_SIZE`
- `WeightedWingLoss.__init__`: `image_size: int = 224` -> `image_size: int = DEFAULT_IMAGE_SIZE`
- `CentralAlignmentLoss.__init__`: `image_size: int = 224` -> `image_size: int = DEFAULT_IMAGE_SIZE`
- `SoftSymmetryLoss.__init__`: `image_size: int = 224` -> `image_size: int = DEFAULT_IMAGE_SIZE`
- `CombinedLandmarkLoss.__init__`: `image_size: int = 224` -> `image_size: int = DEFAULT_IMAGE_SIZE`

**Razon:** Usar constante nombrada para claridad y consistencia.

### 3. Actualizar imports en test_losses.py

**Antes:**
```python
from src_v2.models.losses import (
    WingLoss,
    WeightedWingLoss,
    CentralAlignmentLoss,
    SoftSymmetryLoss,
    CombinedLandmarkLoss,
    get_landmark_weights,
    SYMMETRIC_PAIRS,
    CENTRAL_LANDMARKS
)
```

**Despues:**
```python
from src_v2.models.losses import (
    WingLoss,
    WeightedWingLoss,
    CentralAlignmentLoss,
    SoftSymmetryLoss,
    CombinedLandmarkLoss,
    get_landmark_weights,
)
from src_v2.constants import SYMMETRIC_PAIRS, CENTRAL_LANDMARKS
```

**Razon:** Las constantes ahora vienen del modulo centralizado.

## Verificaciones Ejecutadas

```bash
# Sin constantes duplicadas SYMMETRIC_PAIRS
grep -c "SYMMETRIC_PAIRS = \[" src_v2/models/losses.py
# Resultado: 0

# Sin constantes duplicadas CENTRAL_LANDMARKS
grep -c "CENTRAL_LANDMARKS = \[" src_v2/models/losses.py
# Resultado: 0

# Sin magic number 224
grep -c "= 224" src_v2/models/losses.py
# Resultado: 0

# Tests
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed

# Imports funcionan
python -c "from src_v2.models.losses import WingLoss, CombinedLandmarkLoss; print('OK')"
# Resultado: Imports OK
```

## Estado Final
- [x] SYMMETRIC_PAIRS eliminado de losses.py, importado desde constants.py
- [x] CENTRAL_LANDMARKS eliminado de losses.py, importado desde constants.py
- [x] Magic number 224 reemplazado por DEFAULT_IMAGE_SIZE (5 lugares)
- [x] Logging agregado
- [x] test_losses.py actualizado para importar constantes desde constants.py
- [x] Tests pasan (50/50)
- [x] Documento de sesion creado

## Notas para Proxima Sesion
- **Modulo 5: models/resnet_landmark.py** - Importar constantes, reemplazar magic numbers (224, 30, 512)
