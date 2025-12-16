# Sesion 05: Modulo Models - ResNet Landmark

## Fecha: 2025-12-07

## Objetivo
Refactorizar `src_v2/models/resnet_landmark.py` para importar constantes desde constants.py y reemplazar magic numbers.

## Archivos Modificados
- `src_v2/models/resnet_landmark.py`

## Cambios Realizados

### 1. Agregar imports de logging y constantes

**Antes:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Tuple
```

**Despues:**
```python
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Tuple

from src_v2.constants import (
    NUM_LANDMARKS,
    DEFAULT_IMAGE_SIZE,
    BACKBONE_FEATURE_DIM,
)


logger = logging.getLogger(__name__)
```

**Razon:** Centralizar constantes en constants.py (DRY) y habilitar logging.

### 2. Reemplazar num_landmarks=15 por NUM_LANDMARKS

Aplicado en 2 lugares:

- `ResNet18Landmarks.__init__`: `num_landmarks: int = 15` -> `num_landmarks: int = NUM_LANDMARKS`
- `create_model()`: `num_landmarks: int = 15` -> `num_landmarks: int = NUM_LANDMARKS`

### 3. Reemplazar feature_dim=512 por BACKBONE_FEATURE_DIM

**Antes:**
```python
self.coord_attention = CoordinateAttention(512, reduction=32) if use_coord_attention else None
...
self.feature_dim = 512
```

**Despues:**
```python
self.coord_attention = CoordinateAttention(BACKBONE_FEATURE_DIM, reduction=32) if use_coord_attention else None
...
self.feature_dim = BACKBONE_FEATURE_DIM
```

**Razon:** BACKBONE_FEATURE_DIM=512 es especifico de ResNet-18 y esta definido en constants.py.

### 4. Reemplazar image_size=224 por DEFAULT_IMAGE_SIZE

**Antes:**
```python
def predict_landmarks(
    self,
    x: torch.Tensor,
    image_size: int = 224
) -> torch.Tensor:
```

**Despues:**
```python
def predict_landmarks(
    self,
    x: torch.Tensor,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
```

## Verificaciones Ejecutadas

```bash
# Sin magic numbers como valores por defecto
grep -n "= 15\|= 224\|= 512" src_v2/models/resnet_landmark.py
# Resultado: (vacio - no hay magic numbers)

# Tests
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed

# Imports funcionan
python -c "from src_v2.models.resnet_landmark import create_model, ResNet18Landmarks; print('OK')"
# Resultado: Imports OK
```

## Estado Final
- [x] NUM_LANDMARKS importado desde constants.py
- [x] DEFAULT_IMAGE_SIZE importado desde constants.py
- [x] BACKBONE_FEATURE_DIM importado desde constants.py
- [x] Magic numbers reemplazados (15, 224, 512)
- [x] Logging agregado
- [x] Tests pasan (50/50)
- [x] Documento de sesion creado

## Notas para Proxima Sesion
- **Modulo 6: models/hierarchical.py** - Eliminar prints de debug, importar constantes
