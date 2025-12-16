# Sesion 06: Modulo Models - Hierarchical

## Fecha: 2025-12-07

## Objetivo
Refactorizar `src_v2/models/hierarchical.py` para eliminar print() de debug e importar constantes desde constants.py.

## Archivos Modificados
- `src_v2/models/hierarchical.py`

## Cambios Realizados

### 1. Agregar imports de logging y constantes

**Antes:**
```python
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
```

**Despues:**
```python
import logging

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

from src_v2.constants import (
    SYMMETRIC_PAIRS,
    CENTRAL_LANDMARKS,
    BACKBONE_FEATURE_DIM,
)


logger = logging.getLogger(__name__)
```

**Razon:** Centralizar constantes y habilitar logging estructurado.

### 2. Reemplazar print() por logging.debug() en bloque __main__

**Antes:**
```python
if __name__ == "__main__":
    # Test
    model = HierarchicalLandmarkModel()
    ...
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    ...
    print(f"L10 perpendicular distance to axis: {dist.mean():.6f} (should be ~0)")
```

**Despues:**
```python
if __name__ == "__main__":
    # Test - configurar logging para ver salida
    logging.basicConfig(level=logging.DEBUG)

    # Test
    model = HierarchicalLandmarkModel()
    ...
    logger.debug("Input shape: %s", x.shape)
    logger.debug("Output shape: %s", out.shape)
    logger.debug("Output range: [%.3f, %.3f]", out.min(), out.max())
    ...
    logger.debug("L10 perpendicular distance to axis: %.6f (should be ~0)", dist.mean())
```

**Razon:** Usar logging estructurado en lugar de print() para debug.

## Verificaciones Ejecutadas

```bash
# Sin print() statements
grep -c "print(" src_v2/models/hierarchical.py
# Resultado: 0

# Tests
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed

# Imports funcionan
python -c "from src_v2.models.hierarchical import HierarchicalLandmarkModel; print('OK')"
# Resultado: Imports OK
```

## Estado Final
- [x] print() reemplazados por logger.debug() (4/4)
- [x] Constantes importadas desde constants.py
- [x] Logging agregado
- [x] Tests pasan (50/50)
- [x] Documento de sesion creado

## Notas para Proxima Sesion
- **Modulo 7: training/trainer.py y callbacks.py** - Reemplazar 16+5 print() por logging
