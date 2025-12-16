# Sesion 07: Modulo Training (Trainer + Callbacks)

## Fecha: 2025-12-07

## Objetivo
Refactorizar `src_v2/training/trainer.py` y `src_v2/training/callbacks.py` para reemplazar print() por logging.

## Archivos Modificados
- `src_v2/training/trainer.py`
- `src_v2/training/callbacks.py`

## Cambios Realizados

### trainer.py

#### 1. Agregar imports de logging y constantes

**Antes:**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
...
```

**Despues:**
```python
import logging
import time
from pathlib import Path
...
from src_v2.constants import DEFAULT_IMAGE_SIZE
...
logger = logging.getLogger(__name__)
```

#### 2. Reemplazar magic number 224 por DEFAULT_IMAGE_SIZE

```python
# Antes
image_size: int = 224

# Despues
image_size: int = DEFAULT_IMAGE_SIZE
```

#### 3. Reemplazar 16 print() por logger.info()

Ejemplos:

**Antes:**
```python
print("=" * 60)
print("PHASE 1: Training head only (backbone + coord_attention frozen)")
print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s)")
print(f"  Train: loss={train_metrics['loss']:.4f}, error={train_metrics['error_px']:.2f}px")
```

**Despues:**
```python
logger.info("=" * 60)
logger.info("PHASE 1: Training head only (backbone + coord_attention frozen)")
logger.info(
    "Epoch %d/%d (%.1fs) - Train: loss=%.4f, error=%.2fpx - Val: loss=%.4f, error=%.2fpx",
    epoch + 1, epochs, elapsed,
    train_metrics['loss'], train_metrics['error_px'],
    val_metrics['val_loss'], val_metrics['val_error_px']
)
```

### callbacks.py

#### 1. Agregar imports de logging

```python
import logging
...
logger = logging.getLogger(__name__)
```

#### 2. Reemplazar 5 print() por logger.info()/logger.warning()

- `EarlyStopping.__call__`: 2 print() -> logger.info()
- `ModelCheckpoint.__call__`: 1 print() -> logger.info()
- `ModelCheckpoint.load_best`: 2 print() -> logger.warning() y logger.info()

## Verificaciones Ejecutadas

```bash
# Sin print() en trainer.py
grep -c "print(" src_v2/training/trainer.py
# Resultado: 0

# Sin print() en callbacks.py
grep -c "print(" src_v2/training/callbacks.py
# Resultado: 0

# Tests
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed

# Imports funcionan
python -c "from src_v2.training.trainer import LandmarkTrainer; print('OK')"
# Resultado: Imports OK
```

## Estado Final
- [x] trainer.py: 16 print() reemplazados por logger.info()
- [x] trainer.py: DEFAULT_IMAGE_SIZE importado
- [x] callbacks.py: 5 print() reemplazados por logger.info()/warning()
- [x] Logging agregado a ambos archivos
- [x] Tests pasan (50/50)
- [x] Documento de sesion creado

## Notas para Proxima Sesion
- **Modulo 8: evaluation/metrics.py** - Eliminar SYMMETRIC_PAIRS y LANDMARK_NAMES duplicados, corregir type hints
