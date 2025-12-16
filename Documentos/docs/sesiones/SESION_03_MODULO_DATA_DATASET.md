# Sesion 03: Modulo Data - Dataset

## Fecha: 2025-12-07

## Objetivo
Refactorizar `src_v2/data/dataset.py` para reemplazar print() por logging, importar constantes desde constants.py y agregar manejo de errores para imagenes corruptas.

## Archivos Modificados
- `src_v2/data/dataset.py`

## Cambios Realizados

### 1. Agregar logging y eliminar constante duplicada

**Antes:**
```python
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
...
from .transforms import get_train_transforms, get_val_transforms


# Pesos por categoria para sobremuestreo
# COVID es mas dificil (11.74 px vs 7.79 px Normal), necesita mas peso
DEFAULT_CATEGORY_WEIGHTS = {
    'COVID': 2.0,           # Doble peso - categoria mas dificil
    'Normal': 1.0,          # Peso base
    'Viral_Pneumonia': 1.2  # Ligeramente mas peso
}
```

**Despues:**
```python
import logging

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
...
from src_v2.constants import DEFAULT_CATEGORY_WEIGHTS, ORIGINAL_IMAGE_SIZE
from .utils import load_coordinates_csv, get_image_path, get_landmarks_array
from .transforms import get_train_transforms, get_val_transforms


logger = logging.getLogger(__name__)
```

**Razon:** Centralizar constantes en constants.py (DRY) y habilitar logging estructurado.

### 2. Reemplazar magic number 299 por ORIGINAL_IMAGE_SIZE

**Antes:**
```python
def __init__(
    self,
    df: pd.DataFrame,
    data_root: str,
    transform: Optional[Callable] = None,
    original_size: int = 299
):
```

**Despues:**
```python
def __init__(
    self,
    df: pd.DataFrame,
    data_root: str,
    transform: Optional[Callable] = None,
    original_size: int = ORIGINAL_IMAGE_SIZE
):
```

**Razon:** Usar constante nombrada para claridad y consistencia.

### 3. Reemplazar print() por logging en create_dataloaders()

**Antes:**
```python
print(f"Dataset split:")
print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
```

**Despues:**
```python
logger.info(
    "Dataset split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
    len(train_df), len(train_df) / len(df) * 100,
    len(val_df), len(val_df) / len(df) * 100,
    len(test_df), len(test_df) / len(df) * 100
)
```

**Razon:** Logging estructurado permite controlar niveles y destinos de salida.

### 4. Reemplazar print() del WeightedRandomSampler

**Antes:**
```python
print(f"  Using WeightedRandomSampler with category weights:")
weights_dict = category_weights if category_weights else DEFAULT_CATEGORY_WEIGHTS
for cat, weight in weights_dict.items():
    count = len(train_df[train_df['category'] == cat])
    print(f"    {cat}: weight={weight}, count={count}")
```

**Despues:**
```python
weights_dict = category_weights if category_weights else DEFAULT_CATEGORY_WEIGHTS
weight_info = ", ".join(
    f"{cat}(w={w}, n={len(train_df[train_df['category'] == cat])})"
    for cat, w in weights_dict.items()
)
logger.info("Using WeightedRandomSampler: %s", weight_info)
```

**Razon:** Consolidar en una linea de log estructurada.

### 5. Agregar manejo de errores para imagenes corruptas

**Antes:**
```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    row = self.df.iloc[idx]

    # Cargar imagen
    image_path = get_image_path(
        row['image_name'],
        row['category'],
        self.data_root
    )
    image = Image.open(image_path).convert('RGB')
    ...
```

**Despues:**
```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    ...
    Raises:
        FileNotFoundError: Si la imagen no existe
        IOError: Si la imagen esta corrupta o no se puede leer
    """
    row = self.df.iloc[idx]
    image_name = row['image_name']
    category = row['category']

    # Cargar imagen con manejo de errores
    image_path = get_image_path(image_name, category, self.data_root)

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        logger.error("Image not found: %s (idx=%d)", image_path, idx)
        raise FileNotFoundError(f"Image not found: {image_path}")
    except (OSError, IOError) as e:
        logger.error("Corrupt or unreadable image: %s (idx=%d): %s", image_path, idx, e)
        raise IOError(f"Corrupt or unreadable image: {image_path}") from e
    ...
```

**Razon:** Proporcionar mensajes de error claros y logging para debugging de datos corruptos.

## Verificaciones Ejecutadas

```bash
# Sin print() statements
grep -c "print(" src_v2/data/dataset.py
# Resultado: 0

# Sin constantes duplicadas
grep -c "DEFAULT_CATEGORY_WEIGHTS = {" src_v2/data/dataset.py
# Resultado: 0

# Tests
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed

# Imports funcionan
python -c "from src_v2.data.dataset import LandmarkDataset, create_dataloaders; print('OK')"
# Resultado: Imports OK
```

## Estado Final
- [x] print() reemplazados por logging (6/6)
- [x] DEFAULT_CATEGORY_WEIGHTS importado desde constants.py
- [x] ORIGINAL_IMAGE_SIZE importado desde constants.py
- [x] Magic number 299 reemplazado
- [x] Manejo de errores para imagenes corruptas agregado
- [x] Tests pasan (50/50)
- [x] Documento de sesion creado

## Notas para Proxima Sesion
- **Modulo 4: models/losses.py** - Eliminar SYMMETRIC_PAIRS y CENTRAL_LANDMARKS duplicados, actualizar test imports
