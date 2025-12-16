# Sesión 01: Módulo Data - Utils

## Fecha: 2025-12-07

## Objetivo
Refactorizar `src_v2/data/utils.py` para eliminar constantes duplicadas, agregar logging y manejo de errores.

## Archivos Modificados
- `src_v2/data/utils.py`

## Cambios Realizados

### 1. Eliminación de constantes duplicadas

**Antes:**
```python
# Pares de landmarks simetricos (indices 0-based)
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]

# Landmarks centrales (deben estar sobre el eje L1-L2)
CENTRAL_LANDMARKS = [8, 9, 10]

# Nombres de landmarks
LANDMARK_NAMES = [
    'L1',  'L2',  'L3',  'L4',  'L5',  'L6',  'L7',  'L8',
    'L9',  'L10', 'L11', 'L12', 'L13', 'L14', 'L15'
]
```

**Después:**
```python
from src_v2.constants import (
    SYMMETRIC_PAIRS,
    CENTRAL_LANDMARKS,
    LANDMARK_NAMES,
    NUM_LANDMARKS,
    CATEGORIES,
)
```

**Razón:** Centralizar constantes para evitar duplicación y facilitar mantenimiento.

### 2. Agregar logging

**Agregado:**
```python
import logging
logger = logging.getLogger(__name__)
```

**Uso en funciones:**
```python
logger.info("Cargando coordenadas desde %s", csv_path)
logger.warning("Categoría desconocida para imagen: %s", name)
logger.info("Cargadas %d muestras: %s", len(df), df['category'].value_counts().to_dict())
```

### 3. Agregar manejo de errores en load_coordinates_csv()

**Agregado:**
```python
# Validar que el archivo existe
csv_path = Path(csv_path)
if not csv_path.exists():
    raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")

try:
    df = pd.read_csv(csv_path, header=None, names=columns)
except Exception as e:
    raise ValueError(f"Error al leer CSV {csv_path}: {e}") from e

# Validar número de columnas
expected_cols = 1 + (NUM_LANDMARKS * 2) + 1
if len(df.columns) != expected_cols:
    raise ValueError(...)
```

### 4. Usar constantes en lugar de magic numbers

**Antes:**
```python
for i in range(1, 16):
for i in range(15):
```

**Después:**
```python
for i in range(1, NUM_LANDMARKS + 1):
for i in range(NUM_LANDMARKS):
```

## Verificaciones Ejecutadas

```bash
# Verificar que no hay constantes duplicadas
grep -c "SYMMETRIC_PAIRS = \[" src_v2/data/utils.py
# Resultado: 0

# Verificar imports
.venv/bin/python -c "from src_v2.data.utils import SYMMETRIC_PAIRS"
# Resultado: OK

# Tests
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed
```

## Estado Final
- [x] Constantes eliminadas e importadas desde constants.py
- [x] Logging agregado
- [x] Manejo de errores agregado
- [x] Magic numbers reemplazados por constantes
- [x] Tests pasan (50/50)
- [x] Documento de sesión creado

## Notas para Próxima Sesión
- **Módulo 2: data/transforms.py** - Eliminar SYMMETRIC_PAIRS duplicado (línea 20)
