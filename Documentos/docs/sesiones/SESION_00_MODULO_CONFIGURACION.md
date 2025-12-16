# Sesión 00: Módulo Configuración y Constantes

## Fecha: 2025-12-07

## Objetivo
Crear la infraestructura base para configuración centralizada con Hydra y constantes del dominio en un único archivo.

## Archivos Creados
- `src_v2/constants.py` - Constantes centralizadas del dominio
- `src_v2/conf/config.yaml` - Configuración principal Hydra
- `src_v2/conf/model/resnet18.yaml` - Configuración del modelo
- `src_v2/conf/training/default.yaml` - Configuración de entrenamiento
- `src_v2/conf/data/default.yaml` - Configuración de datos
- `.claude/ESTADO_PROYECTO.md` - Archivo de estado para continuidad
- `docs/sesiones/SESION_00_MODULO_CONFIGURACION.md` - Este documento

## Archivos Modificados
- `requirements.txt` - Agregadas dependencias: hydra-core, omegaconf, typer, pytest-cov

## Cambios Realizados

### 1. Creación de constants.py

**Contenido principal:**
```python
# Landmarks
NUM_LANDMARKS = 15
NUM_COORDINATES = 30
LANDMARK_NAMES = ['L1', ..., 'L15']

# Pares simétricos (indices 0-based)
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]

# Landmarks centrales
CENTRAL_LANDMARKS = [8, 9, 10]

# Dimensiones
DEFAULT_IMAGE_SIZE = 224

# Normalización ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Categorías
CATEGORIES = ['COVID', 'Normal', 'Viral_Pneumonia']
```

**Razón:** Centralizar todas las constantes que estaban duplicadas en múltiples archivos (utils.py, transforms.py, losses.py, metrics.py).

### 2. Configuración Hydra

Estructura creada:
```
src_v2/conf/
├── config.yaml       # Config principal con defaults
├── model/
│   └── resnet18.yaml # Arquitectura del modelo
├── training/
│   └── default.yaml  # Entrenamiento en dos fases
└── data/
    └── default.yaml  # Preprocesamiento y augmentación
```

**Razón:** Reemplazar el uso inconsistente de final_config.json por una solución tipada y composable.

### 3. Actualización de requirements.txt

**Agregado:**
```
hydra-core>=1.3.0
omegaconf>=2.3.0
typer>=0.9.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

## Verificaciones Ejecutadas

```bash
# Import de constants.py
.venv/bin/python -c "from src_v2.constants import *; print('OK')"
# Resultado: OK

# Import de Hydra
.venv/bin/python -c "from hydra import initialize_config_dir; print('OK')"
# Resultado: OK

# Tests existentes
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed, 20 warnings
```

## Dependencias Instaladas
- hydra-core 1.3.2
- omegaconf 2.3.0
- typer 0.20.0
- pytest-cov 7.0.0
- pytest 9.0.2

## Estado Final
- [x] constants.py creado con todas las constantes
- [x] Hydra configs creados
- [x] requirements.txt actualizado
- [x] Import funciona sin errores
- [x] Tests existentes pasan (50/50)
- [x] Documento de sesión creado

## Notas para Próxima Sesión
- **Módulo 1: data/utils.py** - Eliminar constantes duplicadas (SYMMETRIC_PAIRS, CENTRAL_LANDMARKS, LANDMARK_NAMES) e importar desde constants.py
- Actualizar imports en tests/test_losses.py líneas 21-22 cuando se refactorice losses.py
