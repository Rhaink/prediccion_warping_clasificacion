# Sesion 09: Modulo CLI con Typer

## Fecha: 2025-12-07

## Objetivo
Crear interfaz CLI mejorada con Typer que unifique los scripts principales del proyecto, integrando con Hydra para configuracion.

## Archivos Creados
- `src_v2/cli.py` - CLI principal con comandos train/evaluate/predict/warp
- `src_v2/__main__.py` - Entry point para `python -m src_v2`

## Comandos Implementados

### 1. train
Entrenar modelo de prediccion de landmarks en dos fases.

```bash
python -m src_v2 train --help
python -m src_v2 train --data-root data/ --phase1-epochs 15 --phase2-epochs 100
python -m src_v2 train --config-path src_v2/conf --config-name config
```

**Opciones principales:**
- `--config-path`, `-c`: Path al directorio de configuracion Hydra
- `--config-name`, `-n`: Nombre del archivo de configuracion
- `--data-root`, `-d`: Directorio raiz de datos
- `--csv-path`: Path al CSV de coordenadas
- `--checkpoint-dir`: Directorio para checkpoints
- `--device`: auto, cuda, cpu, mps
- `--seed`, `-s`: Seed para reproducibilidad
- `--phase1-epochs`: Epocas fase 1 (backbone congelado)
- `--phase2-epochs`: Epocas fase 2 (fine-tuning)
- `--batch-size`, `-b`: Tamano de batch

### 2. evaluate
Evaluar modelo en dataset de test con metricas detalladas.

```bash
python -m src_v2 evaluate checkpoint.pt
python -m src_v2 evaluate checkpoint.pt --tta --output results.json
```

**Opciones principales:**
- `CHECKPOINT`: Path al checkpoint del modelo (requerido)
- `--data-root`, `-d`: Directorio raiz de datos
- `--csv-path`: Path al CSV de coordenadas
- `--device`: auto, cuda, cpu, mps
- `--batch-size`, `-b`: Tamano de batch
- `--output`, `-o`: Guardar resultados en JSON
- `--tta`: Usar Test-Time Augmentation

### 3. predict
Predecir landmarks en una imagen de rayos X.

```bash
python -m src_v2 predict xray.png --checkpoint model.pt
python -m src_v2 predict xray.png -c model.pt --output viz.png --json coords.json
```

**Opciones principales:**
- `IMAGE`: Path a la imagen de rayos X (requerido)
- `--checkpoint`, `-c`: Path al checkpoint del modelo (requerido)
- `--device`: auto, cuda, cpu, mps
- `--output`, `-o`: Guardar imagen con landmarks visualizados
- `--json`, `-j`: Guardar coordenadas en JSON

### 4. warp
Aplicar warping geometrico a un dataset de imagenes.

```bash
python -m src_v2 warp input/ output/ --checkpoint model.pt
python -m src_v2 warp data/images/ data/warped/ -c model.pt -m 1.05
```

**Opciones principales:**
- `INPUT_DIR`: Directorio con imagenes de entrada (requerido)
- `OUTPUT_DIR`: Directorio de salida para imagenes warpeadas (requerido)
- `--checkpoint`, `-c`: Path al checkpoint del modelo (requerido)
- `--canonical`: Path a la forma canonica (.npy)
- `--triangles`: Path a los triangulos de Delaunay (.npy)
- `--margin-scale`, `-m`: Factor de escala para margenes
- `--device`: auto, cuda, cpu, mps
- `--pattern`, `-p`: Patron glob para buscar imagenes

### 5. version
Mostrar version del paquete.

```bash
python -m src_v2 version
# Output: COVID-19 Landmark Detection v2.0.0
```

## Arquitectura del CLI

```
src_v2/
├── __main__.py    # Entry point: python -m src_v2
└── cli.py         # Comandos Typer
    ├── app = typer.Typer()
    ├── get_device()           # Deteccion automatica de dispositivo
    ├── setup_hydra_config()   # Carga de configuracion Hydra
    ├── @app.command() train
    ├── @app.command() evaluate
    ├── @app.command() predict
    ├── @app.command() warp
    └── @app.command() version
```

## Integracion con Hydra

El comando `train` integra con Hydra para cargar configuracion:

```python
# CLI con override de valores
python -m src_v2 train \
    --config-path src_v2/conf \
    --config-name config \
    --data-root /path/to/data

# Los valores de CLI tienen prioridad sobre la config
```

## Verificaciones Ejecutadas

```bash
# CLI help funciona
python -m src_v2 --help
# Resultado: OK - Muestra 5 comandos

python -m src_v2 train --help
# Resultado: OK - Muestra todas las opciones

python -m src_v2 evaluate --help
# Resultado: OK - Muestra todas las opciones

python -m src_v2 predict --help
# Resultado: OK - Muestra todas las opciones

python -m src_v2 warp --help
# Resultado: OK - Muestra todas las opciones

python -m src_v2 version
# Resultado: COVID-19 Landmark Detection v2.0.0

# Tests
.venv/bin/python -m pytest tests/ -v
# Resultado: 50 passed
```

## Dependencias Utilizadas

Ya presentes en requirements.txt:
- `typer>=0.9.0` - CLI framework
- `hydra-core>=1.3.0` - Configuracion
- `omegaconf>=2.3.0` - YAML config

## Imports y Modulos Integrados

```python
# Modulos de src_v2 utilizados por el CLI
from src_v2.constants import DEFAULT_IMAGE_SIZE, NUM_LANDMARKS, LANDMARK_NAMES, IMAGENET_MEAN, IMAGENET_STD
from src_v2.data import LandmarkDataset, get_train_transforms, get_val_transforms
from src_v2.models import create_model, CombinedLandmarkLoss
from src_v2.training.trainer import LandmarkTrainer
from src_v2.evaluation.metrics import evaluate_model, evaluate_model_with_tta

# Script externo para warp
from scripts.piecewise_affine_warp import piecewise_affine_warp
```

## Estado Final
- [x] CLI creado con Typer
- [x] Comando `train` implementado
- [x] Comando `evaluate` implementado
- [x] Comando `predict` implementado
- [x] Comando `warp` implementado
- [x] Comando `version` implementado
- [x] Integracion con Hydra
- [x] `python -m src_v2` funciona
- [x] Tests pasan (50/50)
- [x] Documento de sesion creado

## Notas para Proxima Sesion
- **Modulo 10: Tests de Integracion** - Crear tests para CLI y pipeline completo
- Considerar agregar smoke tests para cada comando del CLI
