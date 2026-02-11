# Instalación y Requisitos

**Documentación Completa de Instalación del Sistema**

Este documento proporciona instrucciones detalladas para instalar el sistema de detección automática de landmarks pulmonares y clasificación COVID-19.

---

## Tabla de Contenidos

1. [Requisitos de Hardware](#requisitos-de-hardware)
2. [Requisitos de Software](#requisitos-de-software)
3. [Dependencias Python Explicadas](#dependencias-python-explicadas)
4. [Instalación Paso a Paso](#instalación-paso-a-paso)
5. [Descarga y Organización de Datos](#descarga-y-organización-de-datos)
6. [Colocación de Checkpoints](#colocación-de-checkpoints)
7. [Verificación de la Instalación](#verificación-de-la-instalación)
8. [Instalación para Desarrollo](#instalación-para-desarrollo)
9. [Troubleshooting Detallado](#troubleshooting-detallado)

---

## Requisitos de Hardware

### Mínimos (Inferencia)
- **CPU:** Procesador de 4 núcleos (Intel Core i5 o AMD Ryzen 5)
- **RAM:** 8 GB
- **Almacenamiento:** 10 GB disponibles
  - 2 GB: Dataset original (15,153 imágenes)
  - 184 MB: Modelos del ensemble (4 checkpoints)
  - 1 GB: Dataset normalizado (warped)
  - 2 GB: Código, dependencias y salidas
  - 5 GB: Margen de seguridad
- **GPU:** Opcional
- **Sistema operativo:** Linux, macOS, o Windows 10/11 con WSL2

### Recomendados (Entrenamiento + Inferencia)
- **CPU:** Procesador de 8+ núcleos
- **RAM:** 16 GB o más
- **GPU:** NVIDIA con 6+ GB VRAM (ej: GTX 1660, RTX 3060) o AMD con ROCm
- **Almacenamiento:** 20 GB disponibles (SSD recomendado)

### Notas sobre GPU

**Con GPU:**
- Inferencia completa del dataset (15k imágenes): ~5-10 minutos
- Entrenamiento de un modelo (100 epochs): ~2-3 horas
- Cross-validation (5 folds): ~10-15 horas

**Sin GPU (CPU únicamente):**
- Inferencia completa del dataset: ~45-60 minutos
- Entrenamiento de un modelo: ~24-48 horas
- Cross-validation: ~5-10 días

**Recomendación:** Para evaluación y reproducción de resultados, GPU no es estrictamente necesaria pero acelera significativamente el proceso. Para entrenamiento desde cero, GPU es altamente recomendada.

---

## Requisitos de Software

### Python
- **Versión:** 3.8 o superior (recomendado: 3.9, 3.10, o 3.11)
- **NO compatible con:** Python 3.7 o anterior

**Verificación:**
```bash
python --version
# o
python3 --version
```

Si no tiene Python instalado:

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip
```

**macOS:**
```bash
brew install python@3.9
```

**Windows:**
Descargue desde https://www.python.org/downloads/ y asegúrese de marcar "Add Python to PATH" durante la instalación.

### Otras Dependencias del Sistema

**Linux:**
```bash
# OpenCV dependencies
sudo apt install libgl1-mesa-glx libglib2.0-0

# Para visualización de imágenes
sudo apt install python3-tk
```

**macOS:**
```bash
# OpenCV suele funcionar sin dependencias adicionales
# Si tiene problemas:
brew install opencv
```

**Windows (WSL2 recomendado):**
WSL2 proporciona mejor compatibilidad. Siga las instrucciones de Linux dentro de WSL.

---

## Dependencias Python Explicadas

Esta sección explica **por qué** cada dependencia es necesaria.

### Deep Learning Framework

#### PyTorch >= 2.0.0
**Propósito:** Motor de deep learning para entrenamiento e inferencia de modelos.

**Por qué PyTorch:**
- API flexible y pythónica
- Excelente soporte para investigación
- ResNet-18 pre-entrenado en ImageNet disponible

**Tamaño:** ~800 MB (CPU) o ~2 GB (GPU con CUDA)

#### torchvision >= 0.15.0
**Propósito:** Utilidades de visión por computadora, modelos pre-entrenados, y transformaciones de datos.

**Uso en el proyecto:**
- `torchvision.models.resnet18`: Backbone pre-entrenado
- `torchvision.transforms`: Aumentación de datos

### Scientific Computing

#### numpy >= 2.0.0
**Propósito:** Operaciones numéricas eficientes con arrays multidimensionales.

**Uso en el proyecto:**
- Manipulación de coordenadas de landmarks (N×30 arrays)
- Cache de predicciones (.npz format)
- Álgebra lineal en GPA (Generalized Procrustes Analysis)

#### scipy >= 1.10.0
**Propósito:** Algoritmos científicos avanzados.

**Uso en el proyecto:**
- `scipy.spatial.Delaunay`: Triangulación de landmarks para warping
- Estadísticas descriptivas

#### pandas >= 2.0.0
**Propósito:** Manipulación de datos tabulares.

**Uso en el proyecto:**
- Lectura de anotaciones CSV (coordenadas_maestro.csv)
- Análisis de resultados y métricas

### Computer Vision

#### opencv-python >= 4.8.0
**Propósito:** Procesamiento de imágenes y visión por computadora.

**Uso crítico en el proyecto:**
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Preprocesamiento
- **Warping afín por partes** (`cv2.warpAffine`): Normalización geométrica
- Manipulación de imágenes (lectura, escritura, redimensionamiento)

**Alternativas:** `opencv-python-headless` (sin GUI, más ligero) funciona igual.

#### Pillow >= 10.0.0
**Propósito:** Manipulación de imágenes en Python.

**Uso en el proyecto:**
- Carga de imágenes PNG
- Conversión entre formatos
- Interfaz con PyTorch DataLoader

### Machine Learning

#### scikit-learn >= 1.3.0
**Propósito:** Herramientas de machine learning clásico.

**Uso en el proyecto:**
- Splits train/val/test estratificados
- Métricas de clasificación (accuracy, F1, precision, recall)
- Matrices de confusión
- Normalización de datos

### Visualization

#### matplotlib >= 3.7.0
**Propósito:** Generación de gráficos científicos.

**Uso en el proyecto:**
- Visualización de landmarks sobre imágenes
- Curvas de entrenamiento (loss, accuracy)
- Gráficos de error por landmark

#### seaborn >= 0.12.0
**Propósito:** Visualizaciones estadísticas de alto nivel.

**Uso en el proyecto:**
- Matrices de confusión mejoradas
- Distribuciones de error
- Gráficos de comparación

### Utilities

#### tqdm >= 4.65.0
**Propósito:** Barras de progreso para loops largos.

**Uso en el proyecto:**
- Monitoreo de inferencia (15k imágenes)
- Progreso de epochs durante entrenamiento
- Feedback visual al usuario

#### typer >= 0.9.0
**Propósito:** Framework para crear CLIs modernas.

**Uso en el proyecto:**
- Interfaz de línea de comandos (`python -m src_v2`)
- Parsing automático de argumentos
- Documentación automática (`--help`)

### Testing (Opcional, solo para desarrollo)

#### pytest >= 7.0.0
**Propósito:** Framework de testing.

#### pytest-cov >= 4.0.0
**Propósito:** Cobertura de código.

### GUI (Opcional)

#### gradio >= 4.0.0
**Propósito:** Interfaz web interactiva para demostración.

**Nota:** No está incluida en `requirements.txt` base. Instalar manualmente si se desea usar la GUI demo.

---

## Instalación Paso a Paso

### Paso 1: Preparar el Entorno

Navegue al directorio del código:

```bash
cd 02_Codigo/src_v2
```

Cree un entorno virtual (aislado del sistema):

```bash
python -m venv .venv
```

**¿Por qué un entorno virtual?**
- Evita conflictos con otras instalaciones de Python
- Permite tener diferentes versiones de librerías por proyecto
- Facilita reproducibilidad

### Paso 2: Activar el Entorno Virtual

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Verificación:**
Debería ver `(.venv)` al inicio de la línea de comandos. Además:

```bash
which python
# Linux/macOS: debe mostrar .../venv/bin/python
# Windows: debe mostrar ...\venv\Scripts\python.exe
```

### Paso 3: Actualizar pip

```bash
pip install --upgrade pip setuptools wheel
```

**Tiempo:** ~30 segundos

### Paso 4: Instalar Dependencias

#### Opción A: Instalación Básica (Solo Inferencia)

```bash
pip install -r ../../05_Documentacion/requirements.txt
```

**Tiempo:** 3-5 minutos (descarga ~1-2 GB)

**Nota:** Por defecto, instala PyTorch para CPU. Ver Paso 5 para GPU.

#### Opción B: Instalación para Desarrollo (Incluye Tests)

```bash
pip install -e ".[dev]"
```

Esto instala:
- Todas las dependencias de `requirements.txt`
- Dependencias de testing (pytest, pytest-cov)
- El paquete en modo editable (cambios en el código se reflejan inmediatamente)

### Paso 5: Configurar PyTorch para GPU (Opcional pero Recomendado)

Si tiene GPU NVIDIA con CUDA:

```bash
# Desinstalar versión CPU
pip uninstall torch torchvision

# Instalar versión CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verificar instalación:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CPU only')"
```

**Salida esperada (con GPU):**
```
CUDA available: True
CUDA version: 12.1
```

Si tiene GPU AMD con ROCm (Linux):

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

### Paso 6: Verificar Instalación

```bash
python -c "
import torch
import torchvision
import cv2
import numpy as np
import sklearn
import scipy
import pandas as pd
import matplotlib
import seaborn as sns
import tqdm
import typer
print('✓ Todas las dependencias instaladas correctamente')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
"
```

**Salida esperada:**
```
✓ Todas las dependencias instaladas correctamente
PyTorch: 2.x.x
CUDA disponible: True  (o False si no tiene GPU)
```

---

## Descarga y Organización de Datos

### Dataset COVID-19 Radiography Database

**Fuente:** Kaggle - COVID-19 Radiography Database
**URL:** https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
**Tamaño:** ~2 GB (15,153 imágenes PNG)

#### Paso 1: Descargar el Dataset

**Opción A: Interfaz Web de Kaggle**
1. Visite el enlace arriba
2. Haga clic en "Download" (requiere cuenta de Kaggle)
3. Descomprima el archivo ZIP

**Opción B: Kaggle API (Recomendado)**

```bash
# Instalar Kaggle CLI
pip install kaggle

# Configurar credenciales (ver https://www.kaggle.com/docs/api)
# Descargar ~/.kaggle/kaggle.json desde Kaggle → Account → API → Create New Token

# Descargar dataset
kaggle datasets download -d tawsifurrahman/covid19-radiography-database

# Descomprimir
unzip covid19-radiography-database.zip -d data/dataset/
```

#### Paso 2: Organizar Estructura

El dataset descargado debe quedar así:

```
data/dataset/COVID-19_Radiography_Dataset/
├── COVID/
│   ├── images/
│   │   ├── COVID-1.png
│   │   ├── COVID-2.png
│   │   └── ... (3,616 imágenes)
│   └── masks/          # Mascarillas (no usadas en este proyecto)
├── Normal/
│   ├── images/
│   │   ├── Normal-1.png
│   │   └── ... (10,192 imágenes)
│   └── masks/
└── Viral Pneumonia/
    ├── images/
    │   ├── Viral Pneumonia-1.png
    │   └── ... (1,345 imágenes)
    └── masks/
```

**Verificación:**
```bash
# Desde la raíz del proyecto
find data/dataset/COVID-19_Radiography_Dataset -type d -name "images" -exec sh -c 'echo "$(basename $(dirname {})): $(find {} -name "*.png" | wc -l)"' \;
```

**Salida esperada:**
```
COVID: 3616
Normal: 10192
Viral Pneumonia: 1345
```

### Anotaciones de Landmarks

Las anotaciones deben estar en:
```
data/coordenadas/coordenadas_maestro.csv
```

**Formato del CSV:**
```csv
image_name,category,L1_x,L1_y,L2_x,L2_y,...,L15_x,L15_y
COVID-1.png,COVID,112.5,45.3,...,156.8,201.2
Normal-1.png,Normal,108.2,43.1,...,159.3,198.5
...
```

**Estructura:**
- Columna 1: Nombre de la imagen
- Columna 2: Categoría (COVID, Normal, Viral_Pneumonia)
- Columnas 3-32: Coordenadas x,y de 15 landmarks (30 valores)

**Verificación:**
```bash
head -n 2 data/coordenadas/coordenadas_maestro.csv
wc -l data/coordenadas/coordenadas_maestro.csv
# Debe mostrar: 15154 (1 header + 15153 imágenes)
```

---

## Colocación de Checkpoints

Los modelos entrenados del ensemble deben colocarse en las rutas esperadas.

### Estructura de Checkpoints

```
checkpoints/
├── session10/
│   └── ensemble/
│       └── seed123/
│           └── final_model.pt    (46 MB)
├── session13/
│   └── seed321/
│       └── final_model.pt        (46 MB)
├── repro_split111/
│   └── session14/
│       └── seed111/
│           └── final_model.pt    (46 MB)
└── repro_split666/
    └── session16/
        └── seed666/
            └── final_model.pt    (46 MB)
```

### Copiar Modelos desde el USB

```bash
# Crear estructura de directorios
mkdir -p checkpoints/session10/ensemble/seed123
mkdir -p checkpoints/session13
mkdir -p checkpoints/repro_split111/session14
mkdir -p checkpoints/repro_split666/session16

# Copiar modelos desde 03_Modelos/ del USB
cp ../../03_Modelos/seed123_final_model.pt checkpoints/session10/ensemble/seed123/final_model.pt
cp ../../03_Modelos/seed321_final_model.pt checkpoints/session13/seed321/final_model.pt
cp ../../03_Modelos/seed111_final_model.pt checkpoints/repro_split111/session14/seed111/final_model.pt
cp ../../03_Modelos/seed666_final_model.pt checkpoints/repro_split666/session16/seed666/final_model.pt
```

**Verificación:**
```bash
# Verificar que existen y tienen el tamaño correcto
ls -lh checkpoints/**/final_model.pt
# Cada modelo debe tener ~46 MB
```

### Configuraciones

Copie las configuraciones JSON:

```bash
mkdir -p configs
cp ../../04_Configuraciones/*.json configs/
```

**Verificación:**
```bash
ls configs/*.json
# Debe listar al menos:
# - ensemble_best.json
# - landmarks_train_base.json
# - warping_best.json
# - classifier_warped_base.json
```

---

## Verificación de la Instalación

### Test Completo de Instalación

Ejecute este script para verificar que todo está correctamente instalado:

```bash
python -c "
import sys
import torch
import torchvision
import cv2
import numpy as np
import sklearn
import os

print('=== Verificación de Instalación ===\n')

# Verificar Python
print(f'✓ Python {sys.version.split()[0]}')

# Verificar dependencias
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ torchvision {torchvision.__version__}')
print(f'✓ OpenCV {cv2.__version__}')
print(f'✓ NumPy {np.__version__}')
print(f'✓ scikit-learn {sklearn.__version__}')

# Verificar GPU
if torch.cuda.is_available():
    print(f'✓ GPU disponible: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA {torch.version.cuda}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ GPU no disponible (ejecutará en CPU)')

# Verificar estructura de archivos
print('\n=== Verificación de Archivos ===\n')

checks = [
    ('Dataset', 'data/dataset/COVID-19_Radiography_Dataset'),
    ('Anotaciones', 'data/coordenadas/coordenadas_maestro.csv'),
    ('Config ensemble', 'configs/ensemble_best.json'),
    ('Modelo seed123', 'checkpoints/session10/ensemble/seed123/final_model.pt'),
    ('Modelo seed321', 'checkpoints/session13/seed321/final_model.pt'),
    ('Modelo seed111', 'checkpoints/repro_split111/session14/seed111/final_model.pt'),
    ('Modelo seed666', 'checkpoints/repro_split666/session16/seed666/final_model.pt'),
]

for name, path in checks:
    if os.path.exists(path):
        print(f'✓ {name}: {path}')
    else:
        print(f'✗ {name}: {path} (NO ENCONTRADO)')

print('\n=== Resumen ===')
print('Si todos los checks son ✓, la instalación es correcta.')
print('Si hay ✗, revise las secciones correspondientes de este documento.')
"
```

### Test de Inferencia Rápida

Pruebe que puede cargar un modelo y hacer una predicción:

```bash
python -c "
import torch
model_path = 'checkpoints/session10/ensemble/seed123/final_model.pt'
print(f'Cargando modelo: {model_path}')
model = torch.load(model_path, map_location='cpu')
model.eval()
print('✓ Modelo cargado correctamente')
print(f'Arquitectura: {model.__class__.__name__}')
"
```

---

## Instalación para Desarrollo

Si planea modificar el código o ejecutar tests:

### Instalación Editable

```bash
pip install -e ".[dev]"
```

Esto permite:
- Modificar código sin reinstalar
- Ejecutar tests con pytest
- Generar reportes de cobertura

### Ejecutar Tests

```bash
# Todos los tests
python -m pytest tests/ -v

# Con cobertura
python -m pytest tests/ -v --cov=src_v2 --cov-report=html

# Ver reporte de cobertura
# Linux: xdg-open htmlcov/index.html
# macOS: open htmlcov/index.html
```

### Pre-commit Hooks (Opcional)

Para mantener calidad de código:

```bash
pip install pre-commit
pre-commit install
```

---

## Troubleshooting Detallado

### Problema: "No module named 'torch'"

**Causa:** PyTorch no instalado o entorno virtual no activado.

**Solución:**
```bash
# Verificar que el entorno está activo
which python
# Debe mostrar ruta con .venv/

# Si no está activo
source .venv/bin/activate  # Linux/macOS
# o
.venv\Scripts\Activate.ps1  # Windows

# Reinstalar PyTorch
pip install torch torchvision
```

### Problema: ImportError: libGL.so.1

**Causa:** OpenCV requiere librerías del sistema no instaladas (Linux).

**Solución:**
```bash
sudo apt update
sudo apt install libgl1-mesa-glx libglib2.0-0
```

### Problema: "Permission denied" al crear venv

**Causa:** Falta de permisos de escritura.

**Solución:**
```bash
# Verificar permisos del directorio
ls -ld .

# Si no tiene permisos, cambie a un directorio donde los tenga
cd ~
mkdir mi_proyecto
cd mi_proyecto
# Copie el código aquí
```

### Problema: pip install muy lento

**Causa:** Conexión lenta o servidor de PyPI sobrecargado.

**Solución:**
```bash
# Usar mirror alternativo (ej: mirror de China Tsinghua)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# O aumentar timeout
pip install -r requirements.txt --timeout 300
```

### Problema: "No space left on device"

**Causa:** Disco lleno.

**Solución:**
```bash
# Verificar espacio
df -h .

# Limpiar pip cache si es necesario
pip cache purge

# Usar otro directorio para el venv
python -m venv /ruta/con/espacio/.venv
```

### Problema: PyTorch no detecta GPU

**Síntoma:** `torch.cuda.is_available()` retorna `False` aunque tiene GPU NVIDIA.

**Diagnóstico:**
```bash
# Verificar que NVIDIA drivers están instalados
nvidia-smi

# Verificar versión de CUDA del sistema
nvcc --version
```

**Solución:**
1. Instale/actualice drivers NVIDIA
2. Instale PyTorch con la versión de CUDA correcta:
   ```bash
   pip uninstall torch torchvision
   # Para CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   # Para CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Problema: ModuleNotFoundError: No module named 'src_v2'

**Causa:** Python no encuentra el módulo (problema de PYTHONPATH).

**Solución:**
```bash
# Asegúrese de estar en el directorio correcto
pwd
# Debe estar en el directorio que contiene src_v2/

# O instale en modo editable
pip install -e .
```

### Problema: "RuntimeError: DataLoader worker ... exited unexpectedly"

**Causa:** Problema con multiprocessing en DataLoader.

**Solución:**
Edite configs para usar `num_workers=0`:

```bash
# Temporal: forzar num_workers=0 en la variable de entorno
export FORCE_NUM_WORKERS_ZERO=1
```

O modifique los configs JSON:
```json
{
  "num_workers": 0
}
```

### Problema: Versiones de NumPy incompatibles

**Síntoma:** `ValueError: numpy.dtype size changed` o similar.

**Solución:**
```bash
pip install --upgrade numpy
pip install --force-reinstall --no-cache-dir numpy
```

---

## Requisitos de Almacenamiento Detallados

Resumen de espacio necesario:

| Componente | Tamaño | Ubicación |
|------------|--------|-----------|
| Dataset original | ~2 GB | `data/dataset/` |
| Anotaciones | ~2 MB | `data/coordenadas/` |
| Checkpoints ensemble | 184 MB | `checkpoints/` |
| Código fuente | ~3 MB | `src_v2/` |
| Dependencias Python | ~2-4 GB | `.venv/` |
| Dataset warpeado | ~1 GB | `outputs/warped_*` |
| Outputs intermedios | ~500 MB | `outputs/` |
| **Total** | **~6-10 GB** | |

**Recomendación:** Tener al menos 15 GB libres para margen de seguridad.

---

## Próximos Pasos

Una vez completada la instalación:

1. **Prueba rápida:** Siga `01_GUIA_INICIO_RAPIDO.md`
2. **Reproducción completa:** Consulte `05_REPRODUCIBILIDAD_COMPLETA.md`
3. **Uso del sistema:** Lea `03_GUIA_USO_CLI.md`
4. **Entender el código:** Revise `04_ARQUITECTURA_CODIGO.md`

---

**Última actualización:** 28 de enero de 2026

**Contacto para soporte técnico:**
- Estudiante: Rafael Alejandro Cruz Ovando, BUAP
- Director: Dr. Leopoldo Altamirano Robles (robles@inaoep.mx), INAOE
