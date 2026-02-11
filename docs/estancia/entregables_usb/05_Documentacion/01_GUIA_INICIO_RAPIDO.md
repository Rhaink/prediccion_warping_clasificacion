# Guía de Inicio Rápido

**Sistema de Detección Automática de Landmarks Pulmonares y Clasificación COVID-19**

Esta guía le permitirá ejecutar el sistema completo en **menos de 30 minutos** para validar rápidamente los resultados reportados.

---

## Requisitos Previos

Antes de comenzar, verifique que cuenta con:

### Hardware Mínimo
- **RAM:** 8 GB (recomendado: 16 GB)
- **Espacio en disco:** 5 GB libres
- **GPU:** Opcional para inferencia, recomendada para entrenamiento
  - Sin GPU: ~45 min para inferencia completa del dataset
  - Con GPU: ~5 min para inferencia completa del dataset

### Software
- **Python:** 3.8 o superior (recomendado: 3.9+)
- **Sistema operativo:** Linux, macOS o Windows con WSL

### Verificar Instalación de Python

```bash
python --version
# Debe mostrar: Python 3.8.x o superior
```

---

## Instalación en 3 Pasos

### Paso 1: Crear Entorno Virtual

Abra una terminal y navegue hasta la carpeta `02_Codigo/src_v2`:

```bash
cd 02_Codigo/src_v2
python -m venv .venv
```

Active el entorno virtual:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Verificación:**
Debería ver `(.venv)` al inicio de la línea de comandos.

### Paso 2: Instalar Dependencias

Con el entorno virtual activo:

```bash
pip install --upgrade pip
pip install -r ../../05_Documentacion/requirements.txt
```

**Tiempo estimado:** 3-5 minutos

**Notas importantes:**
- Si tiene GPU NVIDIA con CUDA, instale PyTorch con soporte CUDA:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- Si tiene GPU AMD con ROCm:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
  ```

### Paso 3: Organizar Archivos

Copie los modelos del ensemble a la ubicación esperada:

```bash
mkdir -p checkpoints/session10/ensemble/seed123
mkdir -p checkpoints/session13
mkdir -p checkpoints/repro_split111/session14
mkdir -p checkpoints/repro_split666/session16

# Copiar modelos desde el USB
cp ../../03_Modelos/seed123_final_model.pt checkpoints/session10/ensemble/seed123/final_model.pt
cp ../../03_Modelos/seed321_final_model.pt checkpoints/session13/seed321/final_model.pt
cp ../../03_Modelos/seed111_final_model.pt checkpoints/repro_split111/session14/seed111/final_model.pt
cp ../../03_Modelos/seed666_final_model.pt checkpoints/repro_split666/session16/seed666/final_model.pt
```

Copie las configuraciones:

```bash
mkdir -p configs
cp ../../04_Configuraciones/*.json configs/
```

---

## Ejecución Rápida

### Opción A: Evaluación del Ensemble (Solo Verificación)

Si solo desea **verificar las métricas reportadas** sin entrenar nada:

**IMPORTANTE:** Necesita el dataset de prueba. Si no lo tiene, descárguelo de:
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Una vez descargado, organícelo así:
```
data/dataset/COVID-19_Radiography_Dataset/
├── COVID/
│   └── images/
├── Normal/
│   └── images/
└── Viral Pneumonia/
    └── images/
```

Luego ejecute:

```bash
python scripts/evaluate_ensemble_from_config.py \
  --config configs/ensemble_best.json
```

**Tiempo estimado:**
- Con GPU: ~5-8 minutos
- Sin GPU: ~30-45 minutos

**Salida esperada:**
```
=== Ensemble Evaluation Results ===
Mean Pixel Error: 3.61 px
Std: 2.48 px
Median: 3.07 px
Max Error: 27.3 px

Error by Category:
  Normal:           3.22 px
  COVID:            3.93 px
  Viral_Pneumonia:  4.11 px
```

### Opción B: Pipeline Completo Automático

Para ejecutar el pipeline completo (GPA + Predicciones + Warping):

```bash
bash scripts/quickstart_warping.sh
```

Este script ejecuta automáticamente:
1. **Forma canónica (GPA):** Calcula la forma pulmonar de consenso
2. **Predicción de landmarks:** Genera predicciones para todo el dataset
3. **Warping:** Normaliza las imágenes geométricamente

**Tiempo estimado:**
- Con GPU: ~15-20 minutos
- Sin GPU: ~1-2 horas

**Salida generada:**
```
outputs/shape_analysis/
├── canonical_shape_gpa.json          # Forma canónica
├── canonical_delaunay_triangles.json # Triangulación
└── canonical_shape_visualization.png # Visualización

outputs/landmark_predictions/session_warping/
└── predictions.npz                    # Cache de predicciones

outputs/warped_lung_best/session_warping/
├── train/                             # Dataset normalizado
├── val/
└── test/
```

---

## Verificación de Resultados

### 1. Verificar Forma Canónica

Abra la imagen generada:

```bash
# Linux
xdg-open outputs/shape_analysis/canonical_shape_visualization.png

# macOS
open outputs/shape_analysis/canonical_shape_visualization.png

# Windows
start outputs/shape_analysis/canonical_shape_visualization.png
```

Debería ver:
- 15 landmarks marcados en rojo
- Triangulación de Delaunay en azul
- Forma simétrica que representa el pulmón promedio

### 2. Verificar Predicciones Cacheadas

```bash
python -c "
import numpy as np
cache = np.load('outputs/landmark_predictions/session_warping/predictions.npz', allow_pickle=True)
print(f'Imágenes procesadas: {len(cache[\"image_paths\"])}')
print(f'Forma predicciones: {cache[\"predictions\"].shape}')
print(f'Modelos usados: {len(cache[\"models\"])}')
print(f'TTA activado: {cache[\"tta\"]}')
print(f'CLAHE activado: {cache[\"clahe\"]}')
"
```

**Salida esperada:**
```
Imágenes procesadas: 15153
Forma predicciones: (15153, 30)
Modelos usados: 4
TTA activado: True
CLAHE activado: True
```

### 3. Verificar Dataset Normalizado

```bash
# Contar imágenes warpeadas
find outputs/warped_lung_best/session_warping -name "*.png" | wc -l
# Debería mostrar: 15153

# Ver distribución por split
for split in train val test; do
  echo "$split: $(find outputs/warped_lung_best/session_warping/$split -name "*.png" | wc -l)"
done
```

**Salida esperada:**
```
train: 11364  (75%)
val: 1894     (12.5%)
test: 1895    (12.5%)
```

### 4. Inspeccionar Imagen Warpeada

Abra una imagen normalizada para visualizar el resultado:

```bash
# Linux
xdg-open outputs/warped_lung_best/session_warping/train/COVID/COVID-1.png

# macOS
open outputs/warped_lung_best/session_warping/train/COVID/COVID-1.png
```

Debería ver:
- Imagen de 224×224 píxeles
- Pulmones centrados y alineados
- Forma geométrica normalizada

---

## Errores Comunes y Soluciones

### Error: "No module named 'torch'"

**Causa:** PyTorch no está instalado.

**Solución:**
```bash
pip install torch torchvision
```

### Error: "FileNotFoundError: data/dataset/COVID-19_Radiography_Dataset"

**Causa:** El dataset no está descargado o no está en la ubicación esperada.

**Solución:**
1. Descargue el dataset de Kaggle (link arriba)
2. Descomprímalo en `data/dataset/COVID-19_Radiography_Dataset`
3. Verifique la estructura con:
   ```bash
   ls -R data/dataset/COVID-19_Radiography_Dataset
   ```

### Error: "FileNotFoundError: checkpoints/session10/ensemble/seed123/final_model.pt"

**Causa:** Los modelos no están en la ubicación esperada.

**Solución:**
Vuelva a ejecutar el Paso 3 de la instalación (Organizar Archivos).

### Error: "RuntimeError: CUDA out of memory"

**Causa:** GPU sin memoria suficiente.

**Soluciones:**
1. Reduzca el batch size en los configs:
   ```bash
   # Editar configs/landmarks_train_base.json
   # Cambiar "batch_size": 16 → "batch_size": 8
   ```
2. O ejecute en CPU (más lento pero funciona):
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

### Error: "OSError: [Errno 28] No space left on device"

**Causa:** Disco lleno.

**Solución:**
Libere al menos 5 GB de espacio o cambie el directorio de salida:
```bash
export OUTPUT_DIR="/ruta/con/espacio/outputs"
```

### Advertencia: "UserWarning: CLAHE tile size adjusted"

**Esto NO es un error.** OpenCV ajusta automáticamente el tamaño de tile si no es compatible con las dimensiones de la imagen. El sistema funciona correctamente.

---

## Próximos Pasos

Una vez completada esta guía de inicio rápido:

1. **Para entender el sistema en profundidad:**
   - Lea `04_ARQUITECTURA_CODIGO.md`

2. **Para reproducir resultados exactos:**
   - Siga `05_REPRODUCIBILIDAD_COMPLETA.md`

3. **Para usar comandos específicos:**
   - Consulte `03_GUIA_USO_CLI.md`

4. **Para entrenar sus propios modelos:**
   - Revise `02_INSTALACION_REQUISITOS.md` (sección "Entrenamiento desde Cero")
   - Estudie `06_CONFIGURACIONES_JSON.md`

5. **Para entender los modelos del ensemble:**
   - Lea `07_MODELOS_ENTRENADOS.md`

---

## Resumen de Comandos

```bash
# 1. Crear entorno
python -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r ../../05_Documentacion/requirements.txt

# 3. Organizar archivos (ver sección completa arriba)

# 4. Verificar ensemble (rápido)
python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json

# 5. Pipeline completo (automático)
bash scripts/quickstart_warping.sh

# 6. Verificar resultados
python -c "import numpy as np; cache = np.load('outputs/landmark_predictions/session_warping/predictions.npz', allow_pickle=True); print(f'Imágenes: {len(cache[\"image_paths\"])}')"
```

---

## Soporte

Si encuentra problemas no listados aquí:

1. Revise `10_PREGUNTAS_FRECUENTES.md`
2. Consulte `02_INSTALACION_REQUISITOS.md` (sección "Troubleshooting Detallado")
3. Verifique que todas las métricas coincidan con `GROUND_TRUTH.json`

**Contacto:**
- Estudiante: Rafael Alejandro Cruz Ovando, BUAP
- Director: Dr. Leopoldo Altamirano Robles (robles@inaoep.mx), INAOE

---

**Última actualización:** 28 de enero de 2026
