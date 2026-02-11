# Guía de Reproducibilidad Completa

**Reproducción Paso a Paso de Resultados Reportados**

Este documento le permitirá reproducir **exactamente** los resultados experimentales reportados en el documento "REPORTE_ESTANCIA_INAOE.pdf". Es crucial para la validación académica del trabajo.

---

## Tabla de Contenidos

1. [Objetivos de Reproducción](#objetivos-de-reproducción)
2. [Prerrequisitos](#prerrequisitos)
3. [Fase 1: Validación del Ensemble de Landmarks](#fase-1-validación-del-ensemble-de-landmarks)
4. [Fase 2: Generación del Dataset Normalizado](#fase-2-generación-del-dataset-normalizado)
5. [Fase 3: Clasificación COVID-19](#fase-3-clasificación-covid-19)
6. [Checklist de Verificación](#checklist-de-verificación)
7. [Tiempos Estimados](#tiempos-estimados)
8. [Troubleshooting de Reproducción](#troubleshooting-de-reproducción)

---

## Objetivos de Reproducción

Al completar esta guía, habrá reproducido:

### Detección de Landmarks (Objetivo Principal)
- **Error medio del ensemble:** 3.61 píxeles (en imágenes de 224×224)
- **Desviación estándar:** 2.48 px
- **Error mediano:** 3.07 px
- **Error por categoría:**
  - Normal: 3.22 px
  - COVID: 3.93 px
  - Viral Pneumonia: 4.11 px

### Clasificación COVID-19
- **Accuracy (5-fold CV):** 98.60% ± 0.26%
- **F1-macro (5-fold CV):** 98.00% ± 0.36%
- **F1-weighted (5-fold CV):** 98.60% ± 0.25%

### Ensemble de Clasificadores (con TTA)
- **Accuracy (test set):** 98.26%
- **F1-macro:** 97.12%
- **Impacto de TTA:** +0.16% accuracy
  - Casos mejorados: 6
  - Casos empeorados: 3
  - Mejora neta: +3 casos

**Fuente de referencia:** Todas las métricas están documentadas en `05_Documentacion/GROUND_TRUTH.json` (versión 2.1.0).

---

## Prerrequisitos

### 1. Instalación Completa

Debe haber completado la instalación descrita en `02_INSTALACION_REQUISITOS.md` o en `01_GUIA_INICIO_RAPIDO.md`.

Verifique que tiene:
```bash
# Entorno virtual activo
which python
# Debe mostrar: .../venv/bin/python

# Dependencias instaladas
python -c "import torch, torchvision, cv2, sklearn; print('OK')"
# Debe mostrar: OK
```

### 2. Dataset Completo

Descargue el dataset COVID-19 Radiography Database:
- **Fuente:** https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **Tamaño:** ~2 GB (15,153 imágenes)

Organice así:
```
data/dataset/COVID-19_Radiography_Dataset/
├── COVID/
│   └── images/
│       ├── COVID-1.png
│       ├── COVID-2.png
│       └── ... (3,616 imágenes)
├── Normal/
│   └── images/
│       ├── Normal-1.png
│       └── ... (10,192 imágenes)
└── Viral Pneumonia/
    └── images/
        ├── Viral Pneumonia-1.png
        └── ... (1,345 imágenes)
```

**Verificación:**
```bash
find data/dataset/COVID-19_Radiography_Dataset -name "*.png" | wc -l
# Debe mostrar: 15153
```

### 3. Anotaciones de Landmarks

Las anotaciones ya deben estar en:
```
data/coordenadas/coordenadas_maestro.csv
```

**Verificación:**
```bash
wc -l data/coordenadas/coordenadas_maestro.csv
# Debe mostrar: 15154 (header + 15153 imágenes)
```

### 4. Modelos del Ensemble

Los 4 checkpoints deben estar en:
```
checkpoints/
├── session10/ensemble/seed123/final_model.pt
├── session13/seed321/final_model.pt
├── repro_split111/session14/seed111/final_model.pt
└── repro_split666/session16/seed666/final_model.pt
```

**Verificación:**
```bash
du -sh checkpoints/session10/ensemble/seed123/final_model.pt
# Debe mostrar: ~46M

ls checkpoints/**/final_model.pt | wc -l
# Debe mostrar: 4 (o más si hay otros modelos)
```

---

## Fase 1: Validación del Ensemble de Landmarks

**Objetivo:** Verificar que el ensemble de 4 modelos alcanza **3.61 px de error medio** con TTA y CLAHE.

**Tiempo estimado:**
- Con GPU: 5-8 minutos
- Sin GPU: 30-45 minutos

### Paso 1.1: Verificar Configuración

Inspeccione `configs/ensemble_best.json`:

```bash
cat configs/ensemble_best.json
```

**Contenido esperado:**
```json
{
  "name": "ensemble_best_20260111",
  "models": [
    "checkpoints/session10/ensemble/seed123/final_model.pt",
    "checkpoints/session13/seed321/final_model.pt",
    "checkpoints/repro_split111/session14/seed111/final_model.pt",
    "checkpoints/repro_split666/session16/seed666/final_model.pt"
  ],
  "tta": true,
  "clahe": true
}
```

### Paso 1.2: Ejecutar Evaluación del Ensemble

```bash
python scripts/evaluate_ensemble_from_config.py \
  --config configs/ensemble_best.json
```

### Paso 1.3: Verificar Salida

**Salida esperada en consola:**

```
=== Evaluating Ensemble ===
Config: configs/ensemble_best.json
Models: 4
TTA: True
CLAHE: True

Loading models...
  ✓ checkpoints/session10/ensemble/seed123/final_model.pt
  ✓ checkpoints/session13/seed321/final_model.pt
  ✓ checkpoints/repro_split111/session14/seed111/final_model.pt
  ✓ checkpoints/repro_split666/session16/seed666/final_model.pt

Evaluating on test set (3030 images)...
100%|████████████████████████████████| 3030/3030 [05:23<00:00, 9.37it/s]

=== Ensemble Evaluation Results ===
Mean Pixel Error: 3.61 px
Std:              2.48 px
Median:           3.07 px
Max Error:        27.3 px

Error by Category:
  Normal:           3.22 px
  COVID:            3.93 px
  Viral_Pneumonia:  4.11 px

Individual Model Performance (with TTA):
  seed123:  4.20 px
  seed321:  4.15 px
  seed111:  4.18 px
  seed666:  4.10 px

Ensemble improvement over best individual: 0.49 px (12.1%)
```

**Archivos generados:**
```
outputs/ensemble_evaluation/
├── results.json                    # Métricas completas
├── predictions_per_image.csv       # Predicciones por imagen
├── error_distribution.png          # Histograma de errores
└── error_by_landmark.png           # Error por cada landmark
```

### Paso 1.4: Validación vs. GROUND_TRUTH.json

Compare los resultados obtenidos con los valores de referencia:

```bash
python -c "
import json
with open('05_Documentacion/GROUND_TRUTH.json') as f:
    gt = json.load(f)
ref = gt['landmarks']['ensemble_4_models_tta_best_20260111']
print(f'GROUND_TRUTH.json:')
print(f'  Mean:   {ref[\"mean_error_px\"]} px')
print(f'  Std:    {ref[\"std_px\"]} px')
print(f'  Median: {ref[\"median_px\"]} px')
"
```

**Tolerancia aceptable:**
- Mean error: ± 0.05 px (debido a precisión de punto flotante)
- Std: ± 0.05 px
- Median: ± 0.05 px

**¿Qué hacer si no coincide?**
1. Verifique que los 4 modelos sean exactamente los especificados
2. Confirme que TTA=true y CLAHE=true
3. Verifique que está usando el split de test correcto (seed=42)
4. Revise la sección "Troubleshooting" al final

---

## Fase 2: Generación del Dataset Normalizado

**Objetivo:** Generar el dataset de imágenes normalizadas geométricamente mediante warping afín por partes.

**Tiempo estimado:**
- Con GPU: 15-20 minutos
- Sin GPU: 1-2 horas

### Paso 2.1: Calcular Forma Canónica (GPA)

Ejecute el análisis de Procrustes Generalizado para obtener la forma pulmonar de consenso:

```bash
python -m src_v2 compute-canonical \
  data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis \
  --visualize
```

**Tiempo:** ~30 segundos

**Salida esperada:**
```
Computing canonical shape via GPA...
  Initial alignment...
  Iteration 1: mean shape change = 0.0234
  Iteration 2: mean shape change = 0.0045
  Iteration 3: mean shape change = 0.0008
  Iteration 4: mean shape change = 0.0001
  Converged after 4 iterations.

Computing Delaunay triangulation...
  Generated 24 triangles from 15 landmarks.

Saved:
  ✓ outputs/shape_analysis/canonical_shape_gpa.json
  ✓ outputs/shape_analysis/canonical_delaunay_triangles.json
  ✓ outputs/shape_analysis/canonical_shape_visualization.png
```

**Verificación visual:**
Abra `outputs/shape_analysis/canonical_shape_visualization.png`. Debe ver:
- 15 landmarks en rojo sobre forma simétrica
- Triangulación de Delaunay en azul
- Simetría bilateral evidente

### Paso 2.2: Generar Predicciones de Landmarks (Cache)

**IMPORTANTE:** Este paso genera un archivo `.npz` con predicciones para **todas las 15,153 imágenes**. Se ejecuta una sola vez y se reutiliza.

```bash
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/session_warping/predictions.npz \
  --ensemble-config configs/ensemble_best.json \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4
```

**Tiempo:**
- Con GPU: 10-15 minutos
- Sin GPU: 45-60 minutos

**Salida esperada:**
```
Loading ensemble (4 models)...
  ✓ seed123
  ✓ seed321
  ✓ seed111
  ✓ seed666

Processing dataset...
  Found 15153 images across 3 categories.

Predicting landmarks with TTA...
100%|████████████████████████████| 15153/15153 [12:34<00:00, 20.11it/s]

Saved predictions to:
  outputs/landmark_predictions/session_warping/predictions.npz
  Size: 3.6 MB
  Images: 15153
  Landmarks per image: 15 (30 coordinates)
```

**Verificación del cache:**
```bash
python -c "
import numpy as np
cache = np.load('outputs/landmark_predictions/session_warping/predictions.npz', allow_pickle=True)
print(f'Imágenes: {len(cache[\"image_paths\"])}')
print(f'Shape predicciones: {cache[\"predictions\"].shape}')
print(f'Modelos: {cache[\"models\"]}')
print(f'TTA: {cache[\"tta\"]}')
print(f'CLAHE: {cache[\"clahe\"]}')
"
```

**Salida esperada:**
```
Imágenes: 15153
Shape predicciones: (15153, 30)
Modelos: ['seed123', 'seed321', 'seed111', 'seed666']
TTA: True
CLAHE: True
```

### Paso 2.3: Generar Dataset Warpeado

Use el cache de predicciones para aplicar warping a todas las imágenes:

```bash
python -m src_v2 generate-dataset \
  data/dataset/COVID-19_Radiography_Dataset \
  outputs/warped_lung_best/session_warping \
  --canonical outputs/shape_analysis/canonical_shape_gpa.json \
  --triangles outputs/shape_analysis/canonical_delaunay_triangles.json \
  --margin 1.05 \
  --splits 0.75,0.125,0.125 \
  --seed 42 \
  --clahe --clahe-clip 2.0 --clahe-tile 4 \
  --no-full-coverage \
  --predictions outputs/landmark_predictions/session_warping/predictions.npz
```

**Parámetros críticos:**
- `--margin 1.05`: Expansión del 5% desde el centroide de landmarks (optimizado experimentalmente)
- `--splits 0.75,0.125,0.125`: Train 75%, Val 12.5%, Test 12.5%
- `--seed 42`: Semilla para reproducibilidad de splits
- `--no-full-coverage`: Usa ROI basado en landmarks, NO expande a imagen completa

**Tiempo:** 5-10 minutos

**Salida esperada:**
```
Loading cached predictions...
  ✓ Loaded 15153 predictions from predictions.npz

Applying piecewise affine warping...
Processing: 100%|████████████████| 15153/15153 [07:23<00:00, 34.16it/s]

Dataset statistics:
  Total images: 15153
  Train: 11364 (75.0%)
  Val:    1894 (12.5%)
  Test:   1895 (12.5%)

Fill rate distribution:
  Mean: 47.2%
  Std:   8.3%
  Min:  28.1%
  Max:  68.9%

Saved to: outputs/warped_lung_best/session_warping/
```

**Verificación:**
```bash
# Contar imágenes por split
for split in train val test; do
  count=$(find outputs/warped_lung_best/session_warping/$split -name "*.png" | wc -l)
  echo "$split: $count"
done
```

**Salida esperada:**
```
train: 11364
val: 1894
test: 1895
```

**Verificación del fill_rate:**
```bash
python -c "
import json
with open('outputs/warped_lung_best/session_warping/metadata.json') as f:
    meta = json.load(f)
print(f'Mean fill_rate: {meta[\"fill_rate_mean\"]:.1f}%')
print(f'Margin scale: {meta[\"margin_scale\"]}')
print(f'CLAHE tile: {meta[\"clahe_tile_size\"]}')
"
```

**Salida esperada:**
```
Mean fill_rate: 47.2%
Margin scale: 1.05
CLAHE tile: 4
```

**IMPORTANTE:** El fill_rate de ~47% es **correcto y esperado** con `--no-full-coverage`. No intente aumentarlo a 99%; la metodología actual usa ROI basado en landmarks.

---

## Fase 3: Clasificación COVID-19

**Objetivo:** Entrenar y evaluar clasificador ResNet-18 en imágenes normalizadas, reproduciendo **98.60% ± 0.26% accuracy** con 5-fold CV.

**Tiempo estimado:**
- Con GPU: 2-3 horas (entrenamiento completo de CV)
- Sin GPU: 10-15 horas

### Opción A: Validación Rápida (Usar Modelo Pre-entrenado)

Si solo desea **verificar** las métricas sin entrenar desde cero:

**NOTA:** Los modelos de clasificación pre-entrenados NO están incluidos en el USB (solo los de landmarks). Debe entrenar o descargarlos por separado.

### Opción B: Entrenamiento Completo (Reproducción Total)

#### Paso 3.1: Entrenamiento con 5-Fold Cross-Validation

```bash
python -m src_v2 cross-validate-classifier \
  --data-dir outputs/warped_lung_best/session_warping \
  --output-dir outputs/classifier_cv \
  --backbone resnet18 \
  --folds 5 \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --seed 42 \
  --use-class-weights
```

**Tiempo:** 2-3 horas con GPU

**Salida esperada (resumen):**
```
=== 5-Fold Cross-Validation ===
Data: outputs/warped_lung_best/session_warping
Train+Val size: 13258
Test size: 1895 (held out, not evaluated)

Fold 1/5...
  Epoch 50/50 - Val Acc: 98.72% - F1: 98.14%
  ✓ Best model saved

Fold 2/5...
  Epoch 50/50 - Val Acc: 98.53% - F1: 97.89%
  ✓ Best model saved

Fold 3/5...
  Epoch 50/50 - Val Acc: 98.65% - F1: 98.02%
  ✓ Best model saved

Fold 4/5...
  Epoch 50/50 - Val Acc: 98.42% - F1: 97.75%
  ✓ Best model saved

Fold 5/5...
  Epoch 50/50 - Val Acc: 98.67% - F1: 98.21%
  ✓ Best model saved

=== Cross-Validation Results ===
Validation Accuracy:  98.60% ± 0.26%
Validation F1-macro:  98.00% ± 0.36%
Validation F1-weighted: 98.60% ± 0.25%

Results saved to:
  outputs/classifier_cv/cross_validation_results.json
  outputs/classifier_cv/fold_*/best_classifier.pt (5 models)
```

**Verificación vs. GROUND_TRUTH.json:**
```bash
python -c "
import json
with open('05_Documentacion/GROUND_TRUTH.json') as f:
    gt = json.load(f)
cv = gt['classification']['cross_validation']['metrics_percent']
print('GROUND_TRUTH.json (CV):')
print(f'  Accuracy:  {cv[\"val_accuracy_mean\"]:.2f}% ± {cv[\"val_accuracy_std\"]:.2f}%')
print(f'  F1-macro:  {cv[\"val_f1_macro_mean\"]:.2f}% ± {cv[\"val_f1_macro_std\"]:.2f}%')
"
```

**Tolerancia aceptable:**
- Mean accuracy: ± 0.5% (variabilidad estocástica del entrenamiento)
- Std accuracy: ± 0.1%

#### Paso 3.2: Evaluación del Ensemble de Clasificadores en Test Set

Evalúe el ensemble de los 5 modelos de CV en el conjunto de test:

```bash
python -m src_v2 evaluate-classifier-ensemble \
  --data-dir outputs/warped_lung_best/session_warping \
  --models outputs/classifier_cv/fold_*/best_classifier.pt \
  --output outputs/classifier_cv/ensemble_test_results.json \
  --split test
```

**Salida esperada:**
```
Loading 5 models from CV folds...
Evaluating ensemble on test set (1895 images)...

=== Ensemble Test Results (No TTA) ===
Accuracy:    98.10%
F1-macro:    97.03%
F1-weighted: 98.09%

Per-class F1:
  COVID:            96.82%
  Normal:           98.45%
  Viral_Pneumonia:  95.81%

Saved to: outputs/classifier_cv/ensemble_test_results.json
```

#### Paso 3.3: Evaluación con TTA (Test-Time Augmentation)

```bash
python -m src_v2 evaluate-classifier-ensemble \
  --data-dir outputs/warped_lung_best/session_warping \
  --models outputs/classifier_cv/fold_*/best_classifier.pt \
  --output outputs/classifier_cv/ensemble_test_results_tta.json \
  --split test \
  --tta
```

**Salida esperada:**
```
=== Ensemble Test Results (With TTA) ===
Accuracy:    98.26%
F1-macro:    97.12%
F1-weighted: 98.25%

Improvement over No TTA:
  Accuracy:    +0.16%
  F1-macro:    +0.09%

TTA Impact on Cases:
  Helped:   6 cases (corrected by TTA)
  Hurt:     3 cases (worsened by TTA)
  Neutral:  1886 cases (no change)
  Net improvement: +3 cases

Saved to: outputs/classifier_cv/ensemble_test_results_tta.json
```

**Verificación vs. GROUND_TRUTH.json:**
```bash
python -c "
import json
with open('05_Documentacion/GROUND_TRUTH.json') as f:
    gt = json.load(f)
tta = gt['classification']['classifier_ensemble_cv']['with_tta']
print('GROUND_TRUTH.json (Ensemble + TTA):')
print(f'  Accuracy:  {tta[\"accuracy\"]*100:.2f}%')
print(f'  F1-macro:  {tta[\"f1_macro\"]*100:.2f}%')
print(f'  TTA helped: {tta[\"case_level_impact\"][\"helped\"]} cases')
print(f'  TTA hurt:   {tta[\"case_level_impact\"][\"hurt\"]} cases')
print(f'  Net:        {tta[\"case_level_impact\"][\"net_improvement\"]} cases')
"
```

---

## Checklist de Verificación

Use esta checklist para confirmar que ha reproducido correctamente todos los resultados:

### Landmarks (Fase 1)
- [ ] Ensemble error medio: 3.61 px (± 0.05 px tolerancia)
- [ ] Std: 2.48 px (± 0.05 px)
- [ ] Mediana: 3.07 px (± 0.05 px)
- [ ] Error Normal: ~3.22 px
- [ ] Error COVID: ~3.93 px
- [ ] Error Viral Pneumonia: ~4.11 px
- [ ] 4 modelos cargados correctamente
- [ ] TTA activado
- [ ] CLAHE activado

### Dataset Normalizado (Fase 2)
- [ ] Forma canónica generada (15 landmarks, 24 triángulos)
- [ ] Cache de predicciones: 15,153 imágenes
- [ ] Dataset warpeado: 15,153 imágenes (224×224)
- [ ] Splits: 11364 train / 1894 val / 1895 test
- [ ] Fill rate medio: ~47% (NO 99%)
- [ ] Margin scale: 1.05
- [ ] CLAHE tile: 4
- [ ] Visualización canónica: simétrica y coherente

### Clasificación (Fase 3)
- [ ] CV accuracy: 98.60% ± 0.26% (± 0.5% tolerancia)
- [ ] CV F1-macro: 98.00% ± 0.36%
- [ ] 5 modelos de CV entrenados
- [ ] Ensemble test accuracy (no TTA): ~98.10%
- [ ] Ensemble test accuracy (with TTA): ~98.26%
- [ ] TTA net improvement: +3 casos
- [ ] Class weights usados durante entrenamiento
- [ ] Seed 42 para reproducibilidad

---

## Tiempos Estimados

### Con GPU NVIDIA (CUDA)
| Fase | Paso | Tiempo |
|------|------|--------|
| 1 | Evaluación ensemble landmarks | 5-8 min |
| 2.1 | Forma canónica (GPA) | 30 seg |
| 2.2 | Predicciones cache (15k imágenes) | 10-15 min |
| 2.3 | Warping dataset | 5-10 min |
| 3.1 | Cross-validation (5 folds × 50 epochs) | 2-3 horas |
| 3.2 | Ensemble test (no TTA) | 2-3 min |
| 3.3 | Ensemble test (with TTA) | 4-6 min |
| **Total** | | **~3 horas** |

### Sin GPU (CPU)
| Fase | Paso | Tiempo |
|------|------|--------|
| 1 | Evaluación ensemble landmarks | 30-45 min |
| 2.1 | Forma canónica (GPA) | 30 seg |
| 2.2 | Predicciones cache (15k imágenes) | 45-60 min |
| 2.3 | Warping dataset | 10-15 min |
| 3.1 | Cross-validation (5 folds × 50 epochs) | 10-15 horas |
| 3.2 | Ensemble test (no TTA) | 15-20 min |
| 3.3 | Ensemble test (with TTA) | 30-40 min |
| **Total** | | **~12-16 horas** |

**Recomendación:** Ejecute Phase 3.1 (CV training) con `nohup` en background:
```bash
nohup python -m src_v2 cross-validate-classifier \
  --data-dir outputs/warped_lung_best/session_warping \
  --output-dir outputs/classifier_cv \
  --backbone resnet18 --folds 5 --epochs 50 \
  > cv_training.log 2>&1 &

# Monitorear progreso
tail -f cv_training.log
```

---

## Troubleshooting de Reproducción

### Problema: Error de landmarks ≠ 3.61 px

**Posibles causas:**
1. Modelos incorrectos cargados
2. TTA o CLAHE desactivados
3. Split de test diferente

**Diagnóstico:**
```bash
# Verificar checksums de modelos
md5sum checkpoints/**/final_model.pt

# Verificar config
cat configs/ensemble_best.json | jq '.tta, .clahe'
# Debe mostrar: true, true

# Verificar split seed
grep -r "seed.*42" configs/
```

**Solución:**
- Re-descargue los modelos desde el USB
- Confirme que `ensemble_best.json` tiene `"tta": true, "clahe": true`
- Use `--seed 42` en todos los comandos

### Problema: Fill rate ~99% en vez de ~47%

**Causa:** Usó `--full-coverage` o no usó `--no-full-coverage`.

**Solución:**
Regenere el dataset con:
```bash
python -m src_v2 generate-dataset ... --no-full-coverage
```

El fill rate de ~47% es **correcto** para la metodología actual (ROI basado en landmarks con margin=1.05).

### Problema: CV accuracy muy baja (<95%)

**Posibles causas:**
1. Dataset warpeado incorrecto
2. Hiperparámetros incorrectos
3. Class weights no activados

**Diagnóstico:**
```bash
# Verificar metadata del dataset
cat outputs/warped_lung_best/session_warping/metadata.json | jq '.margin_scale, .fill_rate_mean'

# Verificar hiperparámetros
python -m src_v2 cross-validate-classifier --help
```

**Solución:**
- Use `--use-class-weights` (crítico para dataset desbalanceado)
- Verifique `margin_scale=1.05` y `clahe_tile=4`
- Use lr=0.0001, batch_size=32

### Problema: OOM (Out of Memory) durante CV

**Causa:** GPU sin memoria suficiente para batch_size=32.

**Solución:**
Reduzca batch size:
```bash
python -m src_v2 cross-validate-classifier \
  --batch-size 16 \  # o incluso 8
  ...
```

**Nota:** Reducir batch size puede afectar ligeramente las métricas (± 0.2%).

### Problema: Cache de predicciones corrupto

**Síntoma:** Error al cargar `predictions.npz` o shape incorrecta.

**Solución:**
Regenere el cache:
```bash
rm -f outputs/landmark_predictions/session_warping/predictions.npz
python scripts/predict_landmarks_dataset.py ...
```

### Problema: Splits train/val/test diferentes

**Causa:** Seed diferente o dataset modificado.

**Solución:**
Use **siempre** `--seed 42` en todos los comandos que generen splits.

---

## Referencias

### Documentos Relacionados
- `01_GUIA_INICIO_RAPIDO.md` - Instalación y ejecución básica
- `02_INSTALACION_REQUISITOS.md` - Instalación detallada
- `03_GUIA_USO_CLI.md` - Referencia completa de comandos
- `04_ARQUITECTURA_CODIGO.md` - Estructura del código
- `GROUND_TRUTH.json` - Métricas de referencia (versión 2.1.0)

### Configuraciones Usadas
- `configs/ensemble_best.json` - Ensemble de landmarks
- `configs/warping_best.json` - Parámetros de warping
- `configs/classifier_warped_base.json` - Clasificador base

### Reportes
- `01_Reporte/REPORTE_ESTANCIA_INAOE.pdf` - Reporte completo de la estancia
- `AUDITORIA_REPORTE_INAOE.md` - Auditoría de cumplimiento

---

**Última actualización:** 28 de enero de 2026

**Versión de GROUND_TRUTH.json:** 2.1.0
