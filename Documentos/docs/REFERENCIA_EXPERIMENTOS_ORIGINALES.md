# Referencia: Experimentos Originales vs CLI

## Fecha de Creacion: 2025-12-08
## Proposito: Documentar funcionalidades de scripts originales para guiar desarrollo CLI

---

## 1. RESUMEN EJECUTIVO

### Hipotesis Principal del Proyecto (DEMOSTRADA)
> "La normalizacion geometrica mediante landmarks anatomicos mejora la generalizacion y robustez DENTRO de un dominio de distribucion."

### Resultados Clave Establecidos

| Aspecto | Resultado | Sessions |
|---------|-----------|----------|
| Cross-evaluation interno | Warped generaliza **11x mejor** | 30 |
| Robustez JPEG | **30x mejor** con Warped | 29 |
| Robustez blur | **3x mejor** con Warped | 29 |
| Validacion externa | ~55% ambos (domain shift domina) | 36-37 |
| Modelo recomendado | DenseNet-121 Warped 1.05 | 32 |

### Limitacion Identificada
La normalizacion geometrica NO resuelve domain shift entre datasets diferentes (FedCOVIDx).

---

## 2. SCRIPTS ORIGINALES Y SU FUNCION

### 2.1 Prediccion de Landmarks

| Script | Funcion | CLI Equivalente |
|--------|---------|-----------------|
| `train.py` | Entrenar modelo de landmarks | `train` ✅ |
| `predict.py` | Predecir landmarks en imagen | `predict` ✅ |
| `evaluate_ensemble.py` | Evaluar ensemble de modelos | `evaluate` ✅ |

### 2.2 Generacion de Datasets Warped

| Script | Funcion | CLI Equivalente |
|--------|---------|-----------------|
| `generate_warped_dataset.py` | Dataset warped pequeno (957) | `warp` ✅ |
| `generate_full_warped_dataset.py` | Dataset warped completo (15K) | `warp` ✅ |
| `piecewise_affine_warp.py` | Funcion de warping | Integrado ✅ |

### 2.3 Clasificacion

| Script | Funcion | CLI Equivalente |
|--------|---------|-----------------|
| `train_classifier.py` | Entrenar en warped | `train-classifier` ✅ |
| `train_classifier_original.py` | Entrenar en original | `train-classifier` ✅ |
| `train_all_architectures.py` | Multi-arquitectura | Parcial (manual) |

### 2.4 Evaluacion y Validacion

| Script | Funcion | CLI Equivalente |
|--------|---------|-----------------|
| `compare_classifiers.py` | Comparar modelos | ❌ NO IMPLEMENTADO |
| `session30_cross_evaluation.py` | Cross-evaluation | ❌ NO IMPLEMENTADO |
| `session31_cross_evaluation.py` | Cross multi-arq | ❌ NO IMPLEMENTADO |
| `evaluate_external_baseline.py` | Validacion FedCOVIDx | ❌ NO IMPLEMENTADO |
| `evaluate_external_warped.py` | Validacion warped ext | ❌ NO IMPLEMENTADO |

### 2.5 Analisis y Visualizacion

| Script | Funcion | CLI Equivalente |
|--------|---------|-----------------|
| `gradcam_comparison.py` | Grad-CAM | ❌ NO IMPLEMENTADO |
| `validation_session26.py` | PFS + artefactos | ❌ NO IMPLEMENTADO |
| `session30_robustness_figure.py` | Figuras robustez | ❌ NO IMPLEMENTADO |

---

## 3. FUNCIONALIDADES FALTANTES EN CLI (PRIORIDAD)

### Alta Prioridad

#### 3.1 Cross-Evaluation
**Script original**: `session30_cross_evaluation.py`

**Funcion**: Evaluar modelo entrenado en un dominio sobre otro dominio
```
Original→Original (baseline)
Original→Warped (cross)
Warped→Warped (baseline)
Warped→Original (cross)
```

**Propuesta CLI**:
```bash
python -m src_v2 cross-evaluate \
    --model-original outputs/classifier_original/best.pt \
    --model-warped outputs/classifier_warped/best.pt \
    --data-original data/original_test/ \
    --data-warped data/warped_test/ \
    --output outputs/cross_eval_results.json
```

**Importancia**: Demuestra la hipotesis principal (11x mejor generalizacion)

#### 3.2 Evaluacion Externa
**Script original**: `evaluate_external_baseline.py`

**Funcion**: Evaluar en dataset completamente independiente (FedCOVIDx)
- Mapeo de clases 3→2 (COVID vs No-COVID)
- Calculo de gap de generalizacion

**Propuesta CLI**:
```bash
python -m src_v2 evaluate-external \
    --classifier outputs/classifier/best.pt \
    --external-data outputs/external_validation/dataset3/ \
    --class-mapping "COVID:positive,Normal+Viral:negative" \
    --output outputs/external_results.json
```

### Media Prioridad

#### 3.3 Robustez a Perturbaciones
**Script original**: `test_robustness_artifacts.py`, `test_robustness_geometric.py`

**Funcion**: Evaluar degradacion ante JPEG, blur, ruido, rotacion, etc.

**Propuesta CLI**:
```bash
python -m src_v2 test-robustness \
    --classifier outputs/classifier/best.pt \
    --data-dir data/test/ \
    --perturbations jpeg,blur,noise,rotation \
    --output outputs/robustness_results.json
```

#### 3.4 Grad-CAM Comparison
**Script original**: `gradcam_comparison.py`

**Funcion**: Visualizar atencion del modelo, calcular Pulmonary Focus Score

**Propuesta CLI**:
```bash
python -m src_v2 gradcam \
    --classifier outputs/classifier/best.pt \
    --image imagen.png \
    --lung-mask masks/lung_mask.png \
    --output gradcam_output.png
```

### Baja Prioridad

#### 3.5 Entrenamiento Multi-Arquitectura
**Script original**: `train_all_architectures.py`

**Funcion**: Entrenar multiples arquitecturas automaticamente

#### 3.6 Generacion de Figuras
**Script original**: Varios scripts de visualizacion

---

## 4. EXPERIMENTOS CLAVE Y COMO REPRODUCIRLOS

### 4.1 Experimento: Cross-Evaluation (Session 30)

**Objetivo**: Demostrar que Warped generaliza mejor

**Resultados esperados**:
| Evaluacion | Accuracy |
|------------|----------|
| Original→Original | 98.81% |
| Original→Warped | 73.45% |
| Warped→Warped | 98.02% |
| Warped→Original | 95.78% |

**Como reproducir (scripts originales)**:
```bash
python scripts/session30_cross_evaluation.py
```

**Archivos necesarios**:
- `outputs/classifier_comparison/resnet18_original/best_model.pt`
- `outputs/classifier_comparison/resnet18_warped/best_model.pt`
- `outputs/full_warped_dataset/test/`
- `data/dataset/.../test/` (original)

### 4.2 Experimento: Validacion Externa (Session 36-37)

**Objetivo**: Evaluar en FedCOVIDx

**Resultados esperados**:
| Modelo | Accuracy | Gap |
|--------|----------|-----|
| Original | 57.5% | 41.3% |
| Warped | 53.5% | 44.5% |

**Como reproducir**:
```bash
# Preparar dataset externo
python scripts/prepare_dataset3.py

# Evaluar modelos originales
python scripts/evaluate_external_baseline.py

# Warpear dataset externo
python scripts/warp_dataset3.py

# Evaluar modelos warped
python scripts/evaluate_external_warped.py
```

### 4.3 Experimento: Robustez (Session 29)

**Objetivo**: Medir degradacion ante perturbaciones

**Resultados esperados**:
| Perturbacion | Original | Warped | Mejora |
|--------------|----------|--------|--------|
| JPEG Q=50 | 16.14% | 0.53% | 30x |
| JPEG Q=30 | 29.97% | 1.32% | 23x |
| Blur fuerte | 46.05% | 16.27% | 3x |

**Como reproducir**:
```bash
python scripts/test_robustness_artifacts.py
```

---

## 5. DATASETS UTILIZADOS

### 5.1 Dataset Principal: COVID-19 Radiography Database

**Ubicacion**: `data/dataset/COVID-19_Radiography_Dataset/`

**Estructura**:
```
COVID-19_Radiography_Dataset/
├── COVID/
│   ├── images/     (3,616 imagenes)
│   └── masks/
├── Normal/
│   ├── images/     (10,192 imagenes)
│   └── masks/
├── Viral Pneumonia/
│   ├── images/     (1,345 imagenes)
│   └── masks/
└── Lung_Opacity/   (12,024 - NO USADO en experimentos)
    ├── images/
    └── masks/
```

**NOTA IMPORTANTE**: Solo se usan 3 clases (COVID, Normal, Viral_Pneumonia). Lung_Opacity fue excluido.

### 5.2 Dataset Warped

**Ubicacion**: `outputs/full_warped_dataset/`

**Generado con**: Landmarks de ground truth, margin_scale=1.05

### 5.3 Dataset Externo: FedCOVIDx (Dataset3)

**Ubicacion**: `outputs/external_validation/dataset3/`

**Estructura**:
```
dataset3/
├── test/
│   ├── positive/   (4,241 - COVID)
│   └── negative/   (4,241 - No COVID)
```

---

## 6. MODELOS ENTRENADOS DISPONIBLES

### 6.1 Landmarks

| Modelo | Ubicacion | Error |
|--------|-----------|-------|
| Seed 123 | `checkpoints/session10/ensemble/seed123/final_model.pt` | ~4.05px |
| Seed 456 | `checkpoints/session10/ensemble/seed456/final_model.pt` | ~4.04px |
| Seed 321 | `checkpoints/session13/seed321/final_model.pt` | ~4.0px |
| Seed 789 | `checkpoints/session13/seed789/final_model.pt` | ~4.0px |
| **Ensemble 4** | Combinacion | **3.71px** |

### 6.2 Clasificadores

| Modelo | Ubicacion | Accuracy |
|--------|-----------|----------|
| ResNet-18 Warped | `outputs/classifier_full/best_classifier.pt` | 97.50% |
| EfficientNet Warped | `outputs/classifier_efficientnet/best_classifier.pt` | 97.76% |
| Multi-arq Original | `outputs/classifier_comparison/*_original/` | ~98% |
| Multi-arq Warped | `outputs/classifier_comparison/*_warped/` | ~97% |

---

## 7. CONFIGURACION OPTIMA ESTABLECIDA

### Landmarks
- **Arquitectura**: ResNet-18 con regresion
- **Ensemble**: 4 modelos (mejor balance costo/beneficio)
- **TTA**: Habilitado (flip horizontal)
- **Error**: 3.71px

### Clasificacion
- **Arquitectura recomendada**: DenseNet-121 (mejor generalizacion)
- **Dataset**: Warped con margin_scale=1.05
- **Augmentation**: Standard (flip, rotation, color jitter)
- **Class weights**: Habilitados para desbalance

### Warping
- **Margin scale**: 1.05 (optimo para arquitecturas profundas)
- **Triangulacion**: Delaunay con 23 puntos (15 landmarks + 8 borde)
- **Fill rate**: ~96% (con extension de dominio)

---

## 8. CONCLUSIONES CIENTIFICAS ESTABLECIDAS

### Confirmadas

1. **Normalizacion geometrica mejora generalizacion interna** (11x)
2. **Mejora robustez a perturbaciones** (30x JPEG, 3x blur)
3. **DenseNet-121 es la arquitectura optima** para generalizacion
4. **Margin 1.05 es optimo** para arquitecturas profundas

### No Confirmadas

1. **NO resuelve domain shift externo** (FedCOVIDx ~55%)
2. Requiere tecnicas adicionales (domain adaptation) para cross-dataset

### Recomendacion Final

> Para despliegue clinico, usar **DenseNet-121 entrenado en Warped margin 1.05**.
> Sacrifica ~0.8% accuracy por 11x mejor generalizacion y 30x mejor robustez.

---

## 9. TRABAJO FUTURO PARA CLI

### Inmediato (Sesion 18+)
1. [ ] Implementar comando `cross-evaluate`
2. [ ] Implementar comando `evaluate-external`
3. [ ] Agregar soporte para DenseNet-121

### Medio Plazo
4. [ ] Comando `test-robustness`
5. [ ] Comando `gradcam`
6. [ ] Comparacion automatica de arquitecturas

### Largo Plazo
7. [ ] Integracion con domain adaptation
8. [ ] Pipeline de entrenamiento federado

---

*Documento generado: 2025-12-08*
*Proyecto: Prediccion de Warping y Clasificacion COVID-19*
