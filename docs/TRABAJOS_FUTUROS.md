# Trabajos Futuros: Experimentos a Replicar con warped_lung_best

**Documento generado:** 2026-01-14
**Método actual:** warped_lung_best (98.05% accuracy, fill_rate=47%)
**Estado:** Pendiente de implementación

---

## Resumen Ejecutivo

Durante la actualización de la metodología de tesis (FASES 1-5, completadas 2026-01-14), se identificaron **experimentos obsoletos** realizados con configuraciones anteriores (warped_96, warped_99).

**Prioridad actual:** ejecutar una comparación controlada entre clasificación en imágenes originales vs. warpeadas usando el mismo protocolo (mismos splits, hiperparámetros y preprocesamiento) para que los resultados sean directamente comparables.

**Postergado:** cross-evaluation entre dominios y validación externa.  
**Obsoleto:** robustez ante perturbaciones.  
**Futuro lejano:** evaluar normalización de contraste SAHS (ver `docs/SAHS`).

---

## 0. Comparación Controlada Original vs Warped (PRIORIDAD ACTUAL)

**Estado:** ✅ Completado  
**Prioridad:** ALTA

### Objetivo

Medir el impacto del warping en la clasificación comparando el mismo clasificador entrenado con:
- Imágenes originales (sin warping).
- Imágenes warpeadas (`warped_lung_best`).

La comparación debe usar el **mismo protocolo** (splits, hiperparámetros, arquitectura y semillas) para que sea válida.

### Protocolo recomendado (mismo split)

Usar los splits ya generados en `outputs/warped_lung_best/session_warping` para asegurar consistencia:

```bash
python -m src_v2 compare-architectures \
  outputs/warped_lung_best/session_warping \
  --architectures resnet18 \
  --original-data-dir data/dataset/COVID-19_Radiography_Dataset \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --patience 10 \
  --seed 321 \
  --output-dir outputs/compare_original_vs_warped
```

### Resultados obtenidos (seed 321, lr 2e-4)

**Test (n=1,895):**
- Original: Accuracy 98.89%, F1-Macro 98.10%, F1-Weighted 98.89% (21 errores)
- Warpeado: Accuracy 98.05%, F1-Macro 97.12%, F1-Weighted 98.04% (37 errores)
- Delta (Original - Warpeado): +0.84 pp Accuracy, +0.98 pp F1-Macro

**Artefactos:**
- Original: `outputs/classifier_original_warped_lung_best_seed321/results_original.json`
- Warpeado: `outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/results.json`

### Métricas a reportar

- Accuracy, F1-Macro, F1-Weighted (test).
- Matriz de confusión (ambos dominios).
- Diferencia directa (warped - original) en métricas clave.

---

## 1. Evaluación de Robustez ante Perturbaciones

**Estado:** ⛔ Obsoleto (no requerido)
**Referencia obsoleta:** GROUND_TRUTH.json sección `robustness` (marcada obsoleta 2026-01-14)
**Prioridad:** NULA

### Descripción

Evaluar la resistencia del clasificador ante perturbaciones comunes en imágenes médicas.

**Nota:** Esta línea se considera obsoleta y no se ejecutará salvo nueva indicación.

### Perturbaciones a Aplicar

#### 1.1. Compresión JPEG

**Objetivo:** Medir degradación de accuracy bajo compresión JPEG (simulando transmisión/almacenamiento con pérdida)

**Protocolo:**
```bash
# Generar versiones perturbadas del test set
python scripts/generate_perturbed_dataset.py \
  --input outputs/warped_lung_best/session_warping/test \
  --output outputs/robustness/jpeg_q50 \
  --perturbation jpeg --quality 50

python scripts/generate_perturbed_dataset.py \
  --input outputs/warped_lung_best/session_warping/test \
  --output outputs/robustness/jpeg_q30 \
  --perturbation jpeg --quality 30

# Evaluar clasificador
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/robustness/jpeg_q50 --split test

python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/robustness/jpeg_q30 --split test
```

**Métricas a reportar:**
- Accuracy limpio (baseline): 98.05%
- Accuracy JPEG Q=50
- Accuracy JPEG Q=30
- Degradación Q50 = (Acc_limpio - Acc_Q50) / Acc_limpio × 100%
- Degradación Q30 = (Acc_limpio - Acc_Q30) / Acc_limpio × 100%

**Valores obsoletos (warped_96) para comparación:**
- JPEG Q50 degradación: 3.06%
- JPEG Q30 degradación: 5.28%

---

#### 1.2. Desenfoque Gaussiano (Blur)

**Objetivo:** Medir degradación ante desenfoque (simulando movimiento del paciente, equipos desajustados)

**Protocolo:**
```bash
# Generar test set con blur
python scripts/generate_perturbed_dataset.py \
  --input outputs/warped_lung_best/session_warping/test \
  --output outputs/robustness/blur_sigma1 \
  --perturbation blur --sigma 1.0

# Evaluar
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/robustness/blur_sigma1 --split test
```

**Métricas a reportar:**
- Accuracy blur σ=1
- Degradación = (Acc_limpio - Acc_blur) / Acc_limpio × 100%

**Valores obsoletos (warped_96):**
- Blur σ=1 degradación: 2.43%

---

#### 1.3. Experimentos Adicionales (Opcionales)

**Ruido Gaussiano:**
```bash
python scripts/generate_perturbed_dataset.py \
  --input outputs/warped_lung_best/session_warping/test \
  --output outputs/robustness/noise_std10 \
  --perturbation noise --std 10
```

**Cambios de brillo/contraste:**
```bash
python scripts/generate_perturbed_dataset.py \
  --input outputs/warped_lung_best/session_warping/test \
  --output outputs/robustness/brightness_mult0.7 \
  --perturbation brightness --multiplier 0.7
```

---

### Análisis Esperado

**Comparación con warped_96 (obsoleto):**
- ¿warped_lung_best mantiene robustez similar?
- ¿El fill_rate menor (47% vs 96%) afecta robustez?
- Hipótesis: Mayor eliminación de fondo (47%) podría MEJORAR robustez al reducir información irrelevante

**Sección en tesis:** Capítulo 5 (Resultados Experimentales)

---

## 2. Cross-Evaluation: Generalización entre Dominios

**Estado:** ⏸ Postergado
**Referencia obsoleta:** GROUND_TRUTH.json sección `cross_evaluation` (marcada obsoleta 2026-01-14)
**Prioridad:** BAJA (después de comparación original vs warped)

### Descripción

Evaluar la capacidad de generalización comparando:
1. **Modelo entrenado en ORIGINAL** → evaluado en ORIGINAL vs WARPED
2. **Modelo entrenado en WARPED** → evaluado en ORIGINAL vs WARPED

### Protocolo

#### 2.1. Entrenar Clasificador en Dataset Original

```bash
# Preparar dataset original (sin warping)
python -m src_v2 prepare-original-dataset \
  --input data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/original_dataset \
  --preprocessing clahe --clahe-clip 2.0 --clahe-tile 4

# Entrenar clasificador
python -m src_v2 train-classifier \
  --config configs/classifier_original_base.json \
  --output outputs/classifier_original
```

#### 2.2. Evaluaciones Cruzadas

```bash
# 1. Modelo ORIGINAL → evaluado en ORIGINAL (baseline)
python -m src_v2 evaluate-classifier \
  outputs/classifier_original/best_classifier.pt \
  --data-dir outputs/original_dataset --split test

# 2. Modelo ORIGINAL → evaluado en WARPED (domain shift)
python -m src_v2 evaluate-classifier \
  outputs/classifier_original/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping --split test

# 3. Modelo WARPED → evaluado en ORIGINAL (domain shift)
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/original_dataset --split test

# 4. Modelo WARPED → evaluado en WARPED (baseline)
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping --split test
```

### Métricas a Reportar

**Modelo Original:**
- Accuracy en original (baseline)
- Accuracy en warped (cross-domain)
- Gap de generalización = |Acc_original - Acc_warped|

**Modelo Warped:**
- Accuracy en warped (baseline): 98.05%
- Accuracy en original (cross-domain)
- Gap de generalización = |Acc_warped - Acc_original|

**Análisis clave:**
- Factor de mejora = Gap_original / Gap_warped
- Si Gap_warped < Gap_original → warped generaliza MEJOR

**Valores obsoletos (warped_99):**
- Gap modelo original: 7.7%
- Gap modelo warped: 3.17%
- Factor de mejora: 2.43×

---

## 3. PFS (Pulmonary Focus Score)

**Estado:** ❌ NO computado con warped_lung_best
**Referencia obsoleta:** GROUND_TRUTH.json sección `pfs` (marcada obsoleta 2026-01-14)
**Prioridad:** BAJA

### Descripción

Analizar si el clasificador enfoca su atención en la región pulmonar o en el fondo/artefactos mediante visualización de mapas de activación (GradCAM, Attention).

### Protocolo

```bash
# Generar mapas de atención para test set
python scripts/compute_pfs.py \
  --classifier outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping \
  --split test \
  --output outputs/pfs_analysis/warped_lung_best

# Computar PFS score
python scripts/analyze_pfs.py \
  --attention-maps outputs/pfs_analysis/warped_lung_best \
  --landmarks-file data/coordenadas/coordenadas_maestro.csv \
  --output outputs/pfs_analysis/pfs_scores.json
```

### Métrica

**PFS (Pulmonary Focus Score):**
```
PFS = (Atención en región pulmonar) / (Atención total)
```

- PFS ≈ 0.50 → Modelo no discrimina (azar)
- PFS > 0.70 → Modelo enfoca correctamente en pulmones
- PFS < 0.50 → Modelo enfoca fuera de región de interés

**Valor obsoleto:**
- PFS = 0.487 (≈ 50%, NO evidencia de foco pulmonar forzado)

**Hipótesis:** Con fill_rate=47% (solo región pulmonar), esperamos PFS > 0.70

---

## 4. Validación Externa (Dataset3 FedCOVIDx)

**Estado:** ⏸ Postergado
**Referencia obsoleta:** GROUND_TRUTH.json sección `external_validation` (marcada obsoleta 2026-01-14)
**Prioridad:** BAJA (no requerido ahora)

### Descripción

Evaluar el clasificador en un dataset externo (fuera de distribución) para medir generalización real a nuevos hospitales/equipos.

**Dataset:** FedCOVIDx (8,482 muestras, fuentes: BIMCV ~95%, RICORD, RSNA)
**Mapeo de clases:** COVID → positive, Normal+Viral_Pneumonia → negative

### Protocolo

**Nota:** Este experimento queda pospuesto. Cuando se retome, revisar la pipeline archivada
en `scripts/archive/classification/` y alinear el preprocesamiento con el método actual.

### Métricas a Reportar

**Evaluación en D3 (8,482 muestras):**
- Accuracy interna (test set): 98.05%
- Accuracy en D3 original
- Accuracy en D3 warped
- Sensitivity (recall COVID)
- Specificity (recall Normal+Viral)
- F1-Score
- AUC-ROC
- Gap de generalización = Acc_interno - Acc_externo

**Valores obsoletos (warped_96 en D3 original):**
- Accuracy: 53.36% (≈ random para binaria)
- Gap: 45.74% (99.10% - 53.36%)

**Interpretación esperada:**
- Si Acc_D3 ≈ 50-57% → Domain shift severo (NO problema del warping, también ocurre con original)
- Solución: Domain adaptation, fine-tuning local, transfer learning

---

## 5. Fill Rate Trade-off Analysis

**Estado:** ❌ NO realizado con warped_lung_best
**Referencia obsoleta:** GROUND_TRUTH.json sección `fill_rate_tradeoff` (marcada obsoleta 2026-01-14)
**Prioridad:** BAJA (warped_lung_best ya usa fill_rate óptimo)

### Descripción

Análisis del trade-off entre fill_rate, accuracy y robustez. Este experimento fue realizado para **seleccionar** el parámetro `margin_scale` óptimo (resultado: 1.05, fill_rate=47%).

**Estado actual:** warped_lung_best YA usa los parámetros óptimos (margin_scale=1.05, fill_rate=47%).

### Protocolo (si se desea revalidar)

```bash
# Grid search de margin_scale
for MARGIN in 1.00 1.05 1.10 1.15 1.20 1.25; do
  python -m src_v2 generate-dataset \
    --config configs/warping_best.json \
    --margin-scale $MARGIN \
    --output outputs/warping_margin_${MARGIN}

  python -m src_v2 train-classifier \
    --data-dir outputs/warping_margin_${MARGIN} \
    --output outputs/classifier_margin_${MARGIN}

  python -m src_v2 evaluate-classifier \
    outputs/classifier_margin_${MARGIN}/best_classifier.pt \
    --data-dir outputs/warping_margin_${MARGIN} --split test
done
```

**Métricas por margin_scale:**
- Fill rate
- Accuracy
- Degradación JPEG Q50
- Composite score = Accuracy - α × Degradación (α=0.5)

---

## 6. Clasificación Binaria (COVID+Neumonía vs Normal)

**Estado:** ❌ NO realizado
**Prioridad:** MEDIA

### Descripción

Evaluar el rendimiento del sistema en una tarea de clasificación binaria, agrupando COVID-19 y Neumonía Viral en una sola clase "patológica" vs Normal. Esto simula un escenario de screening inicial donde el objetivo es detectar cualquier anomalía pulmonar.

### Justificación

La clasificación binaria puede ofrecer:
1. **Mayor sensibilidad**: Detectar cualquier anomalía pulmonar sin distinguir tipo específico
2. **Datasets balanceados**: Reducir desbalance de clases (actualmente 24% COVID + 9% Viral vs 67% Normal)
3. **Aplicación clínica**: Screening inicial antes de diagnóstico diferencial

### Protocolo

#### 6.1. Preparar Dataset Binario

```bash
# Crear dataset binario agrupando COVID + Viral_Pneumonia
python scripts/create_binary_dataset.py \
  --input outputs/warped_lung_best/session_warping \
  --output outputs/warped_lung_best_binary \
  --group-covid-viral
```

**Mapeo de clases:**
- `Pathological` (COVID + Viral_Pneumonia): 33% del dataset
- `Normal`: 67% del dataset

#### 6.2. Entrenar Clasificador Binario

```bash
python -m src_v2 train-classifier \
  --data-dir outputs/warped_lung_best_binary \
  --backbone resnet18 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --use-class-weights \
  --output outputs/classifier_warped_lung_best_binary
```

#### 6.3. Evaluar

```bash
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best_binary/best_classifier.pt \
  --data-dir outputs/warped_lung_best_binary --split test
```

### Métricas a Reportar

**Comparación 3 clases vs 2 clases:**

| Métrica | 3 Clases (actual) | 2 Clases (esperado) |
|---------|-------------------|---------------------|
| Accuracy | 98.05% | ? |
| F1-Macro | 97.12% | ? |
| F1-Weighted | 98.04% | ? |
| Sensitivity (Patológico) | - | ? |
| Specificity (Normal) | - | ? |
| AUC-ROC | - | ? |

**Análisis esperado:**
- ¿Mejora accuracy al simplificar el problema?
- ¿Cómo afecta el balanceo de clases?
- ¿Es más robusto ante perturbaciones?

---

## 7. Evaluación de Arquitecturas Alternativas

**Estado:** ❌ NO realizado
**Prioridad:** BAJA

### Descripción

Evaluar arquitecturas CNN alternativas en el dataset warped_lung_best para determinar si ResNet-18 es realmente la opción óptima o si otras arquitecturas ofrecen mejor rendimiento.

### Justificación

Actualmente solo se utilizó ResNet-18. Otras arquitecturas modernas podrían ofrecer:
1. **Mejor accuracy**: Arquitecturas más profundas o con diseños optimizados
2. **Menor costo computacional**: MobileNetV2, EfficientNet-B0
3. **Mejor generalización**: DenseNet-121 con conexiones densas

### Arquitecturas a Evaluar

Basadas en la Tabla 4.1 (eliminada de tesis pero disponibles en código):

| Arquitectura | Parámetros | Ventaja Principal |
|--------------|------------|-------------------|
| AlexNet | 57.0M | Baseline histórico |
| VGG-16 | 134.3M | Simplicidad arquitectónica |
| ResNet-18 | 11.2M | **Actual (baseline)** |
| ResNet-50 | 23.5M | Mayor profundidad |
| DenseNet-121 | 7.0M | Conexiones densas, eficiente |
| MobileNetV2 | 2.2M | Máxima eficiencia |
| EfficientNet-B0 | 4.0M | Balance accuracy/eficiencia |

### Protocolo

#### 7.1. Entrenar Cada Arquitectura

```bash
for ARCH in alexnet vgg16 resnet50 densenet121 mobilenet_v2 efficientnet_b0; do
  python -m src_v2 train-classifier \
    --data-dir outputs/warped_lung_best/session_warping \
    --backbone $ARCH \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --use-class-weights \
    --output outputs/classifier_warped_lung_best_${ARCH}
done
```

#### 7.2. Evaluar Todas

```bash
for ARCH in alexnet vgg16 resnet18 resnet50 densenet121 mobilenet_v2 efficientnet_b0; do
  python -m src_v2 evaluate-classifier \
    outputs/classifier_warped_lung_best_${ARCH}/best_classifier.pt \
    --data-dir outputs/warped_lung_best/session_warping --split test \
    --output outputs/architecture_comparison/${ARCH}_results.json
done
```

#### 7.3. Generar Tabla Comparativa

```bash
python scripts/compare_architectures.py \
  --results-dir outputs/architecture_comparison \
  --output docs/architecture_comparison_table.md
```

### Métricas a Reportar

**Tabla de Comparación:**

| Arquitectura | Accuracy | F1-Macro | F1-Weighted | Parámetros | Tiempo/Época | Inferencia (ms) |
|--------------|----------|----------|-------------|------------|--------------|-----------------|
| ResNet-18 (baseline) | 98.05% | 97.12% | 98.04% | 11.2M | ~40s | ? |
| AlexNet | ? | ? | ? | 57.0M | ? | ? |
| VGG-16 | ? | ? | ? | 134.3M | ? | ? |
| ResNet-50 | ? | ? | ? | 23.5M | ? | ? |
| DenseNet-121 | ? | ? | ? | 7.0M | ? | ? |
| MobileNetV2 | ? | ? | ? | 2.2M | ? | ? |
| EfficientNet-B0 | ? | ? | ? | 4.0M | ? | ? |

**Análisis esperado:**
- Ranking por accuracy
- Trade-off accuracy vs eficiencia
- Curvas de entrenamiento (convergencia)
- Posible ensemble de mejores arquitecturas

### Tiempo Estimado

- Entrenamiento: ~6-8 horas (50 épocas × 7 arquitecturas)
- Evaluación y análisis: 2-3 horas
- **Total: ~10 horas**

### Scripts a Crear

- ⚠️ `scripts/compare_architectures.py` (generar tabla comparativa)
- Resto de comandos ya soportados por CLI actual

---

## 8. Análisis Adicionales (Opcionales)

### 8.1. Error por Landmark Individual (con warped_lung_best)

Verificar si la distribución de errores por landmark se mantiene consistente.

```bash
python scripts/analyze_per_landmark_errors.py \
  --config configs/ensemble_best.json \
  --output outputs/analysis/per_landmark_warped_lung_best.json
```

### 8.2. Matriz de Confusión Detallada

Analizar patrones de confusión entre clases con warped_lung_best.

```bash
python scripts/generate_confusion_matrix.py \
  --classifier outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping \
  --split test \
  --output outputs/analysis/confusion_matrix.png
```

### 8.3. Ablation Study: Impacto de TTA

Medir contribución de Test-Time Augmentation al rendimiento del ensemble.

```bash
# Evaluar ensemble SIN TTA
python scripts/evaluate_ensemble_from_config.py \
  --config configs/ensemble_best.json \
  --no-tta \
  --output outputs/analysis/ensemble_no_tta.json

# Evaluar ensemble CON TTA (baseline)
python scripts/evaluate_ensemble_from_config.py \
  --config configs/ensemble_best.json \
  --output outputs/analysis/ensemble_with_tta.json
```

---

## 9. Normalización de Contraste SAHS (Futuro)

**Estado:** ⏳ Pendiente  
**Prioridad:** BAJA (futuro)

Reemplazar CLAHE por el método SAHS para evaluar su impacto en la clasificación.
Referencias y consideraciones técnicas en `docs/SAHS`.

---

## Priorización Recomendada

### ALTA PRIORIDAD (actual)

1. **Comparación controlada original vs. warped**
   - Tiempo estimado: 4-6 horas
   - Impacto: Alto (cuantifica el efecto real del warping)

### MEDIA PRIORIDAD (después)

2. **Cross-Evaluation**
   - Tiempo estimado: 6-8 horas
   - Impacto: Medio (generalización entre dominios)

### BAJA PRIORIDAD (postergado)

3. **Validación Externa (Dataset3/FedCOVIDx)**
   - Impacto: Bajo por ahora (no requerido)

4. **Clasificación Binaria (COVID+Neumonía vs Normal)**
   - Impacto: Bajo por ahora

5. **Evaluación de Arquitecturas Alternativas**
   - Impacto: Bajo

6. **PFS (Pulmonary Focus Score)**
   - Impacto: Bajo

7. **Fill Rate Trade-off**
   - Impacto: Bajo

8. **Normalización de contraste SAHS**
   - Impacto: Bajo (futuro lejano)

### OBSOLETO

- **Evaluación de Robustez** (JPEG, blur): no requerida.

---

## Checklist de Implementación

Para cada experimento activo, seguir este workflow:

- [ ] Implementar scripts necesarios para la comparación original vs warped (si aplica)
- [ ] Ejecutar protocolo documentado
- [ ] Registrar resultados en archivo JSON
- [ ] Actualizar GROUND_TRUTH.json con nuevos valores (eliminar flag `obsolete`)
- [ ] Documentar en sesión correspondiente (docs/sesiones/SESION_XX.md)
- [ ] Agregar resultados a tesis (Capítulo 5: Resultados Experimentales)
- [ ] Generar figuras/tablas para tesis
- [ ] Commit con tag de sesión

---

## Archivos de Referencia

**Configuraciones actuales:**
- `configs/ensemble_best.json` (landmarks 3.61 px)
- `configs/warping_best.json` (margin_scale=1.05, fill_rate=47%)
- `configs/classifier_warped_base.json` (ResNet-18, 98.05%)

**Ground truth:**
- `GROUND_TRUTH.json` (v2.1.0, actualizado 2026-01-14)

**Scripts existentes:**
- `scripts/evaluate_ensemble_from_config.py`
- `scripts/predict_landmarks_dataset.py`
- `src_v2/cli.py` (comandos: generate-dataset, train-classifier, evaluate-classifier)

**Scripts a crear:**
- Ninguno requerido para la prioridad actual (comparación original vs warped).

---

## Notas Finales

**Este documento es un roadmap, NO un requisito inmediato.** Los experimentos marcados como obsoletos fueron realizados con configuraciones anteriores y deben replicarse SOLO si:

1. Se requieren para completar el Capítulo 5 de la tesis
2. Los revisores solicitan evidencia de robustez/generalización
3. Se desea publicar resultados en conferencia/journal

**Método actual (warped_lung_best) es VÁLIDO y FUNCIONAL sin estos experimentos.** Los trabajos futuros son extensiones para fortalecer la evaluación.

---

**Documento generado por:** Claude Sonnet 4.5
**Fecha:** 2026-01-14
**Plan de actualización:** deep-meandering-reef.md (FASES 1-5 completadas)
