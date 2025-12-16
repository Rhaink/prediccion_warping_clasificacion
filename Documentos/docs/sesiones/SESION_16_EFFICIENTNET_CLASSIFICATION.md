# Sesión 16: Clasificador EfficientNet-B0

**Fecha:** 2025-12-08
**Objetivo:** Entrenar clasificador con EfficientNet-B0 para validar reproducibilidad de resultados ~98%

## Resumen Ejecutivo

Se entrenó exitosamente un clasificador EfficientNet-B0 sobre el dataset warped (margin 1.05), logrando **97.76% accuracy en test**, validando que la CLI reproduce resultados competitivos con la implementación original.

## Configuración del Entrenamiento

```bash
python -m src_v2 train-classifier outputs/full_warped_dataset \
    --backbone efficientnet_b0 \
    --epochs 50 \
    --output-dir outputs/classifier_efficientnet \
    --seed 42 \
    --patience 10
```

### Parámetros
- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Épocas:** 50 (early stopping en época 50)
- **Mejor época:** 40
- **Learning Rate:** 1e-4 (default)
- **Batch Size:** 32
- **Dropout:** 0.3
- **Class Weights:** [1.40, 0.50, 3.76] (balanceo automático)

## Dataset

| Split | Total | COVID | Normal | Viral_Pneumonia |
|-------|-------|-------|--------|-----------------|
| Train | 11,364 | 2,712 (24%) | 7,644 (67%) | 1,008 (9%) |
| Val | 2,271 | 542 | 1,528 | 201 |
| Test | 1,518 | 362 | 1,020 | 136 |

**Margin de warping:** 1.05x

## Resultados

### Validación (Mejor Modelo - Epoch 40)

| Métrica | Valor |
|---------|-------|
| **Val Accuracy** | **98.59%** |
| **Val F1 Macro** | **97.98%** |

### Test (Evaluación Final)

| Métrica | Valor |
|---------|-------|
| **Test Accuracy** | **97.76%** |
| Test F1 Macro | 96.80% |
| Test F1 Weighted | 97.76% |

### Métricas por Clase (Test)

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| COVID | 0.96 | 0.98 | 0.97 | 362 |
| Normal | 0.98 | 0.98 | 0.98 | 1,020 |
| Viral_Pneumonia | 0.97 | 0.93 | 0.95 | 136 |

### Matriz de Confusión (Test)

```
                  Predicted
              COVID  Normal  Viral
Actual COVID   353      8      1
       Normal   13   1004      3
       Viral     1      8    127
```

**Errores principales:**
- 13 Normal clasificados como COVID (falsos positivos COVID)
- 8 Viral_Pneumonia clasificados como Normal
- 8 COVID clasificados como Normal

## Comparación con Sesión 15 (ResNet-18)

| Backbone | Val Acc | Val F1 | Test Acc | Test F1 |
|----------|---------|--------|----------|---------|
| ResNet-18 | 97.50% | 96.34% | - | - |
| **EfficientNet-B0** | **98.59%** | **97.98%** | **97.76%** | **96.80%** |

**Mejora:** EfficientNet-B0 supera a ResNet-18 en ~1% en validación.

## Comparación con Referencia (Session 31 Original)

| Arquitectura | Original Warped 1.25 | CLI Warped 1.05 | Diferencia |
|--------------|---------------------|-----------------|------------|
| EfficientNet-B0 | 98.48% | 97.76% | -0.72% |
| ResNet-18 | 98.09% | 97.50%* | -0.59% |

*ResNet-18 de Sesión 15

**Análisis:** La diferencia de ~0.7% se explica por:
1. **Margin 1.05 vs 1.25:** El margin 1.25 incluye más contexto pulmonar
2. **Dataset diferente:** Posibles diferencias en split train/val/test
3. **Resultados dentro del margen esperado** (~95-97% para margin 1.05)

## Curva de Entrenamiento

```
Epoch  1: Val F1=0.9305 (baseline)
Epoch  4: Val F1=0.9657 -> Nuevo mejor
Epoch  8: Val F1=0.9745 -> Nuevo mejor (Val Acc=98.41%)
Epoch 16: Val F1=0.9754 -> Nuevo mejor
Epoch 23: Val F1=0.9764 -> Nuevo mejor
Epoch 32: Val F1=0.9787 -> Nuevo mejor
Epoch 40: Val F1=0.9798 -> Nuevo mejor (FINAL)
Epoch 50: Early stopping (sin mejora desde época 40)
```

**Tiempo total:** ~85 minutos (50 épocas x ~100s/época)

## Archivos Generados

```
outputs/classifier_efficientnet/
├── best_classifier.pt      # Checkpoint del mejor modelo
└── results.json            # Métricas y configuración
```

## Comandos de Evaluación

```bash
# Evaluar en test
python -m src_v2 evaluate-classifier \
    outputs/classifier_efficientnet/best_classifier.pt \
    --data-dir outputs/full_warped_dataset \
    --split test

# Evaluar en validación
python -m src_v2 evaluate-classifier \
    outputs/classifier_efficientnet/best_classifier.pt \
    --data-dir outputs/full_warped_dataset \
    --split val
```

## Conclusiones

1. **Objetivo cumplido:** EfficientNet-B0 alcanza 97.76% accuracy, validando la reproducibilidad
2. **Supera ResNet-18:** +1% en validación sobre el modelo de Sesión 15
3. **Margin 1.05 funciona:** Resultados competitivos sin necesidad de margin 1.25
4. **Viral_Pneumonia sigue siendo la clase más difícil:** 93% recall (vs 98% para COVID y Normal)

## Próximos Pasos (Sesión 17)

1. **Opcional:** Entrenar con margin 1.25 para comparación directa
2. **Pipeline end-to-end:** Probar `classify --warp` con EfficientNet
3. **Análisis de errores:** Investigar los 9 errores de Viral_Pneumonia
4. **DenseNet-121:** Completar la comparación de arquitecturas

## Estado del Proyecto

- **CLI:** 9 comandos funcionales
- **Tests:** 224 pasando
- **Predicción landmarks:** 3.71px error (ensemble 4 modelos)
- **Clasificación warped:** 97.76% (EfficientNet-B0, margin 1.05)
