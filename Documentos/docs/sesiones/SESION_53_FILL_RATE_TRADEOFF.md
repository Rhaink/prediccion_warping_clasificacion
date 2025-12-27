# SESION 53: INVESTIGACION FILL RATE Y TRADE-OFF

**Fecha:** 14 Diciembre 2025
**Branch:** audit/main
**Objetivo:** Investigar discrepancia de fill rate y documentar trade-off

---

## RESUMEN EJECUTIVO

Se investigó la discrepancia de fill rate entre datasets (99% vs 96%) y se documentó el trade-off óptimo. El dataset de 96% fill rate representa el **punto óptimo** con mejor accuracy (99.10%) y mejor robustez que el de 99%.

### HALLAZGO PRINCIPAL

La diferencia de fill rate se debe al **preprocesamiento de imagen**, no al uso de landmarks diferentes:

| Dataset | Preprocesamiento | Fill Rate | Accuracy | JPEG Q50 Deg |
|---------|------------------|-----------|----------|--------------|
| warped_99 | RGB + CLAHE (LAB) | 99.11% | 98.73% | 7.34% |
| warped_96 | Grayscale + CLAHE | 96.15% | 99.10% | 3.06% |

---

## 1. INVESTIGACION DE CAUSA

### 1.1 Hipótesis Inicial

Se sospechaba que el dataset de 99% fill rate usó landmarks de ground truth mientras que el de 96% usó landmarks predichos.

### 1.2 Hallazgo Real

**Ambos datasets usan landmarks predichos.** La diferencia está en:

1. **Procesamiento RGB+CLAHE en espacio LAB:**
   - Convierte imagen a LAB
   - Aplica CLAHE al canal L (luminancia)
   - Convierte de vuelta a RGB
   - **Resultado:** min pixel value = 2-3 (NO hay píxeles negros verdaderos)

2. **Procesamiento Grayscale+CLAHE:**
   - Carga imagen en escala de grises
   - Aplica CLAHE directamente
   - **Resultado:** min pixel value = 0 (preserva píxeles negros)

### 1.3 Verificación Experimental

```python
# Imagen COVID-1000.png procesada con cada método:

RGB + CLAHE:
  min=2, max=254
  Zeros: 0 out of 150528 (0%)

Grayscale + CLAHE:
  min=0, max=252
  Zeros: 3048 out of 50176 (6%)
```

### 1.4 Implicación para Fill Rate

La función `compute_fill_rate()` cuenta píxeles con valor = 0:

- **RGB+CLAHE:** Solo los bordes del warping quedan en negro (~0.89%)
- **Grayscale:** Bordes + píxeles originales negros quedan en cero (3-8%)

Por lo tanto:
- **99.11%** es un fill rate **artificialmente inflado**
- **96.15%** es un fill rate **más honesto**

---

## 2. ANALISIS DE TRADE-OFF

### 2.1 Datos Comparativos

| Dataset | Fill Rate | Accuracy | JPEG Q50 | JPEG Q30 | Blur σ1 | Score* |
|---------|-----------|----------|----------|----------|---------|--------|
| warped_47 | 47% | 98.02% | 0.53% | 1.32% | 6.06% | 97.49 |
| warped_96 | 96% | **99.10%** | 3.06% | 5.28% | 2.43% | **96.04** |
| warped_99 | 99% | 98.73% | 7.34% | 16.73% | 11.35% | 91.39 |
| original | 100% | 98.84% | 16.14% | 29.97% | 14.43% | 82.70 |

*Score = Accuracy - JPEG_Q50_Degradation (mayor es mejor)

### 2.2 Punto Óptimo

**warped_96 es el punto óptimo:**

1. **Mejor accuracy:** 99.10% (supera a warped_99 por 0.37%)
2. **Mejor robustez que warped_99:**
   - JPEG Q50: 2.4x mejor (3.06% vs 7.34%)
   - JPEG Q30: 3.2x mejor (5.28% vs 16.73%)
   - Blur σ1: 4.7x mejor (2.43% vs 11.35%)
3. **Mejor score compuesto:** 96.04 (vs 91.39 de warped_99)

### 2.3 Visualización

![Fill Rate Trade-off](../../outputs/fill_rate_tradeoff_analysis.png)

---

## 3. CONCLUSIONES

### 3.1 Recomendación

**Usar warped_96 como dataset/clasificador principal:**

```bash
# Dataset recomendado
outputs/warped_replication_v2/  # INVALIDADO; ver reportes/RESUMEN_DEPURACION_OUTPUTS.md

# Clasificador recomendado
outputs/classifier_replication_v2/best_classifier.pt  # INVALIDADO; ver reportes/RESUMEN_DEPURACION_OUTPUTS.md
```

### 3.2 Cuándo Usar Cada Dataset

| Caso de Uso | Dataset Recomendado | Razón |
|-------------|---------------------|-------|
| Producción general | warped_96 | Mejor balance accuracy/robustez |
| Máxima robustez | warped_47 | Mínima degradación JPEG |
| Legacy/compatibilidad | warped_99 | Mantener comportamiento anterior |

### 3.3 Actualización de GROUND_TRUTH.json

Se actualizó GROUND_TRUTH.json (v2.1.0) con:
- Nuevo dataset warped_96 y métricas
- Robustez del clasificador warped_96
- Sección fill_rate_tradeoff con análisis completo
- Explicación de diferencia de fill rate

---

## 4. ARCHIVOS MODIFICADOS

- `GROUND_TRUTH.json` - Actualizado a v2.1.0
- `outputs/fill_rate_tradeoff_analysis.png` - Nueva figura
- `docs/sesiones/SESION_53_FILL_RATE_TRADEOFF.md` - Esta documentación

---

## 5. COMANDOS DE REFERENCIA

### Generar Dataset con 96% Fill Rate

```bash
.venv/bin/python -m src_v2 generate-dataset \
    data/dataset/COVID-19_Radiography_Dataset \
    outputs/my_warped_dataset \
    --checkpoint checkpoints/final_model.pt \
    --margin 1.05 \
    --use-full-coverage
```

### Entrenar Clasificador

```bash
.venv/bin/python -m src_v2 train-classifier \
    outputs/my_warped_dataset \
    --output-dir outputs/my_classifier \
    --epochs 50 \
    --backbone resnet18
```

### Test de Robustez

```bash
.venv/bin/python -m src_v2 test-robustness \
    outputs/my_classifier/best_classifier.pt \
    --data-dir outputs/my_warped_dataset
```

---

## 6. NOTAS TECNICAS

### Fill Rate Calculation

```python
def compute_fill_rate(warped_image):
    black_pixels = np.sum(warped_image == 0)
    fill_rate = 1 - (black_pixels / warped_image.size)
    return fill_rate
```

Para imágenes RGB (3 canales), cuenta zeros en cada canal:
- Total pixels = H * W * 3
- Un pixel RGB es "negro" si R=0 AND G=0 AND B=0

### Diferencia de Preprocesamiento

| Aspecto | RGB+CLAHE (warped_99) | Grayscale+CLAHE (warped_96) |
|---------|----------------------|----------------------------|
| Carga | `Image.convert("RGB")` | `cv2.IMREAD_GRAYSCALE` |
| CLAHE | En canal L (LAB) | Directo |
| Resultado | Sin píxeles negros | Preserva negros |
| Fill Rate | Inflado (~99%) | Honesto (~96%) |

---

**Sesión completada:** 14 Diciembre 2025
