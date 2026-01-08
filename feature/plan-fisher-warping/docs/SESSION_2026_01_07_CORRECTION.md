# Sesion 2026-01-07: Correccion Metodologica de Experimentos

## Resumen Ejecutivo

Esta sesion corrigio un error metodologico critico en los experimentos de clasificacion
de 3 clases y actualizo todos los notebooks con los resultados corregidos.

---

## Problema Identificado

### Error Original
El experimento de 3 clases reutilizaba features de Fase 4 (generadas del CSV de 2 clases
con 12,402 imagenes) y simplemente re-etiquetaba por nombre de image_id.

**Consecuencia:** Test size incorrecto (1,245 en lugar de 680) y mezcla de datasets.

### Solucion Implementada
Cada experimento ahora usa su dataset optimizado con pipeline completo independiente:

| Experimento | CSV | Dataset | Test Size |
|-------------|-----|---------|-----------|
| 2 clases | `02_full_balanced_2class_*.csv` | 12,402 imgs | 1,245 |
| 3 clases | `01_full_balanced_3class_*.csv` | 6,725 imgs | 680 |
| 2C Comparable | `01_full_balanced_3class_*.csv` (reagrupado) | 6,725 imgs | 680 |

---

## Archivos Modificados

### 1. Script Principal Corregido
**Archivo:** `src/generate_phase7.py`

**Cambios:**
- Carga imagenes directamente desde CSV de 3 clases (no reutiliza phase4_features)
- Ejecuta pipeline completo: PCA -> Z-score -> Fisher multiclase -> KNN
- Agrega experimento 2C-Comparable (mismas imagenes que 3C, reagrupadas)
- Permite comparacion directa 2C vs 3C con mismas imagenes

### 2. Notebooks Actualizados

| Notebook | Cambios |
|----------|---------|
| `00_Resumen_Ejecutivo.ipynb` | Tabla de resultados, nota metodologica, metricas |
| `01_Pipeline_Completo.ipynb` | Nueva seccion de resultados experimentales |
| `08_Hallazgos_Resultados.ipynb` | Resultados, visualizaciones, conclusiones |

### 3. Nuevas Visualizaciones Generadas

```
results/figures/phase7_comparison/
├── comparacion_final.png              # Grafico principal
├── accuracy_comparison_all.png        # Barras comparativas
├── matrices_confusion_comparativas.png # 4 matrices de confusion
├── 3class_warped/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── fisher_ratios.png
│   └── k_optimization.png
├── 3class_original/
│   └── [mismos archivos]
├── 2class_comparable_warped/
│   └── [mismos archivos]
└── 2class_comparable_original/
    └── [mismos archivos]
```

### 4. Nuevos Archivos de Metricas

```
results/metrics/phase7_comparison/
├── summary.json                    # Resumen completo en JSON
├── comparacion_2c_vs_3c.csv       # Tabla comparativa
├── 3class_warped_results.csv      # Metricas 3C warped
├── 3class_warped_predictions.csv  # Predicciones 3C warped
├── 3class_original_results.csv
├── 3class_original_predictions.csv
├── 2class_comparable_warped_results.csv
├── 2class_comparable_warped_predictions.csv
├── 2class_comparable_original_results.csv
└── 2class_comparable_original_predictions.csv
```

---

## Resultados Corregidos

### Tabla Comparativa Final

| Experimento | Dataset | Test | Original | Warped | Mejora |
|-------------|---------|------|----------|--------|--------|
| **2C (principal)** | 12,402 imgs | 1,245 | 77.75% | **81.69%** | **+3.94%** |
| 2C Comparable | 6,725 imgs | 680 | 79.26% | **82.79%** | **+3.53%** |
| 3C | 6,725 imgs | 680 | 77.06% | **80.44%** | **+3.38%** |

### Metricas Detalladas

| Experimento | K optimo | Test Accuracy | Macro F1 |
|-------------|----------|---------------|----------|
| 2C Warped (12K) | 7 | 81.69% | 0.8052 |
| 2C Original (12K) | 5 | 77.75% | 0.7670 |
| 2C Comp. Warped (6K) | 9 | 82.79% | 0.8185 |
| 2C Comp. Original (6K) | 15 | 79.26% | 0.7790 |
| 3C Warped (6K) | 21 | 80.44% | 0.8106 |
| 3C Original (6K) | 21 | 77.06% | 0.7809 |

### Verificacion de Test Sizes

| Experimento | Test Size | Esperado | Estado |
|-------------|-----------|----------|--------|
| 3class_warped | 680 | 680 | OK |
| 3class_original | 680 | 680 | OK |
| 2class_comparable_warped | 680 | 680 | OK |
| 2class_comparable_original | 680 | 680 | OK |

---

## Conclusiones Clave

### 1. El Warping Mejora Consistentemente
- 2C (12K): +3.94%
- 2C (6K): +3.53%
- 3C (6K): +3.38%

### 2. Comparacion Directa 2C vs 3C (mismas imagenes)
- 2C Warped: 82.79%
- 3C Warped: 80.44%
- Diferencia: +2.35% (clasificar 2 clases es mas facil)

### 3. Concentracion de Varianza
- PC1 Warped: 46.4%
- PC1 Original: 27.1%
- Ratio: 1.71x mas varianza en PC1 con warping

---

## Numeros para la Tesis

### Experimento Principal (2 clases, 12K imgs)
```
Original: 77.75% -> Warped: 81.69% = +3.94%
Test size: 1,245 imagenes
```

### Experimento Secundario (3 clases, 6K imgs)
```
Original: 77.06% -> Warped: 80.44% = +3.38%
Test size: 680 imagenes
```

### Comparacion Directa
```
Mismas 6,725 imagenes:
- 2 clases: 82.79%
- 3 clases: 80.44%
"Clasificar 2 clases es ~2.4% mas facil que 3 clases"
```

---

## Proximos Pasos Sugeridos

1. **Revision de documentacion**: Actualizar docs/02_PIPELINE.md si es necesario
2. **Validacion cruzada**: Considerar agregar k-fold cross validation
3. **Analisis de errores**: Investigar imagenes que fallan consistentemente
4. **Preparacion de tesis**: Usar los numeros corregidos en el documento final

---

## Comandos para Reproducir

```bash
# Desde feature/plan-fisher-warping/
cd /home/donrobot/Projects/prediccion_warping_clasificacion/feature/plan-fisher-warping

# Ejecutar experimentos corregidos
python src/generate_phase7.py

# Verificar resultados
cat results/metrics/phase7_comparison/summary.json
```

---

## Fecha y Autor

- **Fecha:** 2026-01-07
- **Sesion:** Correccion metodologica Phase 7
- **Estado:** Completado
