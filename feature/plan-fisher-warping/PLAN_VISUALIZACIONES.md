# Plan de Visualizaciones y Mejoras - Proyecto Fisher-Warping

**Objetivo:** Reducir la "caja negra" del proyecto mediante visualizaciones explicativas y análisis estadístico robusto.

**Fecha:** 2026-01-06
**Estado:** Pendiente aprobación

---

## 📊 Matriz de Priorización (Impacto vs Esfuerzo)

| Visualización | Impacto | Esfuerzo | Prioridad | Fase |
|--------------|---------|----------|-----------|------|
| **1. Landmarks + Triangulación** | 🔴 CRÍTICO | 🟢 Bajo | P0 | 1 |
| **2. Scatter 2D PCA (PC1 vs PC2)** | 🔴 CRÍTICO | 🟢 Bajo | P0 | 1 |
| **3. Warping Step-by-Step** | 🔴 CRÍTICO | 🟡 Medio | P0 | 1 |
| **4. Curvas ROC** | 🟠 Alto | 🟢 Bajo | P1 | 2 |
| **5. Tests Estadísticos** | 🟠 Alto | 🟢 Bajo | P1 | 2 |
| **6. Galería Vecinos KNN** | 🟠 Alto | 🟡 Medio | P1 | 2 |
| **7. Fisher 2D Boundary** | 🟡 Medio | 🟢 Bajo | P2 | 3 |
| **8. Reconstrucción Progresiva PCA** | 🟡 Medio | 🟡 Medio | P2 | 3 |
| **9. Feature Importance Map** | 🟡 Medio | 🟡 Medio | P2 | 3 |
| **10. Journey Completo** | 🟢 Bajo | 🔴 Alto | P3 | 4 |
| **11. TSNE/UMAP** | 🟢 Bajo | 🟡 Medio | P3 | 4 |
| **12. Dashboard Interactivo** | 🟢 Bajo | 🔴 Alto | P3 | Futuro |

**Leyenda:**
- 🔴 CRÍTICO/Alto = Necesario para la reunión con asesor
- 🟠 Alto/Medio = Muy recomendable, demuestra rigor
- 🟡 Medio = Mejora la presentación
- 🟢 Bajo = Nice to have

---

## 🎯 FASE 1: Fundamentos Críticos (ANTES de reunión)

### Objetivo
Cerrar las brechas más grandes en explicabilidad. El asesor PREGUNTARÁ sobre el warping y la separación en PCA.

### Duración Estimada
3-4 horas de trabajo

---

### ✅ TAREA 1.1: Visualización de Landmarks sobre Imagen

**Archivo a crear:** `scripts/visualize_landmarks_overlay.py`

**¿Qué hace?**
- Carga una imagen del dataset
- Carga los landmarks predichos
- Dibuja círculos sobre cada uno de los 15 puntos anatómicos
- Agrega labels (L1, L2, ... L15)
- Guarda imagen resultado

**Input necesario:**
- Imagen original (299x299)
- Archivo de landmarks predichos (.npz)
- Lista de nombres de landmarks

**Output:**
- `results/figures/warping_explained/landmarks_overlay_example.png`

**Dependencias:**
- OpenCV para dibujo
- NumPy para cargar landmarks
- Matplotlib para labels

**Pasos de implementación:**
```python
1. Cargar imagen de ejemplo (1 COVID, 1 Normal)
2. Cargar landmarks correspondientes
3. Crear figura con subplots (1x2)
4. Para cada imagen:
   - Dibujar círculos en coordenadas de landmarks
   - Agregar números/labels
   - Agregar título con nombre de archivo
5. Guardar figura final
```

**Criterio de éxito:**
- Se ven claramente los 15 puntos
- Los landmarks están en posiciones anatómicas correctas
- La imagen es legible y profesional

---

### ✅ TAREA 1.2: Visualización de Triangulación Delaunay

**Archivo a crear:** `scripts/visualize_delaunay_triangulation.py`

**¿Qué hace?**
- Toma imagen + landmarks
- Calcula triangulación de Delaunay
- Dibuja las aristas de los triángulos sobre la imagen
- Muestra que toda la imagen está cubierta

**Input necesario:**
- Imagen original
- Landmarks (15 puntos)
- Puntos de borde adicionales (8 puntos)

**Output:**
- `results/figures/warping_explained/delaunay_triangulation_example.png`

**Pasos de implementación:**
```python
1. Cargar imagen + landmarks
2. Agregar 8 puntos de borde (esquinas + medios)
3. Calcular triangulación usando scipy.spatial.Delaunay
4. Dibujar triángulos sobre imagen
5. Resaltar landmarks en un color
6. Resaltar puntos de borde en otro color
7. Guardar
```

**Criterio de éxito:**
- Se ven ~23 triángulos claramente
- Toda la imagen está cubierta
- Landmarks y bordes distinguibles

---

### ✅ TAREA 1.3: Panel Warping Paso a Paso

**Archivo a crear:** `scripts/visualize_warping_pipeline.py`

**¿Qué hace?**
- Crea un panel 2x2 mostrando la evolución:
  1. Imagen original
  2. Original + Landmarks
  3. Original + Triangulación
  4. Imagen Warped final

**Input necesario:**
- Imagen original
- Landmarks
- Imagen warped (resultado final)

**Output:**
- `results/figures/warping_explained/warping_step_by_step.png`

**Pasos de implementación:**
```python
1. Cargar todos los inputs
2. Crear figura con subplots (2x2)
3. Panel 1: Imagen original limpia
4. Panel 2: Original + landmarks dibujados
5. Panel 3: Original + triangulación completa
6. Panel 4: Resultado warped
7. Agregar títulos descriptivos
8. Guardar
```

**Criterio de éxito:**
- Historia visual clara del proceso
- Fácil de entender para alguien que no conoce el proyecto
- Calidad de presentación profesional

---

### ✅ TAREA 1.4: Scatter 2D del Espacio PCA

**Archivo a crear:** `scripts/visualize_pca_2d_space.py`

**¿Qué hace?**
- Proyecta los datos en PC1 vs PC2
- Crea scatter plot coloreado por clase
- Agrega elipses de confianza (95%)
- Muestra la separación visual

**Input necesario:**
- Ponderantes PCA (ya calculados)
- Labels de clase
- Para cada dataset: full_warped, full_original

**Output:**
- `results/figures/pca_explained/pca_2d_scatter_full_warped.png`
- `results/figures/pca_explained/pca_2d_scatter_full_original.png`
- `results/figures/pca_explained/pca_2d_scatter_comparison.png` (ambos lado a lado)

**Pasos de implementación:**
```python
1. Cargar ponderantes PCA del training+validation+test
2. Tomar solo PC1 y PC2
3. Crear scatter plot:
   - Enfermo = rojo
   - Normal = azul
4. Calcular y dibujar elipses de confianza
5. Agregar leyenda, labels de ejes
6. Repetir para warped y original
7. Crear comparación lado a lado
8. Guardar todas las versiones
```

**Criterio de éxito:**
- Se ve claramente la separación (o falta de ella)
- Warped muestra mejor separación que original
- Gráfica profesional lista para presentar

---

### 📝 ENTREGABLES FASE 1

Al completar Fase 1 tendrás:

```
results/figures/
├── warping_explained/
│   ├── landmarks_overlay_example.png
│   ├── delaunay_triangulation_example.png
│   └── warping_step_by_step.png
└── pca_explained/
    ├── pca_2d_scatter_full_warped.png
    ├── pca_2d_scatter_full_original.png
    └── pca_2d_scatter_comparison.png
```

**Scripts creados:**
- `scripts/visualize_landmarks_overlay.py`
- `scripts/visualize_delaunay_triangulation.py`
- `scripts/visualize_warping_pipeline.py`
- `scripts/visualize_pca_2d_space.py`

**Agregado a notebooks:**
- 01_Pipeline_Completo.ipynb: agregar las 3 imágenes de warping
- 02_Fase1_PCA_Eigenfaces.ipynb: agregar los 3 scatters 2D

---

## 🎯 FASE 2: Validación Estadística (PARA tesis)

### Objetivo
Demostrar rigor científico mediante análisis estadístico apropiado.

### Duración Estimada
2-3 horas de trabajo

---

### ✅ TAREA 2.1: Curvas ROC y AUC

**Archivo a crear:** `scripts/generate_roc_curves.py`

**¿Qué hace?**
- Convierte predicciones KNN a probabilidades
- Genera curvas ROC para cada configuración
- Calcula AUC
- Compara warped vs original

**Input necesario:**
- Predicciones KNN
- Distancias a vecinos (para calcular probabilidades)
- Labels verdaderos

**Output:**
- `results/figures/statistical_analysis/roc_curves_comparison.png`
- `results/figures/statistical_analysis/auc_table.png`

**Criterio de éxito:**
- Curvas ROC bien formadas
- AUC warped > AUC original
- Tabla de métricas clara

---

### ✅ TAREA 2.2: Tests de Significancia Estadística

**Archivo a crear:** `scripts/statistical_significance_tests.py`

**¿Qué hace?**
- T-test pareado entre warped y original
- Calcula intervalos de confianza
- Bootstrap para robustez
- Genera tabla de p-values

**Input necesario:**
- Accuracies por fold (necesitas implementar k-fold CV)
- O usar bootstrap sobre el test set

**Output:**
- `results/figures/statistical_analysis/significance_tests.png`
- `results/tables/statistical_tests.csv`

**Criterio de éxito:**
- p-value < 0.05 demuestra significancia
- Visualización clara de resultados
- Intervalos de confianza calculados

---

### ✅ TAREA 2.3: Galería de Vecinos KNN

**Archivo a crear:** `scripts/visualize_knn_neighbors.py`

**¿Qué hace?**
- Selecciona casos ejemplo (correcto, error FP, error FN)
- Muestra imagen central + sus K vecinos más cercanos
- Indica distancias y clases

**Input necesario:**
- Imágenes del test set
- Predicciones KNN
- Vecinos más cercanos (indices + distancias)

**Output:**
- `results/figures/knn_explained/neighbors_correct_example.png`
- `results/figures/knn_explained/neighbors_false_positive_example.png`
- `results/figures/knn_explained/neighbors_false_negative_example.png`

**Criterio de éxito:**
- Se entiende POR QUÉ se clasificó así
- Imágenes legibles
- 3 casos diferentes bien documentados

---

### 📝 ENTREGABLES FASE 2

```
results/figures/
├── statistical_analysis/
│   ├── roc_curves_comparison.png
│   ├── auc_table.png
│   └── significance_tests.png
└── knn_explained/
    ├── neighbors_correct_example.png
    ├── neighbors_false_positive_example.png
    └── neighbors_false_negative_example.png

results/tables/
└── statistical_tests.csv
```

---

## 🎯 FASE 3: Profundización (Para responder preguntas)

### Objetivo
Tener material de respaldo si el asesor profundiza en algún tema específico.

### Duración Estimada
3-4 horas de trabajo

---

### ✅ TAREA 3.1: Fisher 2D Decision Boundary

**¿Qué hace?**
- Scatter de PC1 vs PC2 DESPUÉS de amplificación Fisher
- Muestra "decision boundary" conceptual
- Violin plots de distribuciones

**Criterio de éxito:**
- Separación más clara que scatter PCA original
- Se entiende el efecto de amplificación

---

### ✅ TAREA 3.2: Reconstrucción Progresiva PCA

**¿Qué hace?**
- Reconstruye una imagen con 10, 20, 30, 40, 50 componentes
- Muestra cómo mejora la calidad
- Panel 2x3 con comparación visual

**Criterio de éxito:**
- Demuestra que 50 componentes es suficiente
- Calidad visual profesional

---

### ✅ TAREA 3.3: Feature Importance Heatmap

**¿Qué hace?**
- Proyecta Fisher ratios de vuelta al espacio de imagen
- Crea heatmap mostrando qué regiones discriminan
- ¿Son pulmones? ¿Bordes? ¿Centro?

**Criterio de éxito:**
- Se ven regiones anatómicas claras
- Correlaciona con conocimiento médico esperado

---

### 📝 ENTREGABLES FASE 3

```
results/figures/
├── fisher_explained/
│   ├── fisher_2d_boundary.png
│   └── violin_plots_top_features.png
├── pca_explained/
│   └── reconstruction_progressive.png
└── interpretation/
    └── feature_importance_heatmap.png
```

---

## 🎯 FASE 4: Extras (Solo si hay tiempo)

### Contenido
- Journey completo de imagen
- TSNE/UMAP embedding
- Dashboard interactivo
- Ablation study detallado

**Nota:** Solo hacer si las Fases 1-3 están completas y hay tiempo antes de la reunión.

---

## 📋 CHECKLIST DE EJECUCIÓN

### Antes de Empezar
- [ ] Aprobar este plan
- [ ] Verificar que todos los datos necesarios están disponibles
- [ ] Crear directorios de output

### Durante Ejecución
Para cada tarea:
- [ ] Crear script correspondiente
- [ ] Generar visualización
- [ ] Verificar calidad visual
- [ ] Agregar a notebook correspondiente
- [ ] Documentar en el plan (marcar como ✅)
- [ ] Commit a git

### Al Finalizar Cada Fase
- [ ] Review de todas las visualizaciones
- [ ] Actualizar notebooks
- [ ] Probar que las rutas de imágenes funcionan
- [ ] Commit de la fase completa

---

## 🔧 ESTRUCTURA DE CÓDIGO RECOMENDADA

```python
# Cada script debe seguir este patrón:

"""
Script: visualize_XXX.py
Propósito: [Descripción breve]
Input: [Qué archivos necesita]
Output: [Qué genera]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración
BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "results" / "figures"
DATA_DIR = BASE_DIR / "data"

def main():
    # 1. Cargar datos
    # 2. Procesar
    # 3. Generar visualización
    # 4. Guardar
    # 5. Print confirmación

if __name__ == "__main__":
    main()
```

---

## 📊 MÉTRICAS DE ÉXITO

### Para Fase 1 (Crítico)
- ✅ Asesor puede ENTENDER el warping visualmente
- ✅ Asesor puede VER la separación en PCA
- ✅ No quedan dudas sobre "caja negra" del warping

### Para Fase 2 (Rigor)
- ✅ Resultados estadísticamente significativos
- ✅ Curvas ROC demuestran mejora cuantificable
- ✅ Vecinos KNN explican las clasificaciones

### Para Fase 3 (Profundidad)
- ✅ Material de respaldo listo
- ✅ Todas las preguntas anticipadas tienen respuesta visual

---

## ⏱️ ESTIMACIÓN TEMPORAL TOTAL

| Fase | Tiempo Estimado | Dependencias |
|------|----------------|--------------|
| Fase 1 | 3-4 horas | Ninguna |
| Fase 2 | 2-3 horas | Fase 1 completa |
| Fase 3 | 3-4 horas | Fase 1 completa |
| Fase 4 | 4-6 horas | Todo lo anterior |

**Total mínimo viable (Fases 1+2):** 5-7 horas
**Total recomendado (Fases 1+2+3):** 8-11 horas
**Total completo (Todas):** 12-17 horas

---

## 📅 TIMELINE DE 2 SEMANAS (APROBADO)

**Periodo:** 2026-01-06 al 2026-01-20 (14 días)

### Semana 1: Fundamentos + Estadística (Fases 1 y 2)

**Días 1-3 (Lun-Mié):** FASE 1 - Fundamentos Críticos
- Día 1: Tareas 1.1 y 1.2 (Landmarks + Triangulación)
- Día 2: Tarea 1.3 (Warping step-by-step)
- Día 3: Tarea 1.4 (Scatter 2D PCA) + Revisión Fase 1

**Días 4-6 (Jue-Sáb):** FASE 2 - Validación Estadística
- Día 4: Tarea 2.1 (Curvas ROC)
- Día 5: Tarea 2.2 (Tests estadísticos)
- Día 6: Tarea 2.3 (Galería vecinos KNN)

**Día 7 (Dom):** Descanso / Buffer / Revisión Semana 1

### Semana 2: Profundización + Refinamiento (Fase 3 + pulido)

**Días 8-10 (Lun-Mié):** FASE 3 - Profundización
- Día 8: Tarea 3.1 (Fisher 2D boundary)
- Día 9: Tarea 3.2 (Reconstrucción progresiva PCA)
- Día 10: Tarea 3.3 (Feature importance map)

**Días 11-13 (Jue-Sáb):** Refinamiento y Documentación
- Día 11: Agregar TODAS las imágenes a notebooks
- Día 12: Review completo de calidad visual
- Día 13: Preparar material para reunión (slides, talking points)

**Día 14 (Dom):** Ensayo de presentación / Buffer final

### Puntos de Control (Checkpoints)

- ✅ **Checkpoint 1 (Día 3):** Fase 1 completa - Warping explicado visualmente
- ✅ **Checkpoint 2 (Día 6):** Fase 2 completa - Estadística sólida
- ✅ **Checkpoint 3 (Día 10):** Fase 3 completa - Material de profundización listo
- ✅ **Checkpoint 4 (Día 13):** Todo integrado en notebooks, listo para presentar

### Contingencias

- **Si vamos adelantados:** Trabajar en Fase 4 (TSNE, Journey, Dashboard)
- **Si vamos atrasados:** Priorizar Fases 1 y 2 (son críticas)
- **Buffer days:** Días 7 y 14 pueden usarse para recuperar retrasos

---

## 🚀 PRÓXIMOS PASOS

1. **REVISAR este plan** - ¿Estás de acuerdo con las prioridades?
2. **APROBAR** - Dar luz verde para empezar
3. **EJECUTAR Fase 1** - Tarea por tarea
4. **EVALUAR** - ¿Continuamos a Fase 2?

---

## 📝 NOTAS IMPORTANTES

- **Cada visualización debe ser autocontenida** - entendible sin explicación adicional
- **Calidad > Cantidad** - Mejor 5 gráficas excelentes que 15 mediocres
- **Consistencia visual** - Mismo estilo, colores, fuentes en todas
- **Rutas relativas** - Para que funcionen en notebooks
- **Git commits frecuentes** - Una tarea = un commit

---

**¿Listo para aprobar y comenzar con Fase 1, Tarea 1.1?**
