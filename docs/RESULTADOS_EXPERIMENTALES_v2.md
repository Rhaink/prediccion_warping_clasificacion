# Resultados Experimentales - Version Corregida

**Fecha:** 2025-12-10
**Estado:** Claims reformulados post-Sesion 35/36/39
**Ultima actualizacion:** Sesion 39 - Cross-evaluation valido completado

---

## Resumen de Resultados Validados

### 1. Prediccion de Landmarks (VALIDADO)

| Metrica | Valor |
|---------|-------|
| Error ensemble (4 modelos + TTA) | **3.71 px** |
| Desviacion estandar | 2.42 px |
| Error mediano | 3.17 px |
| Mejor modelo individual | 4.04 px |

**Arquitectura:** ResNet-18 + Coordinate Attention + Deep Head (768 dim)

### 2. Clasificacion COVID-19 (VALIDADO)

| Dataset | Accuracy Test | F1-Score |
|---------|---------------|----------|
| Original 3 clases | 98.84% | 98.16% |
| Warped 47% fill | 98.02% | - |
| Warped 99% fill | 98.73% | 97.95% |
| Original Cropped 47% | 98.89% | 98.25% |

**Nota:** Todos los modelos usan 3 clases (COVID, Normal, Viral_Pneumonia) para comparacion valida.

---

## 3. Robustez a Perturbaciones (VALIDADO)

### 3.1 Tabla Comparativa Completa (Sesion 39)

| Modelo | Fill Rate | JPEG Q50 deg | JPEG Q30 deg | Blur σ1 deg |
|--------|-----------|--------------|--------------|-------------|
| **Original 100%** | 100% | 16.14% | 29.97% | 14.43% |
| **Original Cropped 47%** | 47% | 2.11% | 7.65% | 7.65% |
| **Warped 47%** | 47% | **0.53%** | **1.32%** | **6.06%** |
| **Warped 99%** | 99% | 7.34% | 16.73% | 11.35% |

### 3.2 Analisis Causal de Robustez

El experimento de control (Original Cropped 47%) revela que la robustez tiene **DOS componentes**:

| Componente | Contribucion | Evidencia |
|------------|--------------|-----------|
| **Reduccion de informacion** | ~75% | Original Cropped 47% es 7.6x mas robusto que Original 100% |
| **Normalizacion geometrica** | ~25% adicional | Warped 47% es 4x mas robusto que Original Cropped 47% |

### 3.3 Conclusiones de Robustez

- **JPEG Q50:** Warped 47% es **30x mas robusto** que Original 100%
- **JPEG Q30:** Warped 47% es **23x mas robusto** que Original 100%
- **Blur σ1:** Warped 47% es **2.4x mas robusto** que Original 100%

**Mecanismo:** La robustez proviene principalmente de regularizacion implicita por fill rate reducido,
con contribucion adicional de la normalizacion geometrica.

---

## 4. Cross-Evaluation Valido (Sesion 39 - NUEVO)

### 4.1 Resultados Cross-Evaluation 3 Clases

**Configuracion:**
- Model A: Clasificador Original 3 clases
- Model B: Clasificador Warped 99% fill (full_coverage)
- Ambos datasets: 1,895 muestras test, 3 clases identicas

| Modelo | En Dataset A (Original) | En Dataset B (Warped 99%) |
|--------|-------------------------|---------------------------|
| **Model A (Original)** | 98.84% | 91.13% |
| **Model B (Warped)** | 95.57% | 98.73% |

### 4.2 Gaps de Generalizacion

| Modelo | Gap de Generalizacion | Interpretacion |
|--------|----------------------|----------------|
| Model A (Original) | **7.70%** | Pierde 7.70% al evaluar en warped |
| Model B (Warped) | **3.17%** | Pierde 3.17% al evaluar en original |

**Ratio: 2.4x** - El modelo warped generaliza **2.4x mejor** que el original.

### 4.3 Correccion de Claim Anterior

| Claim | Valor Anterior | Valor Correcto |
|-------|----------------|----------------|
| Generalizacion | ~~11x mejor~~ (INVALIDO) | **2.4x mejor** (VALIDO) |
| Razon invalidez | 4 clases vs 3 clases | Ahora 3 clases vs 3 clases |

---

## 5. Pulmonary Focus Score (PFS) - Actualizado Sesion 39

### 5.1 Resultados con Mascaras Warped (VALIDO)

| Metrica | Valor |
|---------|-------|
| **Mean PFS** | **0.487** (+/- 0.091) |
| Median PFS | 0.486 |
| Range | [0.185, 0.722] |

### 5.2 PFS por Clase

| Clase | PFS Mean | Std | n |
|-------|----------|-----|---|
| COVID | 0.478 | 0.076 | 362 |
| Normal | 0.510 | 0.118 | 138 |

### 5.3 Interpretacion

**Conclusion:** El modelo warped **NO** enfoca exclusivamente en los pulmones.
- PFS ~0.49 significa ~49% de atencion en region pulmonar
- Esto es aproximadamente igual al chance (~50%)
- **No hay evidencia de que el warping fuerce atencion pulmonar**

### 5.4 Correccion de Claim Anterior

| Claim Anterior | Estado |
|----------------|--------|
| "Warping fuerza atencion pulmonar" | **INVALIDO** |

**Nota metodologica:** Resultados calculados con mascaras correctamente warped usando `warp_mask()`.
Ver `outputs/pfs_warped_valid_full/pfs_warped_summary.json` para datos completos.

---

## 6. Reformulacion de Claims

### INCORRECTO (version anterior):
> "La normalizacion geometrica reduce el gap de generalizacion de 25.36% a 2.24%,
> demostrando que el modelo warped generaliza 11x mejor"

### CORRECTO (version actual - Sesion 39):
> "El pipeline de warping proporciona:
>
> **1. Robustez superior a perturbaciones:**
> - **30x mas robusto** a compresion JPEG (Q50)
> - **23x mas robusto** a compresion JPEG severa (Q30)
> - **2.4x mas robusto** a blur gaussiano
>
> **2. Mejor generalizacion cross-dataset:**
> - El modelo warped generaliza **2.4x mejor** que el original
> - Gap de generalizacion: 3.17% (warped) vs 7.70% (original)
>
> **3. Mecanismo de robustez identificado:**
> - ~75% por reduccion de informacion (regularizacion implicita)
> - ~25% adicional por normalizacion geometrica
>
> **4. PFS (Pulmonary Focus Score):**
> - El warping **NO** fuerza atencion pulmonar
> - PFS ~0.49 (aproximadamente igual a chance)
> - Conclusion: la robustez NO proviene de forzar atencion en pulmones"

---

## 7. Trabajo Futuro Requerido

### Alta Prioridad - COMPLETADO
1. [x] Generar dataset warped con `use_full_coverage=True` (~99% fill rate) - **COMPLETADO**
2. [x] Re-evaluar cross-validation con datasets informativamente equivalentes - **COMPLETADO (2.4x)**
3. [x] Warpear mascaras pulmonares para PFS valido - **COMPLETADO (PFS ~0.49)**

### Media Prioridad
4. [ ] Evaluar en datasets externos (Montgomery, Shenzhen)
5. [x] Documentar trade-offs de normalizacion geometrica - **COMPLETADO (Sesion 39)**
6. [ ] Implementar tests criticos faltantes

---

## 8. Referencias de Verificacion

### Sesion 35: Analisis critico inicial
- `/docs/sesiones/SESION_35_ANALISIS_CRITICO.md` - Analisis completo

### Sesion 39: Experimento de control y cross-evaluation valido
- `/docs/sesiones/SESION_39_EXPERIMENTO_CONTROL.md` - Experimento Original Cropped 47%
- `/outputs/cross_evaluation_valid_3classes/cross_evaluation_results.json` - Cross-eval 3 clases
- `/outputs/original_3_classes/dataset_summary.json` - Dataset filtrado 3 clases
- `/outputs/robustness_original_cropped_47.json` - Robustez control

### Datasets generados
- `/outputs/original_3_classes/` - 15,153 imagenes (3 clases, sin Lung_Opacity)
- `/outputs/full_coverage_warped_dataset/` - 15,153 imagenes (99% fill rate)
- `/outputs/original_cropped_47/` - 15,153 imagenes (47% fill rate, control)

---

## Apendice: Bugs Corregidos

### Sesion 36
| Bug | Estado | Commit |
|-----|--------|--------|
| Import warping en cli.py L1025 | CORREGIDO | 11fb902 |
| Import warping en cli.py L1684 | CORREGIDO | 11fb902 |
| Tests de validacion de imports | AGREGADOS | 11fb902 |

### Sesion 37
| Bug | Estado | Commit |
|-----|--------|--------|
| Bounding box exceeds image bounds | CORREGIDO | dd2bfb4 |
