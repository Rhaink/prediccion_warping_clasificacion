# Resultados Experimentales - Version Corregida

**Fecha:** 2025-12-10
**Estado:** Claims reformulados post-Sesion 35/36

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

| Dataset | Accuracy | F1-Score |
|---------|----------|----------|
| Original (train) | 98.81% | - |
| Original (test) | 73.45% | - |
| Warped (train) | 98.02% | - |
| Warped (test) | 95.78% | - |

**Nota:** Los modelos fueron entrenados y evaluados en sus respectivos datasets.

---

## 3. Robustez a Perturbaciones (VALIDADO)

### 3.1 Robustez JPEG (Compresion Q=10)

| Modelo | Accuracy Original | Accuracy JPEG | Degradacion |
|--------|-------------------|---------------|-------------|
| Original | 98.81% | 82.67% | -16.14% |
| **Warped** | 98.02% | 97.50% | **-0.53%** |

**Conclusion:** El modelo warped es **30x mas robusto** a compresion JPEG.

### 3.2 Robustez Blur (Gaussian sigma=2)

| Modelo | Accuracy Original | Accuracy Blur | Degradacion |
|--------|-------------------|---------------|-------------|
| Original | 98.81% | ~92% | ~-7% |
| **Warped** | 98.02% | ~96% | ~-2% |

**Conclusion:** El modelo warped es **~3x mas robusto** a blur.

---

## 4. Limitaciones Metodologicas (CRITICO)

### 4.1 Cross-Evaluation: INVALIDA

El claim anterior de "generaliza 11x mejor" (gap 25.36% -> 2.24%) es **INVALIDO** debido a:

| Aspecto | Dataset Original | Dataset Warped |
|---------|------------------|----------------|
| Fill rate | ~100% | ~47% |
| Fondo negro | 0% | ~53% |
| Marcas hospitalarias | Presentes | Ausentes |
| Bordes de imagen | Presentes | Ausentes |

**Razon:** El gap mide **perdida de informacion**, no sobreajuste geometrico.

### 4.2 Pulmonary Focus Score (PFS): NO CONCLUYENTE

| Dataset | PFS Promedio |
|---------|--------------|
| Original | 35.5% |
| Warped | 35.6% |
| Diferencia | +0.1% (p=0.856) |

**Problema:** Las mascaras pulmonares NO estan warped, haciendo la comparacion invalida.

---

## 5. Reformulacion de Claims

### INCORRECTO (version anterior):
> "La normalizacion geometrica reduce el gap de generalizacion de 25.36% a 2.24%,
> demostrando que el modelo warped generaliza 11x mejor"

### CORRECTO (version actual):
> "La normalizacion geometrica proporciona **robustez significativamente mayor**
> a perturbaciones comunes en imagenes medicas:
> - **30x mas robusto** a compresion JPEG
> - **3x mas robusto** a blur gaussiano
>
> La comparacion cross-evaluation entre datasets original y warped no es valida
> debido a diferencias en el contenido informacional (~47% vs ~100% fill rate).
> Se requiere un dataset warped con full_coverage (~96% fill rate) para una
> evaluacion justa de generalizacion."

---

## 6. Trabajo Futuro Requerido

### Alta Prioridad
1. [ ] Generar dataset warped con `use_full_coverage=True` (~96% fill rate)
2. [ ] Re-evaluar cross-validation con datasets informativamente equivalentes
3. [ ] Warpear mascaras pulmonares para PFS valido

### Media Prioridad
4. [ ] Evaluar en datasets externos (Montgomery, Shenzhen)
5. [ ] Documentar trade-offs de normalizacion geometrica

---

## 7. Referencias de Verificacion

Todos los resultados fueron verificados en Sesion 35 con 5 agentes independientes:

- `/docs/sesiones/SESION_35_ANALISIS_CRITICO.md` - Analisis completo
- `/outputs/full_warped_dataset/dataset_summary.json` - Fill rate ~47%
- `/outputs/classifier_test_results.json` - Metricas de clasificacion
- `/outputs/robustness_test_results.json` - Tests JPEG/Blur

---

## Apendice: Bugs Corregidos en Sesion 36

| Bug | Estado | Commit |
|-----|--------|--------|
| Import warping en cli.py L1025 | CORREGIDO | 11fb902 |
| Import warping en cli.py L1684 | CORREGIDO | 11fb902 |
| Tests de validacion de imports | AGREGADOS | 11fb902 |
