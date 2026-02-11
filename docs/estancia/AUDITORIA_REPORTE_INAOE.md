# Auditoría del Reporte de Estancia INAOE

**Fecha de auditoría:** 2026-01-27
**Documento auditado:** `docs/estancia/REPORTE_ESTANCIA_INAOE.tex`
**Plan de referencia:** `docs/Tesis/anexos/inaoe.pdf`
**Fuente de verdad:** `GROUND_TRUTH.json` v2.1.0

---

## 1. Datos Generales

| Campo | Plan Firmado | Reporte | Estado |
|-------|-------------|---------|--------|
| Nombre proyecto | Normalización y alineación automática... | Idéntico (línea 117) | ✅ |
| Director | Dr. Leopoldo Altamirano Robles | Idéntico (línea 121) | ✅ |
| Estudiante | Rafael Alejandro Cruz Ovando | Idéntico (línea 120) | ✅ |
| Fechas | 9 oct - 9 nov 2025 | Idéntico (línea 123) | ✅ |
| Lugar | Lab. Visión por Computadora, INAOE | Idéntico (línea 153) | ✅ |

**Resultado:** ✅ TODOS LOS DATOS GENERALES COINCIDEN

---

## 2. Objetivos del Plan vs Reporte

### 2.1 Objetivo General

| Plan | Reporte | Estado |
|------|---------|--------|
| Desarrollar prototipo de detección automática de landmarks pulmonares usando DL y modelado geométrico | Sección 1-2: Sistema completo landmarks + normalización geométrica | ✅ CUMPLIDO |

### 2.2 Objetivos Específicos

| # | Objetivo del Plan | Evidencia en Reporte | Estado |
|---|------------------|---------------------|--------|
| 1 | Recopilar y normalizar dataset con anotaciones de landmarks | Sec. 3.1: Dataset 21,165 imágenes, 15 landmarks (línea 247) | ✅ |
| 2 | Implementar ResNet-18 con funciones de pérdida geométricas | Sec. 3.2-3.3: ResNet-18 + Wing Loss + Symmetry + Alignment | ✅ |
| 3 | Plan de entrenamiento que evalúe contribución de componentes | Sec. 3.4: Entrenamiento 2 fases (líneas 436-496) | ✅ |
| 4 | Protocolo de evaluación cuantitativa/cualitativa | Sec. 4: Métricas por landmark, categoría, CV 5-fold | ✅ |
| 5 | Documentar flujo de trabajo y resultados | Sec. 5-7: Entregables + cronograma + documentación | ✅ |

**Resultado:** ✅ TODOS LOS OBJETIVOS ESPECÍFICOS CUMPLIDOS

---

## 3. Metodología

### 3.1 Preparación de Datos

| Requisito del Plan | Implementación Reportada | Estado |
|--------------------|-------------------------|--------|
| Diagnóstico inicial del dataset | Tabla línea 240: 21,165 imágenes, 3 clases | ✅ |
| Procedimientos de limpieza/normalización | CLAHE clip=2.0, tile=4 (línea 263-264) | ✅ |
| División train/val/test | 75%/15%/10% (línea 250) | ✅ |

### 3.2 Diseño del Modelo

| Requisito del Plan | Implementación Reportada | Estado |
|--------------------|-------------------------|--------|
| Arquitectura ResNet | ResNet-18 pretrained ImageNet (línea 302) | ✅ |
| Cabeza de regresión FC | Deep Head 768 dim (línea 304) | ✅ |
| Funciones de pérdida (Wing, Symmetry, Distance) | Wing Loss + Soft Symmetry + Central Alignment (líneas 355-434) | ✅ |

### 3.3 Plan de Entrenamiento

| Fase del Plan | Implementación Reportada | Estado |
|---------------|-------------------------|--------|
| Fase 1: Cabeza con backbone congelado | 15 épocas, lr=1e-3 (línea 451) | ✅ |
| Fase 2: Descongelar gradualmente | 100 épocas, LR diferenciado 2e-5/2e-4 (líneas 474-476) | ✅ |
| Fase 3: Pérdidas geométricas incrementales | Combined Loss (línea 421) | ✅ |
| Fase 4: Ajuste de hiperparámetros | Sweep de semillas, ensemble (líneas 498-517) | ✅ |

### 3.4 Evaluación y Análisis

| Requisito del Plan | Implementación Reportada | Estado |
|--------------------|-------------------------|--------|
| Error medio por landmark en píxeles | 3.61 px ensemble (línea 665) | ✅ |
| Distribución por categorías clínicas | Normal/COVID/Viral_Pneumonia (líneas 679-681) | ✅ |
| Validación cruzada | 5-fold CV (líneas 710-724) | ✅ |
| Visualizaciones (superposición, mapas) | Figuras F4.x, F5.x (líneas 982-998) | ✅ |

**Resultado:** ✅ METODOLOGÍA IMPLEMENTADA SEGÚN PLAN

---

## 4. Verificación de Métricas vs GROUND_TRUTH.json

### 4.1 Métricas de Landmarks

| Métrica | Valor en Reporte | Valor en GROUND_TRUTH | Estado |
|---------|-----------------|----------------------|--------|
| Error ensemble (TTA) | **3.61 px** (línea 665) | 3.61 px (`ensemble_4_models_tta_best_20260111`) | ✅ CORRECTO |
| Std ensemble | 2.48 px (línea 665) | 2.48 px | ✅ CORRECTO |
| Mediana ensemble | 3.07 px (línea 665) | 3.07 px | ✅ CORRECTO |
| Error mejor individual | **4.04 px** (línea 663) | 4.04 px (`best_individual_tta`) | ✅ CORRECTO |

### 4.2 Error por Categoría

| Categoría | Reporte (líneas 679-681) | GROUND_TRUTH (`ensemble_4_tta_best_20260111`) | Estado |
|-----------|---------|--------------|--------|
| Normal | 3.22 px | 3.22 px | ✅ |
| COVID | 3.93 px | 3.93 px | ✅ |
| Viral_Pneumonia | 4.11 px | 4.11 px | ✅ |

### 4.3 Error por Landmark Individual

| Landmark | Reporte (líneas 695-700) | GROUND_TRUTH (`values_best_20260111`) | Estado |
|----------|---------|--------------|--------|
| L10 | 2.44 px | 2.44 px | ✅ |
| L9 | 2.76 px | 2.76 px | ✅ |
| L5 | 2.88 px | 2.88 px | ✅ |
| L12 | 5.43 px | 5.43 px | ✅ |
| L13 | 5.35 px | 5.35 px | ✅ |
| L14 | 4.39 px | 4.39 px | ✅ |

### 4.4 Métricas de Clasificación

| Métrica | Reporte (líneas 718-720) | GROUND_TRUTH (`cross_validation`) | Estado |
|---------|-----------------|---------------------------|--------|
| Accuracy CV | **98.60%** | 98.5971% | ✅ (redondeo correcto) |
| F1-macro CV | **98.00%** | 98.0035% | ✅ (redondeo correcto) |
| F1-weighted CV | **98.60%** | 98.5983% | ✅ (redondeo correcto) |
| Std accuracy | ±0.26% | 0.2551% | ✅ |
| Std F1-macro | ±0.36% | 0.3613% | ✅ |

### 4.5 Ensemble Clasificadores + TTA

| Métrica | Reporte (líneas 742-744) | GROUND_TRUTH (`classifier_ensemble_cv.with_tta`) | Estado |
|---------|-----------------|---------------------------|--------|
| Accuracy sin TTA | 98.10% | 98.10% (0.9810) | ✅ |
| Accuracy con TTA | **98.26%** | 98.26% (0.9826) | ✅ |
| F1-macro con TTA | 97.12% | 97.12% (0.9712) | ✅ |
| Casos mejorados TTA | 6 | 6 | ✅ |
| Casos empeorados TTA | 3 | 3 | ✅ |
| Mejora neta | +3 | +3 | ✅ |

**Resultado:** ✅ TODAS LAS MÉTRICAS COINCIDEN CON GROUND_TRUTH.json

---

## 5. Verificación de Entregables Físicos

### 5.1 Código Fuente (src_v2/)

| Módulo | Reporte (líneas 794-812) | Existe | Estado |
|--------|-------------------------|--------|--------|
| `models/resnet_landmark.py` | ✓ | ✓ | ✅ |
| `models/losses.py` | ✓ | ✓ | ✅ |
| `models/classifier.py` | ✓ | ✓ | ✅ |
| `training/trainer.py` | ✓ | ✓ | ✅ |
| `training/callbacks.py` | ✓ | ✓ | ✅ |
| `data/dataset.py` | ✓ | ✓ | ✅ |
| `data/transforms.py` | ✓ | ✓ | ✅ |
| `processing/gpa.py` | ✓ | ✓ | ✅ |
| `processing/warp.py` | ✓ | ✓ | ✅ |
| `evaluation/metrics.py` | ✓ | ✓ | ✅ |
| `evaluation/ensemble.py` | ✓ | ✓ | ✅ |
| `cli.py` | ✓ | ✓ | ✅ |

**Total módulos verificados:** 12/12 ✅

### 5.2 Checkpoints del Ensemble

| Checkpoint | Reporte (líneas 825-828) | Existe | Tamaño |
|------------|-------------------------|--------|--------|
| `session10/ensemble/seed123/final_model.pt` | ✓ | ✓ | 47.6 MB |
| `session13/seed321/final_model.pt` | ✓ | ✓ | 47.6 MB |
| `repro_split111/session14/seed111/final_model.pt` | ✓ | ✓ | 47.6 MB |
| `repro_split666/session16/seed666/final_model.pt` | ✓ | ✓ | 47.6 MB |

**Total checkpoints verificados:** 4/4 ✅

### 5.3 Configuraciones JSON

| Config | Reporte (líneas 837-841) | Existe | Estado |
|--------|-------------------------|--------|--------|
| `configs/ensemble_best.json` | ✓ | ✓ | ✅ |
| `configs/landmarks_train_base.json` | ✓ | ✓ | ✅ |
| `configs/warping_best.json` | ✓ | ✓ | ✅ |
| `configs/classifier_warped_base.json` | ✓ | ✓ | ✅ |

**Total configs verificados:** 4/4 ✅

### 5.4 Documentación

| Documento | Reporte (líneas 852-857) | Existe | Estado |
|-----------|-------------------------|--------|--------|
| `CLAUDE.md` | ✓ | ✓ | ✅ |
| `GROUND_TRUTH.json` | ✓ | ✓ | ✅ |
| `docs/manual/README.md` | (referenciado) | ✓ | ✅ |

**Resultado:** ✅ TODOS LOS ENTREGABLES EXISTEN FÍSICAMENTE

---

## 6. Cronograma

| Semana | Plan Original | Reporte (líneas 875-878) | Estado |
|--------|--------------|-------------------------|--------|
| 1 (9-15 Oct) | Revisión estado arte, análisis dataset | ✓ Análisis 21,165 imágenes | ✅ |
| 2 (16-22 Oct) | Arquitectura base, Fase 1 | ✓ ResNet-18 + CoordAttention | ✅ |
| 3 (23-29 Oct) | Fine-tuning, Fases 2-3 | ✓ LR diferenciado, pérdidas geométricas | ✅ |
| 4 (30 Oct-9 Nov) | Experimentos finales, documentación | ✓ Ensemble, TTA, clasificación | ✅ |

**Resultado:** ✅ CRONOGRAMA EJECUTADO SEGÚN PLAN

---

## 7. Resumen de Auditoría

### ✅ Elementos Correctos

1. **Datos generales:** Coinciden exactamente con plan firmado
2. **Objetivos:** 5/5 objetivos específicos cumplidos
3. **Metodología:** Implementada según plan con extensiones adicionales
4. **Métricas landmarks:** Todas coinciden con GROUND_TRUTH.json
5. **Métricas clasificación:** Todas coinciden con GROUND_TRUTH.json
6. **Entregables:** 100% existen físicamente (código, checkpoints, configs, docs)
7. **Cronograma:** Ejecutado según plan original

### ⚠️ Nota sobre CLAUDE.md (No afecta al reporte)

El archivo `CLAUDE.md` menciona "99.10% (warped_96)" que está marcado como obsoleto en GROUND_TRUTH.json. Sin embargo, **el reporte de estancia usa correctamente 98.60% CV** como métrica principal de clasificación, que coincide con GROUND_TRUTH.json.

### Extensiones Más Allá del Plan Original

El reporte documenta trabajo adicional no contemplado en el plan original:

1. **Ensemble de 4 modelos:** Mejora error de 4.04 → 3.61 px (-10.6%)
2. **Test-Time Augmentation (TTA):** Mejora adicional en landmarks y clasificación
3. **Sistema de configuración JSON:** Reproducibilidad mejorada
4. **Validación cruzada 5-fold:** Evaluación más robusta que holdout simple
5. **Pipeline completo de clasificación:** Incluyendo warping afín por partes

---

## 8. Conclusión Final

### ✅ EL REPORTE CUMPLE CON TODOS LOS REQUISITOS DEL PLAN FIRMADO

| Aspecto | Verificación | Resultado |
|---------|-------------|-----------|
| Datos generales | Coinciden exactamente | ✅ |
| Objetivo general | Cumplido | ✅ |
| Objetivos específicos | 5/5 cumplidos | ✅ |
| Metodología | Implementada según plan | ✅ |
| Métricas reportadas | Coinciden con GROUND_TRUTH.json | ✅ |
| Entregables físicos | 100% existen | ✅ |
| Cronograma | Ejecutado según plan | ✅ |

### No se requieren correcciones al reporte

El reporte `REPORTE_ESTANCIA_INAOE.tex` está completo, correcto y listo para entrega.

---

**Auditoría realizada por:** Claude Code
**Fecha:** 2026-01-27
