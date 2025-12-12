# SESION 47 - INTROSPECCION FINAL Y HALLAZGOS CRITICOS

**Fecha:** 2025-12-11
**Rama:** feature/restructure-production
**Metodologia:** Analisis con 5 agentes paralelos + verificacion manual

---

## RESUMEN EJECUTIVO

Se realizo una introspeccion exhaustiva del proyecto usando 5 agentes especializados en paralelo:
1. Scripts de visualizacion
2. Documentacion
3. Codigo fuente (bugs)
4. Tests (valores hardcodeados)
5. Consistencia con GROUND_TRUTH.json

**Resultado:** Se identificaron **23 problemas** organizados por prioridad.

---

## HALLAZGOS CRITICOS (PRIORIDAD 1)

### 1.1 GROUND_TRUTH.json tiene valores incorrectos de per_category_landmarks

**Archivo:** `GROUND_TRUTH.json` lineas 152-157

**Problema:** Los valores mezclan datos de Sesion 12 (2 modelos) con Sesion 13 (4 modelos)

| Categoria | GROUND_TRUTH actual | Valor correcto (S13) | Diferencia |
|-----------|--------------------|--------------------|------------|
| COVID | 3.83 | **3.77** | -0.06 px |
| Normal | 3.53 | **3.42** | -0.11 px |
| Viral_Pneumonia | 4.42 | **4.40** | -0.02 px |

**Fuente correcta:** `docs/sesiones/SESION_13_ENSEMBLE_4_MODELOS.md` lineas 73-75

---

### 1.2 Valores por landmark DIFERENTES entre archivos de visualizacion

**Archivos afectados:**
- `scripts/visualization/generate_bloque6_resultados.py` lineas 88-92
- `scripts/visualization/generate_results_figures.py` lineas 117-118

**Ejemplo de discrepancia:**

| Landmark | bloque6 | results_figures | Diferencia |
|----------|---------|-----------------|-----------|
| L1 | 3.29 | 3.1 | 0.19 |
| L2 | 4.34 | 4.1 | 0.24 |
| L12 | 5.63 | 5.3 | 0.33 |

**Problema:** Ninguno de los dos archivos cita fuente verificable.

---

### 1.3 Valores INVENTADOS en tablas de visualizacion

**Archivo:** `scripts/visualization/generate_bloque5_ensemble_tta.py` lineas 453-456

```python
['Baseline (ResNet-18)', '9.08 px', '8.15 px', '28.4 px', '-'],
['Ensemble + TTA', '3.71 px', '3.18 px', '13.5 px', '59%'],
```

**Valores sin fuente verificable:**
- Mediana: 8.15, 3.45, 3.31, 3.18 px (NO en GROUND_TRUTH.json)
- Maximo: 28.4, 15.2, 14.1, 13.5 px (NO en GROUND_TRUTH.json)

---

### 1.4 Distribucion sintetica fabricada

**Archivo:** `scripts/visualization/generate_bloque6_resultados.py` lineas 446-451

```python
errors = np.concatenate([
    np.random.normal(2.9, 0.6, 300),   # Centrales (~2.9 px)
    np.random.normal(3.4, 0.8, 480),   # Pulmonares (~3.4 px)
    np.random.normal(5.0, 1.2, 540),   # Costofrenicos (~5.0 px)
])
```

**Problema:** Genera datos aleatorios que PARECEN reales pero son fabricados.

---

## HALLAZGOS ALTOS (PRIORIDAD 2)

### 2.1 Accuracy 98.81% vs 98.84%

**Problema:** Scripts de session30 usan 98.81% pero GROUND_TRUTH.json dice 98.84%

**Archivos afectados:**
- `scripts/session30_cross_evaluation.py` linea 8
- `scripts/session30_robustness_figure.py` linea 287
- `scripts/session30_error_analysis.py` lineas 41, 403, 511, 627

**Valor correcto:** 98.84% (GROUND_TRUTH.json linea 37)

---

### 2.2 Factor robustez 30.5 vs 30.45

**Archivos con valor incorrecto:**
- `scripts/create_thesis_figures.py` linea 238: `30.5`
- `docs/sesiones/SESION_34_VISUAL_GALLERY.md` linea 61: `30.5`
- `docs/PROMPT_SESION_35_COMPLETO.md` linea 22: `30.62x`

**Valor correcto:** 30.45 (16.14 / 0.53 = 30.4528)

---

### 2.3 Import no usado en resnet_landmark.py

**Archivo:** `src_v2/models/resnet_landmark.py` linea 9
```python
import torch.nn.functional as F  # NUNCA SE USA
```

---

### 2.4 Codigo duplicado: BILATERAL_PAIRS

**Problema:** `BILATERAL_PAIRS` definido en hierarchical.py es duplicado de `SYMMETRIC_PAIRS` en constants.py

**Archivo:** `src_v2/models/hierarchical.py` lineas 55-61

**Solucion:** Usar `SYMMETRIC_PAIRS` de constants.py

---

### 2.5 Magic number 3 en losses.py

**Archivo:** `src_v2/models/losses.py` linea 221
```python
loss = total_dist / 3  # Deberia ser len(CENTRAL_LANDMARKS)
```

---

### 2.6 Tests con tolerancias irreales

**Archivo:** `tests/test_evaluation_metrics.py`
- Linea 132: `abs=1e-5` (deberia ser 0.5 segun GROUND_TRUTH.json)
- Linea 213: `abs=1e-6` (irreal para imagenes medicas)

---

### 2.7 Tests que siempre se skipean

**Archivo:** `tests/test_robustness_comparative.py`

9 tests usan `pytest.skip()` si no existen archivos de datos:
- Lineas: 52, 69, 93, 108, 123, 138, 152, 167, 191, 203, 227

**Problema:** CI puede reportar "todos pasaron" cuando se omitieron 10+ tests.

---

### 2.8 Umbral de robustez muy bajo

**Archivo:** `tests/test_robustness_comparative.py` linea 83
```python
assert ratio >= 5  # Deberia ser >= 20 segun GROUND_TRUTH (30.45x)
```

---

## HALLAZGOS MEDIOS (PRIORIDAD 3)

### 3.1 Vector perpendicular duplicado en data/utils.py

**Archivo:** `src_v2/data/utils.py` linea 267

Implementa su propia version en lugar de usar `compute_perpendicular_vector()` de geometry.py.

---

### 3.2 Indices hardcodeados sin constantes

**Archivo:** `src_v2/models/hierarchical.py` linea 207
```python
for i, (landmark_idx, t_base) in enumerate([(8, 0.25), (9, 0.50), (10, 0.75)]):
```

**Deberia usar:** `CENTRAL_LANDMARKS_T` constante

---

### 3.3 Learning rates hardcodeados

**Archivo:** `src_v2/models/hierarchical.py` linea 250
```python
def get_trainable_params(self, backbone_lr: float = 2e-5, head_lr: float = 2e-4):
```

**Deberia usar:** `DEFAULT_PHASE2_BACKBONE_LR` y `DEFAULT_PHASE2_HEAD_LR` de constants.py

---

### 3.4 Test CLI acepta exit_code 1 como exito

**Archivo:** `tests/test_cli_integration.py` linea 61
```python
assert result.exit_code in [0, 1]  # Esconde fallos de entrenamiento
```

---

### 3.5 Valores hardcodeados en tests sin importar de GROUND_TRUTH

**Archivo:** `tests/test_robustness_comparative.py` lineas 241-243
```python
orig_deg = original.get("jpeg", {}).get("degradation", 16.14)  # Hardcodeado
crop_deg = cropped.get("jpeg_q50", {}).get("degradation", 2.11)  # Hardcodeado
warp_deg = warped.get("jpeg", {}).get("degradation", 0.53)  # Hardcodeado
```

---

## HALLAZGOS BAJOS (PRIORIDAD 4)

### 4.1 Porcentaje de Viral redondeado incorrectamente

**Archivo:** `scripts/visualization/generate_bloque6_resultados.py` linea 124
- Muestra: 51%
- Correcto: 50.5% ((8.93-4.42)/8.93*100)

---

### 4.2 Tests triviales de --help

**Archivo:** `tests/test_cli.py` lineas 27, 37, 45, 53, 61, 69

Tests de `--help` siempre pasan, no prueban funcionalidad real.

---

### 4.3 Funciones sin tests directos en warp.py

**Archivo:** `src_v2/processing/warp.py`

Funciones sin unit test:
- `_triangle_area_2x()` (linea 24)
- `scale_landmarks_from_centroid()` (linea 41)
- `get_affine_transform_matrix()` (linea 117)
- `warp_triangle()` (linea 178)

---

## RESUMEN POR ARCHIVO

| Archivo | Problemas | Severidad |
|---------|-----------|-----------|
| GROUND_TRUTH.json | per_category_landmarks incorrectos | CRITICA |
| generate_bloque5_ensemble_tta.py | Valores inventados (mediana, max) | CRITICA |
| generate_bloque6_resultados.py | Landmark errors inconsistentes, distribucion fabricada | CRITICA |
| generate_results_figures.py | Landmark errors diferentes | CRITICA |
| test_robustness_comparative.py | Tolerancias bajas, skips excesivos | ALTA |
| test_evaluation_metrics.py | Tolerancias irreales | ALTA |
| session30_*.py | 98.81% vs 98.84% | ALTA |
| hierarchical.py | Codigo duplicado, magic numbers | MEDIA |
| resnet_landmark.py | Import no usado | MEDIA |
| losses.py | Magic number 3 | MEDIA |

---

## METRICAS FINALES

- **Tests totales:** 613 pasando, 6 skipped
- **Problemas identificados:** 23
- **Criticos:** 4
- **Altos:** 8
- **Medios:** 6
- **Bajos:** 5

---

## SIGUIENTE PASO

Ver `Prompt para Sesion 48.txt` para las correcciones a implementar.
