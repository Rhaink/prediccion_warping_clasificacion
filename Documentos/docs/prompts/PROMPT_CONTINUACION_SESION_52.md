# Prompt de Continuación - Sesión 52
## Objetivo: Corregir clasificador para alcanzar 98.73% accuracy

**NOTA (2025-12-21):** `outputs/warped_replication_v2` y
`outputs/full_coverage_warped_dataset` fueron invalidados. Cualquier
clasificador derivado (por ejemplo `outputs/classifier_replication_v2` y
`outputs/classifier_warped_full_coverage`) no debe usarse.

---

## CONTEXTO DEL PROBLEMA

En la sesión anterior (51) se validó el pipeline completo de replicación:
- **Landmarks:** ✅ 3.71 px (EXACTO)
- **Clasificación:** ❌ 97.73% (esperado: 98.73%)

### Diferencia: -1.00% respecto al GROUND_TRUTH

---

## ROOT CAUSE IDENTIFICADO

El comando `generate-dataset` tiene un **bug/limitación**: el parámetro `use_full_coverage` está hardcodeado a `False` en `src_v2/cli.py` línea 3727.

```python
# src_v2/cli.py:3727 (PROBLEMA)
warped = piecewise_affine_warp(
    image=image,
    source_landmarks=scaled_landmarks,
    target_landmarks=canonical,
    triangles=tri,
    use_full_coverage=False  # ← HARDCODED A FALSE
)
```

### Efecto del parámetro:

| use_full_coverage | Fill Rate | Accuracy | Puntos Triangulación |
|-------------------|-----------|----------|---------------------|
| `False` (actual)  | ~47%      | 97.73%   | 15 landmarks        |
| `True` (correcto) | ~99%      | 98.73%   | 15 + 8 boundary = 23 |

---

## EVIDENCIA

### Dataset actual (warped_replication):
```json
// outputs/warped_replication/dataset_summary.json
{
  "fill_rate_mean": 0.4707657258347836,  // 47.08%
  "train": { "processed": 11364 }
}
```

### Dataset correcto existente (full_coverage_warped_dataset):
```json
// outputs/full_coverage_warped_dataset/dataset_summary.json
{
  "use_full_coverage": true,
  "fill_rate_mean": 0.9910898857868565,  // 99.11%
  "train": { "processed": 11364 }
}
```

### Clasificador correcto existente:
```json
// outputs/classifier_warped_full_coverage/results.json
{
  "test_metrics": {
    "accuracy": 0.987335092348285  // 98.73% ✓
  }
}
```

---

## OPCIONES DE SOLUCIÓN

### OPCIÓN A: Usar recursos existentes (RÁPIDA - ~5 min)

Ya existen el dataset y clasificador correctos:

```bash
# El dataset con 99% fill ya existe:
ls outputs/full_coverage_warped_dataset/

# El clasificador con 98.73% ya existe:
ls outputs/classifier_warped_full_coverage/best_classifier.pt

# Verificar accuracy:
python -m src_v2 evaluate-classifier \
    outputs/classifier_warped_full_coverage/best_classifier.pt \
    --data-dir outputs/full_coverage_warped_dataset \
    --split test
```

**Pros:** Inmediato, ya validado
**Contras:** No corrige el bug del CLI

---

### OPCIÓN B: Corregir el CLI y regenerar (COMPLETA - ~40 min)

1. **Agregar flag `--use-full-coverage` al CLI** (`src_v2/cli.py`):

```python
# Agregar en parámetros del comando generate-dataset (~línea 3402):
use_full_coverage: bool = typer.Option(
    True,  # Default True para coincidir con GROUND_TRUTH
    "--use-full-coverage/--no-full-coverage",
    help="Agregar puntos de borde para cobertura completa (fill_rate ~99%)"
),
```

2. **Pasar el parámetro a la función** (~línea 3727):

```python
warped = piecewise_affine_warp(
    image=image,
    source_landmarks=scaled_landmarks,
    target_landmarks=canonical,
    triangles=tri,
    use_full_coverage=use_full_coverage  # ← Usar parámetro
)
```

3. **Regenerar dataset:**

```bash
python -m src_v2 generate-dataset \
    data/dataset/COVID-19_Radiography_Dataset \
    outputs/warped_replication_v2 \
    --checkpoint checkpoints/final_model.pt \
    --use-full-coverage
```

4. **Reentrenar clasificador:**

```bash
python -m src_v2 train-classifier \
    outputs/warped_replication_v2 \
    --output-dir outputs/classifier_replication_v2 \
    --epochs 50
```

**Pros:** Corrige el bug permanentemente, documentación consistente
**Contras:** Requiere más tiempo

---

### OPCIÓN C: Entrenar con dataset existente (INTERMEDIA - ~25 min)

Usar el dataset `full_coverage_warped_dataset` existente para entrenar un nuevo clasificador:

```bash
python -m src_v2 train-classifier \
    outputs/full_coverage_warped_dataset \
    --output-dir outputs/classifier_replication_v2 \
    --epochs 50 \
    --batch-size 32
```

**Pros:** No requiere regenerar dataset
**Contras:** No corrige el bug del CLI

---

## ARCHIVOS RELEVANTES

| Archivo | Descripción | Líneas clave |
|---------|-------------|--------------|
| `src_v2/cli.py` | CLI principal | L3402-3468 (params), L3727 (bug) |
| `src_v2/processing/warp.py` | Función de warping | L241-301 (implementación correcta) |
| `GROUND_TRUTH.json` | Valores esperados | L125 (use_full_coverage: true) |
| `outputs/full_coverage_warped_dataset/` | Dataset correcto | fill_rate 99% |
| `outputs/classifier_warped_full_coverage/` | Clasificador correcto | 98.73% |

---

## CRITERIO DE ÉXITO

```
Accuracy del clasificador ≥ 98.73% (GROUND_TRUTH)

O dentro de tolerancia:
- Mínimo aceptable: 96.73% (98.73% - 2%)
- Objetivo: 98.73% exacto
```

---

## RECOMENDACIÓN

**Ejecutar OPCIÓN B** (corregir CLI) porque:
1. El bug afectará a usuarios futuros
2. La documentación asume `use_full_coverage=True` por defecto
3. GROUND_TRUTH.json especifica `use_full_coverage: true`
4. Mantiene consistencia con el sistema

Si el tiempo es crítico, **OPCIÓN A** valida inmediatamente que el pipeline puede alcanzar 98.73%.

---

## COMANDO DE INICIO SUGERIDO

```
Continúa la sesión 51. El clasificador obtuvo 97.73% pero necesitamos 98.73%.

El problema está identificado: src_v2/cli.py línea 3727 tiene use_full_coverage=False hardcodeado.

Ejecuta la Opción B:
1. Corrige el CLI agregando --use-full-coverage flag
2. Regenera el dataset con el flag activado
3. Reentrena el clasificador
4. Valida que alcance ≥98.73%

Archivos a modificar: src_v2/cli.py (líneas ~3402 y ~3727)
```

---

## NOTAS ADICIONALES

### Trade-off de robustez (Sesión 39)

| Fill Rate | Accuracy | Degradación JPEG Q50 |
|-----------|----------|---------------------|
| 47%       | 98.02%   | 0.53% (muy robusto) |
| 99%       | 98.73%   | 7.34% (menos robusto) |

El modelo con 99% fill es menos robusto a compresión JPEG. Esto es un trade-off documentado - más información = mayor accuracy en datos limpios pero menor robustez a perturbaciones.

### Referencia de sesiones anteriores
- Sesión 39: Experimento de control que identificó este trade-off
- Sesión 50: Introspección que consolidó estos valores
- Sesión 51: Auditoría de checkpoints + intento de replicación
