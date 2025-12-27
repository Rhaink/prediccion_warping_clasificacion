# Prompt de Continuación - Sesión 53

## Objetivo: Investigar discrepancia de fill rate y documentar trade-off

**NOTA (2025-12-21):** `outputs/full_coverage_warped_dataset` y
`outputs/warped_replication_v2` fueron invalidados. Los resultados y comandos
asociados en este prompt quedan solo como registro histórico.

---

## CONTEXTO

En la sesión 52 se corrigió el bug del CLI (`use_full_coverage` hardcodeado a `False`) y se logró:
- **Accuracy:** 99.10% (supera GROUND_TRUTH de 98.73%)
- **Robustez:** 2-5x mejor que el clasificador existente

Sin embargo, se identificó una **discrepancia importante**:

| Dataset | Fill Rate | Accuracy | JPEG Q50 Degradación |
|---------|-----------|----------|---------------------|
| Existente (`full_coverage_warped_dataset`) | **99.11%** | 98.73% | 7.34% |
| Nuevo (`warped_replication_v2`) | **96.15%** | 99.10% | 3.06% |

**Paradoja:** El dataset nuevo tiene MENOR fill rate pero MEJOR accuracy y robustez.

---

## TRABAJO FUTURO A ABORDAR

### 1. Investigar causa de diferencia en fill rate

**Hipótesis a verificar:**
- El dataset existente usó landmarks de ground truth (anotados manualmente)
- El dataset nuevo usa landmarks predichos por el modelo
- Los landmarks predichos tienen más error → menor fill rate

**Acciones:**
```bash
# Verificar cómo se generó el dataset existente
cat outputs/full_coverage_warped_dataset/dataset_summary.json

# Comparar landmarks entre datasets
# (si hay metadata de landmarks disponible)
```

### 2. Evaluar si 96% fill rate es óptimo

**Referencia - Sesión 39 (Experimento de Control):**

| Fill Rate | Accuracy | JPEG Q50 Degradación |
|-----------|----------|---------------------|
| 47% | 98.02% | 0.53% (muy robusto) |
| 99% | 98.73% | 7.34% (menos robusto) |
| **96%** | **99.10%** | **3.06%** |

**Hipótesis:** 96% podría ser un punto óptimo entre accuracy y robustez.

**Acciones:**
- Graficar la curva fill_rate vs accuracy vs robustez
- Determinar si hay un "sweet spot"

### 3. Documentar trade-off en GROUND_TRUTH.json

Actualizar GROUND_TRUTH.json con:
- Nuevo dataset y clasificador como alternativa válida
- Documentar el trade-off fill_rate/robustez
- Agregar métricas del clasificador nuevo

---

## ARCHIVOS RELEVANTES

| Archivo | Descripción |
|---------|-------------|
| `outputs/warped_replication_v2/` | Dataset nuevo (96% fill) |
| `outputs/classifier_replication_v2/` | Clasificador nuevo (99.10%) |
| `outputs/full_coverage_warped_dataset/` | Dataset existente (99% fill) |
| `outputs/classifier_warped_full_coverage/` | Clasificador existente (98.73%) |
| `GROUND_TRUTH.json` | Fuente de verdad a actualizar |
| `docs/sesiones/SESION_52_CLI_FIX_FULL_COVERAGE.md` | Documentación sesión anterior |

---

## DATOS VERIFICADOS (Sesión 52)

### Clasificador Nuevo (warped_replication_v2)
```json
{
  "accuracy": 0.9910,
  "f1_macro": 0.9845,
  "robustness": {
    "jpeg_q50_degradation": 3.06,
    "jpeg_q30_degradation": 5.28,
    "blur_sigma1_degradation": 2.43
  }
}
```

### Clasificador Existente (full_coverage)
```json
{
  "accuracy": 0.9873,
  "f1_macro": 0.9795,
  "robustness": {
    "jpeg_q50_degradation": 7.34,
    "jpeg_q30_degradation": 16.73,
    "blur_sigma1_degradation": 11.35
  }
}
```

---

## COMANDO DE INICIO SUGERIDO

```
Continúa desde la sesión 52. Identificamos una discrepancia interesante:

El dataset nuevo (96% fill rate) produce un clasificador con mejor accuracy (99.10% vs 98.73%)
y mejor robustez (2-5x) que el dataset existente (99% fill rate).

Tareas:
1. Investigar por qué el dataset existente tiene 99% fill rate vs 96% del nuevo
2. Analizar si 96% fill rate es un punto óptimo (trade-off accuracy/robustez)
3. Documentar hallazgos en GROUND_TRUTH.json

Archivos clave:
- outputs/full_coverage_warped_dataset/dataset_summary.json
- outputs/warped_replication_v2/dataset_summary.json
- GROUND_TRUTH.json
```

---

## CRITERIOS DE ÉXITO

1. **Causa identificada:** Explicar por qué fill rates difieren
2. **Trade-off documentado:** Actualizar GROUND_TRUTH.json con ambas opciones
3. **Recomendación clara:** Cuál dataset/clasificador usar según el caso de uso

---

## NOTAS ADICIONALES

### Estado del Proyecto
- **Auditoría:** Completada, APROBADO PARA DEFENSA
- **Tests:** 627 passed
- **Sesión 52:** Bug corregido, clasificador entrenado y verificado

### Commits de Sesión 52
```
3d5bf0e docs(session-52): documentar correccion CLI y resultados verificados
17309e3 docs: actualizar documentación con flag --use-full-coverage
1f756d5 fix(cli): agregar flag --use-full-coverage a generate-dataset
```

### Referencia de Trade-off (Sesión 39)
El hallazgo clave de la Sesión 39 fue:
- **75% de la robustez** viene de reducción de información (menor fill rate)
- **25% adicional** viene de normalización geométrica

Esto explica por qué 96% fill rate podría ser mejor que 99%:
menos información = más robustez, sin sacrificar mucha accuracy.
