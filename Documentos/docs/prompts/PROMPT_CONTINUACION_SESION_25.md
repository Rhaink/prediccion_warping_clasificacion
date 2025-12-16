# Prompt de Continuación - Sesión 25

**Fecha:** 2025-12-09
**Sesión anterior:** 24 (Análisis PFS - Pospuesto)
**Objetivo:** Completar CLI al ~95% de cobertura

---

## PROMPT PARA COPIAR Y PEGAR

```
Lee este prompt y comienza a trabajar en la Sesión 25. Utiliza ultrathink y múltiples agentes cuando sea necesario.

## Contexto del Proyecto

Este es un proyecto de tesis sobre clasificación de COVID-19 en radiografías de tórax usando normalización geométrica (warping). El CLI en `src_v2/` permite reproducir los experimentos.

## Estado Actual (Fin Sesión 24)

- **Comandos CLI:** 19 implementados
- **Tests:** 387 pasando
- **Cobertura:** ~85% de experimentos reproducibles
- **Rama:** feature/restructure-production

## Objetivo de Esta Sesión

Implementar el comando `optimize-margin` - el último comando pendiente para alcanzar ~95% de cobertura.

### Comando a Implementar: `optimize-margin`

**Propósito:** Buscar automáticamente el margen óptimo para warping.

**Uso esperado:**
```bash
python -m src_v2 optimize-margin \
    --data-dir data/COVID-19_Radiography_Dataset \
    --landmarks-csv data/landmarks.csv \
    --margins 1.00,1.05,1.10,1.15,1.20,1.25,1.30 \
    --epochs 10 \
    --output-dir outputs/margin_optimization
```

**Qué debe hacer:**
1. Para cada margen en la lista:
   - Generar dataset warped temporal con ese margen
   - Entrenar clasificador rápido (epochs reducidos)
   - Evaluar accuracy en validation/test
   - Guardar resultados parciales
2. Al finalizar:
   - Identificar margen con mejor accuracy
   - Generar gráfico accuracy vs margen
   - Guardar `margin_optimization_results.json`

**Referencia:** Basarse en `scripts/margin_optimization_experiment.py` y `scripts/experiment_extended_margins.py`

**Resultado esperado histórico:** Margen óptimo = 1.25

### Archivos Clave

- CLI principal: `src_v2/cli.py`
- Warping: `src_v2/processing/warp.py`
- GPA: `src_v2/processing/gpa.py`
- Clasificador: `src_v2/models/classifier.py`
- Tests: `tests/test_cli.py`

### Criterios de Éxito

1. Comando `optimize-margin` funcionando
2. Tests nuevos para el comando (~10-15 tests)
3. Todos los tests pasando (387+ tests)
4. Documentación en `docs/sesiones/SESION_25_OPTIMIZE_MARGIN.md`

### Notas Importantes

- El PFS fue pospuesto (ver `docs/TRABAJO_FUTURO_PFS.md`)
- Reutilizar lógica existente de `generate-dataset` y `train-classifier`
- Usar arquitectura simple (ResNet-18) para rapidez
- Considerar modo `--quick` para pruebas rápidas
```

---

## Contexto Adicional para el Agente

### Scripts de Referencia

1. **`scripts/margin_optimization_experiment.py`** - Experimento original de optimización
2. **`scripts/experiment_extended_margins.py`** - Experimento con rango extendido

### Comandos Existentes Relacionados

```bash
# Generar dataset warped (ya implementado)
python -m src_v2 generate-dataset --margin 1.25 ...

# Entrenar clasificador (ya implementado)
python -m src_v2 train-classifier --epochs 10 ...

# Evaluar clasificador (ya implementado)
python -m src_v2 evaluate-classifier ...
```

### Estructura de Salida Esperada

```
outputs/margin_optimization/
├── margin_optimization_results.json   # Resultados consolidados
├── accuracy_vs_margin.png             # Gráfico principal
├── per_margin/
│   ├── margin_1.00/
│   │   ├── dataset/                   # Dataset temporal
│   │   ├── checkpoint.pt              # Modelo entrenado
│   │   └── eval_results.json          # Resultados evaluación
│   ├── margin_1.05/
│   └── ...
└── summary.csv                        # Tabla resumen
```

### Parámetros Sugeridos

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--data-dir` | Path | Requerido | Dataset original |
| `--landmarks-csv` | Path | Requerido | Archivo de landmarks |
| `--margins` | str | "1.00,1.05,1.10,1.15,1.20,1.25,1.30" | Márgenes a probar |
| `--epochs` | int | 10 | Epochs por entrenamiento |
| `--batch-size` | int | 32 | Batch size |
| `--architecture` | str | resnet18 | Arquitectura a usar |
| `--output-dir` | Path | outputs/margin_optimization | Salida |
| `--quick` | bool | False | Modo rápido (menos epochs) |
| `--keep-datasets` | bool | False | Mantener datasets temporales |

---

## Historial de Sesiones Relevantes

- **Sesión 20:** Implementó `compute-canonical` y `generate-dataset`
- **Sesión 22:** Implementó `compare-architectures`
- **Sesión 23:** Implementó `gradcam` y `analyze-errors`
- **Sesión 24:** Implementó `pfs-analysis` (pospuesto para mejoras)

---

**Última actualización:** 2025-12-09
