# Sesion 11: Comando evaluate-ensemble y Reproducibilidad

## Fecha: 2025-12-07

## Objetivo
Implementar el comando `evaluate-ensemble` en el CLI para poder reproducir el resultado de **3.71 px** con un ensemble de 4 modelos, y documentar el proceso de reproducibilidad.

## Contexto
El CLI tenía 5 comandos (`train`, `evaluate`, `predict`, `warp`, `version`) pero no podía evaluar ensembles de múltiples modelos. El mejor resultado documentado (3.71 px) requería evaluar 4 modelos simultáneamente con TTA.

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `src_v2/cli.py` | Agregado comando `evaluate-ensemble` (~340 líneas) |
| `tests/test_cli.py` | Agregados 6 tests para evaluate-ensemble |
| `README.md` | Corregido 4.50→3.71 px, 2→4 modelos |
| `configs/final_config.json` | Agregada sección `ensemble_4models` |

## Archivos Creados

| Archivo | Descripción |
|---------|-------------|
| `REPRODUCIBILITY.md` | Guía paso a paso para reproducir 3.71 px |

## Comando evaluate-ensemble

### Implementación
```python
@app.command("evaluate-ensemble")
def evaluate_ensemble(
    checkpoints: list[str],  # Múltiples checkpoints
    --tta/--no-tta,          # TTA habilitado por defecto
    --clahe/--no-clahe,      # CLAHE habilitado por defecto
    --split,                 # test, val, train, all
    --output,                # Guardar JSON
    ...
)
```

### Características
- Acepta 2+ checkpoints como argumentos
- Auto-detecta arquitectura de cada modelo
- Aplica TTA a cada modelo (opcional, default=True)
- Promedia predicciones con pesos iguales
- Reporta métricas por landmark y categoría
- Soporta CLAHE (default=True)

### Uso
```bash
# Reproducir 3.71 px con 4 modelos
python -m src_v2 evaluate-ensemble \
    checkpoints/session10/ensemble/seed123/final_model.pt \
    checkpoints/session10/ensemble/seed456/final_model.pt \
    checkpoints/session13/seed321/final_model.pt \
    checkpoints/session13/seed789/final_model.pt \
    --tta --clahe --split test
```

## Tests Agregados

### test_cli.py - Clase TestEvaluateEnsemble (6 tests)

| Test | Descripción |
|------|-------------|
| `test_evaluate_ensemble_help` | Verifica --help del comando |
| `test_evaluate_ensemble_requires_checkpoints` | Sin argumentos debe fallar |
| `test_evaluate_ensemble_minimum_two_checkpoints` | Requiere mínimo 2 checkpoints |
| `test_evaluate_ensemble_missing_checkpoint` | Checkpoint inexistente debe fallar |
| `test_evaluate_ensemble_tta_default_enabled` | TTA habilitado por defecto |
| `test_evaluate_ensemble_module_execution` | python -m src_v2 evaluate-ensemble --help |

## Resumen de Tests

| Archivo | Tests Sesión 10 | Tests Agregados | Total |
|---------|-----------------|-----------------|-------|
| test_cli.py | 22 | +6 | 28 |
| test_config.py | 36 | 0 | 36 |
| test_constants.py | 43 | 0 | 43 |
| test_losses.py | 30 | 0 | 30 |
| test_pipeline.py | 26 | 0 | 26 |
| test_transforms.py | 25 | 0 | 25 |
| **TOTAL** | **182** | **+6** | **188** |

## Verificaciones Ejecutadas

### Resultados de Ensemble

| Configuración | Error | Esperado | Estado |
|---------------|-------|----------|--------|
| 4 modelos + TTA | 3.71 px | 3.71 px | ✅ |
| 2 modelos + TTA | 3.79 px | 3.79 px | ✅ |
| 2 modelos sin TTA | 4.07 px | >3.79 px | ✅ |

### Manejo de Errores

| Caso | Comportamiento | Estado |
|------|----------------|--------|
| 1 checkpoint | Exit code 1, mensaje error | ✅ |
| Checkpoint inexistente | Exit code 1, mensaje error | ✅ |
| Sin argumentos | Exit code 2 | ✅ |

### Todos los Comandos CLI Probados

| Comando | Funciona | Prueba |
|---------|----------|--------|
| `train` | ✅ | Modelo existente en checkpoints_v2_correct/ |
| `evaluate` | ✅ | 4.04 px (original), 3.98 px (CLI) |
| `predict` | ✅ | Genera 15 landmarks |
| `warp` | ✅ | Genera imágenes warpeadas |
| `version` | ✅ | Muestra versión |
| `evaluate-ensemble` | ✅ | Reproduce 3.71 px |

## Documentación Actualizada

### README.md
- Línea 17: `4.50 px` → `3.71 px`
- Línea 39: `2 models` → `4 models`
- Agregado ejemplo de `evaluate-ensemble`

### configs/final_config.json
```json
"ensemble_4models": {
  "description": "Ensemble optimo: 4 modelos (Sesion 13) - Mejor resultado",
  "models": [
    {"seed": 123, "checkpoint": "checkpoints/session10/ensemble/seed123/final_model.pt"},
    {"seed": 456, "checkpoint": "checkpoints/session10/ensemble/seed456/final_model.pt"},
    {"seed": 321, "checkpoint": "checkpoints/session13/seed321/final_model.pt"},
    {"seed": 789, "checkpoint": "checkpoints/session13/seed789/final_model.pt"}
  ],
  "ensemble_error_px": 3.71
}
```

### REPRODUCIBILITY.md (nuevo)
- Prerequisitos
- Checkpoints requeridos
- Comando exacto para 3.71 px
- Comparación de configuraciones
- Troubleshooting

## Arquitectura de Modelos Verificada

Todos los modelos (originales y CLI) usan la misma arquitectura:
- `use_coord_attention`: True
- `deep_head`: True
- `hidden_dim`: 768

## Estado Final

- [x] Comando `evaluate-ensemble` implementado
- [x] 6 tests agregados (188 total)
- [x] README.md corregido
- [x] REPRODUCIBILITY.md creado
- [x] configs/final_config.json actualizado
- [x] Resultado 3.71 px verificado
- [x] Todos los comandos CLI probados
- [x] Documento de sesión creado

## Pendiente para Próximas Sesiones

### Próxima Sesión Inmediata
1. **Integración con clasificador COVID-19** - Usar imágenes warpeadas para clasificación
   - Warping ya funciona con comando CLI
   - Clasificador existente en `scripts/train_classifier.py`
   - Evaluar impacto de normalización geométrica en clasificación

### Sesión Futura (cuando usuario se vaya a dormir)
2. **Entrenar 4 modelos con CLI** - Verificar pipeline end-to-end
   ```bash
   for seed in 123 456 321 789; do
       python -m src_v2 train --seed $seed --checkpoint-dir checkpoints_cli/seed${seed}
   done
   ```
   - Tiempo estimado: ~4 horas (1 hora por modelo)
   - Ejecutar en background mientras usuario duerme

### Completado en Esta Sesión
- ~~Actualizar docs/sesiones/~~ → Sesiones 12 y 13 documentadas
- ~~Documentar sesiones 12-13~~ → Completado

## Notas

- Los checkpoints originales (session10, session13) fueron entrenados con scripts legacy
- El modelo CLI (`checkpoints_v2_correct/`) da 3.98 px individual, comparable a originales (~4.04 px)
- El CLI reproduce exactamente el resultado documentado de 3.71 px
- La arquitectura se auto-detecta correctamente de cualquier checkpoint
