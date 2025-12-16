# Sesion 12: Optimización de Ensemble

## Fecha: 2024-11-27

## Objetivo
Analizar el impacto de diferentes combinaciones de modelos en el ensemble y optimizar la selección para minimizar el error de predicción de landmarks.

## Contexto
El ensemble original de 3 modelos (seeds 42, 123, 456) obtenía 4.50 px. Se investigó si excluir el modelo seed=42 (que tenía peor rendimiento individual) mejoraría el resultado.

## Análisis Realizado

### Rendimiento Individual de Modelos (con TTA)

| Seed | Error Individual | Checkpoint |
|------|------------------|------------|
| 42 | 6.75 px | `checkpoints/session10/exp4_epochs100/final_model.pt` |
| 123 | 4.05 px | `checkpoints/session10/ensemble/seed123/final_model.pt` |
| 456 | 4.04 px | `checkpoints/session10/ensemble/seed456/final_model.pt` |

### Comparación de Ensembles

| Configuración | Error | Diferencia |
|---------------|-------|------------|
| 3 modelos (42, 123, 456) | 4.50 px | baseline |
| 2 modelos (123, 456) | **3.79 px** | -0.71 px (-16%) |

## Hallazgo Principal

**El modelo seed=42 degrada el ensemble.** Al excluirlo:
- El error baja de 4.50 px a 3.79 px
- Mejora del 16% en precisión
- Menos modelos = mejor resultado

### Análisis de Por Qué seed=42 es Peor

El modelo seed=42 fue entrenado en condiciones diferentes:
- Fue el primer experimento (exp4_epochs100)
- Posible overfitting o configuración subóptima
- Error individual muy alto (6.75 px vs ~4.04 px)

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `scripts/evaluate_ensemble.py` | Agregado flag `--exclude-42` |
| `configs/final_config.json` | Agregada sección `ensemble_optimal` |

## Script de Evaluación

```bash
# Ensemble sin seed=42
python scripts/evaluate_ensemble.py --exclude-42

# Resultado: 3.79 px
```

## Resultados por Categoría (2 modelos)

| Categoría | Error | Muestras |
|-----------|-------|----------|
| Normal | 3.53 px | 47 |
| COVID | 3.83 px | 31 |
| Viral Pneumonia | 4.42 px | 18 |

## Configuración Final Documentada

```json
{
  "ensemble_optimal": {
    "error_px": 3.79,
    "models_count": 2,
    "models_used": ["seed123", "seed456"],
    "tta_enabled": true,
    "note": "Excluir seed=42 mejora de 4.50 a 3.79 px"
  }
}
```

## Estado Final

- [x] Análisis de impacto de seed=42 completado
- [x] Ensemble optimizado a 2 modelos
- [x] Error reducido de 4.50 px a 3.79 px
- [x] Script actualizado con flag --exclude-42
- [x] Configuración documentada

## Conclusión

Menos modelos pueden ser mejor cuando uno de ellos tiene rendimiento significativamente inferior. La calidad de los modelos individuales importa más que la cantidad.

## Siguiente Paso

Entrenar modelos adicionales con seeds diferentes para expandir el ensemble con modelos de alta calidad (→ Sesión 13).
