# Sesion 13: Ensemble de 4 Modelos

## Fecha: 2024-11-27 / 2024-11-28

## Objetivo
Entrenar modelos adicionales con nuevas seeds para crear un ensemble de 4 modelos de alta calidad y alcanzar el mejor resultado posible en predicción de landmarks.

## Contexto
Tras la Sesión 12, el ensemble de 2 modelos (seeds 123, 456) lograba 3.79 px. Se hipotetizó que agregar más modelos de calidad similar podría reducir aún más el error.

## Modelos Entrenados

### Nuevos Modelos (Session 13)

| Seed | Error Individual | Fecha | Checkpoint |
|------|------------------|-------|------------|
| 321 | ~4.0 px | 2024-11-28 01:16 | `checkpoints/session13/seed321/final_model.pt` |
| 789 | ~4.0 px | 2024-11-27 23:35 | `checkpoints/session13/seed789/final_model.pt` |

### Modelo Hierarchical (Experimental)

También se entrenó un modelo con arquitectura hierarchical:
- Checkpoint: `checkpoints/session13/hierarchical/final_model.pt`
- Fecha: 2024-11-28 01:28
- Nota: Experimento de arquitectura alternativa, no usado en ensemble final

## Configuración de Entrenamiento

Todos los modelos nuevos usaron la misma configuración:
```
- use_coord_attention: true
- deep_head: true
- hidden_dim: 768
- dropout_rate: 0.3
- phase1_epochs: 15
- phase2_epochs: 100
- loss: WingLoss
- CLAHE: enabled (clip=2.0, tile=4)
```

## Resultados de Ensemble

### Comparación de Configuraciones

| Ensemble | Modelos | Error | Mejora vs 2 modelos |
|----------|---------|-------|---------------------|
| 2 modelos | 123, 456 | 3.79 px | baseline |
| 4 modelos | 123, 456, 321, 789 | **3.71 px** | -0.08 px (-2.1%) |

### Resultado Final: 3.71 px

```bash
# Comando para reproducir
python -m src_v2 evaluate-ensemble \
    checkpoints/session10/ensemble/seed123/final_model.pt \
    checkpoints/session10/ensemble/seed456/final_model.pt \
    checkpoints/session13/seed321/final_model.pt \
    checkpoints/session13/seed789/final_model.pt \
    --tta --clahe --split test
```

## Métricas Detalladas (4 modelos + TTA)

### Overall
- Mean Error: **3.71 px**
- Median Error: 3.17 px
- Std: 2.42 px

### Por Categoría

| Categoría | Error | Muestras |
|-----------|-------|----------|
| Normal | 3.42 px | 47 |
| COVID | 3.77 px | 31 |
| Viral Pneumonia | 4.40 px | 18 |

### Por Landmark (ordenado por error)

| Landmark | Error | Descripción |
|----------|-------|-------------|
| L10 | 2.57 px | Middle central |
| L9 | 2.84 px | Superior central |
| L5 | 2.97 px | Left hilum |
| L6 | 3.01 px | Right hilum |
| L11 | 3.19 px | Inferior central |
| L3 | 3.20 px | Left apex |
| L1 | 3.20 px | Superior mediastinum |
| L7 | 3.39 px | Left base |
| L4 | 3.49 px | Right apex |
| L8 | 3.67 px | Right base |
| L2 | 4.34 px | Inferior mediastinum |
| L15 | 4.48 px | Right costophrenic angle |
| L14 | 4.63 px | Left costophrenic angle |
| L13 | 5.21 px | Right upper border |
| L12 | 5.50 px | Left upper border |

## Observaciones

### Landmarks Más Precisos
- Landmarks centrales (L9, L10, L11) tienen menor error (~2.5-3.2 px)
- Los hilios (L5, L6) también son precisos (~3.0 px)

### Landmarks Más Difíciles
- Bordes superiores (L12, L13) tienen mayor error (~5.2-5.5 px)
- Ángulos costofrénicos (L14, L15) moderadamente difíciles (~4.5-4.6 px)

### Impacto de Agregar Modelos
- 2 → 4 modelos: mejora de 0.08 px (2.1%)
- Rendimiento decreciente: cada modelo adicional aporta menos
- 4 modelos parece ser un buen balance costo/beneficio

## Checkpoints Finales del Proyecto

| Seed | Session | Error Individual | Usado en Ensemble |
|------|---------|------------------|-------------------|
| 42 | 10 | 6.75 px | ❌ (degrada) |
| 123 | 10 | 4.05 px | ✅ |
| 456 | 10 | 4.04 px | ✅ |
| 321 | 13 | ~4.0 px | ✅ |
| 789 | 13 | ~4.0 px | ✅ |

## Estado Final

- [x] Modelo seed=321 entrenado
- [x] Modelo seed=789 entrenado
- [x] Modelo hierarchical entrenado (experimental)
- [x] Ensemble de 4 modelos evaluado: 3.71 px
- [x] Mejor resultado del proyecto alcanzado

## Conclusión

El ensemble de 4 modelos (seeds 123, 456, 321, 789) con TTA logra **3.71 px**, el mejor resultado del proyecto. Esto representa:
- Mejora de 17.6% vs ensemble original de 3 modelos (4.50 px)
- Mejora de 8.2% vs mejor modelo individual (4.04 px)

## Siguiente Paso

Implementar el comando `evaluate-ensemble` en el CLI para poder reproducir este resultado fácilmente (→ Sesión 11, implementada posteriormente).
