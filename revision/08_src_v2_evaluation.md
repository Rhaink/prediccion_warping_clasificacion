# 08. src_v2 Evaluation Module

Analisis del sistema de evaluacion: metricas de error y evaluacion de ensembles.

**Archivos analizados**: 3
**Lineas totales**: 1,017
**Tamano total**: ~31 KB

---

## Resumen del Modulo

El modulo `src_v2/evaluation/` contiene la logica de evaluacion para los dos tipos de modelos del proyecto: modelos de landmark detection (metricas de error en pixeles, TTA con flip horizontal y correccion de simetria) y ensembles de clasificadores (soft/hard voting, TTA de clasificador sin correccion de simetria). Es un modulo critico que conecta el entrenamiento con la validacion de resultados reportados en GROUND_TRUTH.json.

---

## Analisis por Archivo

### __init__.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/evaluation/__init__.py
- **Lineas/Tamano**: 19 lineas / 399 B
- **Proposito**: Exporta las funciones publicas del submodulo de metricas de landmarks. Sirve como interfaz publica del paquete evaluation.
- **Contenido clave**:
  - Exporta 5 funciones de `metrics.py`: `compute_pixel_error`, `compute_error_per_landmark`, `compute_error_per_category`, `evaluate_model`, `generate_evaluation_report`
  - Define `__all__` con las mismas 5 funciones
- **Dependencias**:
  - Importa de: `.metrics`
  - Importado por: `scripts/train.py` (indirectamente via las funciones re-exportadas)
- **Importancia**: MEDIO
- **Justificacion**: Archivo de conveniencia estandar de Python. Funcional pero incompleto.
- **Observaciones**:
  - **NO exporta** funciones importantes de `metrics.py`: `compute_success_rate`, `predict_with_tta`, `evaluate_model_with_tta`, `_flip_landmarks_horizontal`. Los consumidores como `src_v2/cli.py` y `scripts/train.py` importan directamente desde `src_v2.evaluation.metrics` en vez de usar el `__init__.py`, lo cual hace que este archivo sea parcialmente inconsistente con el uso real.
  - **NO exporta nada** de `ensemble.py`. Todo consumo de funciones de ensemble (desde `cli.py`) usa imports directos a `src_v2.evaluation.ensemble`.
  - Recomendacion: o bien actualizar `__init__.py` para reflejar todas las funciones publicas, o eliminar la re-exportacion parcial y dejar que cada consumidor importe directamente del submodulo que necesita.

---

### metrics.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/evaluation/metrics.py
- **Lineas/Tamano**: 545 lineas / 15.9 KB
- **Proposito**: Define todas las metricas de evaluacion para modelos de landmark detection: error euclidiano en pixeles, desagregacion por landmark y categoria, TTA con flip horizontal, y generacion de reportes textuales.
- **Contenido clave**:
  - **`_prepare_image_size_tensor()`** (lineas 23-76): Funcion auxiliar robusta que normaliza multiples formatos de `image_size` (escalar, tupla, tensor por muestra) a un tensor (B, 1, 2) para broadcasting. Maneja casos edge: None, 0-dim, 1-dim con 1 o 2 elementos, 2-dim con batch 1 vs batch B. Incluye validacion con mensajes de error claros.
  - **`compute_pixel_error()`** (lineas 79-116): Funcion central. Convierte predicciones/targets normalizados [0,1] a coordenadas de pixeles y calcula distancia euclidiana por landmark. Retorna tensor (B, 15). Soporta image_size variable por muestra.
  - **`compute_error_per_landmark()`** (lineas 119-133): Wrapper que promedia errores por landmark y retorna dict {nombre: error}. Usa constante `LANDMARK_NAMES`.
  - **`_get_batch_image_sizes()`** (lineas 136-167): Extrae tamanos de imagen desde metadata del DataLoader, con fallback configurable.
  - **`evaluate_model()`** (lineas 170-266): Evaluacion completa sin TTA. Itera DataLoader, computa errores, agrega por landmark/categoria, calcula percentiles (p50/p75/p90/p95). Retorna dict detallado con metricas + tensores raw.
  - **`compute_error_per_category()`** (lineas 269-302): Agrega error promedio por categoria (COVID/Normal/Viral_Pneumonia). Usa numpy para estadisticas finales.
  - **`generate_evaluation_report()`** (lineas 305-356): Genera reporte textual formateado con metricas globales, percentiles, tabla de error por landmark (ordenada por error), y error por categoria.
  - **`compute_success_rate()`** (lineas 359-379): Calcula porcentaje de predicciones bajo umbrales dados (5, 8, 10, 15 px). No exportada en `__init__.py`.
  - **`_flip_landmarks_horizontal()`** (lineas 382-404): Implementa flip horizontal de landmarks normalizados: refleja coordenada X (1-x) e intercambia pares simetricos usando `SYMMETRIC_PAIRS`. Critica para TTA correcto.
  - **`predict_with_tta()`** (lineas 407-445): TTA de landmarks: promedia prediccion original con prediccion sobre imagen flipped (corrigiendo flip en landmarks). Decorador `@torch.no_grad()`.
  - **`evaluate_model_with_tta()`** (lineas 448-545): Evaluacion completa con TTA. Estructura paralela a `evaluate_model()` pero usando `predict_with_tta()` para cada batch.
- **Dependencias**:
  - Importa de: `logging`, `collections.defaultdict`, `typing`, `numpy`, `torch`, `torch.utils.data.DataLoader`, `src_v2.constants` (LANDMARK_NAMES, SYMMETRIC_PAIRS, DEFAULT_IMAGE_SIZE)
  - Importado por: `src_v2/cli.py` (evaluate-landmarks), `scripts/train.py`, `scripts/train_hierarchical.py`, `scripts/archive/evaluate_ensemble.py`, `src_v2/evaluation/__init__.py`
- **Importancia**: CRITICO
- **Justificacion**: Es la base de toda evaluacion de landmarks. El valor validado de 3.61 px (GROUND_TRUTH.json) depende directamente de `compute_pixel_error()` y `evaluate_model_with_tta()`. Cualquier error aqui invalida los resultados del proyecto.
- **Observaciones tecnicas**:
  1. **Duplicacion con trainer.py**: `LandmarkTrainer` en `src_v2/training/trainer.py` tiene su propia implementacion inline de `compute_pixel_error()` (linea 59) que NO importa de este modulo. Aunque la logica es identica (reshape + escalar + torch.norm), la duplicacion es un riesgo de divergencia si una se modifica y la otra no.
  2. **Duplicacion evaluate_model vs evaluate_model_with_tta**: Las funciones `evaluate_model()` (lineas 170-266) y `evaluate_model_with_tta()` (lineas 448-545) comparten ~80% del codigo (agregacion de errores, percentiles, format de resultado). La unica diferencia es que una usa `model(images)` y la otra `predict_with_tta(model, images, device)`. Se podria refactorizar a una sola funcion con parametro `use_tta=False`.
  3. **Mutable default argument**: `compute_success_rate()` linea 361 usa `thresholds: List[float] = [5, 8, 10, 15]` -- lista mutable como default. En este caso es seguro porque la lista solo se lee (no se modifica), pero es un antipatron Python que linters suelen marcar.
  4. **Inconsistencia en tipo de retorno de estadisticas**: `compute_error_per_category()` usa `np.mean()`/`np.std()` (retorna float64), mientras que `evaluate_model()` usa `tensor.mean().item()` (retorna float). Ambos retornan floats pero de precision potencialmente distinta.
  5. **_prepare_image_size_tensor es robusto pero complejo**: La funcion maneja 7+ variantes de input en 54 lineas. Esta bien documentada y tiene validaciones, pero la complejidad sugiere que la interfaz de `image_size` evoluciono organicamente. Un tipo union explicito o un dataclass `ImageSize` podria simplificar.
  6. **Hardcoded 15 landmarks y 30 coordenadas**: `pred.view(B, 15, 2)` aparece en multiples funciones sin usar una constante `NUM_LANDMARKS`. Si el numero de landmarks cambiara, habria que modificar multiples funciones.

---

### ensemble.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/evaluation/ensemble.py
- **Lineas/Tamano**: 453 lineas / 15.2 KB
- **Proposito**: Implementa evaluacion de ensembles de clasificadores CNN con soporte para soft voting ponderado (por F1-macro de validacion), hard voting (mayoria simple), TTA de clasificador (flip horizontal sin correccion de simetria), y herramientas de analisis de impacto de TTA.
- **Contenido clave**:
  - **`load_ensemble_models()`** (lineas 22-85): Carga modelos desde checkpoints usando `create_classifier()`. Verifica consistencia de arquitectura (backbone_name). Extrae pesos de validacion (F1-macro) desde `results.json` en cada directorio de fold. Errores claros si falta results.json o el campo best_val_f1.
  - **`weighted_soft_voting()`** (lineas 88-121): Promedio ponderado de probabilidades usando `torch.einsum('mni,m->ni', ...)`. Normaliza pesos a suma 1.0. Retorna predicciones (argmax) y probabilidades ponderadas. Implementacion elegante y eficiente.
  - **`hard_voting()`** (lineas 124-151): Voto mayoritario usando `Counter.most_common()`. Itera sample por sample (loop Python), lo cual es menos eficiente que `torch.mode()` pero permite control explicito de desempates.
  - **`ensemble_inference()`** (lineas 154-199): Ejecuta inferencia con todos los modelos sobre todo el dataset. Usa tqdm para progreso. Retorna listas de predicciones/probabilidades por modelo + labels.
  - **`predict_with_tta_classifier()`** (lineas 202-241): TTA para clasificador: promedia probabilidades de imagen original y flipped. Documenta explicitamente que NO necesita correccion de simetria (a diferencia del TTA de landmarks) porque las etiquetas de clase son anatomicamente simetricas.
  - **`ensemble_inference_with_tta()`** (lineas 244-316): Combina ensemble + TTA. Implementa TTA "dual-level": (1) cada modelo promedia orig+flip, (2) el ensemble promedia los K modelos. Retorna detalles de TTA para trazabilidad (probabilidades originales y flipped por modelo).
  - **`validate_ensemble_setup()`** (lineas 319-377): Sanity checks pre-evaluacion: consistencia de arquitectura, modo eval, conteo de muestras (hardcoded 1895), probabilidades suman 1.0, predicciones en rango valido. Usa asserts (no exceptions).
  - **`categorize_tta_impact()`** (lineas 380-427): Categoriza impacto de TTA por muestra: "helped" (baseline incorrecta, TTA correcta), "hurt" (baseline correcta, TTA incorrecta), "neutral" (ambas iguales). Retorna detalle por muestra y resumen de conteos.
  - **`compute_tta_delta_metrics()`** (lineas 430-453): Calcula deltas entre metricas baseline y TTA (accuracy, F1-macro, F1 per-class). Util para reportes de impacto.
- **Dependencias**:
  - Importa de: `json`, `collections.Counter`, `pathlib.Path`, `typing`, `numpy`, `torch`, `torch.nn`, `torch.utils.data.DataLoader`, `tqdm`, `src_v2.models.create_classifier`
  - Importado por: `src_v2/cli.py` (comando evaluate-classifier-ensemble, linea 2665)
- **Importancia**: ALTO
- **Justificacion**: Soporta la evaluacion del ensemble de clasificadores que logra 99.10% accuracy. Es el unico consumidor de cross-validation folds para clasificacion. Bien disenado con multiples estrategias de ensemble y analisis de TTA.
- **Observaciones tecnicas**:
  1. **Hard voting ineficiente**: `hard_voting()` usa un loop Python sobre N muestras con `Counter()` por muestra. Para datasets grandes, `torch.mode(preds_stacked, dim=0)` seria significativamente mas rapido. Sin embargo, el comentario sobre desempate es valido -- `torch.mode` no garantiza el mismo comportamiento de desempate.
  2. **Hardcoded expected_samples=1895**: `validate_ensemble_setup()` tiene `expected_samples: int = 1895` como default. Este valor es especifico al dataset COVID-19 Radiography actual (tamaño del split de test). Deberia ser configurable o al menos documentar de donde sale el numero.
  3. **Emoji en codigo**: Linea 377 usa `print("✓ All sanity checks passed")` -- usa emoji en un print. Deberia usar logger en lugar de print, y evitar el emoji para consistencia con el resto del proyecto.
  4. **Uso de assert para validacion**: `validate_ensemble_setup()` usa `assert` para todas las validaciones. Los asserts se desactivan con `python -O`. Para validacion en produccion, deberia usar `raise ValueError/RuntimeError`. Aunque el docstring documenta `AssertionError` (nota: typo, deberia ser `AssertionError`), el uso de asserts es fragil.
  5. **Typo en docstring**: Linea 341 dice `AssertionError` en lugar de `AssertionError` -- aunque el nombre correcto de la excepcion Python es `AssertionError`, lo cual es correcto. Sin embargo, el uso de assert en vez de raise sigue siendo cuestionable.
  6. **Responsabilidad dual**: Este archivo mezcla funciones de inferencia (ensemble_inference, predict_with_tta_classifier) con funciones de analisis (categorize_tta_impact, compute_tta_delta_metrics). Podria separarse en `ensemble_inference.py` y `ensemble_analysis.py`, pero dado el tamano actual (453 lineas) la cohesion es aceptable.
  7. **Acoplamiento con results.json**: `load_ensemble_models()` asume una estructura de directorio especifica (checkpoint al lado de results.json con campo "best_val_f1"). Esto funciona para el pipeline actual pero es rigido.

---

## Analisis Transversal

### Patron de Duplicacion evaluate_model / evaluate_model_with_tta

La duplicacion mas significativa del modulo es entre `evaluate_model()` (lineas 170-266 de metrics.py) y `evaluate_model_with_tta()` (lineas 448-545). Ambas funciones:
- Iteran el DataLoader
- Extraen image_sizes del metadata
- Computan errores con `compute_pixel_error()`
- Agregan por categoria
- Calculan estadisticas globales, por landmark, percentiles
- Retornan el mismo formato de diccionario

La unica diferencia es la linea de prediccion:
```python
# evaluate_model:
outputs = model(images)

# evaluate_model_with_tta:
outputs = predict_with_tta(model, images, device, use_flip=True)
```

Refactorizacion sugerida:
```python
def evaluate_model(model, data_loader, device, image_size=DEFAULT_IMAGE_SIZE, use_tta=False):
    ...
    if use_tta:
        outputs = predict_with_tta(model, images, device, use_flip=True)
    else:
        outputs = model(images)
    ...
```

### Duplicacion compute_pixel_error entre metrics.py y trainer.py

`LandmarkTrainer.compute_pixel_error()` en trainer.py (linea 59) reimplementa la misma logica que `metrics.compute_pixel_error()`. El trainer no importa de metrics, probablemente para evitar dependencias circulares o por evolucion historica. Sin embargo, la logica del trainer es mas simple (solo acepta escalar image_size, no soporta per-sample sizes), lo cual genera una discrepancia sutil en capacidades.

### Coherencia entre Landmark TTA y Classifier TTA

El modulo maneja correctamente la diferencia fundamental entre TTA para landmarks y para clasificacion:
- **Landmark TTA** (`metrics.py::predict_with_tta`): Flip horizontal + correccion de pares simetricos (`_flip_landmarks_horizontal`). Necesario porque los landmarks L3/L4, L5/L6, etc. se intercambian al reflejar la imagen.
- **Classifier TTA** (`ensemble.py::predict_with_tta_classifier`): Flip horizontal sin correccion. Correcto porque "COVID flipped sigue siendo COVID".

Esta distincion esta bien documentada en los docstrings de ambas funciones.

### Funciones No Exportadas en __init__.py

Funciones publicas que se usan externamente pero no estan en `__init__.py`:
| Funcion | Definida en | Usada por |
|---------|-------------|-----------|
| `compute_success_rate` | metrics.py | (sin uso externo encontrado) |
| `predict_with_tta` | metrics.py | scripts/visualize_predictions.py (reimplementa), cli.py (reimplementa) |
| `evaluate_model_with_tta` | metrics.py | cli.py, scripts/train.py, scripts/train_hierarchical.py |
| Todas las funciones de ensemble.py | ensemble.py | cli.py |

### Cobertura de Tests

No se encontraron tests unitarios para el modulo de evaluacion. El directorio `tests/` mencionado en CLAUDE.md no existe actualmente en el repositorio. Esto es una brecha significativa dado que:
- `compute_pixel_error()` es la metrica principal del proyecto
- `_flip_landmarks_horizontal()` tiene logica de intercambio de indices que es facil de equivocar
- `weighted_soft_voting()` usa einsum que puede tener errores sutiles de dimensiones

---

## Resumen de Importancia

| Archivo | Lineas | Importancia | Justificacion |
|---------|--------|-------------|---------------|
| `__init__.py` | 19 | MEDIO | Re-exportacion parcial, inconsistente con uso real |
| `metrics.py` | 545 | CRITICO | Base de toda evaluacion de landmarks; valor 3.61 px depende de este codigo |
| `ensemble.py` | 453 | ALTO | Soporta ensemble de clasificadores (99.10% accuracy); bien diseñado con multiples estrategias |

## Recomendaciones Priorizadas

1. **[Alta]** Refactorizar `evaluate_model` y `evaluate_model_with_tta` en una sola funcion con parametro `use_tta` para eliminar ~90 lineas de codigo duplicado.
2. **[Alta]** Agregar tests unitarios para `compute_pixel_error`, `_flip_landmarks_horizontal`, `weighted_soft_voting`, y `hard_voting`.
3. **[Media]** Actualizar `__init__.py` para exportar `evaluate_model_with_tta`, `predict_with_tta`, y `compute_success_rate`, o documentar que los imports directos son el patron preferido.
4. **[Media]** Reemplazar `assert` por `raise ValueError` en `validate_ensemble_setup()` y usar `logger` en vez de `print`.
5. **[Baja]** Considerar que `LandmarkTrainer` importe `compute_pixel_error` de metrics.py en vez de reimplementarla, para mantener una sola fuente de verdad.
6. **[Baja]** Extraer constante `NUM_LANDMARKS = 15` para evitar magic numbers en reshape operations.
7. **[Baja]** Cambiar `thresholds: List[float] = [5, 8, 10, 15]` a `thresholds: Optional[List[float]] = None` con default interno.
