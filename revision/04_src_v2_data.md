# 04. src_v2 Data Module

Analisis del modulo de datos: datasets, transformaciones y utilidades de carga.

**Archivos analizados**: 4

---

## Resumen del modulo

El modulo `src_v2/data/` es el nucleo de carga y preparacion de datos para el pipeline de
deteccion de landmarks. Gestiona la lectura de CSV de coordenadas, la construccion de datasets
PyTorch, las transformaciones (augmentation + CLAHE), y utilidades de visualizacion/analisis.
Es usado extensivamente tanto por el CLI principal como por scripts auxiliares.

**Lineas totales**: 1,073 (sin contar `__init__.py`)
**Tamano total**: ~33 KB

---

## Analisis archivo por archivo

### __init__.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/data/__init__.py`
- **Lineas/Tamano**: 15 lineas / 365 bytes
- **Proposito**: Define la API publica del modulo de datos, re-exportando las clases y funciones principales.
- **Contenido clave**:
  - Exporta: `LandmarkDataset`, `get_train_transforms`, `get_val_transforms`, `load_coordinates_csv`, `visualize_landmarks`
  - `__all__` bien definido con 5 simbolos
- **Dependencias**: Importa de `.dataset`, `.transforms`, `.utils`
- **Importancia**: ALTO
- **Justificacion**: Punto de entrada limpio del modulo. Permite imports como `from src_v2.data import LandmarkDataset`. Bien estructurado.

**Observacion**: La API publica no incluye `get_dataframe_splits`, `compute_sample_weights`, `apply_clahe`, `compute_statistics`, ni `compute_symmetry_error`, los cuales se importan directamente desde sus submodulos por scripts externos. Podria considerarse agregar `get_dataframe_splits` y `apply_clahe` al `__all__` dado su uso frecuente.

---

### dataset.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/data/dataset.py`
- **Lineas/Tamano**: 378 lineas / 12.3 KB
- **Proposito**: Define el dataset PyTorch para imagenes con 15 landmarks y la funcion factory para crear DataLoaders con splits estratificados.
- **Contenido clave**:
  - `compute_sample_weights()` (L27-50): Calcula pesos por muestra segun categoria para `WeightedRandomSampler`. Usa `DEFAULT_CATEGORY_WEIGHTS` del modulo constants (COVID=2.0, Normal=1.0, Viral=1.2).
  - `LandmarkDataset` (L53-145): Dataset PyTorch que retorna `(image_tensor, landmarks_tensor, meta)`. Carga imagen via PIL, extrae landmarks del DataFrame, normaliza coordenadas a [0,1], aplica transformaciones, y retorna metadata (nombre, categoria, indice, tamano original).
  - `create_dataloaders()` (L148-337): Funcion factory completa que:
    1. Carga CSV via `load_coordinates_csv()`
    2. Realiza split estratificado train/val/test con `train_test_split` (con fallback no-estratificado)
    3. Crea transformaciones de train y val
    4. Construye `LandmarkDataset` para cada split
    5. Configura `WeightedRandomSampler` opcional para balanceo de clases
    6. Configura seeding determinista opcional (worker_init_fn + generator)
    7. Retorna tupla de 3 DataLoaders con custom `collate_fn`
  - `get_dataframe_splits()` (L340-378): Funcion auxiliar para obtener solo los DataFrames sin crear DataLoaders. Util para analisis y verificacion.

- **Dependencias**:
  - Importa: `torch`, `pandas`, `numpy`, `PIL`, `sklearn.model_selection`, `src_v2.constants`, `.utils`, `.transforms`
  - Importado por: `src_v2/cli.py`, `src_v2/visualization/scientific_viz.py`, `scripts/train.py`, `scripts/extract_predictions.py`, `scripts/verify_*.py`, y multiples scripts de visualizacion y archivo.
- **Importancia**: CRITICO
- **Justificacion**: Es el punto de entrada para toda la carga de datos del pipeline de entrenamiento de landmarks. Sin este archivo, no se puede entrenar ni evaluar ningun modelo.

**Observaciones tecnicas**:

1. **Logica de split duplicada**: `create_dataloaders()` (L196-227) y `get_dataframe_splits()` (L361-378) implementan la misma logica de split estratificado, pero `create_dataloaders` incluye manejo de excepciones (fallback a no-estratificado) que `get_dataframe_splits` no tiene. Si un dataset tiene categorias con pocos ejemplos, `get_dataframe_splits` fallaria donde `create_dataloaders` no lo haria. Se recomienda refactorizar para que `create_dataloaders` llame a `get_dataframe_splits` (o una funcion comun) con manejo uniforme de errores.

2. **Tamano original hardcoded**: `ORIGINAL_IMAGE_SIZE = 299` es el default, pero el dataset maneja correctamente imagenes de tamano diferente gracias al warning y uso de `actual_size` en L124-131. Buen diseno defensivo.

3. **Custom collate_fn inline**: La funcion `collate_fn` (L258-262) esta definida inline dentro de `create_dataloaders`. Esto impide su reutilizacion. Podria extraerse como funcion de modulo.

4. **Iteracion ineficiente en compute_sample_weights**: Usa `df.iterrows()` (L45-48) que es lento para DataFrames grandes. Podria reemplazarse con `df['category'].map(category_weights).fillna(1.0)` para una solucion vectorizada.

---

### transforms.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/data/transforms.py`
- **Lineas/Tamano**: 390 lineas / 12.4 KB
- **Proposito**: Implementa transformaciones coordinadas para imagen + landmarks, incluyendo augmentation (flip, rotacion, color jitter) y preprocesamiento (CLAHE, resize, normalizacion ImageNet).
- **Contenido clave**:
  - `apply_clahe()` (L33-80): Funcion independiente que aplica CLAHE en espacio LAB (solo canal L). Convierte RGB->LAB, aplica CLAHE al canal de luminancia, y reconvierte. Maneja tanto imagenes RGB como escala de grises.
  - `LandmarkTransform` (L83-143): Clase base con metodos compartidos:
    - `resize_image()`: Resize a `output_size x output_size` con `Image.BILINEAR`
    - `apply_clahe_if_enabled()`: Condicional CLAHE
    - `normalize_coords()`: Normaliza landmarks de pixeles a [0,1] con clipping
    - `image_to_tensor()`: PIL -> tensor + normalizacion ImageNet
    - `landmarks_to_tensor()`: numpy (15,2) -> tensor flat (30,)
  - `TrainTransform(LandmarkTransform)` (L146-298): Subclase con augmentations:
    - `horizontal_flip()`: Refleja imagen Y coordenadas X, luego intercambia indices de pares simetricos (L3<->L4, L5<->L6, etc.). Implementacion correcta del flip bilateral.
    - `rotate()`: Rotacion con transformacion afin de landmarks alrededor del centro (0.5, 0.5) en coords normalizadas. Clipping a [0,1].
    - `color_jitter()`: Variaciones aleatorias de brillo y contraste.
    - `__call__()`: Pipeline completo: normalize_coords -> CLAHE -> resize -> flip -> rotate -> color_jitter -> to_tensor
  - `ValTransform(LandmarkTransform)` (L301-352): Subclase sin augmentation: normalize_coords -> CLAHE -> resize -> to_tensor
  - `get_train_transforms()` (L355-373): Factory function para `TrainTransform`
  - `get_val_transforms()` (L376-390): Factory function para `ValTransform`

- **Dependencias**:
  - Importa: `numpy`, `torch`, `PIL`, `torchvision.transforms.functional`, `random`, `cv2`, `src_v2.constants`
  - Importado por: `src_v2/data/dataset.py`, `src_v2/cli.py`, multiples scripts de visualizacion, `scripts/archive/test_dataset.py`
- **Importancia**: CRITICO
- **Justificacion**: Las transformaciones son fundamentales para el entrenamiento correcto del modelo. El flip horizontal con intercambio de indices simetricos es critico para la correcta augmentation de datos bilaterales. Errores aqui corrompen todo el entrenamiento.

**Observaciones tecnicas**:

1. **Import roto en script externo**: `scripts/glass_box_visualizations/block_a_pipeline.py` importa `apply_clahe_transform` que NO existe en `transforms.py`. La funcion se llama `apply_clahe`. Este import fallaria en runtime.

2. **Import invalido de SYMMETRIC_PAIRS desde transforms**: Dos scripts de visualizacion (`generate_animations.py`, `generate_pipeline_visualizations.py`) importan `SYMMETRIC_PAIRS` desde `src_v2.data.transforms`, pero esta constante no se re-exporta desde transforms -- solo se importa internamente desde `src_v2.constants`. Estos imports fallan silenciosamente si los scripts tienen fallbacks, o fallan en runtime.

3. **Orden de CLAHE y resize**: El pipeline aplica CLAHE ANTES del resize (L278-282 en `TrainTransform.__call__`), lo cual es correcto ya que CLAHE se beneficia de la resolucion original mas alta. Bien documentado con comentario.

4. **Image.BILINEAR deprecado**: `Image.BILINEAR` en PIL/Pillow esta deprecado a favor de `Image.Resampling.BILINEAR` desde Pillow 9.1. Funciona pero genera warnings en versiones recientes.

5. **Clipping en rotacion**: Los landmarks rotados se clipean a [0,1] (L243-244), lo cual introduce error geometrico cuando landmarks se acercan a bordes. Aceptable para rotaciones pequenas (10 grados default), pero podria ser problematico si se aumenta el rango de rotacion.

---

### utils.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/data/utils.py`
- **Lineas/Tamano**: 290 lineas / 8.4 KB
- **Proposito**: Funciones utilitarias para carga de CSV, construccion de rutas, extraccion de landmarks, visualizacion y analisis estadistico de datasets de landmarks.
- **Contenido clave**:
  - `load_coordinates_csv()` (L27-106): Parser robusto del CSV de coordenadas. Lee CSV sin encabezados con formato fijo (idx + 30 coords + image_name). Extrae categoria del nombre de imagen (COVID/Normal/Viral). Incluye validacion de formato y deteccion de encabezados incorrectos.
  - `get_image_path()` (L109-116): Construye ruta de imagen: `data_root/dataset/category/image_name.png`. Simple pero importante para la organizacion del dataset.
  - `get_landmarks_array()` (L119-134): Extrae coordenadas de una fila del DataFrame como array numpy (15, 2).
  - `landmarks_to_dict()` (L137-147): Convierte array de landmarks a diccionario con nombres (L1, L2, ...).
  - `visualize_landmarks()` (L150-222): Funcion de visualizacion con matplotlib. Dibuja landmarks ground truth con colores por tipo (eje, central, bilateral, esquinas) y opcionalmente predicciones en rojo. Dibuja el eje L1-L2.
  - `compute_statistics()` (L225-254): Calcula estadisticas basicas del dataset (total, por categoria, media/std por landmark).
  - `compute_symmetry_error()` (L257-290): Mide error de simetria bilateral calculando la diferencia de distancia perpendicular al eje L1-L2 para cada par simetrico. Util para analisis geometrico.

- **Dependencias**:
  - Importa: `pandas`, `numpy`, `pathlib`, `matplotlib.pyplot`, `PIL`, `src_v2.constants`
  - Importado por: `src_v2/data/dataset.py`, `src_v2/data/__init__.py`, `src_v2/cli.py`, `src_v2/visualization/comparison_viz.py`, `scripts/analyze_data.py`, `scripts/verify_data_leakage.py`, `scripts/generate_all_landmarks_npz.py`, `scripts/generate_all_visualizations.py`
- **Importancia**: CRITICO
- **Justificacion**: `load_coordinates_csv()` y `get_image_path()` son el punto de entrada para TODOS los datos del proyecto. Si la carga de CSV falla o la ruta es incorrecta, nada funciona.

**Observaciones tecnicas**:

1. **Extraccion de categoria fragil**: `extract_category()` (L83-91) usa `startswith` para determinar la categoria. Esto funciona para el dataset actual pero fallaria si se agregan nuevas categorias o si los nombres de archivo cambian de formato. No es un problema practico dado que el dataset esta fijo.

2. **matplotlib import a nivel de modulo**: `import matplotlib.pyplot as plt` (L14) se importa siempre aunque solo `visualize_landmarks()` lo necesita. En entornos headless (servers de entrenamiento), esto puede causar problemas si no hay backend configurado. Un import lazy dentro de la funcion seria mas robusto.

3. **compute_symmetry_error sin uso critico**: Solo se usa en `scripts/analyze_data.py`. Es una utilidad de analisis, no del pipeline principal.

4. **visualize_landmarks() llama plt.show()**: Esto bloquea la ejecucion en scripts no interactivos. Idealmente, la decision de mostrar o solo guardar deberia ser parametrizable (o eliminar `plt.show()` cuando se pasa `save_path`).

---

## Dependencias entre archivos del modulo

```
__init__.py
  |-- dataset.py (LandmarkDataset)
  |-- transforms.py (get_train_transforms, get_val_transforms)
  |-- utils.py (load_coordinates_csv, visualize_landmarks)

dataset.py
  |-- utils.py (load_coordinates_csv, get_image_path, get_landmarks_array)
  |-- transforms.py (get_train_transforms, get_val_transforms)
  |-- constants.py (DEFAULT_CATEGORY_WEIGHTS, ORIGINAL_IMAGE_SIZE)

transforms.py
  |-- constants.py (SYMMETRIC_PAIRS, DEFAULT_IMAGE_SIZE, IMAGENET_MEAN/STD, etc.)

utils.py
  |-- constants.py (SYMMETRIC_PAIRS, CENTRAL_LANDMARKS, LANDMARK_NAMES, etc.)
```

**Flujo de datos tipico**:
1. `load_coordinates_csv()` -> DataFrame con columnas image_name, category, L{i}_x, L{i}_y
2. `create_dataloaders()` -> train_test_split estratificado -> 3 DataLoaders
3. `LandmarkDataset.__getitem__()` -> carga imagen, extrae landmarks, aplica transform
4. `TrainTransform.__call__()` -> CLAHE -> resize -> flip/rotate/jitter -> tensor
5. DataLoader retorna batch: `(images [B,3,224,224], landmarks [B,30], metas [list of dict])`

---

## Resumen de problemas encontrados

### Bugs / Imports rotos

| Problema | Archivo afectado | Severidad |
|---|---|---|
| Import `apply_clahe_transform` no existe | `scripts/glass_box_visualizations/block_a_pipeline.py` L30 | MEDIA (script auxiliar, fallaria en runtime) |
| Import `SYMMETRIC_PAIRS` desde `transforms.py` | `scripts/visualization/generate_animations.py` L35, `scripts/visualization/generate_pipeline_visualizations.py` L31 | MEDIA (la constante no se re-exporta, deberia importarse de `src_v2.constants`) |
| `Image.BILINEAR` deprecado | `transforms.py` L107 | BAJA (funciona, genera warnings) |

### Oportunidades de mejora

| Mejora | Archivo | Impacto |
|---|---|---|
| Refactorizar logica de split duplicada entre `create_dataloaders()` y `get_dataframe_splits()` | `dataset.py` | MEDIO - Reduce riesgo de divergencia en el manejo de errores |
| Vectorizar `compute_sample_weights()` con `.map()` en vez de `iterrows()` | `dataset.py` L45-48 | BAJO - Mejora performance en datasets grandes |
| Import lazy de `matplotlib` en `utils.py` | `utils.py` L14 | BAJO - Evita problemas en entornos headless |
| Extraer `collate_fn` como funcion de modulo | `dataset.py` L258-262 | BAJO - Permite reutilizacion |
| Parametrizar `plt.show()` en `visualize_landmarks()` | `utils.py` L222 | BAJO - Mejor para scripts batch |
| Agregar `get_dataframe_splits` y `apply_clahe` al `__all__` | `__init__.py` | BAJO - Refleja uso real del modulo |

### Cobertura de tests

No se encontraron archivos de test en `tests/` (el directorio no existe o esta vacio). Los unicos "tests" son scripts en `scripts/archive/` (`test_dataset.py`, `test_forward_pass.py`, etc.) que son mas bien scripts de verificacion manual, no tests unitarios automatizados con pytest. **Esto es una brecha significativa** dado que CLAUDE.md documenta `python -m pytest tests/ -v` como comando de testing.

---

## Tabla resumen

| Archivo | Lineas | Importancia | Funcion principal |
|---|---|---|---|
| `__init__.py` | 15 | ALTO | API publica del modulo |
| `dataset.py` | 378 | CRITICO | LandmarkDataset + create_dataloaders() |
| `transforms.py` | 390 | CRITICO | Augmentations coordinadas imagen+landmarks |
| `utils.py` | 290 | CRITICO | Carga CSV, rutas, visualizacion, analisis |

**Conclusion**: El modulo de datos esta bien estructurado y es funcionalmente correcto para el pipeline principal. Los 4 archivos son necesarios: 3 son CRITICOS y 1 es ALTO. No hay archivos eliminables. Los problemas principales son: (1) imports rotos en scripts auxiliares que referencian simbolos inexistentes, (2) logica de split duplicada con manejo de errores inconsistente, y (3) ausencia de tests unitarios automatizados para un modulo tan critico.
