# 06. src_v2 Processing Module

Analisis del nucleo algoritmico: Generalized Procrustes Analysis y warping afin por partes.

**Archivos analizados**: 3

---

## Resumen del modulo

El modulo `src_v2/processing/` implementa los dos algoritmos centrales del pipeline de normalizacion geometrica:

1. **GPA (Generalized Procrustes Analysis)**: Computa la forma canonica (consensus shape) a partir de multiples configuraciones de landmarks, eliminando translacion, escala y rotacion.
2. **Piecewise Affine Warping**: Transforma imagenes a la forma canonica usando triangulacion de Delaunay y transformaciones afines por triangulo.

Estos dos modulos son el corazon matematico del proyecto. Sin ellos, el pipeline de normalizacion geometrica no existe.

---

## Analisis archivo por archivo

### __init__.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/processing/__init__.py`
- **Lineas/Tamano**: 42 lineas / 970 bytes
- **Proposito**: Re-exporta las funciones publicas de `gpa.py` y `warp.py` para ofrecer una API limpia a nivel de paquete.
- **Contenido clave**:
  - Importa 8 funciones de `gpa.py`: `center_shape`, `scale_shape`, `optimal_rotation_matrix`, `align_shape`, `procrustes_distance`, `gpa_iterative`, `scale_canonical_to_image`, `compute_delaunay_triangulation`
  - Importa 4 funciones de `warp.py`: `piecewise_affine_warp`, `scale_landmarks_from_centroid`, `clip_landmarks_to_image`, `add_boundary_points`
  - Define `__all__` con las 12 funciones exportadas
- **Dependencias**:
  - Importa de: `src_v2.processing.gpa`, `src_v2.processing.warp`
  - Importado por: Consumidores del paquete que usan `from src_v2.processing import ...`
- **Importancia**: MEDIO
- **Justificacion**: Punto de entrada del paquete. Cumple su funcion correctamente.
- **Observaciones**:
  - `compute_fill_rate` y `warp_mask` (ambas funciones publicas de `warp.py`) **no estan re-exportadas** en `__init__.py` ni incluidas en `__all__`. Los consumidores deben importar directamente de `src_v2.processing.warp`, lo cual es inconsistente con el patron del modulo. Esto no es critico (ambas funciones son usadas por scripts especializados que importan directamente), pero rompe la simetria de la API.
  - `get_affine_transform_matrix`, `create_triangle_mask`, `get_bounding_box`, `warp_triangle` de `warp.py` tampoco se exportan, lo cual es correcto ya que son funciones auxiliares internas.

---

### gpa.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/processing/gpa.py`
- **Lineas/Tamano**: 305 lineas / 9.0 KB
- **Proposito**: Implementa Generalized Procrustes Analysis (GPA) iterativo para computar la forma canonica (consensus) a partir de un conjunto de configuraciones de landmarks, eliminando diferencias de translacion, escala y rotacion.
- **Contenido clave**:

  | Funcion | Lineas | Descripcion |
  |---------|--------|-------------|
  | `center_shape()` | 30-43 | Centra una forma en el origen restando el centroide |
  | `scale_shape()` | 46-62 | Normaliza a norma unitaria (Frobenius) con proteccion contra escala ~0 |
  | `optimal_rotation_matrix()` | 65-92 | Calcula rotacion optima via SVD, con correccion de reflexion (det < 0) |
  | `align_shape()` | 95-108 | Alinea una forma con una referencia aplicando la rotacion optima |
  | `procrustes_distance()` | 111-135 | Distancia Procrustes entre dos formas (centra, escala, alinea, norma residual) |
  | `gpa_iterative()` | 138-248 | Algoritmo GPA completo: centra+escala todas las formas, itera alineacion con la media hasta convergencia |
  | `scale_canonical_to_image()` | 251-291 | Convierte forma canonica normalizada a coordenadas pixel (224x224) con padding |
  | `compute_delaunay_triangulation()` | 294-305 | Wrapper sobre `scipy.spatial.Delaunay` |

- **Dependencias**:
  - Importa: `numpy`, `warnings`, `logging`, `typing`, `scipy.spatial.Delaunay`
  - Importado por: `src_v2/processing/__init__.py`, `src_v2/cli.py` (en `compute-canonical` y otros comandos), `scripts/glass_box_visualizations/block_a_pipeline.py` (con error, ver observaciones)
- **Importancia**: CRITICO
- **Justificacion**: La forma canonica producida por GPA es el objetivo geometrico (target shape) al que se normalizan todas las imagenes. Sin este modulo, no hay forma canonica y el warping no tiene destino. Los resultados validados del proyecto (99.10% accuracy) dependen directamente de la calidad de este consensus.
- **Observaciones**:

  **Calidad del algoritmo**: La implementacion es textbook-correct:
  - El ciclo GPA sigue el algoritmo estandar de Gower (1975) / Dryden & Mardia (1998)
  - Usa SVD para rotacion optima, que es la solucion analitica exacta al problema de Procrustes ortogonal
  - Incluye correccion de reflexion (linea 88-90) para garantizar rotacion propia (det = +1)
  - Criterio de convergencia basado en cambio en la referencia normalizada
  - Proteccion contra formas degeneradas (scale < 1e-10)

  **Detalle de implementacion en `gpa_iterative()`**: En linea 199, la alineacion se hace siempre sobre `normalized_shapes[i]` (las formas originales centra+escaladas), no sobre `aligned_shapes[i]` de la iteracion anterior. Esto es matematicamente correcto porque la rotacion optima se calcula contra la referencia actual, y aplicar rotaciones incrementales no tiene ventaja.

  **Potencial micro-optimizacion**: El calculo de `mean_distance` en lineas 211-214 usa una list comprehension con loop explicito. Podria vectorizarse con `np.linalg.norm(aligned_shapes - new_reference_scaled, axis=(1,2)).mean()`, pero el impacto es negligible dado que GPA se ejecuta una sola vez y converge en pocas iteraciones.

  **Funcion faltante**: El script `scripts/glass_box_visualizations/block_a_pipeline.py` importa `load_canonical_shape` desde `src_v2.processing.gpa`, pero esta funcion **no existe** en el modulo. Solo esta definida en scripts standalone (`scripts/piecewise_affine_warp.py`, `scripts/benchmark_inference.py`, etc.). Esto causaria un `ImportError` al ejecutar ese script. Sin embargo, esto es un problema del script consumidor, no de `gpa.py`.

  **`scale_canonical_to_image()`**: Logica correcta pero asume padding simetrico y que la forma canonica esta centrada en (0,0). El parametro `padding=0.1` (10%) es un default razonable. La funcion escala proporcionalmente (usa `max_range` para mantener aspect ratio) y centra en la imagen.

  **`compute_delaunay_triangulation()`**: Wrapper trivial sobre scipy. Util para mantener consistencia de API, pero apenas agrega valor. Retorna `tri.simplices` (indices de vertices), no el objeto Delaunay completo.

---

### warp.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/processing/warp.py`
- **Lineas/Tamano**: 448 lineas / 13.1 KB (el archivo mas grande del modulo)
- **Proposito**: Implementa piecewise affine warping para normalizar geometricamente imagenes de rayos X usando landmarks predichos y forma canonica, transformando cada triangulo de la triangulacion de Delaunay con su propia transformacion afin.
- **Contenido clave**:

  | Funcion | Lineas | Visibilidad | Descripcion |
  |---------|--------|-------------|-------------|
  | `_triangle_area_2x()` | 24-38 | Privada | Calcula 2x el area de un triangulo (cross product) para detectar degenerados |
  | `scale_landmarks_from_centroid()` | 41-57 | Publica | Escala landmarks desde su centroide (margin expansion) |
  | `clip_landmarks_to_image()` | 60-77 | Publica | Clamp de landmarks dentro de limites de imagen |
  | `add_boundary_points()` | 80-114 | Publica | Agrega 8 puntos (4 esquinas + 4 medios de borde) para cobertura total |
  | `get_affine_transform_matrix()` | 117-134 | Publica (interna) | Wrapper sobre `cv2.getAffineTransform` |
  | `create_triangle_mask()` | 137-154 | Publica (interna) | Crea mascara binaria de un triangulo con `cv2.fillConvexPoly` |
  | `get_bounding_box()` | 157-175 | Publica (interna) | Bounding box de un triangulo con clamping a imagen |
  | `warp_triangle()` | 178-238 | Publica (interna) | Warps un triangulo src->dst con transformacion afin por parches |
  | `piecewise_affine_warp()` | 241-301 | Publica | **Funcion principal**: warping completo imagen->canonica via todos los triangulos |
  | `compute_fill_rate()` | 304-320 | Publica | Calcula proporcion de pixeles no-negros en imagen warpeada |
  | `warp_mask()` | 323-392 | Publica | Warping de mascaras binarias con NEAREST interpolation |
  | `_warp_triangle_nearest()` | 395-448 | Privada | Version NEAREST de `warp_triangle()` para mascaras |

- **Dependencias**:
  - Importa: `numpy`, `cv2`, `warnings`, `logging`, `typing`, `scipy.spatial.Delaunay`
  - Importado por: `src_v2/processing/__init__.py`, `src_v2/cli.py` (multiples comandos: `generate-dataset`, `warp-image`, etc.), `src_v2/gui/model_manager.py`, `scripts/benchmark_inference.py`, `scripts/generate_thesis_figures_master.py`, `scripts/calculate_pfs_warped.py`, `scripts/visualization/generate_feature_maps_pipeline.py`, `scripts/glass_box_visualizations/block_a_pipeline.py`
- **Importancia**: CRITICO
- **Justificacion**: Este es el modulo que produce las imagenes normalizadas geometricamente sobre las que el clasificador alcanza 99.10% de accuracy. Cada imagen del dataset pasa por `piecewise_affine_warp()` antes de ser clasificada. Es probablemente la funcion mas ejecutada del pipeline completo.
- **Observaciones**:

  **Estrategia de warping (bounding box + patch)**: La implementacion en `warp_triangle()` usa una optimizacion clasica:
  1. Calcula bounding box del triangulo fuente y destino
  2. Extrae solo el parche del bounding box
  3. Computa la transformacion afin en coordenadas locales del parche
  4. Aplica `cv2.warpAffine` solo al parche (mucho mas rapido que transformar la imagen completa)
  5. Usa mascara del triangulo destino para copiar solo los pixeles relevantes

  Esto es correcto y eficiente. El uso de `cv2.BORDER_REFLECT_101` en el warping de imagenes evita artefactos en bordes, mientras que `cv2.BORDER_CONSTANT` con valor 0 en el warping de mascaras preserva el semantica binaria.

  **`piecewise_affine_warp()` -- flujo principal**:
  - Cuando `use_full_coverage=True` (default), agrega 8 boundary points a los 15 landmarks (total 23 puntos) y recomputa Delaunay sobre los puntos destino extendidos. Esto garantiza que toda la imagen tenga cobertura de triangulos, no solo la region de los pulmones.
  - Cuando `use_full_coverage=False`, usa solo los 15 landmarks y la triangulacion proporcionada. Esto produce imagenes con regiones negras fuera del area de los pulmones.
  - Segun CLAUDE.md, la configuracion optima validada es `use_full_coverage=false` en `warping_best.json`, lo cual es interesante porque produce imagenes parcialmente negras pero el clasificador funciona mejor con ellas.
  - Proteccion contra triangulos degenerados (area < 1e-6) y excepciones individuales por triangulo con `warnings.warn` y `continue`.

  **`compute_fill_rate()` -- posible imprecision**: La funcion cuenta pixeles con valor exactamente 0 como "negros". Para imagenes uint8 esto funciona bien, pero si la imagen tuviera pixeles de valor 0 que son contenido real (fondo negro natural de radiografias), el fill rate seria sobreestimado. En la practica esto no es un problema grave porque se usa como metrica de calidad, no para decision critica.

  **`warp_mask()` -- duplicacion parcial con `piecewise_affine_warp()`**: La logica de extension de landmarks y triangulacion (lineas 361-372) duplica la logica de `piecewise_affine_warp()` (lineas 265-275). La diferencia clave es la interpolacion (NEAREST vs LINEAR) y el border mode. Se podria refactorizar extrayendo la preparacion de puntos y triangulacion a una funcion comun, pero la duplicacion actual es menor (~12 lineas) y la claridad del codigo no sufre significativamente.

  **`_warp_triangle_nearest()` -- duplicacion mayor con `warp_triangle()`**: Esta funcion es practicamente identica a `warp_triangle()` excepto por:
  - `cv2.INTER_NEAREST` en lugar de `cv2.INTER_LINEAR`
  - `cv2.BORDER_CONSTANT` en lugar de `cv2.BORDER_REFLECT_101`
  - No maneja imagenes 3D (solo 2D para mascaras)

  Esto podria parametrizarse como argumento de `warp_triangle()`, pero la separacion explicita hace que sea mas dificil introducir bugs al modificar una sin afectar la otra.

  **`add_boundary_points()` -- coordenadas x/y**: Los boundary points usan coordenadas `(x, y)` consistentes con el resto del pipeline. Los 4 corners y 4 midpoints cubren los bordes de la imagen. Nota: el tipo `float64` es explicitamente especificado para evitar problemas de precision en la triangulacion Delaunay.

  **`scale_landmarks_from_centroid()`**: Implementacion simple y correcta. Con `scale=1.05` (el optimo validado), expande los landmarks 5% desde su centroide. Esto agrega un margen alrededor de la region pulmonar que mejora la clasificacion.

  **Funciones no exportadas en `__init__.py`**: `compute_fill_rate` y `warp_mask` son funciones publicas utiles que no estan en `__all__`. Los consumidores las importan directamente: `scripts/analyze_hospital_marks.py` importa `compute_fill_rate`, y `scripts/calculate_pfs_warped.py` importa `warp_mask`.

---

## Resumen de hallazgos

### Fortalezas

1. **Implementacion matematica correcta**: Tanto GPA como warping siguen los algoritmos estandar de la literatura (Gower 1975, Dryden & Mardia 1998). La rotacion via SVD con correccion de reflexion es la solucion analitica exacta.

2. **Robustez**: Proteccion contra formas degeneradas (scale ~0), triangulos degenerados (area < 1e-6), bounding boxes invalidos (w/h <= 0), y excepciones individuales por triangulo que no detienen el proceso completo.

3. **Documentacion**: Docstrings completos estilo Google en todas las funciones publicas. Referencias bibliograficas en el modulo GPA. Tipos de entrada/salida claramente especificados.

4. **Separacion de responsabilidades**: GPA (computar forma canonica) y warping (transformar imagenes) estan limpiamente separados. Las funciones auxiliares son granulares y testables independientemente.

5. **Optimizacion practica**: El warping por parches (bounding box extraction) es significativamente mas rapido que transformar la imagen completa para cada triangulo.

### Debilidades / Areas de mejora

1. **`__init__.py` incompleto**: `compute_fill_rate` y `warp_mask` no estan re-exportadas. Inconsistencia menor en la API publica del paquete.

2. **Duplicacion `warp_triangle` / `_warp_triangle_nearest`**: ~50 lineas de logica casi identica. Podria parametrizarse con argumentos `interpolation` y `border_mode`, pero la separacion actual es aceptable para mantenibilidad.

3. **`load_canonical_shape` no existe en `gpa.py`**: Un script (`scripts/glass_box_visualizations/block_a_pipeline.py`) intenta importar esta funcion desde `src_v2.processing.gpa`, pero no esta definida. Esto es un bug del script consumidor, no del modulo.

4. **Sin tests unitarios**: No existe directorio `tests/` en el proyecto. Las funciones de GPA y warping son altamente testeables (operaciones matematicas puras con entradas/salidas bien definidas). CLAUDE.md menciona `pytest tests/` pero el directorio no existe actualmente.

5. **`compute_fill_rate()` sensible a contenido negro real**: Cuenta pixeles con valor exactamente 0, lo cual podria no distinguir entre fondo artificial (resultado del warping) y contenido naturalmente oscuro de radiografias.

### Tabla resumen

| Archivo | Lineas | Tamano | Importancia | Funciones |
|---------|--------|--------|-------------|-----------|
| `__init__.py` | 42 | 970 B | MEDIO | 0 propias, 12 re-exportadas |
| `gpa.py` | 305 | 9.0 KB | CRITICO | 8 funciones publicas |
| `warp.py` | 448 | 13.1 KB | CRITICO | 8 publicas + 2 privadas |
| **Total** | **795** | **23.1 KB** | **CRITICO** | **16 publicas + 2 privadas** |

### Veredicto global del modulo

**CRITICO** -- Este modulo es el nucleo algoritmico del proyecto. Contiene la matematica que hace posible la normalizacion geometrica que lleva la accuracy del clasificador de ~96% (sin warping) a 99.10% (con warping). La implementacion es solida, bien documentada, y sigue fielmente los algoritmos de la literatura. Las debilidades identificadas son menores (duplicacion de codigo, API incompleta en `__init__.py`). La unica carencia significativa es la falta de tests unitarios para funciones que son ideales para testing automatizado.
