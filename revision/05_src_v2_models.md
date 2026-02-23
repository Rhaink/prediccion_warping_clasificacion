# 05. src_v2 Models Module

Analisis de las arquitecturas de redes neuronales: deteccion de landmarks, clasificacion y funciones de perdida.

**Archivos analizados**: 5

---

## Resumen del Modulo

El modulo `src_v2/models/` contiene las definiciones de las arquitecturas neuronales del proyecto y las funciones de perdida especializadas. Implementa tres modelos: el modelo principal de deteccion de landmarks (ResNet18Landmarks), un clasificador CNN multi-backbone (ImageClassifier), y un modelo jerarquico experimental (HierarchicalLandmarkModel). Ademas incluye funciones de perdida geometricamente informadas (Wing Loss, alineacion central, simetria suave).

**Total**: 1564 lineas, 60 KB en 5 archivos.

---

## Analisis Archivo por Archivo

### 1. `__init__.py`
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/models/__init__.py`
- **Lineas/Tamano**: 39 lineas / 4 KB
- **Proposito**: Punto de entrada del modulo que centraliza las exportaciones publicas. Define `__all__` para control explicito de la API publica del paquete de modelos.
- **Contenido clave**:
  - Importa y re-exporta desde `resnet_landmark`: `ResNet18Landmarks`, `create_model`
  - Importa y re-exporta desde `losses`: `WingLoss`, `WeightedWingLoss`, `CentralAlignmentLoss`, `SoftSymmetryLoss`, `CombinedLandmarkLoss`
  - Importa y re-exporta desde `classifier`: `ImageClassifier`, `create_classifier`, `get_classifier_transforms`, `get_class_weights`, `load_classifier_checkpoint`, `GrayscaleToRGB`
  - **No exporta** `HierarchicalLandmarkModel` ni `AxisLoss` (coherente con su estatus experimental)
  - **No exporta** `get_landmark_weights` ni `count_parameters` (utilidades auxiliares accedidas via import directo)
- **Dependencias**:
  - Importa de: `resnet_landmark`, `losses`, `classifier` (sub-modulos propios)
  - Lo importan: `src_v2/cli.py` (>25 imports), `src_v2/evaluation/ensemble.py`, `src_v2/visualization/`, `src_v2/gui/model_manager.py`, multiples scripts
- **Importancia**: ALTO
- **Justificacion**: Es la interfaz publica del modulo de modelos. Casi todo el codigo del proyecto accede a las arquitecturas a traves de este archivo. Bien estructurado con `__all__` explicito. La exclusion de `hierarchical` del __init__ es una decision correcta dado que es experimental.

---

### 2. `resnet_landmark.py`
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/models/resnet_landmark.py`
- **Lineas/Tamano**: 325 lineas / 12 KB
- **Proposito**: Define la arquitectura principal del modelo de deteccion de landmarks anatomicos: un ResNet-18 preentrenado con cabeza de regresion para predecir 15 landmarks (30 coordenadas) normalizados a [0,1].
- **Contenido clave**:
  - **Clase `CoordinateAttention`** (lineas 23-65): Modulo de atencion posicional (CVPR 2021) que captura dependencias de largo alcance con informacion espacial. Opera en features 2D separando pooling horizontal y vertical, luego genera mapas de atencion por canal. Se inserta entre layer4 y avgpool.
  - **Clase `ResNet18Landmarks`** (lineas 68-271): Modelo principal:
    - Backbone: ResNet-18 (conv1 a layer4) con pesos ImageNet
    - Coordinate Attention opcional entre backbone y avgpool
    - Dos variantes de cabeza de regresion:
      - Standard: `Flatten -> Dropout -> Linear(512, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 30) -> Sigmoid`
      - Deep head: `Flatten -> Linear(512, 512) -> GroupNorm -> ReLU -> Dropout -> Linear(512, hidden_dim) -> GroupNorm -> ReLU -> Dropout -> Linear(hidden_dim, 30) -> Sigmoid`
    - Metodos de freeze/unfreeze para entrenamiento en dos fases
    - `get_trainable_params()`: Retorna grupos de parametros diferenciados (features vs head) para LR discriminativo
    - `predict_landmarks()`: Forward + desnormalizacion a coordenadas en pixeles (multiplicar por image_size)
  - **Funcion `create_model()`** (lineas 274-313): Factory con auto-deteccion de GPU
  - **Funcion `count_parameters()`** (lineas 316-325): Cuenta parametros totales vs entrenables
- **Dependencias**:
  - Importa de: `torch`, `torchvision.models`, `src_v2.constants` (NUM_LANDMARKS, DEFAULT_IMAGE_SIZE, BACKBONE_FEATURE_DIM, DEFAULT_HIDDEN_DIM)
  - Lo importan: `src_v2/cli.py`, `src_v2/gui/model_manager.py`, `scripts/train.py`, `scripts/predict_landmarks_dataset.py`, `scripts/visualization/`, `scripts/verify_*.py`, `scripts/glass_box_visualizations/`
- **Importancia**: CRITICO
- **Justificacion**: Es el modelo central del pipeline de landmarks. Todo el pipeline de prediccion-warping-clasificacion depende de este modelo. Los checkpoints validados (3.61 px ensemble error) fueron entrenados con esta arquitectura. La deteccion de arquitectura desde checkpoint en `cli.py::detect_architecture_from_checkpoint()` depende directamente de la estructura de esta clase (busca claves como `coord_attention.*` y `head.9.*` en state_dict).

**Observaciones tecnicas**:
- El uso de `Sigmoid` como activacion final restringe las predicciones a [0,1], lo cual es apropiado dado que los landmarks estan normalizados. Sin embargo, esto implica que landmarks en los bordes de la imagen pueden tener gradientes atenuados por la saturacion de sigmoid.
- La deep head usa `GroupNorm` en vez de `BatchNorm`, decision documentada como mas estable con batches pequenos (batch_size=16). El calculo de `num_groups` para hidden_dim variable (linea 141) tiene un fallback seguro para dimensiones menores a 16.
- El dropout es reducido a la mitad (dropout * 0.5) en la segunda capa del head, patron razonable para evitar sobre-regularizacion en capas mas profundas.
- `get_trainable_params()` agrupa backbone + coord_attention como "features" con un mismo LR, y head como grupo separado. Los nombres `features` y `head` son usados por el trainer para asignar LRs diferenciados en Phase 2.

---

### 3. `classifier.py`
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/models/classifier.py`
- **Lineas/Tamano**: 394 lineas / 12 KB
- **Proposito**: Define el clasificador CNN para COVID-19/Normal/Viral_Pneumonia con soporte para multiples arquitecturas backbone via transfer learning. Incluye utilidades para transforms, pesos de clase, y carga de checkpoints.
- **Contenido clave**:
  - **Clase `GrayscaleToRGB`** (lineas 26-39): Transform que convierte imagenes grayscale a RGB para compatibilidad con backbones ImageNet. Es necesario porque las radiografias son inherentemente monocromo.
  - **Clase `ImageClassifier`** (lineas 42-214):
    - 7 backbones soportados: resnet18, resnet50, efficientnet_b0, densenet121, alexnet, vgg16, mobilenet_v2
    - Cada backbone reemplaza su capa clasificadora final con `Dropout -> Linear(in_features, num_classes)`
    - `forward()`: Retorna logits crudos
    - `predict_proba()`: Aplica softmax para obtener probabilidades
    - `predict()`: Retorna argmax para prediccion de clase
  - **Funcion `create_classifier()`** (lineas 217-287): Factory que soporta creacion desde cero o desde checkpoint. Incluye conversion de formato antiguo (sin prefijo "backbone.") al formato nuevo con prefijo.
  - **Funcion `get_classifier_transforms()`** (lineas 290-330): Transforms para train (con augmentation: flip, rotacion, affine) y eval (solo resize + normalize).
  - **Funcion `get_class_weights()`** (lineas 333-356): Calcula pesos inversamente proporcionales a la frecuencia de clase para balanceo en CrossEntropyLoss.
  - **Funcion `load_classifier_checkpoint()`** (lineas 359-394): Carga modelo con metadatos (backbone, class_names, best_val_f1). Nota: tiene un problema de eficiencia - carga el checkpoint dos veces (una en esta funcion y otra dentro de `create_classifier()`).
- **Dependencias**:
  - Importa de: `torch`, `torchvision.models`, `torchvision.transforms`, `collections.Counter`
  - Lo importan: `src_v2/cli.py` (>15 imports para distintos comandos), `src_v2/evaluation/ensemble.py`, `src_v2/visualization/plot_roc_curves.py`, `src_v2/visualization/plot_failure_cases.py`, `src_v2/gui/model_manager.py`, multiples scripts
- **Importancia**: CRITICO
- **Justificacion**: Es el modelo de clasificacion final del pipeline (99.10% accuracy reportado). Usado extensamente en CLI, evaluacion, visualizacion, y scripts de analisis. Los 7 backbones soportados indican un trabajo de comparacion de arquitecturas (Session 18, 20), aunque ResNet-18 es el default validado.

**Observaciones tecnicas**:
- La inicializacion del backbone tiene codigo repetitivo (patron if/elif para cada backbone). Se podria refactorizar con un diccionario de builders, pero la claridad actual es aceptable dado que solo hay 7 opciones.
- **Bug potencial en `load_classifier_checkpoint()`**: El checkpoint se carga dos veces: una en lineas 373-377 y otra implicitamente dentro de `create_classifier(checkpoint=checkpoint_path)` en linea 388. Esto es ineficiente (dos lecturas de disco del mismo archivo) pero no causa resultados incorrectos. Seria mejor pasar el state_dict ya cargado en vez de el path.
- Las augmentaciones de entrenamiento son moderadas (flip, 10 grados rotacion, 5% translate/scale), apropiadas para radiografias donde grandes distorsiones no son realistas.
- La normalizacion ImageNet se aplica siempre, lo cual es correcto para modelos preentrenados.

---

### 4. `hierarchical.py`
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/models/hierarchical.py`
- **Lineas/Tamano**: 368 lineas / 16 KB
- **Proposito**: Implementa un modelo alternativo que explota la estructura geometrica de los landmarks: primero predice el eje central (L1, L2), luego parametros relativos al eje (posiciones t y distancias perpendiculares) para los demas landmarks. Incluye losses especializados para esta arquitectura.
- **Contenido clave**:
  - **Clase `HierarchicalLandmarkModel`** (lineas 42-259):
    - Backbone compartido: ResNet-18 (mismo que modelo principal)
    - **axis_head**: Predice 4 valores (L1_x, L1_y, L2_x, L2_y) con Sigmoid
    - **relative_head**: Recibe features(512) + axis(4) = 516, predice 18 parametros relativos
      - 3 valores: offsets dt para landmarks centrales (L9, L10, L11)
      - 15 valores: (t, d_left, d_right) para 5 pares bilaterales
    - **`_reconstruct_landmarks()`**: Reconstruye 15 landmarks desde parametros relativos usando vectores de eje y perpendiculares. Usa `tanh` para offsets (rango limitado) y `sigmoid * D_MAX` para distancias (siempre positivas).
    - Usa `GroupNorm` en ambas cabezas (mas estable que BatchNorm con batches pequenos)
    - Metodos `freeze_backbone()`, `get_trainable_params()` para entrenamiento en dos fases
  - **Clase `AxisLoss`** (lineas 262-293): Loss adicional que pondera errores de L1 y L2 (peso=2.0 por defecto) dado que todos los landmarks dependen del eje.
  - **Clase `CentralAlignmentLossHierarchical`** (lineas 296-335): Verifica que L9, L10, L11 esten alineados con el eje. En principio deberia ser ~0 por construccion, sirve como diagnostico.
  - **Bloque `__main__`** (lineas 338-368): Test simple de forward pass con verificacion de alineacion de L10.
- **Dependencias**:
  - Importa de: `torch`, `torchvision.models`, `src_v2.constants` (SYMMETRIC_PAIRS, CENTRAL_LANDMARKS, multiples constantes HIERARCHICAL_*), `src_v2.utils.geometry.compute_perpendicular_vector`
  - Lo importan: `scripts/train_hierarchical.py`, `scripts/archive/test_hierarchical_forward.py`
- **Importancia**: BAJO
- **Justificacion**: Es un enfoque alternativo experimental que NO se usa en el pipeline principal. No esta exportado en `__init__.py`. Solo lo importan un script de entrenamiento dedicado (`scripts/train_hierarchical.py`) y un test archivado. No hay checkpoints validados ni metricas reportadas en `GROUND_TRUTH.json` para este modelo. Los resultados del proyecto se basan exclusivamente en `ResNet18Landmarks`.

**Observaciones tecnicas**:
- La idea de parametrizar landmarks relativamente al eje es geometricamente elegante y aprovecha la estructura conocida del etiquetado (t=0.25, 0.50, 0.75 para centrales; distancias perpendiculares para bilaterales).
- Sin embargo, la reconstruccion diferenciable en `_reconstruct_landmarks()` introduce complejidad: la multiplicacion `d * perp_unit * axis_len` puede amplificar errores pequenos en la prediccion del eje.
- Las constantes HIERARCHICAL_* en `constants.py` (DT_SCALE=0.1, T_SCALE=0.2, D_MAX=0.7) sugieren que se requirio ajuste cuidadoso de rangos.
- `get_trainable_params()` tiene una interfaz ligeramente diferente al modelo principal: acepta LRs directamente en vez de retornar solo los grupos con nombres. Esto podria causar incompatibilidad con el `LandmarkTrainer` si se intentara usarlo alli.
- La clase `CentralAlignmentLossHierarchical` es redundante con `CentralAlignmentLoss` en `losses.py` - ambas calculan la misma distancia perpendicular al eje, con implementacion casi identica.

---

### 5. `losses.py`
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/models/losses.py`
- **Lineas/Tamano**: 438 lineas / 16 KB
- **Proposito**: Define funciones de perdida especializadas para landmark prediction que incorporan conocimiento geometrico del dominio: Wing Loss (robusta a outliers), alineacion central, simetria suave, y su combinacion ponderada.
- **Contenido clave**:
  - **Clase `WingLoss`** (lineas 30-85):
    - Implementacion del Wing Loss de CVPR 2018
    - Logaritmico para errores < omega (gradientes mayores para errores pequenos)
    - Lineal para errores >= omega (estable para errores grandes)
    - Parametro `normalized=True` escala omega y epsilon automaticamente dividiendo por image_size (224), convirtiendo parametros en pixeles a espacio [0,1]
    - Constante C asegura continuidad en el punto de transicion
  - **Clase `WeightedWingLoss`** (lineas 88-160):
    - Extiende WingLoss con pesos por landmark
    - Reshape a (B, 15, 2), promedia sobre coordenadas x,y, luego aplica pesos
    - Los pesos se registran como buffer (no parametro optimizable)
  - **Clase `CentralAlignmentLoss`** (lineas 163-224):
    - Penaliza distancia perpendicular de L9, L10, L11 al eje L1-L2
    - Calcula proyeccion ortogonal y mide componente perpendicular
    - Opera en espacio normalizado [0,1], no escala a pixeles
    - Basado en hallazgo de que en GT, centrales estan a solo ~1.3 px del eje
  - **Clase `SoftSymmetryLoss`** (lineas 227-295):
    - Penaliza SOLO asimetrias que excedan un margen (6 px default)
    - Formula: `max(0, |d_izq| - |d_der| - margin)^2`
    - Importante: el GT tiene asimetria natural de 5.5-7.9 px, por lo que forzar simetria perfecta seria incorrecto
    - Usa `compute_perpendicular_vector()` centralizada de `utils.geometry`
  - **Clase `CombinedLandmarkLoss`** (lineas 298-374):
    - Combina: `total = wing + alpha * central + beta * symmetry`
    - Pesos por defecto: alpha=1.0 (central), beta=0.5 (symmetry)
    - Retorna diccionario con componentes individuales para logging
  - **Funcion `get_landmark_weights()`** (lineas 377-438):
    - 3 estrategias: 'uniform', 'inverse_variance', 'custom'
    - 'inverse_variance': Basado en variabilidad sigma del GT (L14/L15 mas dificiles, menor peso)
    - 'custom': Pesos heuristicos enfocados en landmarks criticos (L14/L15 con peso 2.0)
- **Dependencias**:
  - Importa de: `torch`, `numpy`, `src_v2.constants` (SYMMETRIC_PAIRS, CENTRAL_LANDMARKS, DEFAULT_IMAGE_SIZE), `src_v2.utils.geometry.compute_perpendicular_vector`
  - Lo importan: `src_v2/cli.py` (para train-landmarks y variantes), `scripts/train.py`, `scripts/train_hierarchical.py`, `src_v2/models/__init__.py`
- **Importancia**: CRITICO
- **Justificacion**: Las funciones de perdida son fundamentales para el entrenamiento del modelo de landmarks. El Wing Loss es la loss principal, y las penalizaciones geometricas (central alignment, soft symmetry) codifican conocimiento del dominio que mejora la calidad de las predicciones. El `CombinedLandmarkLoss` es la loss compuesta usada en el pipeline principal via CLI.

**Observaciones tecnicas**:
- La normalizacion automatica de omega/epsilon es un buen diseno defensivo: permite definir parametros en pixeles intuitivos (omega=10px) y escalarlos automaticamente al espacio [0,1] del modelo.
- El `SoftSymmetryLoss` con margen es una decision bien fundamentada. La asimetria natural del GT (5.5-7.9 px) hace que un loss de simetria estricto sea contraproducente. El margen de 6 px esta dentro de este rango.
- `CombinedLandmarkLoss.forward()` retorna un diccionario en vez de un tensor escalar, lo cual requiere que el caller extraiga `['total']` para backward. Esto es intencional para permitir logging detallado de cada componente.
- Los pesos de `get_landmark_weights('inverse_variance')` parecen calculados manualmente (valores hardcoded como 1.16, 0.79, etc.). No hay una funcion que los derive automaticamente de los datos, pero se documenta su fuente en el docstring.
- El import de `numpy` se usa unicamente para `np.log()` en el calculo de la constante C (lineas 63, 121). Podria reemplazarse con `math.log()` para eliminar la dependencia de numpy en este modulo.

---

## Diagrama de Dependencias

```
src_v2/constants.py
    |
    v
src_v2/utils/geometry.py
    |
    v
+---+-------------------+------------------+
|                        |                  |
v                        v                  v
resnet_landmark.py    losses.py       hierarchical.py
|                     |                  |
|                     |                  | (NO exportado)
v                     v                  v
__init__.py  <--------+           scripts/train_hierarchical.py
    |
    v
classifier.py
    |
    v
__init__.py
    |
    v
src_v2/cli.py, src_v2/evaluation/, src_v2/gui/, scripts/
```

## Tabla Resumen

| Archivo | Lineas | Importancia | Uso en pipeline |
|---------|--------|-------------|-----------------|
| `__init__.py` | 39 | ALTO | Interfaz publica del modulo |
| `resnet_landmark.py` | 325 | CRITICO | Modelo principal de landmarks (3.61 px) |
| `classifier.py` | 394 | CRITICO | Clasificador COVID-19 (99.10% acc) |
| `hierarchical.py` | 368 | BAJO | Alternativa experimental, no usada en produccion |
| `losses.py` | 438 | CRITICO | Wing Loss + penalizaciones geometricas |

## Observaciones Generales

### Fortalezas

1. **Diseno modular limpio**: Cada archivo tiene un proposito bien definido. La separacion entre modelos y losses es correcta.
2. **Conocimiento del dominio codificado**: Las losses incorporan propiedades geometricas verificadas experimentalmente (alineacion central, asimetria natural) en vez de usar solo MSE/L1 genericas.
3. **Flexibilidad arquitectural**: ResNet18Landmarks soporta multiples configuraciones (coord_attention, deep_head, hidden_dim variable) detectables automaticamente desde checkpoints. El clasificador soporta 7 backbones.
4. **Entrenamiento en dos fases**: El diseno de freeze/unfreeze con LR diferenciado es un patron probado para transfer learning con datos limitados.
5. **Normalizacion consistente**: Tanto el modelo (output sigmoid en [0,1]) como las losses (normalizacion automatica de omega/epsilon) trabajan en el mismo espacio normalizado.
6. **Documentacion inline**: Los docstrings son informativos y las notas sobre decisiones de diseno (por que GroupNorm, por que margen de 6 px) facilitan comprension.

### Debilidades y Oportunidades

1. **Doble carga de checkpoint en `load_classifier_checkpoint()`**: La funcion carga el archivo .pt, extrae metadatos, y luego llama a `create_classifier(checkpoint=path)` que vuelve a cargar el mismo archivo. Deberia pasar el state_dict directamente.
2. **Codigo repetitivo en `classifier.py`**: La inicializacion de los 7 backbones sigue un patron if/elif con ligeras variaciones. Un diccionario de configuracion reduciria duplicacion.
3. **Modelo jerarquico sin integracion**: `hierarchical.py` no esta integrado con el `LandmarkTrainer` (interfaz de `get_trainable_params` incompatible) ni exportado en `__init__.py`. Si no se planea continuar su desarrollo, podria moverse a `scripts/archive/`.
4. **`CentralAlignmentLossHierarchical` duplicada**: La clase en `hierarchical.py` es funcionalmente identica a `CentralAlignmentLoss` en `losses.py`. Deberia eliminarse o reutilizar la de losses.
5. **Dependencia de numpy en losses.py**: El `import numpy` se usa solo para `np.log()` en 2 lineas. `math.log()` seria suficiente y eliminaria la dependencia.
6. **Sin tests unitarios**: No existe directorio `tests/`. Dado que el CLAUDE.md menciona `pytest tests/ -v`, esto sugiere que los tests existieron pero fueron eliminados o no se incluyen en el repo actual. Las funciones de loss y los modelos se beneficiarian enormemente de tests unitarios (verificar shapes de salida, rangos de valores, gradient flow).
7. **Hardcoded landmark count**: El numero 15 aparece hardcoded en varias partes (reshape a `(B, 15, 2)` en losses, pesos de 15 elementos). Deberia usarse `NUM_LANDMARKS` de constants.py para consistencia.

### Evaluacion de Riesgo

- **resnet_landmark.py**: Sin riesgo. Modelo maduro con resultados validados.
- **classifier.py**: Riesgo bajo. Bug de doble carga no afecta resultados, solo eficiencia.
- **losses.py**: Sin riesgo. Las losses estan correctamente implementadas y documentadas.
- **hierarchical.py**: Sin riesgo para el pipeline (no se usa). Riesgo de confusion si alguien intenta integrarlo sin adaptar la interfaz.
