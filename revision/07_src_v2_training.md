# 07. src_v2 Training Module

Analisis del sistema de entrenamiento: trainer de dos fases con callbacks.

**Archivos analizados**: 3

---

## Resumen del modulo

El modulo `src_v2/training/` implementa el pipeline de entrenamiento para los modelos de deteccion de landmarks. Su diseno central es un entrenamiento en **dos fases** (transfer learning clasico):

- **Fase 1**: Backbone congelado (+ Coordinate Attention congelado). Solo se entrena la cabeza de regresion. Epocas cortas (15), learning rate alto (1e-3).
- **Fase 2**: Fine-tuning completo con learning rates diferenciados. Backbone + CoordAttention usan LR bajo (2e-5), cabeza usa LR alto (2e-4). Se aplica CosineAnnealingLR.

Este esquema ha producido modelos con 3.61 px de error en ensemble (validado en GROUND_TRUTH.json).

---

## Analisis archivo por archivo

### 1. __init__.py

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/training/__init__.py`
- **Lineas/Tamano**: 13 lineas / 267 bytes
- **Proposito**: Exporta las clases publicas del modulo de entrenamiento, proporcionando una interfaz limpia para imports.
- **Contenido clave**:
  - Re-exporta `LandmarkTrainer` desde `trainer.py`
  - Re-exporta `EarlyStopping`, `ModelCheckpoint`, `LRSchedulerCallback` desde `callbacks.py`
  - Define `__all__` con las 4 clases publicas
- **Dependencias**:
  - Importa de: `.trainer`, `.callbacks` (modulos hermanos)
  - Importado por: Ningun archivo lo importa directamente via `from src_v2.training import ...`; los consumidores (`cli.py`, `scripts/train.py`) importan directamente desde `src_v2.training.trainer`
- **Importancia**: BAJO
- **Justificacion**: Archivo estandar de paquete Python. Correcto y limpio, aunque en la practica ningun consumidor externo usa los imports del `__init__.py` (importan directo de los submodulos). No tiene defectos.

---

### 2. trainer.py (LandmarkTrainer)

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/training/trainer.py`
- **Lineas/Tamano**: 435 lineas / 13.6 KB
- **Proposito**: Implementa el trainer de dos fases para entrenamiento de modelos de deteccion de landmarks con backbone congelado/descongelado, metricas en pixeles, y orquestacion de callbacks.
- **Contenido clave**:
  - **Clase `LandmarkTrainer`** (linea 24-435):
    - `__init__()` (31-57): Recibe modelo, device, directorio de checkpoints, tamano de imagen. Crea historial vacio.
    - `compute_pixel_error()` (59-80): Convierte predicciones normalizadas [0,1] a pixeles (multiplicando por `image_size=224`) y calcula error euclidiano medio sobre los 15 landmarks. Hardcoded para 15 landmarks y 2 coordenadas (reshape a B,15,2).
    - `train_epoch()` (82-143): Loop de entrenamiento para una epoca. Maneja criterios que retornan dict (como `CombinedLandmarkLoss`) o escalar. Muestra progreso con tqdm. Soporta `scheduler_callback.step_batch()` opcional.
    - `validate()` (145-187): Loop de validacion con `@torch.no_grad()`. Misma logica de loss que `train_epoch()`.
    - `train_phase1()` (189-274): Fase 1 completa. Congela backbone via `freeze_all_except_head()` (o fallback a `freeze_backbone()`). Usa Adam con LR=1e-3. Instancia `EarlyStopping(patience=5)` y `ModelCheckpoint` para val_error_px. Al final carga el mejor modelo.
    - `train_phase2()` (276-371): Fase 2 completa. Descongela todo via `unfreeze_all()`. Usa Adam con param_groups diferenciados (backbone_lr, head_lr). Aplica `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`. Misma estructura de callbacks.
    - `train_full()` (373-418): Orquestador que ejecuta fase1 seguida de fase2. Retorna historiales combinados como dict `{'phase1': ..., 'phase2': ...}`.
    - `save_model()` (420-427): Guarda modelo con `model_state_dict` y `history`. Nota: el parametro `include_optimizer` esta declarado pero **nunca se usa** en el cuerpo del metodo.
    - `load_model()` (429-435): Carga modelo con `weights_only=False`.
- **Dependencias**:
  - Importa de: `torch`, `torch.nn`, `torch.optim`, `torch.optim.lr_scheduler.CosineAnnealingLR`, `tqdm`, `src_v2.constants.DEFAULT_IMAGE_SIZE`, `.callbacks` (EarlyStopping, ModelCheckpoint, LRSchedulerCallback)
  - Depende fuertemente de: `ResNet18Landmarks` (espera metodos `freeze_all_except_head()`, `unfreeze_all()`, `get_trainable_params()`, `head` attribute)
  - Importado por: `src_v2/cli.py` (linea 334), `scripts/train.py` (linea 32)
- **Importancia**: CRITICO
- **Justificacion**: Es el nucleo del entrenamiento de landmarks. Toda la generacion de modelos pasa por esta clase. Los modelos del ensemble (3.61 px) fueron entrenados con este trainer. Sin el no hay pipeline de entrenamiento.

**Observaciones tecnicas**:

1. **Parametro `include_optimizer` no utilizado** (linea 420): `save_model()` declara `include_optimizer: bool = False` pero el cuerpo del metodo nunca lo consulta. El optimizer state nunca se guarda via este metodo (aunque si se guarda via `ModelCheckpoint`). Esto es un defecto menor -- si alguien pasa `include_optimizer=True` esperando que funcione, no tendra efecto.

2. **Historial de instancia vs historiales de fase**: La clase tiene `self.history` (linea 51-57) que nunca se actualiza durante `train_phase1/2`. Cada fase crea su propio historial local y lo retorna. `self.history` solo se usa en `save_model()` y `load_model()`. Esto significa que despues de `train_full()`, `save_model()` guarda un historial vacio `{'train_loss': [], ...}` en lugar del historial real. Los consumidores (cli.py linea 526) no se ven afectados porque el modelo guardado es el state_dict del mejor checkpoint (cargado por `checkpoint.load_best()`), pero el historial en el archivo guardado esta incorrecto.

3. **Logica duplicada de criterion handling**: Las lineas 111-118 (train_epoch) y 169-176 (validate) tienen la misma logica para manejar criterios que retornan dict vs escalar. Podria extraerse a un metodo privado `_compute_loss()`.

4. **Metricas de batch vs muestra**: `compute_pixel_error()` calcula `errors.mean()` sobre todas las dimensiones (batch y landmarks), lo cual da un promedio por-landmark-por-imagen que es la metrica estandar para este dominio. Sin embargo, `train_epoch()` y `validate()` acumulan `total_error` y dividen por `num_batches`, lo que es correcto solo si todos los batches tienen el mismo tamano. El ultimo batch de una epoca puede tener menor tamano, introduciendo un sesgo leve. En la practica este sesgo es minimo.

5. **Sin gradient clipping**: No se aplica gradient clipping en ninguna fase. Para Wing Loss con su gradiente logaritmico esto no es problematico, pero podria ser util para estabilidad con otros criterios.

6. **Scheduler en Phase 2 pero no en Phase 1**: Phase 1 usa Adam plano sin scheduler, lo cual es razonable dado que son solo 15 epocas. Phase 2 usa CosineAnnealingLR correctamente.

---

### 3. callbacks.py

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/training/callbacks.py`
- **Lineas/Tamano**: 240 lineas / 6.7 KB
- **Proposito**: Implementa tres callbacks reutilizables para el loop de entrenamiento: parada anticipada, guardado de checkpoints, y wrapper de scheduler.
- **Contenido clave**:
  - **Clase `EarlyStopping`** (linea 17-91):
    - Monitorea una metrica y cuenta epocas sin mejora
    - Soporta modos 'min' y 'max'
    - `min_delta` para umbral de mejora minima
    - `__call__(score, epoch)` retorna `True` cuando debe parar
    - `reset()` para reiniciar estado entre fases
    - Logging informativo cuando no hay mejora (muestra contador, mejor score, epoca)
  - **Clase `ModelCheckpoint`** (linea 94-205):
    - Guarda checkpoints cuando la metrica monitoreada mejora
    - Incluye `model_state_dict`, `optimizer_state_dict`, `metrics`, `best_score` y `epoch` en el checkpoint
    - Nombres de archivo con formato: `checkpoint_epoch{NNN}_{metrica}{valor}_{timestamp}.pt`
    - `save_best_only=True` por defecto (solo guarda cuando mejora)
    - `load_best()` carga el mejor checkpoint al finalizar
    - No implementa limpieza de checkpoints antiguos (se acumulan en disco)
  - **Clase `LRSchedulerCallback`** (linea 208-240):
    - Wrapper generico para schedulers de PyTorch
    - Soporta step por 'epoch' o 'batch'
    - Manejo especial para `ReduceLROnPlateau` (requiere metrica)
    - `get_last_lr()` para consultar LR actual
- **Dependencias**:
  - Importa de: `torch`, `numpy` (numpy importado pero **no utilizado**), `datetime`, `pathlib`, `logging`
  - Importado por: `src_v2/training/trainer.py`
- **Importancia**: ALTO
- **Justificacion**: Los callbacks son esenciales para el entrenamiento estable. `EarlyStopping` previene sobreajuste, `ModelCheckpoint` asegura que no se pierda el mejor modelo, y `LRSchedulerCallback` integra CosineAnnealingLR en Phase 2. Son componentes criticos del pipeline pero son componentes de soporte, no la logica central.

**Observaciones tecnicas**:

1. **Import no utilizado**: `numpy` se importa en linea 11 pero no se usa en ningun lugar del archivo.

2. **Checkpoints no se limpian**: `ModelCheckpoint` guarda un nuevo archivo cada vez que la metrica mejora, pero nunca borra el checkpoint anterior. Para entrenamientos largos con muchas mejoras, esto puede acumular muchos archivos (como se vio en la limpieza de 133 GB documentada en CHECKPOINTS_CLEANUP_REPORT.md). Una opcion `max_keep` seria util.

3. **EarlyStopping y ModelCheckpoint no estan sincronizados**: Ambos monitorean independientemente si la metrica mejora. EarlyStopping usa `val_error_px` y ModelCheckpoint tambien usa `val_error_px`, pero tienen logica de mejora separada (EarlyStopping con `min_delta`, ModelCheckpoint sin delta). En la practica esto funciona porque ambos ven los mismos valores.

4. **`load_best()` sin map_location**: La linea 199 (`torch.load(self.best_path, weights_only=False)`) no especifica `map_location`. Si se entrena en GPU y se carga en CPU (poco probable en el mismo run, pero posible), esto podria fallar. En contraste, `trainer.py:load_model()` si usa `map_location=self.device`.

5. **Timestamp en nombres de checkpoint**: El formato `{timestamp}` en el nombre del archivo asegura unicidad pero complica la limpieza manual y el ordenamiento por metrica.

6. **LRSchedulerCallback.step_epoch() con ReduceLROnPlateau**: Solo busca `val_loss` en las metricas (linea 229), pero el trainer pasa `all_metrics` que contiene tanto `loss` como `val_loss`, asi que funciona correctamente. Sin embargo, si se quisiera usar ReduceLROnPlateau con otra metrica, requeriria modificacion.

---

## Diagrama de flujo del entrenamiento

```
train_full()
  |
  +-- train_phase1(epochs=15, lr=1e-3, patience=5)
  |     |
  |     +-- model.freeze_all_except_head()   # backbone + coord_attn frozen
  |     +-- optimizer = Adam(model.head.parameters(), lr=1e-3)
  |     +-- EarlyStopping(patience=5)
  |     +-- ModelCheckpoint(save_dir/phase1, monitor=val_error_px)
  |     +-- for epoch in range(15):
  |     |     +-- train_epoch() -> {loss, error_px}
  |     |     +-- validate()    -> {val_loss, val_error_px}
  |     |     +-- checkpoint(model, optimizer, epoch, metrics)
  |     |     +-- early_stopping(val_error_px) -> stop?
  |     +-- checkpoint.load_best(model)
  |
  +-- train_phase2(epochs=100, backbone_lr=2e-5, head_lr=2e-4, patience=10)
        |
        +-- model.unfreeze_all()             # todo descongelado
        +-- optimizer = Adam([
        |     {features: backbone_lr=2e-5},
        |     {head: head_lr=2e-4}
        |   ])
        +-- CosineAnnealingLR(T_max=100, eta_min=1e-6)
        +-- EarlyStopping(patience=10)
        +-- ModelCheckpoint(save_dir/phase2, monitor=val_error_px)
        +-- for epoch in range(100):
        |     +-- train_epoch() -> {loss, error_px}
        |     +-- validate()    -> {val_loss, val_error_px}
        |     +-- checkpoint(model, optimizer, epoch, metrics)
        |     +-- scheduler.step()
        |     +-- early_stopping(val_error_px) -> stop?
        +-- checkpoint.load_best(model)
```

---

## Relacion entre componentes

```
cli.py / scripts/train.py
  |
  +-- LandmarkTrainer(model, device, save_dir, image_size)
        |
        +-- ResNet18Landmarks (modelo)
        |     +-- .freeze_all_except_head()
        |     +-- .unfreeze_all()
        |     +-- .get_trainable_params()
        |     +-- .head (nn.Module)
        |
        +-- EarlyStopping (callback)
        +-- ModelCheckpoint (callback)
        +-- LRSchedulerCallback (callback)
        |
        +-- WingLoss / CombinedLandmarkLoss (criterion)
```

---

## Resumen de hallazgos

| # | Tipo | Archivo | Descripcion |
|---|------|---------|-------------|
| 1 | Bug menor | trainer.py:420 | `include_optimizer` parametro declarado pero nunca usado |
| 2 | Bug medio | trainer.py:51-57,420-427 | `self.history` nunca se actualiza; `save_model()` guarda historial vacio |
| 3 | Codigo duplicado | trainer.py:111-118,169-176 | Logica de criterion dict/escalar repetida en train_epoch y validate |
| 4 | Import innecesario | callbacks.py:11 | `import numpy as np` no se usa |
| 5 | Acumulacion de archivos | callbacks.py:94 | ModelCheckpoint no limpia checkpoints anteriores |
| 6 | Potencial error | callbacks.py:199 | `torch.load()` sin `map_location` en `load_best()` |
| 7 | Sesgo menor | trainer.py:140-143 | Promedio de metricas por batch (no por muestra), sesgo en ultimo batch |

**Calidad general**: El modulo esta bien estructurado, es legible, y cumple su proposito correctamente. Los defectos identificados son menores y no afectan la funcionalidad en el flujo normal de uso. El diseno de dos fases con callbacks es limpio y extensible. La separacion entre trainer y callbacks sigue buenas practicas de diseno.

**Cobertura de tests**: No existen tests unitarios para este modulo (el directorio `tests/` esta vacio o no existe). Dado que el trainer es critico para reproducir resultados, tests al menos para `compute_pixel_error()`, `EarlyStopping`, y `ModelCheckpoint` serian valiosos.
