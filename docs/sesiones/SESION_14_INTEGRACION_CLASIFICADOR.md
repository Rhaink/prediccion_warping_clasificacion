# Sesion 14: Integracion del Clasificador COVID-19 con Pipeline de Warping

**Fecha**: 2025-12-07
**Rama**: `feature/restructure-production`

## Objetivo

Integrar el clasificador COVID-19 con el pipeline de landmarks/warping existente, agregando comandos CLI para clasificacion end-to-end.

## Cambios Realizados

### Archivos Creados

| Archivo | Descripcion |
|---------|-------------|
| `src_v2/models/classifier.py` | Modulo del clasificador (ImageClassifier, create_classifier, transforms) |
| `tests/test_classifier.py` | 36 tests para clasificador y comandos CLI |

### Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `src_v2/models/__init__.py` | Exporta nuevas clases del clasificador |
| `src_v2/constants.py` | Agrega `CLASSIFIER_CLASSES`, `DEFAULT_CLASSIFIER_BACKBONE` |
| `src_v2/cli.py` | 3 nuevos comandos: `classify`, `train-classifier`, `evaluate-classifier` |
| `tests/test_cli.py` | Actualizado para 9 comandos totales |

## Nuevos Comandos CLI

### 1. `classify` - Clasificar imagenes

```bash
# Clasificacion basica (sin warping)
python -m src_v2 classify imagen.png --classifier modelo.pt

# Con warping geometrico
python -m src_v2 classify imagen.png --classifier clf.pt \
    --warp --landmark-model lm.pt

# Con ensemble de landmarks + TTA
python -m src_v2 classify directorio/ --classifier clf.pt \
    --warp --landmark-ensemble m1.pt m2.pt m3.pt m4.pt --tta \
    --output resultados.json
```

**Opciones principales:**
- `--classifier, -clf`: Checkpoint del clasificador (requerido)
- `--warp/--no-warp`: Aplicar normalizacion geometrica
- `--landmark-model, -lm`: Modelo individual de landmarks
- `--landmark-ensemble, -le`: Modelos de ensemble para landmarks
- `--tta/--no-tta`: Test-Time Augmentation para landmarks
- `--clahe/--no-clahe`: Mejora de contraste CLAHE
- `--output, -o`: Guardar resultados en JSON

### 2. `train-classifier` - Entrenar clasificador

```bash
python -m src_v2 train-classifier outputs/warped_dataset \
    --backbone resnet18 \
    --epochs 50 \
    --batch-size 32 \
    --output-dir outputs/classifier
```

**Estructura de dataset requerida:**
```
data_dir/
├── train/
│   ├── COVID/
│   ├── Normal/
│   └── Viral_Pneumonia/
├── val/
└── test/
```

**Opciones principales:**
- `--backbone, -b`: resnet18 o efficientnet_b0
- `--epochs, -e`: Numero de epocas
- `--batch-size`: Tamano de batch
- `--lr`: Learning rate
- `--class-weights/--no-class-weights`: Balanceo de clases
- `--patience`: Early stopping patience

### 3. `evaluate-classifier` - Evaluar clasificador

```bash
python -m src_v2 evaluate-classifier modelo.pt \
    --data-dir outputs/warped_dataset \
    --split test \
    --output resultados.json
```

**Opciones principales:**
- `--data-dir, -d`: Directorio del dataset
- `--split, -s`: test, val, o all
- `--output, -o`: Guardar resultados en JSON

## Modulo del Clasificador

### Clase ImageClassifier

```python
from src_v2.models import ImageClassifier, create_classifier

# Crear modelo
model = ImageClassifier(
    backbone="resnet18",  # o "efficientnet_b0"
    num_classes=3,
    pretrained=True,
    dropout=0.3
)

# Cargar desde checkpoint
model = create_classifier(checkpoint="modelo.pt", device="cuda")

# Inferencia
probs = model.predict_proba(tensor)  # Probabilidades
pred = model.predict(tensor)          # Clase predicha
```

### Funciones Auxiliares

```python
from src_v2.models import (
    get_classifier_transforms,  # Transforms train/eval
    get_class_weights,          # Pesos para desbalance
    load_classifier_checkpoint, # Carga con metadata
)

# Transforms
train_transform = get_classifier_transforms(train=True)
eval_transform = get_classifier_transforms(train=False)

# Class weights
weights = get_class_weights(labels, num_classes=3)
```

## Compatibilidad de Checkpoints

El sistema soporta dos formatos de checkpoint:

1. **Formato nuevo** (CLI): Claves con prefijo `backbone.`
2. **Formato antiguo** (scripts/train_classifier.py): Claves sin prefijo

La conversion es automatica al cargar.

## Tests

**36 tests del clasificador:**

- `TestImageClassifier`: Creacion, forward pass, predict
- `TestCreateClassifier`: Factory function con/sin checkpoint
- `TestCheckpointCompatibility`: Formatos antiguo/nuevo, preservacion de pesos
- `TestClassifierTransforms`: Augmentaciones, tamaños
- `TestGrayscaleToRGB`: Conversion de canales
- `TestClassWeights`: Calculo de pesos
- `TestClassifyCommand`: CLI classify
- `TestTrainClassifierCommand`: CLI train-classifier
- `TestEvaluateClassifierCommand`: CLI evaluate-classifier

**Total tests del proyecto: 224**

## Verificacion de Comandos

### evaluate-classifier
```
Accuracy: 87.50%
F1 Macro: 0.8823
F1 Weighted: 0.8742
```
Reproduce exactamente los resultados del clasificador existente.

### classify
- Imagen COVID clasificada con 98.85% confianza
- Directorio de 31 imagenes: 87.1% recall (coincide con matriz de confusion)

### train-classifier
- Entrenamiento funcional con early stopping
- Guarda checkpoints compatibles

## Proximos Pasos (Sesion 15)

1. Entrenar clasificador sobre `outputs/full_warped_dataset/` (15,153 imagenes)
2. Comparar clasificacion warped vs original
3. Probar comando `classify --warp` con ensemble de landmarks
4. Evaluar impacto de normalizacion geometrica en metricas

## Dependencias

- PyTorch >= 2.0
- torchvision
- scikit-learn
- typer
- tqdm

## Referencias

- Clasificador base: `scripts/train_classifier.py`
- Warping: `scripts/piecewise_affine_warp.py`
- Forma canonica: `outputs/shape_analysis/canonical_shape_gpa.json`
