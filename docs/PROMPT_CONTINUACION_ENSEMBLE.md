# Prompt para Continuar: Implementación de Ensemble + TTA para Clasificador

## Contexto del Proyecto

Estoy trabajando en mi tesis de maestría sobre detección de COVID-19 en radiografías de tórax usando normalización geométrica mediante landmarks. El proyecto tiene tres componentes principales:

1. **Landmark Detection** (15 landmarks del contorno pulmonar) - ✅ COMPLETADO
   - Error: 3.61 px (ensemble de 4 modelos)
   - Usa TTA (Test-Time Augmentation)
2. **Geometric Normalization** (Piecewise Affine Warping) - ✅ COMPLETADO
   - Basado en GPA (Generalized Procrustes Analysis)
3. **Classification** (ResNet-18) - ✅ ENTRENADO, pero se puede mejorar
   - 5 modelos entrenados con validación cruzada (k=5)
   - Accuracy actual en test set: **97.68% ± 0.16%**

## Problema Identificado y Corregido Recientemente

Acabamos de descubrir que los modelos de validación cruzada **nunca habían sido evaluados en el test set**. Solo se habían evaluado en sus validation folds durante el entrenamiento.

**Trabajo completado:**
- ✅ Evaluamos los 5 modelos CV en el test set fijo (1,895 imágenes)
- ✅ Actualizamos scripts de generación de figuras
- ✅ Regeneramos figuras con métricas correctas
- ✅ Corregimos capítulo LaTeX con métricas del test set

**Archivos importantes generados:**
```
outputs/classifier_cv/fold_01/test_results.json  # Fold 1 test metrics
outputs/classifier_cv/fold_02/test_results.json  # Fold 2 test metrics
outputs/classifier_cv/fold_03/test_results.json  # Fold 3 test metrics
outputs/classifier_cv/fold_04/test_results.json  # Fold 4 test metrics
outputs/classifier_cv/fold_05/test_results.json  # Fold 5 test metrics (mejor: 97.94%)
outputs/classifier_cv/cross_validation_test_results.json  # Métricas agregadas
```

## Métricas Actuales (Test Set)

```
Accuracy:     97.68% ± 0.16%  (rango: 97.52% - 97.94%)
F1-Macro:     96.47% ± 0.27%
F1-Weighted:  97.67% ± 0.16%

Test set size: 1,895 imágenes
Mejor modelo: Fold 5 (97.94% accuracy)
```

**Distribución del test set:**
- COVID-19: 452 imágenes
- Normal: 1,274 imágenes
- Viral_Pneumonia: 169 imágenes

## Objetivo Actual

**IMPLEMENTAR ENSEMBLE + TTA para subir el test set accuracy de 97.68% → ~98.6%**

Estrategia recomendada (ROI máximo):
1. **Ensemble de los 5 modelos CV** (soft voting) → +0.3 a +0.8 puntos
2. **Test-Time Augmentation (TTA)** → +0.2 a +0.5 puntos
3. **Threshold optimization** (opcional) → +0.1 a +0.3 puntos

**Ganancia esperada combinada:** +0.5 a +1.0 puntos
**Accuracy final esperada:** 98.2% - 98.7%

## Tareas Pendientes

### Fase 1: Implementar Ensemble (PRIORIDAD ALTA)

**Archivo a crear:** `scripts/evaluate_classifier_ensemble.py`

Debe implementar:
```python
def ensemble_predict(models, image, method='soft_voting'):
    """
    Ensemble de múltiples modelos.

    Args:
        models: Lista de modelos cargados
        image: Imagen a clasificar (batch)
        method: 'soft_voting' (promediar probs) o 'hard_voting'

    Returns:
        Predicciones del ensemble
    """
    # Soft voting: promediar probabilidades
    probs = []
    for model in models:
        with torch.no_grad():
            output = model(image)
            prob = torch.softmax(output, dim=1)
            probs.append(prob)

    avg_probs = torch.stack(probs).mean(dim=0)
    return avg_probs.argmax(dim=1), avg_probs
```

**Checkpoints de los 5 modelos:**
```
outputs/classifier_cv/fold_01/best_classifier.pt
outputs/classifier_cv/fold_02/best_classifier.pt
outputs/classifier_cv/fold_03/best_classifier.pt
outputs/classifier_cv/fold_04/best_classifier.pt
outputs/classifier_cv/fold_05/best_classifier.pt
```

**Dataset:**
```
outputs/warped_lung_best/session_warping/test/
├── COVID/
├── Normal/
└── Viral_Pneumonia/
```

### Fase 2: Implementar TTA (PRIORIDAD ALTA)

**Agregar a `scripts/evaluate_classifier_ensemble.py`:**

```python
def predict_with_tta(model, image, num_augmentations=5):
    """
    Test-Time Augmentation para reducir varianza.

    Augmentations a aplicar:
    - Original (sin modificar)
    - Horizontal flip
    - Rotación ±5 grados
    - Brightness jitter (±10%)
    - Contrast jitter (±10%)
    """
    predictions = []

    # 1. Predicción original
    with torch.no_grad():
        output = model(image)
        predictions.append(torch.softmax(output, dim=1))

    # 2. Con augmentations
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=5),
        transforms.RandomRotation(degrees=-5),
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.1),
    ]

    for aug in augmentations:
        aug_image = aug(image)
        with torch.no_grad():
            output = model(aug_image)
            predictions.append(torch.softmax(output, dim=1))

    # Promediar todas
    avg_prediction = torch.stack(predictions).mean(dim=0)
    return avg_prediction.argmax(dim=1), avg_prediction
```

### Fase 3: Evaluar en Test Set

**Script debe generar:**
```json
{
  "ensemble_method": "soft_voting",
  "use_tta": true,
  "num_tta_augmentations": 5,
  "n_models": 5,
  "test_set_size": 1895,
  "metrics": {
    "accuracy": 0.XXX,
    "f1_macro": 0.XXX,
    "f1_weighted": 0.XXX
  },
  "per_class_metrics": {...},
  "confusion_matrix": [[...], [...], [...]],
  "comparison_with_individual_models": {
    "individual_avg": 0.9768,
    "ensemble": 0.XXX,
    "improvement": "+X.XX%"
  }
}
```

**Guardar en:** `outputs/classifier_cv/ensemble_test_results.json`

### Fase 4: Actualizar Capítulo LaTeX

**Archivo:** `docs/Tesis/capitulo5/5_3_resultados_clasificacion_CV.tex`

Agregar nueva subsección después de la Tabla de resultados por fold:

```latex
\subsubsection{Ensemble de Modelos con TTA}
\label{subsubsec:ensemble_tta}

Para aprovechar la diversidad de los 5 modelos entrenados con validación cruzada,
se implementó un ensemble mediante \textit{soft voting} (promediado de probabilidades)
combinado con \textit{Test-Time Augmentation} (TTA). El ensemble evalúa cada imagen
del test set con los 5 modelos y cada modelo aplica 6 augmentations (original + 5 variaciones),
generando 30 predicciones por imagen que se promedian para obtener la predicción final.

\begin{table}[htbp]
    \centering
    \caption{Comparación: Modelos Individuales vs Ensemble con TTA en test set.}
    \label{tab:ensemble_comparison}
    \small
    \begin{tabular}{@{}lccc@{}}
        \toprule
        \textbf{Configuración} & \textbf{Accuracy} & \textbf{F1-Macro} & \textbf{Mejora} \\
        \midrule
        Promedio modelos individuales & 97.68\% ± 0.16\% & 96.47\% ± 0.27\% & --- \\
        Mejor modelo individual (Fold 5) & 97.94\% & 96.85\% & +0.26\% \\
        \textbf{Ensemble + TTA} & \textbf{XX.XX\%} & \textbf{XX.XX\%} & \textbf{+X.XX\%} \\
        \bottomrule
    \end{tabular}
\end{table}

El ensemble con TTA alcanza una accuracy de \textbf{XX.XX\%}, superando el promedio
de los modelos individuales en X.XX puntos porcentuales. Este resultado confirma
que la diversidad entre los modelos entrenados con diferentes particiones de datos
aporta información complementaria que mejora el rendimiento final.
```

## Archivos de Referencia Importantes

### Código existente relevante:

1. **Modelo de clasificador:**
   - `src_v2/models/classifier.py` - Clase `ImageClassifier`
   - `src_v2/models/__init__.py` - Función `create_classifier()`

2. **Evaluación actual:**
   - `src_v2/cli.py` - Comando `evaluate-classifier` (líneas ~800-900)
   - Usa `evaluate_model()` de `src_v2/evaluation/metrics.py`

3. **Scripts de referencia:**
   - `scripts/predict_landmarks_dataset.py` - Ejemplo de ensemble para landmarks
   - `scripts/evaluate_ensemble_from_config.py` - Evaluación de ensemble landmarks
   - `scripts/compute_cv_test_aggregated_metrics.py` - Agregación de métricas

### Configuración actual:

```json
// configs/classifier_warped_base.json
{
  "data_dir": "outputs/warped_lung_best/session_warping",
  "backbone": "resnet18",
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.0001,
  "patience": 10,
  "use_class_weights": true,
  "output_dir": "outputs/classifier_warped_lung_best",
  "seed": 42
}
```

## Restricciones y Consideraciones

### ✅ HACER:
- Usar soft voting (promediar probabilidades) para ensemble
- Aplicar TTA con 5-6 augmentations razonables
- Evaluar SOLO en test set (nunca usar test para optimizar hiperparámetros)
- Guardar métricas detalladas en JSON
- Generar matriz de confusión del ensemble
- Comparar con modelos individuales

### ❌ NO HACER:
- NO optimizar thresholds mirando el test set
- NO usar hard voting (mayoría de votos) - soft voting es superior
- NO aplicar augmentations destructivos que cambien semántica médica
- NO entrenar modelos nuevos (usar los 5 existentes)
- NO mezclar datos de train/val/test

## Estructura del Script Recomendada

```python
#!/usr/bin/env python3
"""
Script para evaluar ensemble de clasificadores con TTA en test set.

Uso:
    python scripts/evaluate_classifier_ensemble.py \
        --cv-dir outputs/classifier_cv \
        --data-dir outputs/warped_lung_best/session_warping \
        --use-tta \
        --tta-augmentations 5 \
        --output outputs/classifier_cv/ensemble_test_results.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Importar desde src_v2
from src_v2.models import create_classifier
from src_v2.evaluation.metrics import compute_classification_metrics

class ClassifierEnsemble:
    def __init__(self, checkpoint_paths, device='cuda'):
        """Carga múltiples modelos para ensemble."""
        self.models = []
        for ckpt_path in checkpoint_paths:
            model = create_classifier(checkpoint=str(ckpt_path), device=device)
            model.eval()
            self.models.append(model)
        self.device = device

    def predict_soft_voting(self, images, use_tta=False, num_tta_aug=5):
        """Predicción con soft voting y opcional TTA."""
        # ... implementar aquí ...
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--use-tta', action='store_true')
    parser.add_argument('--tta-augmentations', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # 1. Cargar los 5 modelos
    checkpoint_paths = [
        args.cv_dir / f'fold_{i:02d}' / 'best_classifier.pt'
        for i in range(1, 6)
    ]

    ensemble = ClassifierEnsemble(checkpoint_paths, device=args.device)

    # 2. Preparar test set
    test_dir = args.data_dir / 'test'
    # ... resto de la implementación ...

    # 3. Evaluar con ensemble + TTA
    # ...

    # 4. Guardar resultados
    # ...

if __name__ == '__main__':
    main()
```

## Validación de Resultados

**Al terminar, verificar:**

1. ✅ Accuracy ensemble > accuracy promedio individual
2. ✅ Accuracy ensemble ≥ accuracy mejor modelo individual
3. ✅ F1-macro ensemble > F1-macro promedio individual
4. ✅ Ganancia esperada: +0.3 a +1.0 puntos
5. ✅ Matriz de confusión tiene 1,895 samples (test set completo)
6. ✅ No hay data leakage (solo usamos test set para evaluación final)

**Rango esperado de resultados:**
- Accuracy: 98.0% - 98.8%
- F1-Macro: 96.8% - 98.0%
- Si accuracy < 98.0%: verificar implementación
- Si accuracy > 99.0%: sospechar data leakage

## Documentos de Referencia

- `docs/ESTRATEGIAS_MEJORA_TEST_ACCURACY.md` - Análisis completo de estrategias
- `docs/REPRO_FULL_PIPELINE.md` - Pipeline completo del proyecto
- `CLAUDE.md` - Guía general del proyecto

## Estado del Repositorio

**Último commit relevante:**
- Evaluación de modelos CV en test set
- Scripts de figuras actualizados
- Capítulo LaTeX corregido con métricas de test

**Branch:** main

---

## Prompt Inicial para Claude

"Hola, necesito continuar el trabajo en mi proyecto de tesis de detección de COVID-19.

**Contexto:** Tengo 5 modelos de clasificación entrenados con validación cruzada que ya fueron evaluados individualmente en el test set (accuracy promedio: 97.68% ± 0.16%). Ahora necesito implementar un ensemble de estos 5 modelos con Test-Time Augmentation (TTA) para mejorar el rendimiento en el test set.

**Objetivo:** Crear un script que:
1. Cargue los 5 modelos desde `outputs/classifier_cv/fold_0X/best_classifier.pt`
2. Implemente ensemble con soft voting (promediado de probabilidades)
3. Aplique TTA con ~5 augmentations razonables para radiografías
4. Evalúe en el test set ubicado en `outputs/warped_lung_best/session_warping/test/`
5. Genere `outputs/classifier_cv/ensemble_test_results.json` con métricas completas

**Información importante:**
- Test set: 1,895 imágenes (COVID: 452, Normal: 1,274, Viral_Pneumonia: 169)
- Modelos: ResNet-18 entrenados en imágenes normalizadas geométricamente
- Mejor modelo individual: Fold 5 (97.94% accuracy)
- Ganancia esperada con ensemble+TTA: +0.5 a +1.0 puntos → ~98.2%-98.7%

Por favor, lee el archivo `docs/PROMPT_CONTINUACION_ENSEMBLE.md` que contiene todos los detalles técnicos, estructura recomendada del script, y restricciones metodológicas.

¿Puedes ayudarme a implementar esto?"
