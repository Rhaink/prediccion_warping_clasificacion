# ESTRATEGIAS PARA MEJORAR TEST SET ACCURACY

## 🎯 OBJETIVO
Aumentar test set accuracy de **97.68% → 98.6%** (ganancia de +0.92 puntos)

**IMPORTANTE:** Todas las estrategias deben validarse en test set para evitar overfitting.

---

## 📊 ESTRATEGIA 1: Ensemble de los 5 Modelos CV (ALTA PRIORIDAD)

**Estado actual:** Tenemos 5 modelos entrenados pero solo promediamos métricas.
**Propuesta:** Crear ensemble votando o promediando probabilidades.

### Implementación

```python
# Soft Voting (promediar probabilidades)
def ensemble_predict(models, image):
    probs = []
    for model in models:
        with torch.no_grad():
            output = model(image)
            prob = torch.softmax(output, dim=1)
            probs.append(prob)
    
    # Promediar probabilidades
    avg_probs = torch.stack(probs).mean(dim=0)
    return avg_probs.argmax(dim=1)
```

### Ganancia esperada
- **+0.3 a +0.8 puntos** en accuracy
- Basado en experiencia con landmark ensemble (mejora de ~0.4px)
- Reduce varianza y errores individuales

### Ventajas
✅ No requiere re-entrenar
✅ Usa modelos existentes
✅ Metodológicamente válido
✅ Implementación rápida (~2 horas)

### Riesgos
⚠️  Ninguno - es práctica estándar en ML

---

## 🏗️ ESTRATEGIA 2: Arquitectura Más Profunda

**Estado actual:** ResNet-18 (11.7M parámetros)
**Propuesta:** Probar ResNet-50 o EfficientNet-B0

### Opciones

| Modelo          | Parámetros | ImageNet Top-1 | Tiempo Train | Prioridad |
|----------------|-----------|----------------|--------------|-----------|
| ResNet-50      | 25.6M     | 76.1%          | 1.8x         | ⭐⭐⭐    |
| ResNet-101     | 44.5M     | 77.4%          | 2.5x         | ⭐⭐      |
| EfficientNet-B0| 5.3M      | 77.1%          | 1.0x         | ⭐⭐⭐⭐  |
| EfficientNet-B1| 7.8M      | 79.1%          | 1.3x         | ⭐⭐⭐    |

### Ganancia esperada
- **+0.2 a +0.5 puntos** en accuracy
- Más capacidad → mejores features

### Implementación
```bash
# Modificar config
{
  "backbone": "resnet50",  # o "efficientnet-b0"
  "epochs": 50,
  "batch_size": 16,  # Reducir por memoria
  "lr": 0.00005
}

# Entrenar con CV
python -m src_v2 train-classifier-cv \
  --config configs/classifier_warped_resnet50.json \
  --n-folds 5
```

### Ventajas
✅ Mayor capacidad de representación
✅ Pretrained en ImageNet (mejor inicialización)

### Riesgos
⚠️  Mayor tiempo de entrenamiento (~2-3x)
⚠️  Riesgo de overfitting si no hay suficiente regularización

---

## 🎨 ESTRATEGIA 3: Data Augmentation Avanzado

**Estado actual:** Augmentation básico (flip, rotation, affine)
**Propuesta:** Técnicas modernas de augmentation

### Técnicas Recomendadas

#### A) Mixup
- Mezcla imágenes de diferentes clases
- Regularización muy efectiva
- Paper: "mixup: Beyond Empirical Risk Minimization" (2018)

```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

#### B) RandAugment
- Augmentation automático con magnitud controlada
- State-of-the-art para visión médica
- Paper: "RandAugment" (Google, 2020)

```python
from torchvision.transforms import RandAugment

transform = transforms.Compose([
    RandAugment(num_ops=2, magnitude=9),
    # ... resto de transforms
])
```

#### C) Augmentation Específico para Radiografías
```python
# Variaciones de contraste/brillo más agresivas
transforms.ColorJitter(
    brightness=0.3,  # Aumentar de 0.2
    contrast=0.3,    # Aumentar de 0.2
)

# Elastic deformation (común en imágenes médicas)
from albumentations import ElasticTransform
ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03)

# Grid distortion
from albumentations import GridDistortion
GridDistortion(num_steps=5, distort_limit=0.3)
```

### Ganancia esperada
- Mixup: **+0.3 a +0.6 puntos**
- RandAugment: **+0.2 a +0.5 puntos**
- Combinados: **+0.5 a +0.9 puntos**

### Ventajas
✅ Reduce overfitting significativamente
✅ Mejora generalización
✅ No aumenta inferencia time

### Riesgos
⚠️  Requiere re-entrenamiento completo
⚠️  Aumenta tiempo de entrenamiento (~1.5x)

---

## 🎛️ ESTRATEGIA 4: Optimización de Threshold

**Estado actual:** Threshold fijo de 0.5 para todas las clases
**Propuesta:** Optimizar threshold por clase en validation set

### Implementación

```python
from sklearn.metrics import f1_score

def optimize_thresholds(model, val_loader):
    """Encuentra thresholds óptimos por clase"""
    all_probs = []
    all_labels = []
    
    # Recolectar probabilidades
    for images, labels in val_loader:
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Grid search para encontrar mejores thresholds
    best_f1 = 0
    best_thresholds = [0.5, 0.5, 0.5]
    
    for t1 in np.arange(0.3, 0.7, 0.05):
        for t2 in np.arange(0.3, 0.7, 0.05):
            for t3 in np.arange(0.3, 0.7, 0.05):
                thresholds = [t1, t2, t3]
                preds = apply_thresholds(all_probs, thresholds)
                f1 = f1_score(all_labels, preds, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = thresholds
    
    return best_thresholds
```

### Ganancia esperada
- **+0.1 a +0.3 puntos** en accuracy
- Especialmente útil para clases desbalanceadas

### Ventajas
✅ No requiere re-entrenamiento
✅ Implementación rápida (< 1 hora)
✅ Mejora precision/recall trade-off

### Riesgos
⚠️  Debe optimizarse en validation, NO en test
⚠️  Ganancia modesta comparada con otras estrategias

---

## 🔄 ESTRATEGIA 5: Test-Time Augmentation (TTA)

**Estado actual:** No implementado para clasificador
**Propuesta:** Aplicar TTA como en landmark detection

### Implementación

```python
def predict_with_tta(model, image, num_augmentations=5):
    """Predicción con TTA"""
    predictions = []
    
    # Predicción original
    with torch.no_grad():
        output = model(image)
        predictions.append(torch.softmax(output, dim=1))
    
    # Predicciones con augmentations
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
    
    # Promediar todas las predicciones
    avg_prediction = torch.stack(predictions).mean(dim=0)
    return avg_prediction.argmax(dim=1)
```

### Ganancia esperada
- **+0.2 a +0.5 puntos** en accuracy
- Basado en experiencia con landmarks (TTA aporta ~0.2px mejora)

### Ventajas
✅ No requiere re-entrenamiento
✅ Reduce varianza en predicciones
✅ Metodológicamente válido

### Riesgos
⚠️  Aumenta tiempo de inferencia (~5x)
⚠️  Solo útil en evaluación final

---

## 🧠 ESTRATEGIA 6: Hyperparameter Tuning Exhaustivo

**Estado actual:** Hiperparámetros por defecto
**Propuesta:** Grid/Random search sistemático

### Parámetros a Optimizar

```python
param_grid = {
    'lr': [1e-5, 5e-5, 1e-4, 5e-4],
    'batch_size': [16, 32, 64],
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'epochs': [50, 75, 100],
    'optimizer': ['adam', 'adamw', 'sgd'],
    'scheduler': ['cosine', 'step', 'plateau'],
}
```

### Implementación Práctica

```bash
# Usar Optuna para búsqueda eficiente
pip install optuna

# Script de optimización
python scripts/optimize_classifier_hyperparams.py \
  --n-trials 50 \
  --metric accuracy \
  --cv-folds 5
```

### Ganancia esperada
- **+0.2 a +0.6 puntos** en accuracy
- Depende de cuánto estemos sub-óptimos actualmente

### Ventajas
✅ Sistemático y reproducible
✅ Puede encontrar mejoras inesperadas

### Riesgos
⚠️  MUY costoso computacionalmente (días/semanas)
⚠️  Riesgo de overfitting al validation set

---

## 📈 RESUMEN DE ESTRATEGIAS (ORDENADAS POR ROI)

| # | Estrategia | Ganancia Esperada | Esfuerzo | Tiempo | ROI | Prioridad |
|---|-----------|------------------|----------|--------|-----|-----------|
| 1 | **Ensemble de 5 modelos** | +0.3 a +0.8 | Bajo | 2h | ⭐⭐⭐⭐⭐ | 🔥 ALTA |
| 2 | TTA para clasificador | +0.2 a +0.5 | Bajo | 2h | ⭐⭐⭐⭐ | 🔥 ALTA |
| 3 | Threshold optimization | +0.1 a +0.3 | Bajo | 1h | ⭐⭐⭐⭐ | ALTA |
| 4 | Mixup augmentation | +0.3 a +0.6 | Medio | 1d | ⭐⭐⭐⭐ | ALTA |
| 5 | ResNet-50/EfficientNet | +0.2 a +0.5 | Medio | 2d | ⭐⭐⭐ | MEDIA |
| 6 | RandAugment | +0.2 a +0.5 | Medio | 1d | ⭐⭐⭐ | MEDIA |
| 7 | Hyperparameter tuning | +0.2 a +0.6 | Alto | 1w | ⭐⭐ | BAJA |

### Combinación Recomendada (Máxima Ganancia)

```
Estrategia 1 (Ensemble) + Estrategia 2 (TTA) + Estrategia 3 (Threshold)
─────────────────────────────────────────────────────────────────────────
Ganancia combinada estimada: +0.6 a +1.6 puntos
Esfuerzo total: 5 horas
Accuracy esperada final: 98.28% a 99.28%
```

**⚠️  IMPORTANTE:** Las ganancias NO son aditivas perfectamente. Efecto combinado típicamente 70-80% de la suma.

---

## 🚨 ESTRATEGIAS A EVITAR (METODOLÓGICAMENTE INCORRECTAS)

❌ **Data Leakage**
   - Usar test set para selección de hiperparámetros
   - Entrenar en test set

❌ **Cherry-picking**
   - Reportar mejor fold en lugar de promedio
   - Seleccionar métricas favorables

❌ **Threshold tuning en test**
   - Optimizar thresholds mirando test set
   - Genera sesgo optimista

❌ **Multiple testing sin corrección**
   - Probar 100 configuraciones y reportar la mejor
   - Sin corrección de Bonferroni/FDR

---

## 🎓 RECOMENDACIÓN PARA LA TESIS

### Opción A: Reportar 97.68% (Actual)
**Pros:**
✅ Metodológicamente impecable
✅ Varianza muy baja (robustez demostrada)
✅ Competitivo con estado del arte COVID-19
✅ No requiere trabajo adicional

**Contras:**
⚠️  ~1 punto menor que validation

### Opción B: Implementar Ensemble + TTA → ~98.5%
**Pros:**
✅ Mejora significativa esperada
✅ Metodológicamente válido
✅ Esfuerzo razonable (4-5 horas)
✅ Demuestra conocimiento de técnicas avanzadas

**Contras:**
⚠️  Requiere re-evaluación completa
⚠️  Riesgo de no alcanzar 98.6% exacto

### Opción C: Enfoque Combinado (Recomendado)
**Estrategia:**
1. Reportar 97.68% como baseline metodológicamente correcto
2. Mencionar que validation era 98.60% (optimista)
3. Implementar ensemble y reportar como "mejora experimental"
4. Discutir en detalle por qué test < validation es esperado

**Ventajas:**
✅ Demuestra comprensión metodológica profunda
✅ Muestra transparencia científica
✅ Abre puerta a trabajo futuro
✅ Revisor valorará honestidad

---

## 📚 COMPARACIÓN CON LITERATURA

| Trabajo | Dataset | Test Accuracy | Método |
|---------|---------|---------------|--------|
| **Nuestro (actual)** | COVIDx | **97.68% ± 0.16%** | Warping + CV |
| Wang et al. 2020 | COVIDx | 93.3% | COVID-Net |
| Apostolopoulos 2020 | 1,428 imgs | 96.78% | Transfer Learning |
| Ozturk et al. 2020 | 1,125 imgs | 98.08% | DarkCovidNet |
| Sethy & Behera 2020 | 50 imgs | 95.38% | ResNet-50 + SVM |

**Conclusión:** Nuestro 97.68% está en el rango alto del estado del arte, especialmente considerando la evaluación rigurosa con CV.

---

## 💡 MI RECOMENDACIÓN FINAL

**Para propósitos de tesis:**

1. **Reporta 97.68% ± 0.16%** como resultado principal
   - Es metodológicamente correcto
   - Demuestra rigor científico
   - Ningún revisor lo cuestionará

2. **Explica claramente** la diferencia con validation (98.60%)
   - Test set es estimación no sesgada
   - Validation participó en desarrollo del modelo
   - 0.92 puntos de diferencia es saludable

3. **Implementa ensemble** (Estrategia 1) como "mejora post-tesis"
   - Ganancia esperada: 98.0% - 98.6%
   - Esfuerzo mínimo: 2-4 horas
   - Incluye en sección de "trabajo futuro" o apéndice

4. **Enfatiza fortalezas reales:**
   - Varianza bajísima (0.16% std)
   - Metodología reproducible
   - Normalización geométrica novedosa
   - Comparación justa entre configuraciones

**Mensaje clave:** Un 97.68% bien validado vale más que un 98.6% metodológicamente cuestionable.

