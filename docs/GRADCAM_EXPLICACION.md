# Grad-CAM: Explicabilidad Visual para COVID-19 Detection

## Documento de Referencia para Exposición

---

## ¿Qué es Grad-CAM?

**Grad-CAM** (Gradient-weighted Class Activation Mapping) es una técnica de **explicabilidad e interpretabilidad** para modelos de deep learning en visión por computadora.

### Definición Simple

Las **zonas de atención** (o "regiones de activación") son **mapas de calor que muestran qué partes de la radiografía de tórax fueron más importantes para que el modelo llegara a su predicción**.

### Interpretación de Colores

- **🔴 Rojo/Amarillo**: Regiones donde el modelo "puso más atención" y que tuvieron mayor influencia en la decisión
- **🔵 Azul/Púrpura**: Regiones con menos influencia en la predicción
- **⚫ Negro**: Áreas que el modelo consideró irrelevantes para la clasificación

---

## Funcionamiento Técnico

### Pipeline de Grad-CAM

```
Input Image (224×224)
    ↓
[ResNet-18 Forward Pass]
    ↓
Capture Activations @ layer4 (última capa convolucional)
    ↓
[Backward Pass on Predicted Class]
    ↓
Compute Gradients → Global Average Pooling
    ↓
Weighted Combination: weights × activations
    ↓
ReLU + Normalization [0, 1]
    ↓
Resize to 224×224 + Apply Colormap
    ↓
Overlay on Image (α=0.4 transparency)
```

### Detalles de Implementación

**Archivo**: `src_v2/visualization/gradcam.py`

**Target Layer**: `backbone.layer4` (última capa convolucional de ResNet-18)

**Ecuación clave**:
```python
# Líneas 201-205
weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global Average Pooling
cam = (weights * activations).sum(dim=1)             # Weighted combination
cam = ReLU(cam)                                      # Keep only positive
cam = normalize(cam)                                 # [0, 1] range
```

**Parámetros de visualización**:
- Colormap: `jet` (rojo = alta activación, azul = baja)
- Transparencia (alpha): `0.4` (40% heatmap, 60% imagen original)
- Resolución final: `224×224` píxeles

---

## Justificación en Diagnóstico Médico

### ¿Por qué es importante Grad-CAM en aplicaciones médicas?

#### 1. **Confianza Clínica**
- Los médicos necesitan entender **por qué** el modelo hizo una predicción
- No es suficiente con un porcentaje de confianza
- Permite auditoría visual de las decisiones del modelo

#### 2. **Validación de Aprendizaje Correcto**
- Verifica que el modelo aprende patrones **clínicamente relevantes**:
  - ✅ Consolidaciones pulmonares
  - ✅ Opacidades en vidrio esmerilado (COVID-19)
  - ✅ Infiltrados intersticiales (Neumonía Viral)
- Detecta si aprende **artefactos espurios**:
  - ❌ Marcadores de texto
  - ❌ Tubos endotraqueales
  - ❌ Bordes de la imagen

#### 3. **Detección de Sesgos**
- Identifica si el modelo usa características no relacionadas con patología pulmonar
- Ejemplo: Si activa en la esquina superior (donde suele estar metadata), hay sesgo

#### 4. **Comunicación con Stakeholders**
- Facilita explicar el sistema a personal médico sin background en ML
- Genera confianza en la adopción clínica del sistema

---

## Integración en Nuestro Pipeline

### Posición en el Sistema

```
1. Imagen Original (224×224)
         ↓
2. Detección de 15 Landmarks Anatómicos
         ↓
3. Normalización Geométrica (Piecewise Affine Warping)
         ↓
4. Clasificación ResNet-18 (COVID/Normal/Viral Pneumonia)
         ↓
5. 🔥 Grad-CAM: Visualización de Regiones de Atención 🔥
```

### Implementación en GUI

**Archivo**: `src_v2/gui/app.py`

**Interfaz Gradio** - Tab "Demostración Completa":

```python
# Línea 124-128
img_gradcam = gr.Image(
    label="4️⃣ GradCAM: Regiones de Atención",
    type="pil",
    height=300
)
```

**Pipeline de inferencia** (`src_v2/gui/inference_pipeline.py:167-170`):

```python
# Classify + GradCAM generation
probabilities, gradcam_heatmap, predicted_class_idx = manager.classify_with_gradcam(
    warped,
    target_class=None  # Use predicted class
)
```

---

## Interpretación de Resultados por Clase

### COVID-19
**Patrones esperados**:
- Activación en **periferias pulmonares** (subpleural)
- Opacidades en **vidrio esmerilado** (ground-glass opacities)
- Distribución **bilateral** y **posterior**

**Ejemplo visual**:
```
[Imagen de radiografía]
    ↓
[Grad-CAM muestra rojo en bases pulmonares bilaterales]
    → Consistente con patrón COVID-19
```

### Neumonía Viral (No-COVID)
**Patrones esperados**:
- Infiltrados **difusos** o **focales**
- Consolidaciones en **zonas centrales**
- Patrón **intersticial**

### Normal
**Patrones esperados**:
- Activación **difusa y débil**
- Sin focos específicos de alta activación
- Baja intensidad general del heatmap

---

## Métrica Complementaria: Pulmonary Focus Score (PFS)

### Definición

**Archivo**: `src_v2/visualization/gradcam.py:238-284`

```python
PFS = sum(heatmap * lung_mask) / sum(heatmap)
```

### Interpretación

| PFS Value | Significado |
|-----------|-------------|
| **1.0** | Toda la atención está en tejido pulmonar (✅ ideal) |
| **0.8-0.99** | Alta focalización pulmonar (✅ aceptable) |
| **0.5-0.79** | Atención dividida pulmón/no-pulmón (⚠️ revisar) |
| **< 0.5** | Más atención en no-pulmón (❌ problemático) |

### Uso del PFS

- **Validación automática** de calidad de Grad-CAM
- **Métrica cuantitativa** de interpretabilidad
- **Detección de modelos** que aprenden artefactos

---

## Ejemplo de Explicación para Exposición

### Script Recomendado

> "Cuando un radiólogo o médico carga una imagen en nuestro sistema, obtiene no solo una predicción (ej. 'COVID-19 con 95% de confianza'), sino también un **mapa visual de explicabilidad** generado con Grad-CAM.
>
> Este mapa muestra en **colores cálidos (rojo/amarillo)** las regiones de la radiografía que el modelo consideró más relevantes para llegar a esa decisión. Por ejemplo, si predice COVID-19, esperamos ver activación en las **periferias pulmonares** donde típicamente aparecen las opacidades en vidrio esmerilado características de esta enfermedad.
>
> Esto es fundamental en aplicaciones médicas porque:
> 1. **Genera confianza** - El médico puede verificar que el modelo está mirando las zonas correctas
> 2. **Detecta errores** - Si el modelo activa en áreas no pulmonares, sabemos que hay un problema
> 3. **Facilita la adopción clínica** - Los médicos no usan 'cajas negras', necesitan entender el razonamiento"

---

## Aspectos Técnicos Avanzados

### ¿Por qué layer4?

**Razón**: Es la **última capa convolucional** antes del Global Average Pooling y la capa fully connected.

**Ventajas**:
- Tiene el **mayor campo receptivo** (puede "ver" toda la imagen)
- Sus activaciones son las **más semánticas** (representan conceptos de alto nivel)
- Mantiene cierta **resolución espacial** (7×7 en ResNet-18) que se puede mapear a la imagen original

### Soporte Multi-Arquitectura

**Archivo**: `src_v2/visualization/gradcam.py:20-28`

Nuestro sistema soporta múltiples backbones:

```python
TARGET_LAYER_MAP = {
    "resnet18": "backbone.layer4",
    "resnet50": "backbone.layer4",
    "densenet121": "backbone.features.denseblock4",
    "efficientnet_b0": "backbone.features.8",
    "vgg16": "backbone.features.30",
    # ...
}
```

Esto permite **cambiar de arquitectura** sin modificar la lógica de Grad-CAM.

---

## Limitaciones y Consideraciones

### Limitaciones de Grad-CAM

1. **Resolución limitada**: El mapa de activación original es de baja resolución (7×7), se interpola a 224×224
2. **Solo activaciones positivas**: ReLU elimina contribuciones negativas (que también son informativas)
3. **Promedio espacial**: Puede perder detalles finos de localización

### Alternativas Consideradas

- **Grad-CAM++**: Mejor localización de múltiples objetos (no necesario para pulmones)
- **Score-CAM**: Sin gradientes (más lento)
- **Layer-CAM**: Similar rendimiento, mayor complejidad

**Decisión**: Grad-CAM clásico es suficiente para nuestro caso de uso y bien validado en literatura médica.

---

## Referencias Clave

### Paper Original

**Selvaraju et al. (2017)**
*"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*
ICCV 2017
https://arxiv.org/abs/1610.02391

### Aplicaciones en COVID-19

- **Brunese et al. (2020)**: "Explainable Deep Learning for Pulmonary Disease and Coronavirus COVID-19 Detection from X-rays"
- **Wang et al. (2020)**: "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images"

---

## Preguntas Frecuentes (Q&A para Expo)

### Q1: ¿Grad-CAM es siempre correcto?

**A**: No. Grad-CAM muestra **lo que el modelo está usando**, no necesariamente **lo que debería usar**. Si el modelo aprende incorrectamente, Grad-CAM revelará ese error (lo cual es valioso).

### Q2: ¿Puede Grad-CAM mejorar la precisión del modelo?

**A**: No directamente. Es una herramienta de **interpretabilidad**, no de mejora de rendimiento. Sin embargo, puede ayudar a **identificar problemas** que luego se corrigen (ej. data augmentation para eliminar sesgos).

### Q3: ¿Por qué no usar solo Grad-CAM sin normalización geométrica?

**A**: Grad-CAM se aplica **después del warping** porque queremos ver qué regiones del **pulmón normalizado** son importantes. La normalización mejora la consistencia anatómica, lo que hace las activaciones más interpretables.

### Q4: ¿Qué pasa si dos radiografías diferentes tienen Grad-CAMs similares pero predicciones distintas?

**A**: Esto podría indicar:
1. Diferencias sutiles no capturadas visualmente en el heatmap (resolución limitada)
2. Características fuera del campo de atención principal
3. Problema potencial del modelo (necesita revisión)

### Q5: ¿Es Grad-CAM suficiente para validación clínica?

**A**: Es **una herramienta**, no la única. La validación completa requiere:
- Métricas cuantitativas (sensitivity, specificity)
- Revisión por radiólogos expertos
- Estudios multicéntricos
- Grad-CAM es complementario a estos enfoques

---

## Código de Referencia Rápida

### Generar Grad-CAM manualmente

```python
from src_v2.visualization.gradcam import GradCAM, get_target_layer, overlay_heatmap

# Inicializar
model = load_classifier("path/to/checkpoint.pt")
target_layer = get_target_layer(model, "resnet18")
gradcam = GradCAM(model, target_layer)

# Generar heatmap
heatmap, pred_class, confidence = gradcam(input_tensor, target_class=None)

# Visualizar
overlay = overlay_heatmap(image, heatmap, alpha=0.5, colormap="jet")

# IMPORTANTE: Limpiar hooks
gradcam.remove_hooks()
```

### Calcular PFS

```python
from src_v2.visualization.gradcam import calculate_pfs

pfs_score = calculate_pfs(heatmap, lung_mask)
print(f"Pulmonary Focus Score: {pfs_score:.2%}")
```

---

## Checklist para la Exposición

- [ ] Explicar qué es Grad-CAM en 2 frases
- [ ] Mostrar ejemplo visual con colores (rojo = alta activación)
- [ ] Justificar por qué es importante en medicina
- [ ] Demostrar en vivo con la GUI
- [ ] Mencionar que se aplica después del warping
- [ ] Explicar PFS si hay preguntas técnicas
- [ ] Tener preparada respuesta sobre limitaciones
- [ ] Conectar con validación clínica general del sistema

---

## Notas Adicionales

### Performance

- Tiempo de generación: ~50-100ms adicionales
- Impacto en memoria GPU: mínimo (hooks livianos)
- Se genera solo cuando se solicita (no en quick mode)

### Extensiones Futuras

1. **Grad-CAM a múltiples capas** (layer1-layer4) para ver evolución jerárquica
2. **Integración de PFS en GUI** con umbral de alerta
3. **Comparación Grad-CAM** antes/después del warping
4. **Exportar heatmaps** a formato DICOM para PACS

---

**Última actualización**: 2026-01-18
**Versión del sistema**: v2.1.0
**Contacto**: Rafael Cruz - Tesis de Maestría
