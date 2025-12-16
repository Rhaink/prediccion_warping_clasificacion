# Prompt para Sesion 34: Galeria Visual Original vs Warped

> **Instrucciones:** Copia todo el contenido de este archivo y pegalo como tu primer mensaje en una nueva conversacion con Claude.

---

Comienza la Sesion 34. Usa ultrathink para planificar y ejecutar
las tareas de manera sistematica. El objetivo es crear evidencia
visual para la defensa de tesis.

## Contexto del Proyecto

Estoy desarrollando un sistema CLI para clasificacion de COVID-19 en rayos X usando normalizacion geometrica (warping). El proyecto demuestra que las imagenes warpeadas mejoran la generalizacion y robustez de los clasificadores.

**Hipotesis de tesis (CONFIRMADA Y VERIFICADA):**
> "Las imagenes warpeadas mejoran la generalizacion 11x (gap 25.36% -> 2.24%) y la robustez 30x (degradacion JPEG 16.14% -> 0.53%) porque eliminan variabilidad geometrica no-anatomica (marcas hospitalarias, etiquetas) que causa overfitting."

**Datos verificados en archivos JSON reales (NO inventados):**
- `outputs/session30_analysis/consolidated_results.json` - Generalizacion
- `outputs/session29_robustness/artifact_robustness_results.json` - Robustez

## Estado Actual Post-Sesion 33

### Metricas Verificadas:
- **551 tests** pasando (538 base + 13 nuevos en Sesion 33)
- **21 comandos CLI** funcionando
- **0 bugs criticos**
- **3 bugs medios** encontrados en analisis (nuevos)
- **~50% cobertura** estimada
- **85-90% proyecto completo**

### Logros Sesion 33:
1. +13 tests nuevos (GradCAM: 8, analyze-errors: 9, menos duplicados)
2. Corregido: M1 (test_image_file sin verificacion)
3. Corregido: M4 (mock_classifier_checkpoint sin validacion)
4. Corregido: B1 (num_workers hardcoded - ahora detecta OS)
5. Corregido: M5 (37 assertions debiles corregidas)
6. VERIFICADO: Datos de hipotesis son REALES (11.32x y 30.62x exactos)

### Bugs Nuevos Encontrados (Analisis Sesion 33):
| ID | Bug | Ubicacion | Severidad |
|----|-----|-----------|-----------|
| N1 | CLASSIFIER_CLASSES inconsistente | cli.py:1632,1669 vs 4961,5203 | Alta |
| N2 | Image loading redundante en gradcam | cli.py:5050 | Media |
| N3 | Image.open sin .convert('RGB') | cli.py:4987,5046,5313 | Baja |

## Objetivo Principal - Sesion 34

**Crear galeria visual comparativa original vs warped para evidencia de tesis.**

Esta sesion es CRITICA porque:
1. Los revisores de tesis necesitan EVIDENCIA VISUAL
2. GradCAM muestra DONDE mira el modelo (pulmones vs etiquetas)
3. Las figuras comparativas son el "punch line" de la investigacion

### Entregables Esperados
1. **5-6 figuras comparativas** GradCAM (original vs warped)
2. **Script reproducible** para generar figuras
3. **Documentacion** de hallazgos visuales
4. **Correccion** de bugs N1-N3 (si hay tiempo)

## Plan de Trabajo Detallado

### Fase 1: Preparacion de Datos (30 min)

```bash
# 1. Verificar que tenemos checkpoints entrenados
ls -la outputs/classifier/*.pt 2>/dev/null || echo "Necesitamos entrenar modelos"
ls -la checkpoints_v2/*.pt 2>/dev/null
ls -la checkpoints_v2_full/*.pt 2>/dev/null

# 2. Verificar datasets disponibles
ls -la data/original/test/ 2>/dev/null
ls -la data/warped/test/ 2>/dev/null
ls -la outputs/warped_dataset/test/ 2>/dev/null

# 3. Verificar comandos funcionan
python -m src_v2 gradcam --help
python -m src_v2 classify --help
```

**Si no hay checkpoints entrenados:**
```bash
# Entrenar clasificador en dataset original (quick mode)
python -m src_v2 train-classifier \
    --data-dir data/original \
    --output-dir outputs/classifier_original \
    --backbone resnet18 \
    --epochs 5 \
    --batch-size 32

# Entrenar clasificador en dataset warped
python -m src_v2 train-classifier \
    --data-dir outputs/warped_dataset \
    --output-dir outputs/classifier_warped \
    --backbone resnet18 \
    --epochs 5 \
    --batch-size 32
```

### Fase 2: Generar GradCAM Comparativos (1-2 horas)

```bash
# Crear directorio para figuras de tesis
mkdir -p outputs/thesis_figures/gradcam_comparison

# GradCAM para modelo ORIGINAL en imagenes ORIGINALES
python -m src_v2 gradcam \
    --checkpoint outputs/classifier_original/best.pt \
    --data-dir data/original/test \
    --output-dir outputs/thesis_figures/gradcam_comparison/original_on_original \
    --num-samples 10 \
    --colormap jet

# GradCAM para modelo WARPED en imagenes WARPED
python -m src_v2 gradcam \
    --checkpoint outputs/classifier_warped/best.pt \
    --data-dir outputs/warped_dataset/test \
    --output-dir outputs/thesis_figures/gradcam_comparison/warped_on_warped \
    --num-samples 10 \
    --colormap jet

# CRITICO: GradCAM para modelo ORIGINAL en imagenes WARPED (cross-domain)
python -m src_v2 gradcam \
    --checkpoint outputs/classifier_original/best.pt \
    --data-dir outputs/warped_dataset/test \
    --output-dir outputs/thesis_figures/gradcam_comparison/original_on_warped \
    --num-samples 10

# CRITICO: GradCAM para modelo WARPED en imagenes ORIGINALES (cross-domain)
python -m src_v2 gradcam \
    --checkpoint outputs/classifier_warped/best.pt \
    --data-dir data/original/test \
    --output-dir outputs/thesis_figures/gradcam_comparison/warped_on_original \
    --num-samples 10
```

### Fase 3: Crear Figuras Comparativas (1 hora)

Crear script Python para generar figuras lado-a-lado:

```python
# scripts/create_thesis_figures.py
"""
Session 34: Script para crear figuras comparativas de GradCAM.

Genera figuras lado-a-lado mostrando:
- Imagen original vs warped
- GradCAM de modelo original vs warped
- Diferencias en atencion (pulmones vs etiquetas)
"""
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

def create_comparison_figure(
    original_img_path: Path,
    warped_img_path: Path,
    original_gradcam_path: Path,
    warped_gradcam_path: Path,
    output_path: Path,
    title: str = ""
):
    """Crea figura 2x2 comparativa."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Fila 1: Imagenes originales
    axes[0, 0].imshow(Image.open(original_img_path), cmap='gray')
    axes[0, 0].set_title("Imagen Original", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(Image.open(warped_img_path), cmap='gray')
    axes[0, 1].set_title("Imagen Warped", fontsize=14)
    axes[0, 1].axis('off')

    # Fila 2: GradCAM
    axes[1, 0].imshow(Image.open(original_gradcam_path))
    axes[1, 0].set_title("GradCAM Modelo Original\n(atiende a etiquetas)", fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(Image.open(warped_gradcam_path))
    axes[1, 1].set_title("GradCAM Modelo Warped\n(atiende a pulmones)", fontsize=12)
    axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figura guardada: {output_path}")

# Generar figuras para cada clase
for class_name in ["COVID", "Normal", "Viral_Pneumonia"]:
    # Encontrar imagenes correspondientes
    # ... logica para emparejar imagenes ...
    pass
```

### Fase 4: Analisis de Patrones de Atencion (30 min)

Documentar hallazgos visuales:

1. **Modelo Original tiende a:**
   - Atender a esquinas y bordes (marcas hospitalarias)
   - Atender a etiquetas de texto
   - Distribucion de atencion dispersa

2. **Modelo Warped tiende a:**
   - Atender a regiones pulmonares centrales
   - Patron de atencion concentrado en anatomia
   - Ignorar areas no-pulmonares

### Fase 5: Corregir Bugs (si hay tiempo)

#### Bug N1: CLASSIFIER_CLASSES inconsistente (ALTA prioridad)
```python
# cli.py - Estandarizar uso de constante
# Linea 1632: from src_v2.constants import CLASSIFIER_CLASSES
# Linea 4961, 5203: Cambiar hardcoded a CLASSIFIER_CLASSES

# Verificar constante existe
grep -n "CLASSIFIER_CLASSES" src_v2/constants.py
```

#### Bug N2: Image loading redundante (MEDIA prioridad)
```python
# cli.py:5050 - Eliminar carga duplicada
# ANTES:
img_pil = Image.open(img_path)
img_array = np.array(img_pil)
img_tensor, _ = dataset[idx]  # Carga redundante

# DESPUES:
img_pil = Image.open(img_path).convert('RGB')
img_array = np.array(img_pil)
img_tensor = transform(img_pil).unsqueeze(0).to(device)
```

#### Bug N3: Image.open sin .convert('RGB') (BAJA prioridad)
```python
# cli.py:4987,5046,5313 - Agregar conversion explicita
# ANTES:
img_pil = Image.open(image)

# DESPUES:
img_pil = Image.open(image).convert('RGB')
```

## Estructura de Archivos Relevantes

```
outputs/
  thesis_figures/              # NUEVO - figuras para tesis
    gradcam_comparison/
      original_on_original/
      warped_on_warped/
      original_on_warped/
      warped_on_original/
    combined_figures/          # Figuras lado-a-lado finales
  session30_analysis/
    consolidated_results.json  # Datos de generalizacion
  session29_robustness/
    artifact_robustness_results.json  # Datos de robustez

src_v2/
  cli.py                       # CLI principal (bugs N1, N2, N3)
  visualization/
    gradcam.py                 # Implementacion GradCAM
```

## Comandos CLI Principales

```bash
# GradCAM - imagen individual
python -m src_v2 gradcam \
    --checkpoint PATH_CHECKPOINT \
    --image PATH_IMAGEN \
    --output PATH_SALIDA \
    --alpha 0.5 \
    --colormap jet

# GradCAM - batch (directorio)
python -m src_v2 gradcam \
    --checkpoint PATH_CHECKPOINT \
    --data-dir PATH_DIRECTORIO \
    --output-dir PATH_SALIDA \
    --num-samples 10

# Clasificar imagen
python -m src_v2 classify \
    --image PATH_IMAGEN \
    --classifier-checkpoint PATH_CHECKPOINT \
    --output-json resultado.json
```

## Metricas de Exito - Sesion 34

| Entregable | Meta |
|------------|------|
| Figuras comparativas GradCAM | 5-6 figuras |
| Script reproducible | 1 script |
| Bug N1 corregido | Si hay tiempo |
| Bug N2 corregido | Si hay tiempo |
| Documentacion visual | Seccion en docs/ |

## Figuras Clave para Tesis

### Figura 1: Comparacion GradCAM COVID
- Izquierda: Imagen original con GradCAM (modelo original)
- Derecha: Imagen warped con GradCAM (modelo warped)
- Leyenda: "El modelo warped atiende a regiones pulmonares"

### Figura 2: Comparacion GradCAM Normal
- Similar estructura
- Mostrar diferencia en patron de atencion

### Figura 3: Comparacion GradCAM Viral Pneumonia
- Caso mas dificil (mas similar a COVID)
- Destacar donde mira cada modelo

### Figura 4: Cross-domain GradCAM
- Modelo original → imagen warped (confusion)
- Modelo warped → imagen original (robustez)

### Figura 5: Matriz de Confusion Visual
- 2x2 grid con ejemplos de cada combinacion

### Figura 6: Resumen de Metricas
- Tabla visual con 11x generalizacion, 30x robustez
- Incluir p-values si estan disponibles

## Comandos de Verificacion

```bash
# Verificar GradCAM funciona
python -m src_v2 gradcam --help

# Ejecutar tests de visualizacion
.venv/bin/python -m pytest tests/test_visualization.py -v

# Verificar no regresiones
.venv/bin/python -m pytest tests/ -v --tb=no -q 2>&1 | tail -10
```

## Notas Importantes

1. **Priorizar figuras sobre bugs** - La evidencia visual es critica para tesis
2. **Usar mismas imagenes** para comparacion justa original vs warped
3. **Documentar observaciones** mientras se generan figuras
4. **Guardar en alta resolucion** (dpi=300) para publicacion
5. **Usar colormaps consistentes** (jet para todos los GradCAM)

## Orden de Trabajo Recomendado

1. Leer este prompt completo
2. Verificar checkpoints y datasets disponibles
3. Generar GradCAM en las 4 combinaciones (2x2)
4. Crear script de figuras comparativas
5. Generar 5-6 figuras lado-a-lado
6. Documentar observaciones visuales
7. (Opcional) Corregir bugs N1-N3
8. Documentar en `docs/sesiones/SESION_34_VISUAL_GALLERY.md`

## Contexto Estrategico

El proyecto esta al 85-90% de completitud. Esta sesion es CRITICA porque:
- Las figuras son la "prueba visual" de la hipotesis
- Los revisores necesitan ver la diferencia en atencion
- Sin evidencia visual, los numeros son menos convincentes

**Despues de esta sesion quedan:**
- Sesion 35: Validacion estadistica (p-values, IC 95%)
- Sesion 36: Tests de integracion criticos (optimize-margin)
- Sesion 37: Documentacion final y README ejecutivo

**El objetivo final es demostrar VISUALMENTE que el warping mejora clasificadores medicos al forzar atencion en anatomia pulmonar en lugar de artefactos hospitalarios.**

## Datos de Referencia Verificados

```
GENERALIZACION (outputs/session30_analysis/consolidated_results.json):
- Gap original: 25.36% (modelo original falla en datos warped)
- Gap warped: 2.24% (modelo warped generaliza bien)
- Mejora: 11.32x

ROBUSTEZ JPEG (outputs/session29_robustness/artifact_robustness_results.json):
- Degradacion original Q50: 16.14%
- Degradacion warped Q50: 0.53%
- Mejora: 30.62x
```

Estos datos NO son inventados - verificados en Sesion 33 con analisis exhaustivo de archivos JSON.
