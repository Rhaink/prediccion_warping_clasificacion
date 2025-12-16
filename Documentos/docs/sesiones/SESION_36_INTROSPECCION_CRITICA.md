# Sesion 36: Introspeccion Critica y Analisis de Validez

**Fecha:** 2025-12-10
**Objetivo:** Revision profunda con 6 agentes para verificar validez del proyecto

---

## RESUMEN EJECUTIVO

### Estado General del Proyecto

| Aspecto | Veredicto | Severidad |
|---------|-----------|-----------|
| **Codigo y CLI** | FUNCIONA | - |
| **Datos experimentales** | REALES Y VERIFICADOS | - |
| **Pipeline de warping** | CORRECTO tecnicamente | - |
| **GradCAM/PFS** | IMPLEMENTACION CORRECTA, APLICACION INVALIDA | ALTA |
| **Clasificador** | CORRECTO, sin data leakage | - |
| **Preprocesamiento** | ESTANDAR pero sub-optimo | MEDIA |
| **Hipotesis cientifica** | PARCIALMENTE VALIDADA | **CRITICA** |

### Hallazgo Principal

> **La hipotesis "warped elimina marcas hospitalarias" es una SUPOSICION SIN EVIDENCIA DIRECTA.**
>
> Lo que SI esta demostrado:
> - Robustez JPEG 30x mejor
> - Robustez Blur 3x mejor
>
> Lo que NO esta demostrado:
> - Que el warping ELIMINA marcas (vs solo las recorta)
> - Que las marcas CAUSAN el overfitting
> - Que warped "generaliza 11x mejor" (comparacion invalida)

---

## 1. BUGS CORREGIDOS EN SESION 36

### 1.1 Imports de Warping (CORREGIDO)

**Problema:** cli.py lineas 1025 y 1684 usaban version ANTIGUA de warping.

**Solucion aplicada:**
```python
# ANTES (buggy)
from scripts.piecewise_affine_warp import piecewise_affine_warp

# DESPUES (correcto)
from src_v2.processing.warp import piecewise_affine_warp
```

**Commit:** `11fb902`

### 1.2 Tests de Validacion Agregados

Nueva clase `TestWarpingImports` con 4 tests:
- `test_cli_no_imports_from_scripts_piecewise_affine_warp`
- `test_cli_imports_from_src_v2_processing_warp`
- `test_src_v2_processing_warp_module_exists`
- `test_src_v2_processing_warp_has_required_functions`

**Estado:** 524 tests pasan (520 + 4 nuevos)

---

## 2. VERIFICACION DE PIPELINE DE WARPING

### 2.1 Funcion `piecewise_affine_warp()` - CORRECTA

| Aspecto | Estado | Detalle |
|---------|--------|---------|
| Triangulacion Delaunay | OK | Usa scipy.spatial.Delaunay |
| Deteccion triangulos degenerados | OK | Area < 1e-6 se omite |
| Interpolacion | OK | cv2.INTER_LINEAR + BORDER_REFLECT_101 |
| Boundary points | OK | 8 puntos adicionales para full_coverage |

### 2.2 Fill Rate Verificado

**Dataset actual:** ~47.08% fill rate
```
Verificacion manual en imagen COVID-3146_warped.png:
- Pixeles negros: 26,549 / 50,176 = 52.91%
- Fill rate: 47.09%
- COINCIDE con dataset_summary.json (0.4708)
```

### 2.3 Bug Menor Identificado

```python
# Linea 298 en warp.py - FRAGIL
black_pixels = np.sum(warped_image == 0)

# RECOMENDADO - Robusto
black_pixels = np.sum(warped_image < 5)
```

---

## 3. VERIFICACION DE DATOS EXPERIMENTALES

### 3.1 Archivos Verificados

| Archivo | Existe | Contenido Valido |
|---------|--------|------------------|
| `outputs/full_warped_dataset/dataset_summary.json` | SI | Fill rate 47.08% |
| `checkpoints/` | SI | 136 GB (1,191 archivos) |
| `checkpoints_v2/` | SI | 521 MB |
| Matrices de confusion | SI | Suman correctamente |

### 3.2 Verificacion Matematica de Matrices

**Caso: ResNet-18 Warped 1.25 (Sesion 31)**
```
Matriz:
        COVID  Normal  Viral
COVID    350    10      2
Normal     3  1012      5
Viral      0     9    127

Accuracy calculada: (350+1012+127)/1518 = 98.09%
Accuracy reportada: 98.09%
COINCIDE EXACTAMENTE
```

### 3.3 Conclusion

> **Los datos son REALES, no inventados.**
> - 15+ verificaciones matematicas perfectas
> - Timestamps cronologicamente coherentes
> - Checkpoints fisicos existen (145.6 GB total)

---

## 4. ANALISIS DE GRADCAM Y PFS

### 4.1 Implementacion GradCAM - CORRECTA

```python
# gradcam.py lineas 201-225
weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
cam = (weights * self.activations).sum(dim=1, keepdim=True)
cam = F.relu(cam)
cam = F.interpolate(cam, size=(224,224), mode='bilinear')
```

La implementacion es matematicamente correcta.

### 4.2 PFS - CRITICA: INVALIDO PARA WARPED

**Resultados reportados:**
```
PFS Original: 35.5%
PFS Warped:   35.6%
Diferencia:   +0.1% (p=0.856 = NO significativa)
```

**PROBLEMA CRITICO:**

Las mascaras pulmonares NO estan transformadas junto con las imagenes:
```
Imagen Original ─[Warping]─> Imagen Warped (224x224)
       ↓                            ↓
Mascara Original           Mascara DESALINEADA
(geometria OK)             (geometria INCORRECTA)
```

**Conclusion:** El PFS para imagenes warped es **tecnicamente invalido**.

### 4.3 Interpretacion del PFS Bajo

PFS = 35% significa:
- Solo 35% de atencion del modelo esta en pulmones
- 65% de atencion esta FUERA de la region pulmonar
- **Ambos modelos NO enfocan principalmente en tejido pulmonar**

---

## 5. VERIFICACION DEL CLASIFICADOR

### 5.1 Arquitectura - CORRECTA

```python
# classifier.py
ResNet18 pretrained + Dropout(0.3) + Linear(512→3)
```

### 5.2 Data Leakage - NO EXISTE

```python
# Splits estratificados correctamente
train_df, temp_df = train_test_split(df, test_size=0.25,
                                     stratify=df['category'],
                                     random_state=42)
```

### 5.3 Comparacion Justa - PARCIALMENTE

| Criterio | Estado |
|----------|--------|
| Mismo split de datos | SI |
| Mismo modelo | SI |
| Mismos hiperparametros | SI |
| Misma informacion | **NO** (47% vs 100%) |

### 5.4 Overfitting Detectado

| Modelo | Gap Val→Test | Interpretacion |
|--------|--------------|----------------|
| Warped | 7.1% | Posible overfitting moderado |
| Original | 1.0% | Generalizacion excelente |

---

## 6. VERIFICACION DE PREPROCESAMIENTO

### 6.1 CLAHE - CORRECTO

- Aplicado en espacio LAB (solo canal L)
- clip_limit=2.0, tile_size=4
- Aplicado ANTES del resize

### 6.2 Normalizacion ImageNet - SUB-OPTIMA

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
```

**Problema:** Estos valores son para fotografias naturales, NO rayos X.

**Impacto en fondo negro:**
```
Pixel negro (0,0,0) normalizado:
z = (0 - 0.485) / 0.229 = -2.12 desviaciones estandar
```

El modelo interpreta el fondo negro como **patron anomalo fuerte**.

### 6.3 Fondo Negro - PROBLEMA CRITICO

El modelo USA el fondo negro como senal discriminativa:
- En imagenes originales: No existe fondo negro
- En imagenes warped: Siempre presente (~53%)

Esto introduce un **sesgo artificial** en la clasificacion.

---

## 7. HIPOTESIS CIENTIFICA - ANALISIS CRITICO

### 7.1 Hipotesis Original

> "Las imagenes warpeadas son mejores para entrenar clasificadores porque eliminan marcas hospitalarias y normalizan la geometria"

### 7.2 Evidencia Buscada vs Encontrada

| Claim | Evidencia Buscada | Evidencia Encontrada |
|-------|-------------------|----------------------|
| "Elimina marcas hospitalarias" | Visualizacion antes/despues | **NINGUNA** |
| "Marcas causan overfitting" | Ablation study | **NINGUNA** |
| "Generaliza 11x mejor" | Cross-evaluation valida | **INVALIDA** (info asimetrica) |
| "Fuerza atencion en pulmones" | PFS significativamente mayor | **NO** (p=0.856) |

### 7.3 Lo que SI esta Demostrado

| Claim | Evidencia | Estado |
|-------|-----------|--------|
| Robustez JPEG 30x | Tests reproducibles | **VALIDO** |
| Robustez Blur 3x | Tests reproducibles | **VALIDO** |
| Datos experimentales reales | Verificacion matematica | **VALIDO** |

### 7.4 Explicacion Alternativa Mas Plausible

La mejora en robustez probablemente viene de:
1. **Menos informacion total** → menos features para overfitar
2. **Informacion concentrada** → modelo enfoca en region central
3. **Transformacion consistente** → perturbaciones afectan menos

**NO necesariamente de:** "Eliminacion de marcas hospitalarias"

---

## 8. REFORMULACION DE CLAIMS RECOMENDADA

### INCORRECTO (Narrativa actual):
> "La normalizacion geometrica elimina marcas hospitalarias y generaliza 11x mejor"

### CORRECTO (Reformulacion):
> "La normalizacion geometrica mediante landmarks anatomicos proporciona:
> - 30x mejor robustez a compresion JPEG
> - 3x mejor robustez a blur gaussiano
> - Trade-off: -0.8% accuracy en dominio interno
>
> El mecanismo probable es reduccion de informacion y regularizacion de features,
> no necesariamente eliminacion de marcas hospitalarias (hipotesis no verificada).
>
> La comparacion cross-evaluation tiene sesgo metodologico debido a diferencias
> en fill rate (47% vs 100%) y requiere dataset con full_coverage para validacion."

---

## 9. TRABAJO FUTURO CRITICO

### Alta Prioridad (Antes de publicacion/defensa)

1. **Generar dataset con full_coverage=True** (~96% fill rate)
   ```bash
   python scripts/generate_warped_dataset_full_coverage.py
   ```

2. **Implementar warp_mask()** para PFS valido
   - Transformar mascaras junto con imagenes
   - Recalcular PFS con alineacion correcta

3. **Analisis visual de marcas hospitalarias**
   - Mostrar lado-a-lado: Original vs Warped
   - Identificar que marcas desaparecen vs se distorsionan

4. **Experimento de control**
   - Original croppeada a pulmones (equivalente en informacion a warped)
   - Comparar si tambien es 30x mas robusta

### Media Prioridad

5. **Calcular media/std especifica del dataset** para normalizacion
6. **Agregar mask del fondo** antes de normalizar
7. **Probar Dropout(0.4-0.5)** para regularizacion

---

## 10. PROXIMOS PASOS PARA CLI

### Comandos Faltantes (25% restante)

| Comando | Estado | Prioridad |
|---------|--------|-----------|
| `compare-architectures` | Parcialmente implementado | Media |
| `export-results` | No implementado | Baja |
| `visualize-predictions` | Parcialmente implementado | Media |
| `validate-dataset` | No implementado | Alta |
| `compute-dataset-stats` | No implementado | Media |

### Tests Adicionales Recomendados

1. Test end-to-end: warp → classify
2. Test de visualizacion de landmarks
3. Test de fill_rate con umbral robusto
4. Test de validacion de mascaras warped

---

## 11. CONCLUSION DE LA SESION

### Lo que SE LOGRO:
- Bugs de warping corregidos (commit 11fb902)
- 4 tests nuevos agregados (524 total)
- Documentacion de claims reformulados
- Script para full_coverage dataset

### Lo que SE DESCUBRIO:
- La hipotesis de "marcas hospitalarias" es suposicion sin evidencia
- PFS es invalido para imagenes warped (mascaras desalineadas)
- El fondo negro es una senal discriminativa (problema metodologico)
- La comparacion cross-evaluation tiene sesgo por info asimetrica

### Veredicto Final:

> **El proyecto tiene contribuciones validas (robustez JPEG/Blur), pero la narrativa
> cientifica necesita reformulacion antes de publicacion. Los datos son reales,
> el codigo funciona, pero las conclusiones van mas alla de la evidencia disponible.**

---

## COMMITS DE LA SESION

| Commit | Descripcion |
|--------|-------------|
| `11fb902` | fix: corregir imports de warping en CLI |
| `a67e54c` | docs: agregar resultados reformulados y script full_coverage |

---

## ARCHIVOS CLAVE CREADOS/MODIFICADOS

```
MODIFICADOS:
- src_v2/cli.py (fix imports lineas 1021, 1673)
- tests/test_cli.py (+4 tests TestWarpingImports)

CREADOS:
- docs/RESULTADOS_EXPERIMENTALES_v2.md
- docs/sesiones/SESION_35_ANALISIS_CRITICO.md
- docs/sesiones/SESION_36_INTROSPECCION_CRITICA.md
- scripts/generate_warped_dataset_full_coverage.py
```
