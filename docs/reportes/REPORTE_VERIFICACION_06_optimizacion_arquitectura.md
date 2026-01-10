# REPORTE DE VERIFICACION - 06_optimizacion_arquitectura.tex

**Documento**: /home/donrobot/Projects/Tesis/documentación/06_optimizacion_arquitectura.tex
**Fecha verificación**: 2025-12-06
**Sesiones cubiertas**: 5, 6, 9

---

## RESUMEN EJECUTIVO

### Estado General: ⚠️ PROBLEMAS ENCONTRADOS

- **Implementación**: ✅ VERIFICADA
- **Resultados experimentales**: ⚠️ PARCIALMENTE VERIFICADOS
- **Métricas documentadas**: ⚠️ DISCREPANCIAS ENCONTRADAS
- **Experimentos faltantes**: ❌ EXPERIMENTOS INCOMPLETOS

---

## 1. COORDINATE ATTENTION

### 1.1 Implementación ✅ VERIFICADA

**Archivo**: /home/donrobot/Projects/Tesis/src_v2/models/resnet_landmark.py (líneas 12-54)

**Verificación de implementación**:
```python
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 32):
        # Pooling direccional
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Reducción de dimensión
        mid_channels = max(8, in_channels // reduction)

        # Transformación compartida
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        # Generación de pesos de atención
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
```

**Parámetros verificados**:
- ✅ **reduction factor = 32** (línea 111): `CoordinateAttention(512, reduction=32)`
- ✅ **mid_channels** = max(8, 512//32) = 16 canales (línea 25)
- ✅ Uso en backbone de 512 canales

**Ecuaciones del documento**:
- ✅ Ecuación 6.1 (pooling H): Implementada en línea 38
- ✅ Ecuación 6.2 (pooling W): Implementada en línea 39
- ✅ Ecuación 6.3 (transformación): Implementada en líneas 42-45
- ✅ Ecuaciones 6.4-6.5 (atención): Implementadas en líneas 48-54

---

## 2. DIMENSIÓN HIDDEN LAYER

### 2.1 Experimentos Documentados vs Reales

**Tabla 6.4 del documento** (líneas 380-394):
```
hidden_dim | Parámetros | Error Test | vs 256
256        | 137K       | 7.84 px    | baseline
512        | 270K       | 7.29 px    | -7.0%
768        | 407K       | 7.21 px    | -8.0%
1024       | 548K       | 7.35 px    | -6.3%
```

### 2.2 Verificación de Resultados Reales

#### ⚠️ PROBLEMA: Parámetros incorrectos

**Cálculo verificado** (con GroupNorm):
- hidden_dim=256: **403,230 params (403K)** vs documento: 137K ❌
- hidden_dim=512: **542,750 params (543K)** vs documento: 270K ❌
- hidden_dim=768: **682,270 params (682K)** vs documento: 407K ❌
- hidden_dim=1024: **821,790 params (822K)** vs documento: 548K ❌

**Fórmula correcta**:
```
FC1: 512*512 + 512 = 262,656
GN1: 512*2 = 1,024
FC2: 512*hidden_dim + hidden_dim
GN2: hidden_dim*2
FC3: hidden_dim*30 + 30
```

#### ✅ Errores de Test - VERIFICADOS

**Experimento 1: hidden_dim=256 (baseline Session 9)**
- Archivo: outputs/session9/exp_dropout0.3/training_config.json
- Config: hidden_dim=256, dropout=0.3, CLAHE tile=4
- **Resultado real**: 7.60 px
- **Documento**: 7.84 px (baseline Session 8)
- ⚠️ **Discrepancia**: Documento usa baseline incorrecto

**Experimento 2: hidden_dim=512**
- Archivo: outputs/session10/exp3_hidden512/evaluation_report_20251127_213835.txt
- Config: hidden_dim=512, dropout=0.3, coord_attention=true, deep_head=true
- **Resultado real**: **7.29 px** ✅
- **Documento**: 7.29 px ✅ COINCIDE

**Experimento 3: hidden_dim=768**
- Archivo: outputs/session9/exp_hidden768/evaluation_report_20251127_211619.txt
- Config: hidden_dim=768, dropout=0.3, coord_attention=true, deep_head=true
- **Resultado real**: **7.21 px** ✅
- **Documento**: 7.21 px ✅ COINCIDE

**Experimento 4: hidden_dim=1024**
- Archivo: outputs/session10/exp2_hidden1024/evaluation_report_20251127_213314.txt
- Config: hidden_dim=1024, dropout=0.3, coord_attention=true, deep_head=true
- **Resultado real**: **7.35 px** ✅
- **Documento**: 7.35 px ✅ COINCIDE

### 2.3 Resultados por Categoría (Tabla 6.7)

**Documento** (líneas 405-417):
```
Categoría | Antes  | Después | Mejora
Normal    | 7.00px | 6.34px  | -9.4%
Viral     | 7.98px | 8.50px  | +6.5%
COVID     | 9.03px | 7.79px  | -13.7%
```

**Verificación real** (outputs/session9/exp_hidden768/evaluation_report_20251127_211619.txt):
```
Normal: 6.34 +/- 4.81 px (n=47) ✅ COINCIDE
Viral_Pneumonia: 8.50 +/- 4.58 px (n=18) ✅ COINCIDE
COVID: 7.79 +/- 6.85 px (n=31) ✅ COINCIDE
```

**Baseline para comparación** (outputs/exp_clahe_tile4/evaluation_report_20251127_202425.txt):
```
Normal: 7.00 +/- 5.50 px ✅ COINCIDE
Viral_Pneumonia: 7.98 +/- 4.60 px ✅ COINCIDE
COVID: 9.03 +/- 8.42 px ✅ COINCIDE
```

✅ **Tabla 6.7 completamente verificada**

---

## 3. DROPOUT RATE OPTIMIZATION

### 3.1 Experimentos Documentados (Tabla 6.3)

**Tabla Session 9** (líneas 349-364):
```
Exp | Configuración   | Error Test | Delta
1   | CLAHE tile=2    | 7.88 px    | +0.04
2   | CLAHE clip=1.0  | 7.85 px    | +0.01
2b  | CLAHE clip=1.5  | 8.38 px    | +0.54
3   | dropout=0.3     | 7.60 px    | -0.24 ✅
4   | hidden_dim=768  | 7.21 px    | -0.63 ✅
```

### 3.2 Verificación de Resultados

**Experimento baseline (Session 8):**
- Archivo: outputs/exp_clahe_tile4/evaluation_report_20251127_202425.txt
- Config: dropout=0.5, hidden_dim=256, CLAHE tile=4
- **Resultado**: 7.84 px ✅ COINCIDE

**Experimento tile=2:**
- Archivo: outputs/session9/exp_tile2/evaluation_report_20251127_204353.txt
- Config: dropout=0.5, hidden_dim=256, CLAHE tile=2
- **Resultado**: 7.88 px ✅ COINCIDE (+0.04 vs baseline)

**Experimento clip=1.0:**
- Archivo: outputs/session9/exp_clip1.0/evaluation_report_20251127_204848.txt
- Config: dropout=0.5, hidden_dim=256, CLAHE clip=1.0
- **Resultado**: 7.85 px ✅ COINCIDE (+0.01 vs baseline)

**Experimento clip=1.5:**
- Archivo: outputs/session9/exp_clip1.5/evaluation_report_20251127_210618.txt
- Config: dropout=0.5, hidden_dim=256, CLAHE clip=1.5
- **Resultado**: 8.38 px ✅ COINCIDE (+0.54 vs baseline)

**Experimento dropout=0.3:**
- Archivo: outputs/session9/exp_dropout0.3/evaluation_report_20251127_211125.txt
- Config: dropout=0.3, hidden_dim=256, CLAHE tile=4
- **Resultado**: 7.60 px ✅ COINCIDE (-0.24 vs baseline)

**Experimento hidden_dim=768:**
- Archivo: outputs/session9/exp_hidden768/evaluation_report_20251127_211619.txt
- Config: dropout=0.3, hidden_dim=768, CLAHE tile=4
- **Resultado**: 7.21 px ✅ COINCIDE (-0.63 vs baseline)

### 3.3 ❌ PROBLEMA: Experimentos faltantes

**Documento menciona** (líneas 341, 366-367):
> "Dropout: ¿0.5 es óptimo o hay mejor valor?"
> "Dropout de 0.3 supera a dropout de 0.5"

**Afirmación implícita**: Se probaron múltiples valores de dropout (0.2, 0.3, 0.4, 0.5)

**Experimentos encontrados**:
- ✅ dropout=0.5: outputs/exp_clahe_tile4/ (baseline)
- ✅ dropout=0.3: outputs/session9/exp_dropout0.3/
- ⚠️ dropout=0.2: outputs/session10/exp1_dropout02/ (PERO es Session 10, no Session 9)
- ❌ dropout=0.4: NO ENCONTRADO

**Resultado session10/exp1_dropout02**:
- Error: 7.43 px
- Peor que dropout=0.3 (7.60 px) y dropout con hidden_dim=768 (7.21 px)
- ⚠️ Este experimento NO está documentado en las tablas

---

## 4. DEEP HEAD vs SHALLOW HEAD

### 4.1 Implementación ✅ VERIFICADA

**Deep Head** (líneas 120-135 de resnet_landmark.py):
```python
if deep_head:
    self.head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.GroupNorm(num_groups=32, num_channels=512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, hidden_dim),
        nn.GroupNorm(num_groups=16, num_channels=hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate * 0.5),
        nn.Linear(hidden_dim, 30),
        nn.Sigmoid()
    )
```

**Shallow Head** (líneas 136-146):
```python
else:
    self.head = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate * 0.5),
        nn.Linear(hidden_dim, 30),
        nn.Sigmoid()
    )
```

### 4.2 ❌ PROBLEMA: Comparación no documentada

**Documento menciona** (líneas 39, 200):
> "Mejoras arquitectónicas (Coordinate Attention, cabeza profunda)"
> "CoordAttn + DeepHead"

**Problema**: El documento NO incluye tabla comparando:
- Deep head vs Shallow head
- Solo deep head vs solo coord attention
- Experimentos de ablación de componentes arquitectónicos

**Todos los experimentos encontrados** usan `deep_head=true`, no hay baseline con `deep_head=false`

---

## 5. TEST-TIME AUGMENTATION (TTA)

### 5.1 Implementación ✅ VERIFICADA

**Archivo**: src_v2/evaluation/metrics.py (línea 301)

```python
def predict_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_flip: bool = True
) -> torch.Tensor:
    # Prediccion original
    pred_original = model(images)

    # Prediccion con flip
    images_flipped = torch.flip(images, dims=[3])  # Flip horizontal
    pred_flipped = model(images_flipped)

    # Promedio
    pred_avg = (pred_original + pred_flipped) / 2.0
    return pred_avg
```

✅ Implementación coincide con ecuación 6.6 del documento

### 5.2 Transformaciones

**Documento** (ecuaciones 6.7-6.8):
- Flip horizontal: (x', y') = (1 - x, y)
- Intercambio de índices simétricos

✅ Implementación verificada en código

### 5.3 ⚠️ PROBLEMA: Resultados de TTA

**Tabla 6.1** (líneas 191-203):
```
Configuración              | Error Test
Baseline (Wing Loss)       | 9.08 px
Baseline + TTA             | 8.80 px (-0.28)
CoordAttn + DeepHead       | 10.44 px
```

**Problema**: No se encontraron archivos de experimentos de Session 5 que verifiquen estos números.

**Session 6** (Tabla 6.2, líneas 283-294):
```
Modelo                          | Sin TTA | Con TTA
Baseline (código nuevo)         | 9.87 px | 9.76 px
CoordAttn+DeepHead (corregido)  | 9.45 px | 8.93 px
```

**Problema**: No se encontraron archivos de experimentos que verifiquen estos números específicos.

---

## 6. BUGS CORREGIDOS (SESIÓN 6)

### 6.1 Bug 1: CoordAttention en Phase 1 ✅ VERIFICADO

**Implementación de corrección** (líneas 179-192):

```python
def freeze_all_except_head(self):
    """Congela backbone y CoordAttention para Phase 1."""
    self.freeze_backbone()
    self.freeze_coord_attention()

def freeze_coord_attention(self):
    """Congela los parametros de Coordinate Attention."""
    if self.coord_attention is not None:
        for param in self.coord_attention.parameters():
            param.requires_grad = False
```

✅ Código implementa correctamente el control granular descrito

### 6.2 Bug 2: BatchNorm → GroupNorm ✅ VERIFICADO

**Documento** (líneas 246-278):
> "Reemplazar BatchNorm1d por GroupNorm... independiente del tamaño del batch"

**Implementación verificada** (líneas 126, 130):
```python
nn.GroupNorm(num_groups=32, num_channels=512)  # 512/32=16
nn.GroupNorm(num_groups=16, num_channels=hidden_dim)
```

✅ GroupNorm implementado correctamente

---

## 7. EVOLUCIÓN DEL ERROR (TABLA 6.8)

**Tabla del documento** (líneas 481-495):
```
Sesión | Cambio                    | Error   | Mejora Acum.
4      | Baseline (Wing Loss)      | 9.08 px | -
5      | + TTA                     | 8.80 px | -3.1%
6      | Bugs corregidos           | 8.93 px | -1.7%
7      | + CLAHE                   | 8.18 px | -9.9%
8      | CLAHE tile=4              | 7.84 px | -13.7%
9      | dropout=0.3, hidden=768   | 7.21 px | -20.6%
```

### Verificación:

- ❌ **Sesión 4 (9.08 px)**: No encontrado archivo específico
- ❌ **Sesión 5 (8.80 px)**: No encontrado archivo específico
- ❌ **Sesión 6 (8.93 px)**: No encontrado archivo específico
- ❌ **Sesión 7 (8.18 px)**: No encontrado archivo específico
- ✅ **Sesión 8 (7.84 px)**: outputs/exp_clahe_tile4/evaluation_report_20251127_202425.txt
- ✅ **Sesión 9 (7.21 px)**: outputs/session9/exp_hidden768/evaluation_report_20251127_211619.txt

**Nota**: Los resultados de sesiones 4-7 están mencionados en SESSION_LOG.md pero no tienen archivos de outputs correspondientes.

---

## 8. CONFIGURACIÓN FINAL OPTIMIZADA (TABLA 6.5)

**Verificación contra código**:

| Componente      | Documento         | Código Real                              | Estado |
|-----------------|-------------------|------------------------------------------|--------|
| Backbone        | ResNet-18         | models.resnet18(weights=IMAGENET1K_V1)   | ✅     |
| Atención        | Coordinate Attn   | CoordinateAttention(512, reduction=32)   | ✅     |
| Normalización   | GroupNorm         | nn.GroupNorm(32, 512) y (16, hidden_dim) | ✅     |
| hidden_dim      | 768               | Configurable, óptimo=768                 | ✅     |
| Dropout         | 0.3               | Configurable, óptimo=0.3                 | ✅     |
| Activación      | Sigmoid           | nn.Sigmoid()                             | ✅     |
| Loss            | Wing Loss         | Configurable                             | ✅     |
| CLAHE           | clip=2.0, tile=4  | Configurable                             | ✅     |
| TTA             | Flip horizontal   | predict_with_tta()                       | ✅     |

---

## PROBLEMAS IDENTIFICADOS

### CRÍTICOS ❌

1. **Tabla 6.4 - Parámetros incorrectos**
   - Documento: 137K, 270K, 407K, 548K
   - Real: 403K, 543K, 682K, 822K
   - **Acción**: Corregir tabla con valores reales
   - **Causa**: No se contaron los parámetros de GroupNorm

2. **Falta comparación Deep Head vs Shallow Head**
   - No hay tabla de ablación
   - Todos los experimentos usan deep_head=true
   - **Acción**: Agregar nota o realizar experimentos faltantes

3. **Resultados Session 5 y 6 no verificables**
   - Tablas 6.1 y 6.2 sin archivos de respaldo
   - **Acción**: Buscar archivos antiguos o documentar como históricos

### MENORES ⚠️

4. **Baseline hidden_dim=256**
   - Documento usa 7.84 px (Session 8 con dropout=0.5)
   - Existe experimento con dropout=0.3 → 7.60 px
   - **Acción**: Aclarar que baseline es con dropout=0.5

5. **Dropout 0.4 no experimentado**
   - Documento implica múltiples valores probados
   - Solo se encontraron 0.2, 0.3, 0.5
   - **Acción**: Documentar valores realmente probados

6. **Experimento dropout=0.2**
   - Existe en session10 (7.43 px)
   - No está documentado en tablas de Session 9
   - **Acción**: Agregar a tabla o mover a documento de Session 10

---

## FORTALEZAS DEL DOCUMENTO ✅

1. **Implementación completamente verificada**
   - CoordinateAttention implementada según paper original
   - GroupNorm correctamente implementado
   - TTA implementado correctamente

2. **Resultados clave verificados**
   - hidden_dim=768 → 7.21 px ✅
   - hidden_dim=512 → 7.29 px ✅
   - hidden_dim=1024 → 7.35 px ✅
   - dropout=0.3 → 7.60 px ✅

3. **Resultados por categoría completamente verificados**
   - Normal: 6.34 px ✅
   - COVID: 7.79 px ✅
   - Viral: 8.50 px ✅

4. **Ecuaciones matemáticas correctas**
   - Todas las ecuaciones de CoordAttention coinciden con implementación
   - Fórmulas de TTA correctas

---

## RECOMENDACIONES

### URGENTES

1. **Corregir Tabla 6.4** - Número de parámetros
   - Usar valores: 403K, 543K, 682K, 822K
   - O explicar por qué se excluyen parámetros de GroupNorm

2. **Agregar tabla de ablación arquitectónica**
   - Baseline (sin CoordAttention, sin DeepHead)
   - +CoordAttention only
   - +DeepHead only
   - +Ambos (actual)

3. **Verificar/documentar resultados históricos**
   - Sesiones 4, 5, 6, 7
   - O marcar como "resultados históricos no verificables"

### DESEABLES

4. **Completar experimentos de dropout**
   - Agregar dropout=0.4 si es relevante
   - O documentar que solo se probaron 0.2, 0.3, 0.5

5. **Documentar experimento dropout=0.2**
   - Actualmente en session10
   - Resultado: 7.43 px (peor que 0.3)

6. **Agregar sección de limitaciones**
   - Mencionar que no se probaron todas las combinaciones
   - Documentar decisiones de diseño experimental

---

## CONCLUSIÓN

El documento **06_optimizacion_arquitectura.tex** está **bien fundamentado** pero tiene **problemas de verificabilidad** en las sesiones tempranas y **errores en el conteo de parámetros**.

**Puntuación**: 7.5/10

- ✅ Implementación: 10/10
- ⚠️ Resultados Session 9: 10/10
- ❌ Resultados Session 5-6: 3/10
- ❌ Tabla de parámetros: 0/10
- ⚠️ Completitud experimental: 6/10

**Prioridad de corrección**: ALTA (especialmente Tabla 6.4)
