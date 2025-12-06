# DESCUBRIMIENTOS GEOMETRICOS CRITICOS

## Fecha: 27-Nov-2024
## Analisis realizado sobre 957 muestras del dataset

---

## 1. ESTRUCTURA PARAMETRICA DEL ETIQUETADO

El analisis revela que el proceso de etiquetado manual crea una estructura geometrica **CASI PERFECTA**:

### 1.1 El Eje Central (L1-L2)

| Metrica | Valor |
|---------|-------|
| Longitud promedio | 198.2 +/- 32.8 px |
| Angulo con vertical | -0.21 +/- 4.00 grados |
| **Conclusion** | El eje es CASI PERFECTAMENTE VERTICAL |

### 1.2 Puntos Centrales (L9, L10, L11)

Los puntos centrales estan **CASI PERFECTAMENTE** sobre el eje L1-L2:

| Punto | Distancia al eje | Posicion en eje (t) |
|-------|-----------------|---------------------|
| L9 | 1.37 +/- 1.13 px | 0.249 +/- 0.010 (teorico: 0.25) |
| L10 | 1.30 +/- 1.13 px | 0.500 +/- 0.010 (teorico: 0.50) |
| L11 | 1.35 +/- 1.11 px | 0.749 +/- 0.010 (teorico: 0.75) |

**HALLAZGO CLAVE**: Los puntos centrales dividen el eje en **EXACTAMENTE 4 PARTES IGUALES** con error de solo ~1.4 px.

### 1.3 Representacion Parametrica de Todos los Landmarks

Cada landmark puede representarse con dos parametros relativos al eje L1-L2:
- **t**: Posicion a lo largo del eje (0 = L1, 1 = L2)
- **d**: Distancia perpendicular al eje (negativo = izquierda, positivo = derecha)

```
Landmark    t (pos eje)     d (dist horiz)    Interpretacion
---------------------------------------------------------------
L1          0.000           0.0 px            Origen del eje
L2          1.000           0.0 px            Fin del eje
L3          0.247          -86.2 px           Apice izquierdo
L4          0.248          +86.5 px           Apice derecho
L5          0.499          -99.2 px           Hilio izquierdo
L6          0.499          +98.8 px           Hilio derecho
L7          0.748         -106.6 px           Base izquierda
L8          0.749         +106.0 px           Base derecha
L9          0.249           -0.3 px           Centro 1/4
L10         0.500           -0.2 px           Centro 1/2
L11         0.749           -0.3 px           Centro 3/4
L12        -0.001          -45.4 px           Borde sup izq
L13        -0.001          +46.8 px           Borde sup der
L14         0.998         -112.7 px           Seno costof izq
L15         0.999         +111.6 px           Seno costof der
```

### 1.4 Relacion d/longitud_eje (Proporcionalidad)

Las distancias horizontales son **PROPORCIONALES** a la longitud del eje:

| Par | Ratio d/longitud_eje |
|-----|---------------------|
| L3-L4 (apices) | 0.444 +/- 0.076 |
| L5-L6 (hilios) | 0.511 +/- 0.084 |
| L7-L8 (bases) | 0.549 +/- 0.091 |
| L14-L15 (costof) | 0.581 +/- 0.098 |

**Correlacion longitud_eje vs d**: r = 0.41-0.44 (significativa, p < 1e-39)

---

## 2. ERROR IRREDUCIBLE EN ANOTACIONES

### 2.1 Fuentes de Error

1. **Ruido de etiquetado base**: ~1.3-1.5 px
   - Evidencia: L9, L10, L11 estan a ~1.3 px del eje perfecto
   - Causa: Movimiento discreto con teclas, imprecision del cursor

2. **Asimetria bilateral en GT**: ~5.5-7.9 px
   - L3-L4: 5.51 +/- 4.58 px
   - L5-L6: 5.55 +/- 5.20 px
   - L7-L8: 6.82 +/- 5.85 px
   - L14-L15: 7.89 +/- 6.84 px

### 2.2 Error Minimo Teorico Alcanzable

| Escenario | Error Esperado |
|-----------|---------------|
| Perfecto (aprendizaje ideal de ruido) | 1.5-2 px |
| Excelente (mejor realista) | 5-6 px |
| Muy bueno (objetivo alcanzable) | 6-8 px |
| Actual (baseline) | 10.91 px |

**CONCLUSION**: Un error de <8 px es REALISTA pero DESAFIANTE. Un error de 5-6 px seria EXCELENTE.

---

## 3. IMPLICACIONES PARA EL MODELO

### 3.1 NO Forzar Simetria Perfecta

El Ground Truth **NO ES SIMETRICO**. Si forzamos simetria perfecta, introducimos error.

**Solucion**: Usar **Soft Symmetry Loss** con margen:

```python
def soft_symmetry_loss(asimetria, margin=6.0):
    # Solo penaliza si asimetria > margen
    if asimetria < margin:
        return 0
    else:
        return (asimetria - margin)^2
```

Evaluacion en GT con diferentes margenes:
- Margen 0 px: Loss = 72.31 (muy alto, GT penalizado)
- Margen 5 px: Loss = 28.96
- Margen 8 px: Loss = 16.51 (GT casi sin penalizacion)

**Recomendacion**: Margen de 5-8 px

### 3.2 Explotar Estructura Geometrica

**OPCION A: Loss Geometrico durante Entrenamiento**

```
Total_Loss = Wing_Loss
           + 0.3 * Central_Alignment_Loss
           + 0.1 * Soft_Symmetry_Loss(margin=6)
           + 0.1 * Height_Alignment_Loss
```

Donde:
- **Central_Alignment_Loss**: dist(L9,L10,L11 al eje L1-L2) / 224
- **Soft_Symmetry_Loss**: Solo penaliza asimetria > 6 px
- **Height_Alignment_Loss**: |Y_izq - Y_der| para pares

**OPCION B: Arquitectura Jerarquica (INNOVADORA)**

```
ETAPA 1: Predecir L1, L2 (el eje central)
  - Input: Imagen 224x224
  - Output: 4 valores (L1_x, L1_y, L2_x, L2_y)
  - Objetivo: Error < 5 px en L1, L2

ETAPA 2: Predecir desplazamientos relativos
  - Input: Imagen + Eje de Etapa 1
  - Output: 13 valores
    - 3 refinamientos de t (para L9, L10, L11)
    - 10 distancias d (para 5 pares bilaterales)

RECONSTRUCCION:
  - L9 = L1 + (0.25 + dt9) * (L2 - L1)
  - L3 = punto_en_eje(t=0.25) + d3 * perpendicular
  - etc.
```

Ventajas:
1. Reduce dimensionalidad: 30 → 4 + 13 = 17 parametros efectivos
2. Las restricciones estan EMBEBIDAS en la arquitectura
3. El modelo aprende la estructura geometrica explicitamente
4. Mas facil de interpretar y depurar

### 3.3 Post-Procesamiento Geometrico

Mejora marginal (~0.5 px) pero util:

1. **Proyeccion de centrales**: Forzar L9, L10, L11 sobre el eje predicho
   - Mejora ~2.5 px para L9, L10, L11
   - No afecta otros landmarks

2. **NO simetrizar pares**: Empeora porque las predicciones de L1,L2 tienen error

---

## 4. NUEVAS LOSS FUNCTIONS PROPUESTAS

### 4.1 Central Alignment Loss

```python
def central_alignment_loss(pred):
    """
    Penaliza que L9, L10, L11 no esten sobre el eje L1-L2
    """
    L1, L2 = pred[:, 0], pred[:, 1]
    eje = L2 - L1
    eje_unit = eje / norm(eje)

    total = 0
    for i in [8, 9, 10]:  # L9, L10, L11
        vec = pred[:, i] - L1
        proj = dot(vec, eje_unit) * eje_unit
        perp = vec - proj
        dist = norm(perp)
        total += dist

    return total / 3
```

En GT: 1.34 +/- 0.87 px (muy bajo, restriccion casi perfecta)

### 4.2 Soft Symmetry Loss

```python
def soft_symmetry_loss(pred, margin=6.0):
    """
    Penaliza SOLO asimetrias grandes (> margin)
    """
    L1, L2 = pred[:, 0], pred[:, 1]
    perp = perpendicular(L2 - L1)

    total = 0
    for (l, r) in [(2,3), (4,5), (6,7), (11,12), (13,14)]:
        d_l = abs(dot(pred[:, l] - L1, perp))
        d_r = abs(dot(pred[:, r] - L1, perp))
        asim = abs(d_l - d_r)
        loss = max(0, asim - margin)^2
        total += loss

    return total / 5
```

En GT con margin=6: 28.96 (permitiendo asimetria natural)

### 4.3 Height Alignment Loss

```python
def height_alignment_loss(pred):
    """
    Penaliza que pares bilaterales no esten a la misma altura
    """
    total = 0
    for (l, r) in [(2,3), (4,5), (6,7), (11,12), (13,14)]:
        diff_y = abs(pred[:, l, 1] - pred[:, r, 1])
        total += diff_y

    return total / 5
```

En GT: 8.20 +/- 8.67 px

---

## 5. RESUMEN DE RECOMENDACIONES

### Implementar Obligatoriamente:

1. **Central Alignment Loss** (peso 0.3)
   - Forzar L9, L10, L11 sobre el eje
   - GT ya lo cumple (~1.3 px error)

2. **Soft Symmetry Loss con margen 6 px** (peso 0.1)
   - NO forzar simetria perfecta
   - Solo penalizar asimetrias > 6 px

3. **Post-procesamiento de centrales**
   - Proyectar L9, L10, L11 sobre eje predicho despues de inferencia

### Considerar Experimentalmente:

1. **Arquitectura Jerarquica** (predecir eje primero, luego desplazamientos)
2. **Pesos diferenciados por landmark** (mas peso a L14, L15 que son mas dificiles)
3. **Height Alignment Loss** (peso 0.1)

### NO Hacer:

1. NO forzar simetria perfecta (el GT no es simetrico)
2. NO reconstruir landmarks completamente desde parametros (introduce error)
3. NO ignorar la estructura geometrica del etiquetado

---

## 6. CASOS ESPECIALES Y OUTLIERS

### 6.1 Variabilidad por Categoria

| Categoria | Muestras | Variabilidad (σ) | Outliers |
|-----------|----------|-----------------|----------|
| COVID | 306 (32%) | 20.1 px | 10.5% |
| Normal | 468 (49%) | 17.0 px | 4.5% |
| Viral | 183 (19%) | 12.5 px | 1.1% |

**Observacion critica**: Viral tiene MUCHA MENOS variabilidad - un modelo unico tiene que "comprometerse".

### 6.2 Muestras Problematicas

**Distribucion de asimetria por muestra:**
- Min: 0.4 px
- Percentil 50: 5.2 px
- Percentil 90: 11.2 px
- Max: 39.2 px

**20.7% de muestras tienen geometria inconsistente** (asimetria > 8 px en algun par)

**Top 5 muestras mas problematicas:**
1. [601] Normal-8317: asim=39.2px
2. [369] COVID-1258: asim=33.9px
3. [373] COVID-1558: asim=32.9px
4. [379] COVID-2281: asim=25.7px
5. [285] COVID-2933: asim=24.1px

### 6.3 Recomendacion de Filtrado

| Umbral | Muestras retenidas | Variabilidad |
|--------|-------------------|--------------|
| ≤10 px | 808 (84%) | σ=17.5 px |
| ≤15 px | 919 (96%) | σ=17.6 px |
| ≤20 px | 948 (99%) | σ=17.7 px |

**Recomendacion**: Filtrar muestras con asimetria > 15 px (elimina 4%, reduce ruido)

---

## 7. SOLUCIONES ADICIONALES DESCUBIERTAS

### 7.1 Correlaciones Clave

**Hallazgo**: Las coordenadas Y son ALTAMENTE predecibles desde L1, L2:
- R² > 0.98 para TODAS las coordenadas Y
- Las X de L9, L10, L11 tienen R² > 0.97
- Las X de bilaterales tienen R² ~0.4-0.5 (mas dificiles)

**Correlaciones mas fuertes (r > 0.95):**
- L1_Y ↔ L12_Y, L13_Y (r=0.98) → L12, L13 al nivel de L1
- L2_Y ↔ L14_Y, L15_Y (r=0.97) → L14, L15 al nivel de L2
- L1_X ↔ L9_X (r=0.96) → L9 alineado con L1

### 7.2 Prediccion Dual X/Y

Y es SIEMPRE mas facil de predecir:
- Todos los 15 landmarks tienen mayor variabilidad en Y que en X
- Pero Y tiene altisima correlacion con L1, L2

**Estrategia propuesta:**
```
Cabeza Y: Predice 15 coordenadas Y (alta correlacion, converge rapido)
Cabeza X: Predice 15 coordenadas X (mas capacidad, LR diferente)
```

### 7.3 Curriculum Learning

Orden de dificultad (menor a mayor):
1. L9, L10 (σ ~20 px) - MAS FACILES
2. L1, L4, L13, L3, L12 (σ ~22-24 px)
3. L6, L5, L11 (σ ~24-25 px)
4. L8, L7 (σ ~28-30 px)
5. L2, L15, L14 (σ ~32-35 px) - MAS DIFICILES

**Estrategia:**
- Fase 1: Entrenar en eje central (L1, L9, L10, L11)
- Fase 2: Anadir bilaterales medios
- Fase 3: Anadir L2, L14, L15

### 7.4 Modelos Especializados por Categoria

| Categoria | Variabilidad | Beneficio esperado |
|-----------|-------------|-------------------|
| Viral | σ=12.5 px | Modelo mas preciso |
| Normal | σ=17.0 px | Precision media |
| COVID | σ=20.1 px | Mas dificil |

**Opciones:**
1. 3 modelos separados + clasificador
2. Mixture of Experts con categoria como gate
3. Modelo unico con embedding de categoria

### 7.5 Pesos Adaptativos por Landmark

```python
PESOS_SUGERIDOS = {
    'L1': 1.16, 'L2': 0.79,  # Eje central
    'L3': 1.07, 'L4': 1.11,  # Apices
    'L5': 1.00, 'L6': 1.04,  # Hilios
    'L7': 0.85, 'L8': 0.89,  # Bases
    'L9': 1.30, 'L10': 1.21, 'L11': 0.99,  # Centrales (MAYOR peso)
    'L12': 1.06, 'L13': 1.07,  # Bordes sup
    'L14': 0.71, 'L15': 0.74   # Costofrenicos (MENOR peso)
}
```

**Logica**: Landmarks faciles tienen peso MAYOR para que el modelo los aprenda bien primero.

---

## 8. IMPACTO ESPERADO ACTUALIZADO

### Estrategias Principales (obligatorias)

| Estrategia | Impacto Estimado |
|------------|-----------------|
| Central Alignment Loss | 0.5-1.0 px |
| Soft Symmetry Loss (margen 6px) | 0.3-0.5 px |
| Post-proc de centrales | 0.2-0.5 px |
| Pesos por landmark | 0.3-0.5 px |

### Estrategias Secundarias (si necesario)

| Estrategia | Impacto Estimado |
|------------|-----------------|
| Filtrado outliers (asim>15px) | 0.2-0.5 px |
| Prediccion dual X/Y | 0.3-0.7 px |
| Ensemble 5 modelos | 0.5-1.0 px |
| Modelos por categoria | 0.5-1.5 px |

### Estrategias Avanzadas

| Estrategia | Impacto Estimado |
|------------|-----------------|
| Arquitectura jerarquica | 1.0-2.0 px |
| Heatmap regression | Variable |

### Proyeccion Final

```
Actual:                    10.91 px
Con estrategias 1-4:       ~8.5-9.5 px
Con estrategias 1-8:       ~7-8 px
Con arq. jerarquica:       ~6-7 px (potencial)
Limite teorico:            ~5-6 px
```
