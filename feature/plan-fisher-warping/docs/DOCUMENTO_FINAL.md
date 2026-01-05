# Documento Final: Clasificacion de Radiografias con Eigenfaces, Fisher y KNN

## Resumen Ejecutivo

Este proyecto implementa un pipeline de clasificacion de radiografias de torax
para detectar neumonia (COVID-19 y viral) vs pacientes sanos. El objetivo
principal es demostrar que el **preprocesamiento de alineacion (warping)**
mejora significativamente la clasificacion, incluso con clasificadores simples
como K-Nearest Neighbors (KNN).

### Resultados Principales

| Dataset | Accuracy sin Warping | Accuracy con Warping | Mejora |
|---------|---------------------|----------------------|--------|
| Full (2 clases) | 79.26% | **81.47%** | +2.21% |
| Manual (2 clases) | 66.67% | **71.88%** | +5.21% |
| Full (3 clases) | 77.06% | **77.79%** | +0.74% |
| Manual (3 clases) | 62.50% | **64.58%** | +2.08% |

**Conclusion**: El warping mejora la clasificacion en TODOS los escenarios probados.

---

## 1. Introduccion

### 1.1 Problema

Las radiografias de torax pueden revelar patrones asociados con neumonia.
Sin embargo, las imagenes clinicas presentan variaciones en:
- Posicion del paciente
- Rotacion del torax
- Escala y encuadre

Estas variaciones dificultan la comparacion directa entre imagenes.

### 1.2 Hipotesis

> "Si alineamos las imagenes (warping) para normalizar posicion, rotacion y
> escala de las estructuras anatomicas, la clasificacion mejorara incluso
> con clasificadores simples."

### 1.3 Metodologia

Pipeline implementado:
1. **Warping**: Alineacion de imagenes a una referencia anatomica comun
2. **PCA (Eigenfaces)**: Reduccion de dimensionalidad
3. **Z-score**: Estandarizacion de caracteristicas
4. **Fisher Ratio**: Seleccion/ponderacion de caracteristicas discriminativas
5. **KNN**: Clasificacion por vecinos mas cercanos

---

## 2. Fundamentos Matematicos

### 2.1 PCA (Analisis de Componentes Principales)

#### Intuicion
Cada imagen de 224x224 pixeles es un punto en un espacio de 50,176 dimensiones.
PCA encuentra las direcciones de maxima varianza para reducir este espacio
a K dimensiones (usamos K=50).

#### Formulas

**Imagen promedio:**
```
mu = (1/N) * sum(x_i)  para i = 1..N
```

**Centrado de datos:**
```
X_centrada = X - mu
```

**Matriz de covarianza:**
```
C = (1/N) * X^T * X
```

**Eigenvectores y eigenvalores:**
```
C * v = lambda * v
```
- v_k = Eigenface k (direccion de varianza)
- lambda_k = Varianza capturada por eigenface k

**Proyeccion (extraccion de ponderantes):**
```
ponderantes = X_centrada @ eigenfaces^T
```

Los **ponderantes** son las caracteristicas que usamos para clasificacion.

#### Truco de Dimensionalidad
Para N << D (mas imagenes que pixeles), calculamos la matriz de covarianza
pequena (N x N) en lugar de la grande (D x D):

```
C_pequena = X * X^T / N     (N x N)
C_grande = X^T * X / N      (D x D)
```

Los eigenvectores de C_grande se obtienen de los de C_pequena mediante:
```
v_grande = X^T * v_pequena / sqrt(lambda)
```

### 2.2 Estandarizacion Z-Score

#### Intuicion
Diferentes caracteristicas tienen diferentes escalas. Sin estandarizar, las
caracteristicas con valores grandes dominarian el calculo de distancias.

#### Formula
Para cada caracteristica j:
```
media_j = (1/N) * sum(x_ij)  para i = 1..N
sigma_j = sqrt( (1/N) * sum( (x_ij - media_j)^2 ) )

z_ij = (x_ij - media_j) / sigma_j
```

Despues de estandarizar:
- Media = 0
- Desviacion estandar = 1

**IMPORTANTE**: Media y sigma se calculan SOLO con datos de training.
Se aplican los mismos valores a validation y test.

### 2.3 Criterio de Fisher

#### Intuicion
Mide que tan bien separa cada caracteristica a las dos clases.
Una caracteristica es buena si:
- Las medias de las clases estan lejos (numerador grande)
- Las clases son compactas (denominador pequeno)

#### Formula
Para la caracteristica j con clases Enfermo (e) y Normal (n):
```
J_j = (mu_e - mu_n)^2 / (sigma_e^2 + sigma_n^2)
```

Donde:
- mu_e = Media de caracteristica j para pacientes enfermos
- mu_n = Media de caracteristica j para pacientes normales
- sigma_e = Desviacion estandar para enfermos
- sigma_n = Desviacion estandar para normales

#### Ejemplo Numerico
```
Enfermos: media=2.0, sigma=0.5
Normales: media=-2.0, sigma=0.4

J = (2.0 - (-2.0))^2 / (0.5^2 + 0.4^2)
  = 16 / 0.41
  = 39.02   <-- Buena separacion
```

#### Extension a 3 Clases (Pairwise)
Para K clases, promediamos los Fisher ratios de todos los pares:
```
J = (1/C(K,2)) * sum(J_pares)

Para 3 clases:
J = (J_COVID_Normal + J_COVID_Viral + J_Normal_Viral) / 3
```

### 2.4 Amplificacion

#### Intuicion
Usamos el Fisher ratio como ponderador: amplificamos caracteristicas
discriminativas y atenuamos las no discriminativas.

#### Formula
```
x_amplificada_j = x_estandarizada_j * J_j
```

Efecto:
- Si J_j es grande -> caracteristica j se amplifica
- Si J_j es pequeno -> caracteristica j se atenua

### 2.5 KNN (K-Nearest Neighbors)

#### Intuicion
Para clasificar una imagen nueva:
1. Calculamos distancia a todas las imagenes de training
2. Tomamos los K mas cercanos
3. Votamos: la clase mas frecuente gana

#### Distancia Euclidiana
```
d(a, b) = sqrt( sum( (a_j - b_j)^2 ) )
```

#### Seleccion de K
Probamos K = 1, 3, 5, ..., 51 en el conjunto de validacion.
Elegimos el K con mayor accuracy.

---

## 3. Pipeline Completo

```
Imagen Warped (224x224)
        |
        v
    [APLANAR]  -->  Vector (50,176)
        |
        v
    [PCA]  -->  Ponderantes (50 dims)
    (entrenar solo con training)
        |
        v
    [Z-SCORE]  -->  Caracteristicas estandarizadas
    (media/sigma del training)
        |
        v
    [FISHER]  -->  Calcular ratio por caracteristica
    (entrenar solo con training)
        |
        v
    [AMPLIFICAR]  -->  Caracteristicas amplificadas
    (multiplicar por J_j)
        |
        v
    [KNN]  -->  Prediccion: Enfermo / Normal
    (K optimizado en validation)
```

---

## 4. Datasets

### 4.1 Fuentes

| Dataset | Descripcion | Training | Validation | Test |
|---------|-------------|----------|------------|------|
| Full Warped | Imagenes alineadas (224x224) | 5,040 | 1,005 | 680 |
| Full Original | Imagenes sin alinear (299x299) | 5,040 | 1,005 | 680 |
| Manual Warped | Subset manual alineado | 717 | 144 | 96 |
| Manual Original | Subset manual sin alinear | 717 | 144 | 96 |

### 4.2 Clases

**Escenario 2 clases (principal):**
- Enfermo: COVID-19 + Viral Pneumonia
- Normal: Pacientes sanos

**Escenario 3 clases (secundario):**
- COVID: Neumonia por COVID-19
- Viral_Pneumonia: Neumonia viral (no-COVID)
- Normal: Pacientes sanos

El escenario de 2 clases es el principal porque, radiograficamente, COVID y
neumonia viral son dificiles de distinguir (ambos causan patrones similares).

---

## 5. Resultados

### 5.1 PCA - Varianza Explicada

| Dataset | PC1 (%) | Top 10 (%) | K@95% |
|---------|---------|------------|-------|
| Full Warped | **46.4** | 82.0 | 50 |
| Full Original | 27.1 | 72.3 | 50 |
| Manual Warped | **49.0** | 82.0 | 50 |
| Manual Original | 28.3 | 72.7 | 50 |

**Hallazgo**: El warping concentra ~70% mas varianza en PC1 (46-49% vs 27-28%).
Esto indica que la alineacion organiza mejor la informacion.

### 5.2 Fisher Ratio - Caracteristicas Discriminativas

| Dataset | Mejor PC | J_max | J_mean |
|---------|----------|-------|--------|
| Full Warped | **PC1** | 0.262 | 0.018 |
| Full Original | PC2 | 0.357 | 0.020 |
| Manual Warped | **PC1** | 0.300 | 0.019 |
| Manual Original | PC2 | 0.259 | 0.018 |

**Hallazgo**: En datasets WARPED, PC1 es la mejor caracteristica discriminativa.
En datasets ORIGINALES, PC2 es mejor. El warping reorganiza la informacion
discriminativa hacia el componente principal.

### 5.3 Clasificacion KNN - 2 Clases

| Dataset | K optimo | Val Acc | Test Acc | Macro F1 |
|---------|----------|---------|----------|----------|
| Full Warped | 11 | 84.58% | **81.47%** | 0.804 |
| Full Original | 15 | 79.40% | 79.26% | 0.779 |
| Manual Warped | 5 | 76.39% | **71.88%** | 0.719 |
| Manual Original | 31 | 75.69% | 66.67% | 0.666 |

**Mejora por Warping:**
- Full: 81.47% - 79.26% = **+2.21%**
- Manual: 71.88% - 66.67% = **+5.21%**

### 5.4 Clasificacion KNN - 3 Clases

| Dataset | K optimo | Val Acc | Test Acc | Macro F1 |
|---------|----------|---------|----------|----------|
| Full Warped | 41 | 78.71% | **77.79%** | 0.786 |
| Full Original | 21 | 78.11% | 77.06% | 0.781 |
| Manual Warped | 15 | 65.28% | **64.58%** | 0.651 |
| Manual Original | 11 | 63.89% | 62.50% | 0.626 |

**Mejora por Warping:**
- Full: 77.79% - 77.06% = **+0.74%**
- Manual: 64.58% - 62.50% = **+2.08%**

### 5.5 Metricas por Clase (Mejor Modelo: Full Warped 2C)

| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Enfermo | 0.83 | 0.88 | 0.85 |
| Normal | 0.79 | 0.72 | 0.76 |

El modelo tiene mayor recall para Enfermos (88%), lo cual es deseable en
contexto medico: es preferible falsos positivos que falsos negativos.

---

## 6. Analisis de Errores

### 6.1 Resumen

| Escenario | Total Test | Errores | Error Rate |
|-----------|------------|---------|------------|
| Full Warped 2C | 680 | 126 | 18.53% |
| Full Original 2C | 680 | 141 | 20.74% |
| Manual Warped 2C | 96 | 27 | 28.12% |
| Manual Original 2C | 96 | 32 | 33.33% |

### 6.2 Impacto del Warping en Errores

| Grupo | Corregidos | Introducidos | Balance Neto |
|-------|------------|--------------|--------------|
| Full 2C | 73 | 58 | **+15** |
| Full 3C | 76 | 71 | **+5** |
| Manual 2C | 16 | 11 | **+5** |
| Manual 3C | 17 | 15 | **+2** |

**Total global:**
- Clasificaciones corregidas por warping: 182
- Clasificaciones empeoradas por warping: 155
- **Balance neto: +27 clasificaciones correctas**

### 6.3 Tipos de Errores (Full Warped 2C)

| Tipo | Descripcion | Cantidad |
|------|-------------|----------|
| Falso Negativo | Enfermo -> Normal | 51 |
| Falso Positivo | Normal -> Enfermo | 75 |

Los falsos positivos son mas frecuentes, pero son preferibles a los falsos
negativos en un contexto de screening medico.

### 6.4 Imagenes Problematicas

68 imagenes fallan en AMBOS escenarios (warped y original) para Full 2C.
Estas imagenes son inherentemente dificiles y pueden tener:
- Patologia sutil o atipica
- Problemas de calidad de imagen
- Caracteristicas ambiguas entre clases

---

## 7. Discusion

### 7.1 Por que funciona el Warping?

1. **Normalizacion espacial**: El warping alinea estructuras anatomicas
   (costillas, corazon, pulmones) a posiciones consistentes.

2. **Comparacion directa**: Al estar alineadas, pixel (i,j) de una imagen
   corresponde aproximadamente al mismo punto anatomico en otra imagen.

3. **Reduccion de varianza irrelevante**: PCA captura varianza. Sin warping,
   parte de la varianza se debe a variaciones de pose, no de patologia.

4. **Mejor concentracion de informacion**: PC1 en warped captura ~46% de
   varianza vs ~27% en original. La informacion esta mas concentrada.

### 7.2 Limitaciones

1. **Clasificador simple**: KNN es un baseline. Clasificadores mas sofisticados
   podrian mejorar resultados (pero ese no es el objetivo del estudio).

2. **Dataset**: COVID-19 Radiography Dataset tiene calidad variable y
   potenciales sesgos de coleccion.

3. **Warping imperfecto**: El proceso de warping no es perfecto y puede
   introducir artefactos en algunos casos.

### 7.3 Comparacion con Deep Learning

El asesor advirtio contra usar CNNs para esta demostracion:

> "La red neuronal convolucional en realidad ya esta preparada para que las
> imagenes no esten alineadas."

Las CNNs tienen mecanismos internos (pooling, convoluciones) que las hacen
parcialmente invariantes a desalineaciones. Usar CNN ocultaria el beneficio
del warping. Por eso usamos un clasificador "naive" (KNN).

---

## 8. Conclusiones

1. **El warping mejora la clasificacion** en todos los escenarios probados
   (2 clases y 3 clases, datasets full y manual).

2. **La mejora es mas notable en datasets pequenos** (Manual: +5.21% vs
   Full: +2.21%), sugiriendo que el warping es especialmente util cuando
   hay pocos datos.

3. **El warping reorganiza la informacion**: PC1 se vuelve la caracteristica
   mas discriminativa despues del warping.

4. **El balance de errores favorece al warping**: Corrige mas errores de los
   que introduce (182 vs 155).

5. **La metodologia es reproducible**: Todo el codigo esta implementado
   desde cero, sin dependencias de alto nivel como sklearn para los
   algoritmos principales.

---

## 9. Archivos Generados

### Metricas
```
results/metrics/
├── phase4_features/      # Caracteristicas PCA
├── phase5_fisher/        # Fisher ratios
├── phase6_classification/ # Resultados 2 clases
└── phase7_comparison/    # Resultados 3 clases
```

### Figuras
```
results/figures/
├── phase2_samples/       # Muestras warped vs original
├── phase3_pca/          # Eigenfaces y varianza
├── phase4_features/     # Distribuciones Z-score
├── phase5_fisher/       # Fisher ratios por caracteristica
├── phase6_classification/ # Matrices de confusion
└── phase7_comparison/   # Comparacion 2C vs 3C
```

### Logs
```
results/logs/
├── analisis_errores.txt  # Analisis detallado de errores
└── *.txt                 # Logs de cada fase
```

---

## 10. Referencias

1. Turk, M., & Pentland, A. (1991). Eigenfaces for Recognition.
   Journal of Cognitive Neuroscience, 3(1), 71-86.

2. Fisher, R.A. (1936). The Use of Multiple Measurements in Taxonomic Problems.
   Annals of Eugenics, 7(2), 179-188.

3. COVID-19 Radiography Database.
   https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

---

*Documento generado: 2026-01-05*
*Pipeline: Eigenfaces + Fisher + KNN*
*Branch: feature/plan-fisher-warping*
