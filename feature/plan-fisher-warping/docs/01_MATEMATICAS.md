# Fundamentos Matematicos del Pipeline

Este documento explica las matematicas detras de cada paso del pipeline,
sin cajas negras, como lo solicito el asesor.

## 1. PCA (Analisis de Componentes Principales) / Eigenfaces

### Intuicion

Imagina que tienes 1,000 radiografias de 224x224 pixeles. Cada imagen es un
punto en un espacio de **50,176 dimensiones** (224 x 224 = 50,176 pixeles).

El problema: es imposible visualizar o trabajar eficientemente en 50,176 dimensiones.

PCA encuentra las **direcciones de maxima varianza** en este espacio. Son las
direcciones donde los datos "se estiran" mas. Proyectar a estas direcciones
reduce dimensiones mientras preserva la informacion mas importante.

### Proceso Paso a Paso

**Paso 1: Aplanar las imagenes**

Cada imagen de 224x224 se convierte en un vector de 50,176 elementos:

```
Imagen (224, 224) --> Vector (50176,)
```

**Paso 2: Calcular la imagen promedio**

Sumamos todas las imagenes del training y dividimos entre el numero total:

```
imagen_promedio = (1/N) * suma(todas_las_imagenes)
```

Esto nos da el "rostro promedio" o en nuestro caso, la "radiografia promedio".

**Paso 3: Centrar los datos**

A cada imagen le restamos la imagen promedio:

```
imagen_centrada = imagen - imagen_promedio
```

Esto mueve el centro de los datos al origen (0,0,...,0).

**Paso 4: Calcular la matriz de covarianza**

La covarianza mide como varian los pixeles juntos:

```
C = (1/N) * X^T * X
```

Donde X es la matriz de imagenes centradas (cada fila es una imagen).

**Paso 5: Encontrar eigenvectores y eigenvalores**

Resolvemos: `C * v = lambda * v`

- **Eigenvector (v)**: Direccion principal de varianza (una "eigenface")
- **Eigenvalor (lambda)**: Cantidad de varianza en esa direccion

**Paso 6: Seleccionar K componentes**

Ordenamos eigenvalores de mayor a menor y tomamos los K primeros.
El asesor sugiere K=10 como ejemplo.

### Formula de Proyeccion

Para proyectar una imagen al espacio reducido:

```
ponderantes = (imagen_centrada) * eigenvectores_K
```

Resultado: Un vector de K ponderantes (ej: 10 numeros).

### Lo que dijo el asesor

> "Las caracteristicas no serian las Eigenfaces. Las caracteristicas serian
> los ponderantes... Los pesos. Esos serian las caracteristicas."

---

## 2. Estandarizacion Z-Score

### Intuicion

Supongamos que tenemos 10 caracteristicas (ponderantes). La caracteristica 1
tiene valores entre -1000 y +1000, mientras que la caracteristica 2 tiene
valores entre -0.01 y +0.01.

Si usamos estas caracteristicas directamente, la caracteristica 1 dominaria
cualquier calculo de distancia simplemente por tener valores mas grandes,
no por ser mas informativa.

La estandarizacion pone todas las caracteristicas en la misma escala.

### Formula

Para cada caracteristica i, con los N valores del training:

```
media_i = (1/N) * suma(x_i)
sigma_i = sqrt( (1/N) * suma( (x_i - media_i)^2 ) )

z_i = (x_i - media_i) / sigma_i
```

Despues de estandarizar:
- Media = 0
- Desviacion estandar = 1

### Lo que dijo el asesor

> "De esos 1,000 valores sacas la media. Y le sacas la desviacion estandar...
> a cada valor le restas la media y luego esa diferencia la divides entre
> la desviacion estandar."

### Nota Importante

La media y sigma se calculan SOLO con el training set.
Luego se aplican esos mismos valores a validation y test.

---

## 3. Criterio de Fisher (Fisher Ratio)

### Intuicion

Queremos saber que tan bien separa cada caracteristica a las dos clases
(Enfermo vs Normal).

Imaginemos la caracteristica 1 graficada como un histograma:
- Los valores de pacientes "Enfermos" forman una campana
- Los valores de pacientes "Normales" forman otra campana

El Fisher Ratio mide:
- **Numerador**: Que tan separadas estan las medias de las dos campanas
- **Denominador**: Que tan anchas (dispersas) son las campanas

Si las campanas estan lejos y son estrechas -> Fisher GRANDE -> buena separacion
Si las campanas se traslapan -> Fisher PEQUENO -> mala separacion

### Formula

Para la caracteristica i:

```
J_i = (mu_enfermo - mu_normal)^2 / (sigma_enfermo^2 + sigma_normal^2)
```

Donde:
- mu_enfermo: Media de la caracteristica i para pacientes enfermos
- mu_normal: Media de la caracteristica i para pacientes normales
- sigma_enfermo: Desviacion estandar para enfermos
- sigma_normal: Desviacion estandar para normales

### Ejemplo Numerico

Supongamos caracteristica 1 con 1000 imagenes (500 enfermos, 500 normales):

**Enfermos:**
- Valores: [2.1, 2.3, 1.9, 2.0, ...]
- Media (mu_e) = 2.0
- Sigma (sigma_e) = 0.5

**Normales:**
- Valores: [-1.8, -2.1, -1.9, -2.0, ...]
- Media (mu_n) = -2.0
- Sigma (sigma_n) = 0.4

**Calculo:**
```
J_1 = (2.0 - (-2.0))^2 / (0.5^2 + 0.4^2)
    = (4.0)^2 / (0.25 + 0.16)
    = 16 / 0.41
    = 39.02
```

Este valor alto indica que la caracteristica 1 separa muy bien las clases.

### Lo que dijo el asesor

> "Si esa Razon de Fisher es grande, significa que esa caracteristica separa
> bien las clases. Si esa Razon de Fisher es cero o chiquita, significa que
> esa caracteristica no separa bien las clases."

---

## 4. Amplificacion

### Intuicion

Una vez que sabemos que tan bien separa cada caracteristica (Fisher ratio),
podemos usar ese conocimiento para "amplificar" las buenas y "atenuar" las malas.

Es como subir el volumen de los instrumentos que suenan bien y bajar los que
hacen ruido.

### Formula

Para cada caracteristica i:

```
caracteristica_amplificada_i = caracteristica_estandarizada_i * J_i
```

### Lo que dijo el asesor

> "Con esa Razon de Fisher, puedes usarla como un ponderante... la estas
> amplificando. Si separaba muy bien, como quien dice la ganancia o la
> amplificacion pues va a ser grande. Si separa mal, la amplificacion
> va a ser chiquita."

---

## 5. KNN (K-Nearest Neighbors)

### Intuicion

Para clasificar una nueva imagen:
1. Calculamos sus caracteristicas amplificadas
2. Medimos la distancia a TODOS los ejemplos del training
3. Tomamos los K mas cercanos
4. Votamos: la clase con mas votos gana

### Distancia Euclidiana

```
distancia(a, b) = sqrt( suma( (a_i - b_i)^2 ) )
```

### Ejemplo

Nueva imagen con caracteristicas amplificadas: [0.5, -0.3, 1.2]

Los 5 vecinos mas cercanos son:
- Vecino 1: Enfermo (distancia 0.8)
- Vecino 2: Normal (distancia 0.9)
- Vecino 3: Enfermo (distancia 1.0)
- Vecino 4: Enfermo (distancia 1.1)
- Vecino 5: Normal (distancia 1.2)

Votos: Enfermo=3, Normal=2 --> Prediccion: **Enfermo**

### Lo que dijo el asesor

> "Puede ser un KNN... No va a ser mucha la diferencia en la clasificacion.
> Lo que hace la diferencia es que hayas alineado esas canijas imagenes."

---

## Resumen del Pipeline Completo

```
Imagen Warped (224x224)
        |
        v
    [APLANAR]
        |
        v
    Vector (50176)
        |
        v
    [PCA] (solo training)
        |
        v
    Ponderantes (K dims, ej: 10)  <-- Estas son las CARACTERISTICAS
        |
        v
    [Z-SCORE] (media/sigma del training)
        |
        v
    Caracteristicas Estandarizadas
        |
        v
    [FISHER RATIO] (por caracteristica)
        |
        v
    [AMPLIFICAR] (multiplicar por J_i)
        |
        v
    Caracteristicas Amplificadas
        |
        v
    [KNN]
        |
        v
    Prediccion: Enfermo / Normal
```
