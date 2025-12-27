# Notas de Defensa: Validación Geométrica (Fisher Linear Analysis)

> **Objetivo de la Tesis:** Demostrar que el *Piecewise Affine Warping* alinea geométricamente las estructuras pulmonares, haciendo que la distinción entre "Sano" y "Enfermo" sea más evidente linealmente.

---

## 1. El Concepto del Asesor (La Intuición)

El asesor planteó una hipótesis geométrica fundamental:

> *"Las imágenes no son píxeles aleatorios; viven en una variedad (manifold) de baja dimensión. Si tu Warping funciona, debería 'desenredar' estas variedades. No necesitamos Deep Learning para probar esto; la geometría clásica basta."*

### La Lógica paso a paso:
1.  **Espacio Latente (Eigen-space):** Al usar PCA, encontramos las direcciones donde los pulmones varían más (forma del tórax, diafragma, etc.).
2.  **Ponderantes (Weights):** Cada imagen es una combinación lineal de "Eigenfaces". Los coeficientes de esa combinación son sus coordenadas en este espacio.
3.  **Estandarización:** Antes de medir nada, debemos poner todas las características en la misma escala (z-score), porque el primer componente principal tiene muchísima más varianza que el último.
4.  **Fisher Ratio:** Medimos matemáticamente la separación. Si el warping es bueno, la distancia entre la nube de puntos "Sanos" y "Enfermos" debe aumentar relativa a su dispersión.

---

## 2. El Algoritmo Matemático

### A. PCA (Reducción de Dimensionalidad)
$$ X_{pca} = (X - \mu) \cdot V^T $$
Donde $V$ son los autovectores de la matriz de covarianza.

### B. Criterio de Fisher (Score $J$)
Para cada dimensión $i$, calculamos qué tanto separa las clases:

$$ J_i = \frac{(\mu_{i, sano} - \mu_{i, enfermo})^2}{\sigma^2_{i, sano} + \sigma^2_{i, enfermo}} $$

*   **Numerador:** Distancia entre los centros de las clases (queremos que sea grande).
*   **Denominador:** Suma de las varianzas internas (queremos que sea pequeña).

### C. Amplificación (Weighting)
Transformamos el espacio multiplicando cada dimensión por su importancia discriminante:
$$ X'_{i} = X_{pca, i} \cdot \sqrt{J_i} $$
*(Nota: En la versión estricta usamos raíz cuadrada para no suprimir excesivamente características débiles).*

---

## 3. Implementación Clave (NumPy)

```python
# 1. Estandarizar Ponderantes (Crucial según Asesor)
# "Tienes que ponerlos a competir en igualdad de condiciones"
mean = np.mean(X_pca, axis=0)
std = np.std(X_pca, axis=0)
X_std = (X_pca - mean) / std

# 2. Calcular Fisher Score (J)
# Separamos las nubes de puntos
c0 = X_std[y==0]
c1 = X_std[y==1]

# Aplicamos la fórmula
numerador = (np.mean(c0) - np.mean(c1))**2
denominador = np.var(c0) + np.var(c1)
J = numerador / denominador

# 3. Amplificar (Proyección Final)
# "Dale volumen a lo que importa, silencia el ruido"
X_final = X_std * np.sqrt(J)
```

---

## 4. Resultados Clave (Evidencia)

| Experimento | Método | Accuracy (Test) | Observaciones |
| :--- | :--- | :--- | :--- |
| **Baseline** | Pixeles Crudos (Raw) + PCA | ~82.0% | Referencia base. |
| **Warped (Basic)** | CPU, Multiplicador $J$ | ~82.8% | Mejora marginal o negativa en algunos casos. |
| **Warped (Strict)** | **GPU, Multiplicador $\sqrt{J}$** | **85.31%** | **El resultado hito.** Supera claramente al baseline. |

**Conclusión:** El warping, combinado con la ponderación correcta ($\sqrt{J}$), expone características discriminantes que estaban ocultas en la geometría original.

---

## 5. Análisis de Discrepancia: CPU vs GPU (3% de Diferencia)

Durante la validación, notamos que el script de GPU (`scripts/fisher/thesis_validation_fisher.py`) superaba sistemáticamente al de CPU (`scripts/fisher/thesis_validation_fisher_basic.py`).

**Causa Raíz:**
No es el hardware, es la **matemática de pre-procesamiento**.

1.  **Normalización de Píxeles vs Centrado (El factor clave):**
    *   **GPU (Strict - Binario):** Realiza **solo centrado** de píxeles (`X - mean`) antes del PCA. NO divide por la desviación estándar de los píxeles. Esto preserva la estructura de intensidad relativa (importante para rayos X) y evita amplificar ruido de fondo.
    *   **Multiclase:** SÍ normaliza píxeles completo (`(X-mean)/std`), probablemente necesario por la varianza entre múltiples fuentes de datos.
    *   **CPU (Basic):** Solo centra (comportamiento default de Sklearn), pero fallaba en la etapa siguiente (ponderación).

2.  **Estandarización de Ponderantes (La Clave del 86%):**
    *   Una vez en el espacio PCA, el método Strict aplica **Z-score a los ponderantes (weights)**. Esto pone a competir todas las características latentes en igualdad de condiciones antes de calcular el Fisher Score.
    *   Sin esto, el primer componente principal domina el cálculo solo por magnitud.

3.  **Factor de Amplificación:**
    *   **GPU (Strict):** Usa $\sqrt{J}$. Esto es una ponderación "suave".
    *   **CPU (Basic):** Usa $J$. Al ser valores menores a 1 (ej: 0.05), elevar al cuadrado (implícitamente al usar $J$ directamente como peso lineal frente a la varianza) suprime demasiado las características secundarias.

**Veredicto:** El método "Estricto" (GPU) es el metodológicamente correcto para imágenes médicas con variaciones de contraste.
