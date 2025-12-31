# Pipeline Completo: Eigenfaces + Fisher + KNN

Este documento describe el pipeline EXACTO solicitado por el asesor.

## Diagrama del Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASE DE ENTRENAMIENTO                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 1. CARGAR IMAGENES WARPED                                       │
│    - Fuente: outputs/full_warped_dataset/train/                 │
│    - Tamano: 224x224 pixeles                                    │
│    - Clases: Enfermo (COVID + Viral_Pneumonia), Normal          │
│    - Salida: Lista de imagenes + etiquetas                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 2. APLANAR IMAGENES                                             │
│    - Entrada: Imagen (224, 224)                                 │
│    - Proceso: Reshape a vector                                  │
│    - Salida: Vector (50176,)                                    │
│    - Matriz X_train: (N_train, 50176)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 3. PCA (EIGENFACES)                                             │
│    - Entrada: X_train (N_train, 50176)                          │
│    - Calcular: imagen_promedio = mean(X_train)                  │
│    - Centrar: X_centrada = X_train - imagen_promedio            │
│    - SVD: U, S, Vt = svd(X_centrada)                            │
│    - Seleccionar: K componentes principales (ej: K=10)          │
│    - Guardar: imagen_promedio, componentes[:K]                  │
│    - Salida: Modelo PCA entrenado                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 4. EXTRAER PONDERANTES (CARACTERISTICAS)                        │
│    - Entrada: X_centrada, componentes[:K]                       │
│    - Proceso: ponderantes = X_centrada @ componentes.T          │
│    - Salida: Matriz de ponderantes (N_train, K)                 │
│    - IMPORTANTE: Estos ponderantes SON las caracteristicas      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 5. ESTANDARIZACION Z-SCORE                                      │
│    - Entrada: ponderantes (N_train, K)                          │
│    - Para cada columna j (caracteristica):                      │
│        media_j = mean(ponderantes[:, j])                        │
│        sigma_j = std(ponderantes[:, j])                         │
│    - Aplicar: z = (x - media) / sigma                           │
│    - Guardar: medias, sigmas (para aplicar a val/test)          │
│    - Salida: ponderantes_estandarizados (N_train, K)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 6. CALCULAR FISHER RATIO                                        │
│    - Entrada: ponderantes_estandarizados, etiquetas             │
│    - Para cada caracteristica j:                                │
│        - Separar valores por clase                              │
│        - Calcular mu_enfermo, mu_normal                         │
│        - Calcular sigma_enfermo, sigma_normal                   │
│        - J_j = (mu_e - mu_n)^2 / (sigma_e^2 + sigma_n^2)        │
│    - Guardar: vector J de K valores                             │
│    - Salida: fisher_ratios (K,)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 7. AMPLIFICAR CARACTERISTICAS                                   │
│    - Entrada: ponderantes_estandarizados, fisher_ratios         │
│    - Proceso: amplificados = ponderantes_estand * fisher_ratios │
│    - Salida: caracteristicas_amplificadas (N_train, K)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 8. GUARDAR MODELO                                               │
│    - imagen_promedio (50176,)                                   │
│    - componentes_pca (K, 50176)                                 │
│    - medias_zscore (K,)                                         │
│    - sigmas_zscore (K,)                                         │
│    - fisher_ratios (K,)                                         │
│    - X_train_amplificado (N_train, K)                           │
│    - y_train (N_train,)                                         │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    FASE DE INFERENCIA                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 1. CARGAR IMAGEN NUEVA                                          │
│    - Imagen warped de test (224, 224)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 2. APLANAR                                                      │
│    - Vector (50176,)                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 3. PROYECTAR CON PCA                                            │
│    - Centrar: x_centrada = x - imagen_promedio                  │
│    - Proyectar: ponderantes = x_centrada @ componentes.T        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 4. ESTANDARIZAR                                                 │
│    - Usar medias y sigmas del training                          │
│    - z = (ponderantes - medias) / sigmas                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 5. AMPLIFICAR                                                   │
│    - Usar fisher_ratios del training                            │
│    - amplificados = z * fisher_ratios                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 6. CLASIFICAR CON KNN                                           │
│    - Calcular distancia a todos los puntos de training          │
│    - Tomar los K=5 mas cercanos                                 │
│    - Votar: clase con mas vecinos gana                          │
│    - Salida: Prediccion (Enfermo / Normal)                      │
└─────────────────────────────────────────────────────────────────┘


## Parametros del Pipeline

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| Tamano imagen | 224x224 | Tamano de imagenes warped existentes |
| Dimensiones aplanadas | 50,176 | 224 * 224 |
| Componentes PCA (K) | 10-50 | El asesor sugiere 10 como ejemplo |
| Vecinos KNN | 5 | Valor comun, puede ajustarse |
| Clases | 2 | Enfermo vs Normal (segun asesor) |

## Rutas de Datos

### Entrada
- Training warped: `outputs/full_warped_dataset/train/`
- Validation warped: `outputs/full_warped_dataset/val/`
- Test warped: `outputs/full_warped_dataset/test/`
- Training original (para comparacion): `data/dataset/COVID-19_Radiography_Dataset/`

### Splits Balanceados (2 clases)
- `results/metrics/02_full_balanced_2class_warped.csv`
- `results/metrics/02_full_balanced_2class_original.csv`

### Salida
- Modelo: `results/models/fisher_pipeline.pkl`
- Metricas: `results/metrics/`
- Figuras: `results/figures/`

## Pseudocodigo

```python
# === ENTRENAMIENTO ===

# 1. Cargar datos
X_train, y_train = cargar_imagenes_warped("train", clases=2)
X_train = aplanar(X_train)  # (N, 50176)

# 2. PCA
imagen_promedio = X_train.mean(axis=0)
X_centrada = X_train - imagen_promedio
U, S, Vt = np.linalg.svd(X_centrada, full_matrices=False)
componentes = Vt[:K]  # (K, 50176)

# 3. Extraer ponderantes
ponderantes = X_centrada @ componentes.T  # (N, K)

# 4. Estandarizar
medias = ponderantes.mean(axis=0)
sigmas = ponderantes.std(axis=0)
ponderantes_std = (ponderantes - medias) / sigmas

# 5. Fisher
fisher_ratios = []
for j in range(K):
    col = ponderantes_std[:, j]
    mu_e = col[y_train == 1].mean()  # Enfermo
    mu_n = col[y_train == 0].mean()  # Normal
    s_e = col[y_train == 1].std()
    s_n = col[y_train == 0].std()
    J = (mu_e - mu_n)**2 / (s_e**2 + s_n**2 + 1e-9)
    fisher_ratios.append(J)
fisher_ratios = np.array(fisher_ratios)

# 6. Amplificar
X_train_amp = ponderantes_std * fisher_ratios

# === INFERENCIA ===

def predecir(imagen):
    x = aplanar(imagen)
    x_centrada = x - imagen_promedio
    ponderantes = x_centrada @ componentes.T
    z = (ponderantes - medias) / sigmas
    amplificado = z * fisher_ratios

    # KNN
    distancias = np.sqrt(((X_train_amp - amplificado)**2).sum(axis=1))
    indices_cercanos = np.argsort(distancias)[:5]
    votos = y_train[indices_cercanos]
    prediccion = 1 if votos.sum() > 2.5 else 0
    return prediccion
```

## Experimento Principal

### Comparacion CON vs SIN Warping

Para demostrar que el warping mejora la clasificacion:

1. **Pipeline con Warping**:
   - Usar imagenes de `outputs/full_warped_dataset/`
   - Ejecutar pipeline completo
   - Registrar accuracy y F1

2. **Pipeline SIN Warping**:
   - Usar imagenes originales de `data/dataset/`
   - Mismo split (por ID de imagen)
   - Mismo K, mismos hiperparametros
   - Registrar accuracy y F1

3. **Comparar**:
   - Tabla: | Configuracion | Accuracy | F1 Macro |
   - Esperado: Warped > Original
