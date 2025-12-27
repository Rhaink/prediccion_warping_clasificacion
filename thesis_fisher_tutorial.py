# thesis_fisher_tutorial.py
# -----------------------------------------------------------------------------
# TUTORIAL DE DEFENSA DE TESIS: VALIDACIÓN GEOMÉTRICA CON FISHER LINEAR DISCRIMINANT
# -----------------------------------------------------------------------------
# AUTOR: Tu Asistente de IA (bajo tu dirección)
# FECHA: 23 de Diciembre, 2025
#
# OBJETIVO:
# Este script es una radiografía matemática del "Método del Asesor".
# Desglosa cada caja negra de Scikit-Learn/PyTorch en operaciones básicas de Álgebra Lineal
# usando SOLAMENTE NumPy.
#
# INSTRUCCIONES DE USO:
# Ejecutar: python thesis_fisher_tutorial.py
# Leer la salida en consola como si fuera una historia.
# -----------------------------------------------------------------------------

import numpy as np
import cv2
from pathlib import Path
import time

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================
RUTA_DATASET = "outputs/warped_dataset" # Ajustar si es necesario
TAMANO_IMAGEN = 64  # Reducido para que el tutorial corra rápido en CPU
NUMERO_COMPONENTES_PRINCIPALES = 20 # "Eigen-spaces" a analizar
VECINOS_KNN = 5

def imprimir_titulo(texto):
    print(f"\n{'='*80}")
    print(f" {texto.upper()}")
    print(f"{ '='*80}")

def imprimir_paso(numero, titulo, explicacion):
    print(f"\n>> PASO {numero}: {titulo}")
    print(f"   Explicación: {explicacion}")

# =============================================================================
# 1. CARGA DE DATOS (Artesanal)
# =============================================================================
def cargar_dataset_didactico(ruta_base, split, limite=200):
    """
    Carga un subconjunto pequeño de datos para demostración.
    """
    ruta_csv = Path(ruta_base) / split / "images.csv"
    if not ruta_csv.exists():
        print(f"   [AVISO] No se encontró {ruta_csv}. Usando datos sintéticos (Ruido) para demostración.")
        # Generar datos falsos si no existe el dataset real
        N = limite
        D = TAMANO_IMAGEN * TAMANO_IMAGEN
        # Simular: Clase 0 (Normal) tiene media 0, Clase 1 (Enfermo) tiene media 0.5 (ligero shift)
        X_sanos = np.random.randn(N // 2, D)
        X_enfermos = np.random.randn(N // 2, D) + 0.2 
        y_sanos = np.zeros(N // 2)
        y_enfermos = np.ones(N // 2)
        return np.vstack([X_sanos, X_enfermos]), np.hstack([y_sanos, y_enfermos])

    import pandas as pd
    df = pd.read_csv(ruta_csv)
    if "split" in df.columns:
        df = df[df["split"] == split]
    
    # Limitar para el tutorial
    df = df.head(limite)
    
    matrices_imagenes = []
    vector_etiquetas = []
    
    print(f"   Leyendo {len(df)} imágenes de disco...")
    for _, fila in df.iterrows():
        # Lógica de ruta (simplificada)
        nombre = fila.get("warped_filename", f"{fila['image_name']}.png")
        cat = fila["category"]
        
        # Intentar encontrar el archivo
        posibles_rutas = [
            Path(ruta_base) / split / cat / nombre,
            Path(ruta_base) / cat / nombre,
        ]
        
        archivo_encontrado = None
        for p in posibles_rutas:
            if p.exists():
                archivo_encontrado = p
                break
        
        if archivo_encontrado:
            # Leer en escala de grises
            img_bgr = cv2.imread(str(archivo_encontrado))
            if img_bgr is None: continue
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (TAMANO_IMAGEN, TAMANO_IMAGEN))
            
            # APLANAR: De matriz (64,64) a vector (4096,)
            matrices_imagenes.append(img_resized.flatten())
            vector_etiquetas.append(0 if cat == "Normal" else 1)

    return np.array(matrices_imagenes), np.array(vector_etiquetas)

# =============================================================================
# BLOQUES DE CONSTRUCCIÓN MATEMÁTICA (REEMPLAZANDO SCIKIT-LEARN)
# =============================================================================

def pca_artesanal_svd(X, n_componentes):
    """
    Implementación de PCA usando Descomposición en Valores Singulares (SVD).
    PCA es esencialmente rotar los datos para alinear la mayor varianza con los ejes.
    """
    imprimir_paso(2, "ANÁLISIS DE COMPONENTES PRINCIPALES (PCA - Manual)", 
                  "Transformamos los píxeles (espacio original) a 'Eigenfaces' (espacio latente).")

    # 1. Centrar los datos (Restar la media de cada píxel)
    #    Es crucial para que el PCA rote alrededor del origen de la nube de puntos.
    print("   -> Calculando vector promedio de la imagen...")
    vector_media = np.mean(X, axis=0)
    X_centrada = X - vector_media
    
    # 2. SVD (Singular Value Decomposition)
    #    Descompone X en U (rotación izquierda), S (estiramiento), Vt (rotación derecha/Eigenvectors)
    #    Usamos 'full_matrices=False' para eficiencia (SVD truncado implícito por dimensiones)
    print("   -> Ejecutando SVD (esto puede tardar un poco)...")
    U, S, Vt = np.linalg.svd(X_centrada, full_matrices=False)
    
    # 3. Seleccionar los primeros N componentes (los más importantes)
    #    Vt tiene los Eigenvectors (las "direcciones" principales) en las filas.
    componentes_principales = Vt[:n_componentes]
    
    # 4. Proyectar los datos al nuevo espacio
    #    Producto punto: Datos_Centrados · Componentes_Transpuestos
    print("   -> Proyectando datos al espacio latente...")
    X_proyectada = np.dot(X_centrada, componentes_principales.T)
    
    return X_proyectada, componentes_principales, vector_media

def estandarizador_manual(X):
    """
    Estandarización Z-Score: (X - media) / desviación_estándar
    """
    imprimir_paso(3, "ESTANDARIZACIÓN Z-SCORE", 
                  "El asesor insistió: 'Estandariza los ponderantes'. Hacemos media=0, var=1.")
    
    media = np.mean(X, axis=0)
    desviacion = np.std(X, axis=0)
    
    # Evitar división por cero
    desviacion[desviacion == 0] = 1.0
    
    X_estandarizada = (X - media) / desviacion
    return X_estandarizada, media, desviacion

def criterio_fisher_score(X, y):
    """
    Calcula el 'Fisher Score' (J) para cada característica (columna).
    J = Dispersión_Entre_Clases / Dispersión_Intra_Clase
    """
    imprimir_paso(4, "CÁLCULO DE FISHER RATIOS (J)", 
                  "Medimos qué tan bien separa cada componente a los enfermos de los sanos.")
    
    clases = np.unique(y)
    num_features = X.shape[1]
    scores_j = np.zeros(num_features)
    
    print(f"   -> Analizando {num_features} componentes...")
    
    for i in range(num_features):
        feature_col = X[:, i]
        
        # Separar por clase
        datos_c0 = feature_col[y == 0] # Sanos
        datos_c1 = feature_col[y == 1] # Enfermos
        
        # Medias
        mu0 = np.mean(datos_c0)
        mu1 = np.mean(datos_c1)
        
        # Varianzas
        var0 = np.var(datos_c0)
        var1 = np.var(datos_c1)
        
        # Fórmula de Fisher
        numerador = (mu0 - mu1)**2         # Distancia entre centros de las campanas
        denominador = var0 + var1 + 1e-9   # Suma de los anchos de las campanas
        
        scores_j[i] = numerador / denominador
        
    return scores_j

def clasificador_knn_manual(X_train, y_train, X_test, k=5):
    """
    Clasificador K-Nearest Neighbors implementado con pura distancia Euclidiana.
    """
    imprimir_paso(6, "CLASIFICACIÓN K-NN (Manual)", 
                  f"Buscamos los {k} vecinos más cercanos usando distancia Euclidiana.")
    
    predicciones = []
    num_test = X_test.shape[0]
    
    print("   -> Calculando distancias (fuerza bruta)...")
    # Para cada punto de prueba...
    for i in range(num_test):
        punto_test = X_test[i]
        
        # 1. Distancia Euclidiana a TODOS los puntos de entrenamiento
        #    D = raiz( suma( (x1-x2)^2 ) )
        #    Truco NumPy: Broadcasting permite restar el punto a toda la matriz X_train
        diferencias = X_train - punto_test
        distancias_cuadradas = np.sum(diferencias**2, axis=1)
        distancias = np.sqrt(distancias_cuadradas)
        
        # 2. Encontrar los índices de los K más cercanos
        indices_k_cercanos = np.argsort(distancias)[:k]
        
        # 3. Votación
        etiquetas_vecinos = y_train[indices_k_cercanos]
        votos_0 = np.sum(etiquetas_vecinos == 0)
        votos_1 = np.sum(etiquetas_vecinos == 1)
        
        prediccion = 0 if votos_0 > votos_1 else 1
        predicciones.append(prediccion)
        
    return np.array(predicciones)

# =============================================================================
# FLUJO PRINCIPAL (MAIN)
# =============================================================================
def main():
    imprimir_titulo("Tutorial Interactivo: Fisher Validation")
    
    # 1. Cargar Datos
    # Usamos limit=300 para que sea rápido pero representativo
    print("\n--- Fase: Carga ---")
    X_train_raw, y_train = cargar_dataset_didactico(RUTA_DATASET, "train", limite=300)
    X_test_raw, y_test = cargar_dataset_didactico(RUTA_DATASET, "test", limite=100)
    
    print(f"   Shape Train: {X_train_raw.shape}")
    print(f"   Shape Test:  {X_test_raw.shape}")
    
    # 2. PCA Manual
    # Entrenamos PCA solo con TRAIN
    X_train_pca, componentes, media_train = pca_artesanal_svd(X_train_raw, NUMERO_COMPONENTES_PRINCIPALES)
    
    # Proyectamos TEST usando los parámetros aprendidos en TRAIN
    print("   -> Aplicando proyección PCA al conjunto de Test...")
    X_test_centrada = X_test_raw - media_train
    X_test_pca = np.dot(X_test_centrada, componentes.T)
    
    # 3. Estandarización de Ponderantes
    # El asesor dice: "Normaliza las características (ponderantes) antes de medir Fisher"
    X_train_std, media_std, dev_std = estandarizador_manual(X_train_pca)
    
    # Estandarizamos TEST con los parámetros de TRAIN
    X_test_std = (X_test_pca - media_std) / dev_std
    
    # 4. Cálculo de Scores de Fisher
    fisher_ratios = criterio_fisher_score(X_train_std, y_train)
    
    print(f"   -> Top 5 Fisher Ratios: {np.round(np.sort(fisher_ratios)[::-1][:5], 4)}")
    
    # 5. AMPLIFICACIÓN (El Corazón del Método)
    imprimir_paso(5, "AMPLIFICACIÓN DE CARACTERÍSTICAS", 
                  "Multiplicamos cada dimensión por su importancia Fisher (J).")
    
    # IMPORTANTE: Aquí está la discrepancia CPU vs GPU.
    # CPU (Basic): Multiplica por J
    # GPU (Strict): Multiplica por sqrt(J)
    # Seguiremos el script BASIC (CPU) como se pidió en el tutorial, pero imprimimos la nota.
    print("   [NOTA DE ESTUDIO] En la versión 'Basic' usamos J. En la 'Strict/GPU' usamos sqrt(J).")
    print("   Usaremos J para este tutorial (más agresivo).")
    
    ponderadores = fisher_ratios # O np.sqrt(fisher_ratios)
    
    X_train_fisher = X_train_std * ponderadores
    X_test_fisher = X_test_std * ponderadores
    
    # 6. Clasificación
    y_pred = clasificador_knn_manual(X_train_fisher, y_train, X_test_fisher, k=VECINOS_KNN)
    
    # Evaluación
    aciertos = np.sum(y_pred == y_test)
    total = len(y_test)
    accuracy = aciertos / total
    
    imprimir_titulo("RESULTADOS FINALES")
    print(f"ACCURACY OBTENIDO: {accuracy*100:.2f}%")
    print(f"({aciertos} de {total} imágenes clasificadas correctamente)")
    
    print("\n[FIN DEL TUTORIAL] Ahora revisa el código fuente para leer los comentarios.")

if __name__ == "__main__":
    main()
