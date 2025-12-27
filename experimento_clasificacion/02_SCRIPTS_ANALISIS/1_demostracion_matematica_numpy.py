#!/usr/bin/env python3
"""
PROYECTO: Validación Geométrica de Warping Pulmonar
SCRIPT: 1_demostracion_matematica_numpy.py
OBJETIVO: Implementación pura de Fisher Linear Analysis (Sin cajas negras de ML).

Este script sigue estrictamente las instrucciones del asesor capturadas en audio.
Se utilizan comentarios literales para mapear la matemática con la solicitud.
"""

import numpy as np
from pathlib import Path
from lib.utils import cargar_dataset

# =============================================================================
# BLOQUE 1: CONSTRUCCIÓN DEL EIGEN-SPACE (SVD)
# =============================================================================
def calcular_espacio_propio(X, n_componentes=50):
    """
    ASESOR: "A esos dos grupos, construyes un solo Eigen-space."
    ASESOR: "Que serían las 10 [o N] que corresponden a las 10 mayores varianzas."
    """
    print(f"    [MATE] Calculando SVD manual sobre matriz de {X.shape}...")
    
    # 1. Centrado (Restar la imagen promedio)
    media_pixeles = np.mean(X, axis=0)
    X_centrada = X - media_pixeles
    
    # 2. SVD (Singular Value Decomposition)
    # Álgebra lineal pura: X = U * S * Vt
    U, S, Vt = np.linalg.svd(X_centrada, full_matrices=False)
    
    # 3. Eigenfaces (Direcciones de varianza)
    eigenfaces = Vt[:n_componentes]
    
    # 4. Proyección para obtener Ponderantes
    # ASESOR: "Las características no serían las Eigenfaces. Las características serían los ponderantes."
    ponderantes = np.dot(X_centrada, eigenfaces.T)
    
    return ponderantes, eigenfaces, media_pixeles

# =============================================================================
# BLOQUE 2: ESTANDARIZACIÓN DE PONDERANTES
# =============================================================================
def estandarizar_ponderantes(P_train, P_test):
    """
    ASESOR: "Vamos a estandarizar solo la característica 1... de esos 1,000 valores sacas la media..."
    ASESOR: "A cada valor le restas la media y luego esa diferencia la divides entre la desviación estándar."
    """
    # Estadísticos sobre entrenamiento
    media_p = np.mean(P_train, axis=0)
    std_p = np.std(P_train, axis=0) + 1e-9 # Estabilidad
    
    # Aplicar transformación Z-score
    P_train_std = (P_train - media_p) / std_p
    P_test_std = (P_test - media_p) / std_p
    
    return P_train_std, P_test_std

# =============================================================================
# BLOQUE 3: CÁLCULO DE RAZÓN DE FISHER (J)
# =============================================================================
def calcular_fisher_scores(P_std, y):
    """
    ASESOR: "De esos 1,000 valores, tienes 500 que son para la neumonía y tienes otros 500 que son para los sanos."
    ASESOR: "Sacas una media número 1... sacas la 'Media número 2'. Haces lo mismo para la desviación estándar."
    ASESOR: "Y con esos sacas el Criterio de Fisher."
    """
    # Separar nubes de puntos
    c1 = P_std[y == 1] # Neumonía/COVID
    c2 = P_std[y == 0] # Sanos
    
    # Medias y Desviaciones
    m1, m2 = np.mean(c1, axis=0), np.mean(c2, axis=0)
    s1, s2 = np.std(c1, axis=0), np.std(c2, axis=0)
    
    # Fórmula de Fisher (Dispersión entre clases vs Dispersión intra clase)
    numerador = (m1 - m2)**2
    denominador = (s1**2) + (s2**2) + 1e-9
    
    return numerador / denominador

# =============================================================================
# BLOQUE 4: AMPLIFICACIÓN Y CLASIFICACIÓN
# =============================================================================
def clasificador_knn_numpy(X_train, y_train, X_test, k=5):
    """
    ASESOR: "Puede ser un KNN... incluso con un clasificador tan chafa o tan simple como un KNN."
    """
    preds = []
    for x in X_test:
        dist = np.sqrt(np.sum((X_train - x)**2, axis=1))
        vecinos = y_train[np.argsort(dist)[:k]]
        preds.append(np.argmax(np.bincount(vecinos)))
    return np.array(preds)

# =============================================================================
# FLUJO PRINCIPAL: COMPARATIVA DE ESCENARIOS
# =============================================================================
def ejecutar_experimento(nombre, ruta, tipo):
    print(f"\n>>> EJECUTANDO ESCENARIO: {nombre} <<<")
    
    # 1. Cargar Datos
    # Límite de Seguridad para Demostración en CPU (SVD es muy costoso en memoria RAM)
    # Nota: El 86% se demuestra con el script de GPU; este script es para validar la matemática.
    # Usamos 1500 para asegurar que el proceso termine exitosamente en esta máquina.
    limite = 1500
    
    X_train, y_train = cargar_dataset(ruta, tipo, split="train", limite=limite)
    X_test, y_test = cargar_dataset(ruta, tipo, split="test", limite=500)
    
    if X_train is None: return
    
    # 2. Espacio Propio (SVD Manual)
    P_train, E, mu_px = calcular_espacio_propio(X_train, n_componentes=50)
    P_test = np.dot(X_test - mu_px, E.T)
    
    # 3. Estandarización
    P_train_std, P_test_std = estandarizar_ponderantes(P_train, P_test)
    
    # 4. Fisher Score y Amplificación
    # ASESOR: "Si separa muy bien, pues el ponderante es muy alto, entonces la estás amplificando."
    j_scores = calcular_fisher_scores(P_train_std, y_train)
    
    # Amplificamos multiplicando los datos estandarizados por su score de Fisher
    P_train_final = P_train_std * j_scores
    P_test_final = P_test_std * j_scores
    
    # 5. Clasificación KNN
    y_pred = clasificador_knn_numpy(P_train_final, y_train, P_test_final, k=5)
    
    # Resultado
    acc = np.mean(y_pred == y_test)
    print(f"    [RESULTADO] Accuracy {nombre}: {acc*100:.2f}%")
    return acc

if __name__ == "__main__":
    BASE = Path(__file__).parent.parent / "01_DATOS_ENTRADA"
    
    # Fijar semilla para reproducibilidad
    np.random.seed(8)
    
    # Escenarios solicitados
    ejecutar_experimento("RAW (Original)", BASE / "COVID-19_Radiography_Dataset", "raw")
    ejecutar_experimento("WARPED (Small)", BASE / "warped_dataset", "warped")
    ejecutar_experimento("WARPED (Full)", BASE / "full_warped_dataset", "warped")