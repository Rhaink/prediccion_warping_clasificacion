# thesis_validation_fisher_basic_standard.py
# -----------------------------------------------------------------------------
# VERSIÓN ESTÁNDAR (Práctica común en ML)
# Flujo: Estandarización de Píxeles -> PCA -> Fisher -> Amplificación
# Objetivo: Comparar contra la versión "Estricta" del asesor.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Configuración básica
IMAGE_SIZE = 224
COMPONENTS = 50

def cargar_datos_simples(image_root_dir, split_name):
    # (Misma función de carga que ya funciona bien)
    csv_path = Path(image_root_dir) / split_name / "images.csv"
    print(f"\n[1] Cargando datos del conjunto '{split_name}' desde {csv_path}...")
    
    if not csv_path.exists():
        print(f"Error: No se encontró el archivo CSV en {csv_path}")
        return np.array([]), np.array([])
    
    df = pd.read_csv(csv_path)
    if 'split' in df.columns: df = df[df['split'] == split_name]
    
    imagenes = []
    etiquetas = []
    total = len(df)
    for i, row in df.iterrows():
        if i % 100 == 0: print(f"    Procesando imagen {i}/{total}...", end='\r')
        nombre = row['image_name']
        categoria = row['category']
        if 'warped_filename' in row: nombre_archivo = row['warped_filename']
        else: nombre_archivo = f"{nombre}.png"
            
        posibles_rutas = [
            Path(image_root_dir) / split_name / categoria / nombre_archivo,
            Path(image_root_dir) / categoria / nombre_archivo,
            Path(image_root_dir) / nombre_archivo
        ]
        
        ruta_valida = None
        for p in posibles_rutas:
            if p.exists(): ruta_valida = p; break
        
        if ruta_valida:
            img = cv2.imread(str(ruta_valida), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            imagenes.append(img.flatten())
            etiquetas.append(0 if categoria == 'Normal' else 1)
            
    print(f"    Carga completada: {len(imagenes)} imágenes encontradas.")
    return np.array(imagenes), np.array(etiquetas)

def criterio_fisher_manual(X_pca, y):
    print("\n[3] Calculando Criterio de Fisher (Manual)...")
    num_componentes = X_pca.shape[1]
    fisher_ratios = []
    for i in range(num_componentes):
        componente = X_pca[:, i]
        datos_clase_0 = componente[y == 0]
        datos_clase_1 = componente[y == 1]
        
        media_0 = np.mean(datos_clase_0)
        media_1 = np.mean(datos_clase_1)
        var_0 = np.var(datos_clase_0)
        var_1 = np.var(datos_clase_1)
        
        numerador = (media_0 - media_1) ** 2
        denominador = var_0 + var_1 + 1e-9
        fisher_ratios.append(numerador / denominador)
        
    return np.array(fisher_ratios)

def main():
    print("="*60)
    print("VALIDACIÓN GEOMÉTRICA - VERSIÓN ESTÁNDAR (Píxeles Normalizados)")
    print("="*60)
    
    dataset_dir = "outputs/warped_dataset"
    
    print("\n--- FASE 1: CARGA DE DATOS ---")
    X_train, y_train = cargar_datos_simples(dataset_dir, "train")
    X_test, y_test = cargar_datos_simples(dataset_dir, "test")
    
    if len(X_train) == 0: return

    # --- DIFERENCIA CLAVE: Normalización ANTES de PCA ---
    print("\n--- FASE 2: NORMALIZACIÓN DE PÍXELES ---")
    print("[2] Estandarizando píxeles (Media=0, Var=1)...")
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    print("\n--- FASE 3: PCA ---")
    print(f"[3] Calculando Eigenfaces sobre datos normalizados...")
    pca = PCA(n_components=COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    print(f"    Varianza explicada: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # Fisher y Amplificación (Igual que siempre)
    print("\n--- FASE 4: CRITERIO DE FISHER ---")
    fisher_scores = criterio_fisher_manual(X_train_pca, y_train)
    
    print("\n--- FASE 5: AMPLIFICACIÓN ---")
    X_train_final = X_train_pca * fisher_scores
    X_test_final = X_test_pca * fisher_scores
    
    print("\n--- FASE 6: CLASIFICACIÓN ---")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_final, y_train)
    preds = knn.predict(X_test_final)
    
    acc = accuracy_score(y_test, preds)
    print("\n" + "#"*40)
    print(f"RESULTADO FINAL ESTÁNDAR: Accuracy = {acc*100:.2f}%")
    print("#"*40)
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    main()
