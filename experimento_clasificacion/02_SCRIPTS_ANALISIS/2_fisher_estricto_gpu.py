#!/usr/bin/env python3
"""
PROYECTO: Validación Geométrica de Warping Pulmonar
SCRIPT: 2_fisher_estricto_gpu.py
OBJETIVO: Ejecución masiva del algoritmo de Fisher usando Aceleración GPU (PyTorch).

Este script es la implementación optimizada para procesar el dataset completo (>15,000 imágenes)
manteniendo la fidelidad matemática de las instrucciones del asesor.

RESULTADO ESPERADO: ~86% Accuracy.
"""

import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 8
OUTPUT_DIR = Path(__file__).parent.parent / "03_EVIDENCIA_RESULTADOS"
DATASET_DIR = Path(__file__).parent.parent / "01_DATOS_ENTRADA" / "full_warped_dataset"
RAW_DIR = Path(__file__).parent.parent / "01_DATOS_ENTRADA" / "COVID-19_Radiography_Dataset"

# Crear directorios de salida
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "comparativa_visual").mkdir(exist_ok=True)

# Fijar semilla
torch.manual_seed(SEED)
np.random.seed(SEED)

class DatasetLoaderGPU:
    def __init__(self, root_dir, raw_dir, img_size=224):
        self.root_dir = Path(root_dir)
        self.raw_dir = Path(raw_dir)
        self.img_size = img_size
        # ASESOR: "estuve trabajando con el CLAHE" -> Vital para el 86%
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def load_split(self, split):
        print(f"[CARGA] Leyendo conjunto '{split}' desde {self.root_dir}...")
        csv_path = self.root_dir / split / "images.csv"
        if not csv_path.exists():
            csv_path = self.root_dir / "images.csv"
        
        df = pd.read_csv(csv_path)
        if "split" in df.columns:
            df = df[df["split"] == split]
            
        X_list = []
        y_list = []
        metadata = [] # Para rastrear nombres de archivo
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cargando {split}"):
            name = row['image_name']
            cat = row['category']
            fname = row.get('warped_filename', row.get('filename', f"{name}.png"))
            
            # Buscar archivo
            p = next((p for p in [self.root_dir/split/cat/fname, self.root_dir/cat/fname] if p.exists()), None)
            
            if p:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize -> CLAHE (Orden correcto)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = self.clahe.apply(img)
                    
                    X_list.append(img.flatten())
                    y_list.append(0 if cat == 'Normal' else 1)
                    metadata.append({'name': name, 'category': cat, 'path': str(p)})
                    
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.long)
        return X, y, metadata

class FisherAnalysisGPU:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.mean_px = None
        self.V = None # Eigenfaces
        self.w_mean = None
        self.w_std = None
        self.fisher_weights = None

    def fit(self, X):
        """
        ASESOR: "Construyes un solo Eigen-space... las 10 [50] mayores varianzas."
        """
        X = X.to(DEVICE)
        
        # 1. Centrado de Píxeles
        self.mean_px = torch.mean(X, dim=0)
        X_centered = X - self.mean_px
        
        # 2. PCA vía SVD (Optimizado Torch)
        print("[GPU] Calculando PCA (SVD Lowrank)...")
        # torch.pca_lowrank es equivalente a RandomizedSVD pero en GPU
        U, S, V = torch.pca_lowrank(X_centered, q=self.n_components, center=False, niter=2)
        self.V = V
        
        # 3. Proyección (Obtener Ponderantes)
        # ASESOR: "Las características serían los ponderantes."
        weights = torch.mm(X_centered, V)
        
        # 4. Estandarización de Ponderantes
        # ASESOR: "Estandarizar solo la característica 1... (X - media) / std"
        self.w_mean = torch.mean(weights, dim=0)
        self.w_std = torch.std(weights, dim=0) + 1e-9
        
        return weights # Retornamos weights crudos para el paso de Fisher externo

    def compute_fisher_and_amplify(self, weights, y):
        """
        ASESOR: "Sacas el Criterio de Fisher... y la usas como ponderante para multiplicar."
        """
        weights = weights.to(DEVICE)
        y = y.to(DEVICE)
        
        # Estandarizar
        weights_std = (weights - self.w_mean) / self.w_std
        
        # Fisher Score J
        c1 = weights_std[y == 1]
        c0 = weights_std[y == 0]
        
        m1, m0 = torch.mean(c1, dim=0), torch.mean(c0, dim=0)
        v1, v0 = torch.var(c1, dim=0), torch.var(c0, dim=0)
        
        numerator = (m1 - m0)**2
        denominator = v1 + v0 + 1e-9
        J = numerator / denominator
        
        # ASESOR: "Si separa muy bien... el ponderante es alto... la estás amplificando."
        # Usamos sqrt(J) para amplitud física correcta
        self.fisher_weights = torch.sqrt(J)
        
        return weights_std * self.fisher_weights

    def transform(self, X):
        X = X.to(DEVICE)
        # Pipeline completo para Test
        X_centered = X - self.mean_px
        weights = torch.mm(X_centered, self.V)
        weights_std = (weights - self.w_mean) / self.w_std
        return weights_std * self.fisher_weights

def knn_predict(X_train, y_train, X_test, k=5):
    """
    ASESOR: "Puede ser un KNN... incluso con un clasificador tan simple."
    Optimizado por lotes para GPU.
    """
    y_preds = []
    batch_size = 1000
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    
    print("[GPU] Ejecutando KNN...")
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size].to(DEVICE)
        
        # Distancia Euclidiana
        dist = torch.cdist(X_batch, X_train, p=2)
        
        # Top K
        knn = dist.topk(k, largest=False)
        votes = y_train[knn.indices]
        
        # Moda (Votación)
        preds, _ = torch.mode(votes, dim=1)
        y_preds.append(preds.cpu())
        
    return torch.cat(y_preds)

def main():
    print("=== VALIDACIÓN FISHER ESTRICTA (GPU) ===")
    print(f"Dataset: {DATASET_DIR}")
    
    loader = DatasetLoaderGPU(DATASET_DIR, RAW_DIR)
    analyzer = FisherAnalysisGPU(n_components=50)
    
    # 1. Cargar Datos Completos
    X_train, y_train, meta_train = loader.load_split("train")
    X_test, y_test, meta_test = loader.load_split("test")
    
    # 2. Entrenamiento
    raw_weights_train = analyzer.fit(X_train)
    X_train_final = analyzer.compute_fisher_and_amplify(raw_weights_train, y_train)
    
    # 3. Transformación Test
    X_test_final = analyzer.transform(X_test)
    
    # 4. Clasificación
    y_pred = knn_predict(X_train_final, y_train, X_test_final, k=5)
    
    # 5. Métricas y Reportes
    acc = accuracy_score(y_test.numpy(), y_pred.numpy())
    print(f"\n>>> ACCURACY FINAL: {acc*100:.2f}% <<<")
    
    # Guardar Matriz de Confusión
    cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sano', 'Enfermo'], yticklabels=['Sano', 'Enfermo'])
    plt.title(f'Matriz de Confusión (Acc: {acc*100:.2f}%)')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.savefig(OUTPUT_DIR / "matriz_confusion_final.png")
    print(f"[REPORTE] Gráfica guardada en {OUTPUT_DIR}/matriz_confusion_final.png")
    
    # Guardar Reporte de Texto
    with open(OUTPUT_DIR / "metricas_oficiales.txt", "w") as f:
        f.write(f"Accuracy Global: {acc*100:.4f}%\n\n")
        f.write("Reporte de Clasificación:\n")
        f.write(classification_report(y_test.numpy(), y_pred.numpy(), target_names=['Sano', 'Enfermo']))
        f.write("\n\nMatriz de Confusión:\n")
        f.write(str(cm))
        
    # Guardar Gráfica de Fisher Scores (Amplificación)
    scores = analyzer.fisher_weights.cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(scores)), scores)
    plt.title("Factor de Amplificación Fisher (Importancia por Característica)")
    plt.xlabel("Componente Principal (Eigen-feature)")
    plt.ylabel("Factor de Amplificación (sqrt(J))")
    plt.savefig(OUTPUT_DIR / "fisher_amplification_plot.png")

    print("[FIN] Análisis completado con éxito.")

if __name__ == "__main__":
    main()
