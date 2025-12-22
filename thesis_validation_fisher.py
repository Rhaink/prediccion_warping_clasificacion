#!/usr/bin/env python3
"""
ROLE: Senior Computer Vision Research Engineer
PROJECT: Thesis - Geometric Validation of Lung Warping via Fisher Linear Analysis
STACK: PyTorch (GPU), Numpy, Matplotlib

OBJETIVO:
Validación científica rigurosa usando aceleración por GPU con gestión inteligente de memoria.
Usa torch.pca_lowrank para evitar OOM en VRAM de 8GB.

VENTAJAS:
- Datos en RAM, Cálculo en VRAM.
- PCA eficiente en memoria.
"""

import numpy as np
import pandas as pd
import cv2
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import json
import gc
import sys
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Verificar GPU
GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[SYSTEM] Usando dispositivo: {GLOBAL_DEVICE}")

class DatasetLoader:
    def __init__(self, raw_root, warped_root, image_size=224):
        self.raw_root = Path(raw_root)
        self.warped_root = Path(warped_root)
        self.image_size = image_size
        self._validate()

    def _validate(self):
        if not self.raw_root.exists(): self.raw_root = Path("data/dataset")
        if not self.warped_root.exists(): self.warped_root = Path.cwd() / self.warped_root

    def load_full_dataset(self, split="train", use_clahe=True):
        csv_path = self.warped_root / split / "images.csv"
        if not csv_path.exists(): csv_path = self.warped_root / "images.csv"
        
        df = pd.read_csv(csv_path)
        if 'split' in df.columns: df = df[df['split'] == split]

        N = len(df)
        D = self.image_size * self.image_size
        X = np.zeros((N, D), dtype=np.float32)
        y = np.zeros(N, dtype=np.int32)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) if use_clahe else None
        
        print(f"[LOADER] Cargando {N} imágenes en memoria (CPU)...")
        loaded_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=N):
            name = row['image_name']
            cat = row['category']
            w_name = row.get('warped_filename', row.get('filename', f"{name}.png"))
            p_candidates = [
                self.warped_root / split / cat / w_name,
                self.warped_root / cat / w_name,
                self.warped_root / w_name
            ]
            
            img_path = None
            for p in p_candidates:
                if p.exists(): img_path = p; break
            
            if not img_path: continue
            
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img = cv2.resize(img, (self.image_size, self.image_size))
            if use_clahe: img = clahe.apply(img)
            
            X[loaded_count] = img.flatten()
            y[loaded_count] = 0 if cat == 'Normal' else 1
            loaded_count += 1

        return torch.from_numpy(X[:loaded_count]), torch.from_numpy(y[:loaded_count])

class TorchAnalysis:
    def __init__(self, device):
        self.device = device

    def fit_pca_efficient(self, X, n_components):
        """
        Realiza PCA usando torch.pca_lowrank (SVD aleatorizado eficiente).
        Esto evita calcular la matriz de covarianza completa o SVD full.
        """
        # X ya debe estar en GPU y centrado si se desea, pero pca_lowrank puede centrarlo
        # center=True por defecto en pca_lowrank
        
        # U: (N, q), S: (q,), V: (D, q)
        U, S, V = torch.pca_lowrank(X, q=n_components, center=True, niter=2)
        
        # Proyección: X @ V = U @ S
        # Pero ojo, pca_lowrank centra internamente. 
        # Para proyectar validación necesitamos la media y V.
        mean = torch.mean(X, dim=0)
        
        # Varianza explicada
        eigvals = S ** 2 / (X.shape[0] - 1)
        total_var = torch.var(X, dim=0, unbiased=True).sum() # Aproximación de varianza total
        explained_variance_ratio = eigvals / total_var
        
        # Proyección entrenamiento
        X_pca = torch.mm(X - mean, V)
        
        return X_pca, explained_variance_ratio, V, mean

    def fisher_score(self, X_pca, y):
        unique_classes = torch.unique(y)
        means = []
        vars = []
        
        for c in unique_classes:
            mask = (y == c)
            Xc = X_pca[mask]
            means.append(torch.mean(Xc, dim=0))
            vars.append(torch.var(Xc, dim=0, unbiased=True))
            
        numerator = (means[0] - means[1]) ** 2
        denominator = vars[0] + vars[1] + 1e-9
        J = numerator / denominator
        return J

    def knn_predict(self, X_train, y_train, X_test, k=5):
        # Lotes pequeños para cdist si es necesario
        dist_matrix = torch.cdist(X_test, X_train, p=2)
        topk_vals, topk_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)
        topk_labels = y_train[topk_indices]
        mode_val, _ = torch.mode(topk_labels, dim=1)
        return mode_val

def run_scientific_validation(args, device):
    print("\n" + "="*70)
    print("VALIDACIÓN CIENTÍFICA HÍBRIDA (RAM + GPU)")
    print("="*70)
    
    print(f">> Cargando dataset desde: {args.dataset_dir}")
    loader = DatasetLoader(args.raw_dir, args.dataset_dir)
    
    # Cargar en CPU (RAM del sistema)
    X_all_cpu, y_all_cpu = loader.load_full_dataset("train", use_clahe=args.clahe)
    print(f"[RAM] Dataset cargado: {X_all_cpu.shape} (Float32)")
    
    analyzer = TorchAnalysis(device)
    
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    components_list = [10, 25, 50, 75, 100, 150, 200]
    history = {k: {'acc': [], 'var': []} for k in components_list}
    
    y_numpy = y_all_cpu.numpy()
    
    print(f"\n[CV] Iniciando {k_folds}-Fold Cross Validation...")
    
    fold = 1
    for train_idx, val_idx in skf.split(np.zeros(len(y_numpy)), y_numpy):
        print(f"--- Fold {fold}/{k_folds} ---")
        
        # Mover SOLO el fold actual a GPU
        # Indices
        idx_tr = torch.from_numpy(train_idx)
        idx_val = torch.from_numpy(val_idx)
        
        # Copia a GPU
        X_train = X_all_cpu[idx_tr].to(device)
        y_train = y_all_cpu[idx_tr].to(device)
        X_val = X_all_cpu[idx_val].to(device)
        y_val = y_all_cpu[idx_val].to(device)
        
        # 1. Standard Scaler (GPU)
        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Liberar memoria original en GPU
        del X_train, X_val
        
        # 2. PCA Eficiente (Low Rank)
        max_k = max(components_list)
        X_train_pca_all, var_ratios_all, V, mean_pca = analyzer.fit_pca_efficient(X_train_norm, max_k)
        
        # Proyectar validación
        X_val_centered = X_val_norm - mean_pca
        X_val_pca_all = torch.mm(X_val_centered, V)
        
        # Liberar X_norm para hacer espacio
        del X_train_norm, X_val_norm, X_val_centered
        torch.cuda.empty_cache()
        
        # Varianza acumulada
        cum_var = torch.cumsum(var_ratios_all, dim=0)
        
        for k in components_list:
            # Slice
            X_tr_k = X_train_pca_all[:, :k]
            X_val_k = X_val_pca_all[:, :k]
            
            # Fisher
            J_weights = analyzer.fisher_score(X_tr_k, y_train)
            weights = torch.sqrt(J_weights)
            
            # Ponderar
            X_tr_w = X_tr_k * weights
            X_val_w = X_val_k * weights
            
            # Clasificar
            preds = analyzer.knn_predict(X_tr_w, y_train, X_val_w, k=5)
            
            acc = (preds == y_val).float().mean().item()
            var_acc = cum_var[k-1].item()
            
            history[k]['acc'].append(acc)
            history[k]['var'].append(var_acc)
            
        fold += 1
        
        # Limpieza masiva
        del V, X_train_pca_all, X_val_pca_all, y_train, y_val
        torch.cuda.empty_cache()
        gc.collect()

    # Reporte
    print("\n" + "="*40)
    print("RESULTADOS FINALES (Media +/- Std)")
    print("="*40)
    print(f"{ ' K':<5} | {'Accuracy':<20} | {'Varianza Exp.':<20}")
    print("-" * 50)
    
    accuracies = []
    ks = []
    
    for k in components_list:
        mean_acc = np.mean(history[k]['acc'])
        std_acc = np.std(history[k]['acc'])
        mean_var = np.mean(history[k]['var'])
        accuracies.append(mean_acc)
        ks.append(k)
        print(f"{k:<5} | {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}) | {mean_var*100:.2f}%")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks, accuracies, 'o-', linewidth=2, color='blue', label='Warped + CLAHE')
    plt.fill_between(ks, 
                     np.array(accuracies) - np.array([np.std(history[k]['acc']) for k in ks]),
                     np.array(accuracies) + np.array([np.std(history[k]['acc']) for k in ks]),
                     alpha=0.2, color='blue')
    plt.title(f"Performance vs Complejidad (CV)")
    plt.xlabel("Componentes PCA")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("results/gpu_validation_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--raw-dir", default="data/dataset/COVID-19_Radiography_Dataset")
    parser.add_argument("--clahe", action="store_true", default=True)
    args = parser.parse_args()
    
    try:
        run_scientific_validation(args, GLOBAL_DEVICE)
    except Exception as e:
        print(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()