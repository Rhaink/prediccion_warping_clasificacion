#!/usr/bin/env python3
"""
ROLE: Senior Computer Vision Research Engineer
PROJECT: Thesis - Geometric Validation of Lung Warping via Fisher Linear Analysis
STACK: PyTorch (GPU), Numpy, Matplotlib

OBJETIVO:
Validación científica rigurosa usando aceleración por GPU.
Incluye generación de evidencia visual (Galería de Clasificación).
"""

import numpy as np
import pandas as pd
import cv2
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import gc
import warnings
import random

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
        
        print(f"[LOADER] Cargando {N} imágenes '{split}' en memoria (CPU)...")
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
        # U: (N, q), S: (q,), V: (D, q)
        U, S, V = torch.pca_lowrank(X, q=n_components, center=True, niter=2)
        mean = torch.mean(X, dim=0)
        
        eigvals = S ** 2 / (X.shape[0] - 1)
        total_var = torch.var(X, dim=0, unbiased=True).sum()
        explained_variance_ratio = eigvals / total_var
        
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
        if X_test.shape[0] > 2000:
            preds = []
            batch_size = 1000
            for i in range(0, X_test.shape[0], batch_size):
                batch = X_test[i:i+batch_size]
                dist = torch.cdist(batch, X_train, p=2)
                vals, idx = torch.topk(dist, k=k, largest=False, dim=1)
                labels = y_train[idx]
                mode, _ = torch.mode(labels, dim=1)
                preds.append(mode)
            return torch.cat(preds)
        else:
            dist_matrix = torch.cdist(X_test, X_train, p=2)
            topk_vals, topk_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)
            topk_labels = y_train[topk_indices]
            mode_val, _ = torch.mode(topk_labels, dim=1)
            return mode_val

    def generate_classification_gallery(self, X_test_cpu, y_test_cpu, y_pred_cpu, output_path, image_size=224):
        y_true = y_test_cpu.numpy()
        y_pred = y_pred_cpu.numpy()
        
        tp = np.where((y_true == 1) & (y_pred == 1))[0]
        tn = np.where((y_true == 0) & (y_pred == 0))[0]
        fp = np.where((y_true == 0) & (y_pred == 1))[0]
        fn = np.where((y_true == 1) & (y_pred == 0))[0]
        
        categories = {
            "TP (Enfermo Correcto)": tp,
            "TN (Sano Correcto)": tn,
            "FP (Sano -> Enfermo)": fp,
            "FN (Enfermo -> Sano)": fn
        }
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.4, wspace=0.1)
        
        print(f"[GALERÍA] TP: {len(tp)}, TN: {len(tn)}, FP: {len(fp)}, FN: {len(fn)}")
        
        for row_idx, (label, indices) in enumerate(categories.items()):
            if len(indices) > 0:
                selected = np.random.choice(indices, min(5, len(indices)), replace=False)
            else:
                selected = []
                
            for col_idx in range(5):
                ax = axes[row_idx, col_idx]
                if col_idx < len(selected):
                    idx = selected[col_idx]
                    img_flat = X_test_cpu[idx].numpy()
                    img = img_flat.reshape(image_size, image_size)
                    
                    ax.imshow(img, cmap='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    color = 'green' if "Correcto" in label else 'red'
                    for spine in ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(2)
                else:
                    ax.axis('off')
            
            axes[row_idx, 2].set_title(label, fontsize=12, fontweight='bold', pad=10)

        plt.suptitle("Evidencia de Clasificación (Test Set)", fontsize=16, y=0.95)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[GALERÍA] Guardada en {output_path}")

def run_scientific_validation(args, device):
    print("\n" + "="*70)
    print("VALIDACIÓN CIENTÍFICA HÍBRIDA (RAM + GPU)")
    print("="*70)
    
    loader = DatasetLoader(args.raw_dir, args.dataset_dir)
    
    # 1. Cargar Train
    X_train_cpu, y_train_cpu = loader.load_full_dataset("train", use_clahe=args.clahe)
    print(f"[RAM] Train Dataset: {X_train_cpu.shape}")
    
    analyzer = TorchAnalysis(device)
    
    # Usaremos k=50 (nuestro óptimo) para la inferencia visual
    best_k = 50 
    print(f"\n[INFERENCIA] Entrenando modelo final con k={best_k} en todo TRAIN...")
    
    try:
        X_train_gpu = X_train_cpu.to(device)
        y_train_gpu = y_train_cpu.to(device)
        
        # Pipeline Final
        mean = X_train_gpu.mean(dim=0)
        std = X_train_gpu.std(dim=0) + 1e-8
        X_train_norm = (X_train_gpu - mean) / std
        
        # PCA
        X_train_pca, _, V, mean_pca = analyzer.fit_pca_efficient(X_train_norm, best_k)
        
        # Fisher
        J_weights = analyzer.fisher_score(X_train_pca, y_train_gpu)
        weights = torch.sqrt(J_weights)
        X_train_final = X_train_pca * weights
        
        del X_train_gpu, X_train_norm
        torch.cuda.empty_cache()
        
    except RuntimeError:
        print("Error OOM entrenando modelo final.")
        return

    # 2. Cargar Test y Predecir
    print("\n[INFERENCIA] Cargando Test Set...")
    X_test_cpu, y_test_cpu = loader.load_full_dataset("test", use_clahe=args.clahe)
    
    try:
        X_test_gpu = X_test_cpu.to(device)
        
        # Aplicar transformación guardada
        X_test_norm = (X_test_gpu - mean) / std
        X_test_centered = X_test_norm - mean_pca
        X_test_pca = torch.mm(X_test_centered, V)
        X_test_final = X_test_pca * weights
        
        # Predecir
        print("[INFERENCIA] Clasificando Test Set...")
        y_pred_gpu = analyzer.knn_predict(X_train_final, y_train_gpu, X_test_final, k=5)
        y_pred_cpu = y_pred_gpu.cpu()
        
        # Métricas Test
        acc_test = accuracy_score(y_test_cpu.numpy(), y_pred_cpu.numpy())
        print(f"\n>>> TEST SET ACCURACY (k={best_k}): {acc_test*100:.2f}% <<<")
        
        # Generar Galería
        analyzer.generate_classification_gallery(X_test_cpu, y_test_cpu, y_pred_cpu, "results/classification_gallery.png")
        
    except RuntimeError as e:
        print(f"Error en inferencia: {e}")
    
    del X_test_gpu, y_train_gpu, V
    torch.cuda.empty_cache()

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
