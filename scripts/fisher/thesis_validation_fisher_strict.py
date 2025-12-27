#!/usr/bin/env python3
"""
ROLE: Senior Computer Vision Research Engineer
PROJECT: Thesis - Geometric Validation (STRICT ADVISOR VERSION)
STACK: PyTorch (GPU), Numpy

OBJETIVO:
Implementación ESTRICTA de las instrucciones del asesor:
1. PCA sobre píxeles crudos (o centrados, pero no estandarizados por varianza).
2. Estandarización sobre LOS PONDERANTES (PCA Output).
3. Fisher sobre Ponderantes Estandarizados.
"""

import numpy as np
import pandas as pd
import cv2
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import warnings
import shutil

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

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

    def get_raw_path(self, image_name, category):
        candidates = [
            self.raw_root / category / "images" / f"{image_name}.png",
            self.raw_root / category / f"{image_name}.png",
            self.raw_root / f"{image_name}.png"
        ]
        for p in candidates:
            if p.exists(): return p
        return None

    def load_full_dataset(self, split="train", use_clahe=True):
        csv_path = self.warped_root / split / "images.csv"
        if not csv_path.exists(): csv_path = self.warped_root / "images.csv"
        
        df = pd.read_csv(csv_path)
        if 'split' in df.columns: df = df[df['split'] == split]

        N = len(df)
        D = self.image_size * self.image_size
        X = np.zeros((N, D), dtype=np.float32)
        y = np.zeros(N, dtype=np.int32)
        names = []
        categories = []
        
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
            names.append(name)
            categories.append(cat)
            loaded_count += 1

        return torch.from_numpy(X[:loaded_count]), torch.from_numpy(y[:loaded_count]), names, categories

class TorchAnalysis:
    def __init__(self, device):
        self.device = device

    def fit_pca_raw(self, X, n_components):
        """PCA directo sobre datos crudos (centrados pero no escalados)."""
        # Centrar datos (Media=0) es necesario para PCA
        mean = torch.mean(X, dim=0)
        X_centered = X - mean
        
        # PCA
        U, S, V = torch.pca_lowrank(X_centered, q=n_components, center=False, niter=2)
        
        # Proyectar para obtener PONDERANTES
        X_weights = torch.mm(X_centered, V)
        return X_weights, V, mean

    def standardize_weights(self, weights):
        """Estandariza los PONDERANTES (Feature-wise)."""
        w_mean = torch.mean(weights, dim=0)
        w_std = torch.std(weights, dim=0) + 1e-8
        weights_std = (weights - w_mean) / w_std
        return weights_std, w_mean, w_std

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

def run_scientific_validation(args, device):
    print("\n" + "="*70)
    print("ANÁLISIS ESTRICTO (INSTRUCCIONES ASESOR)")
    print("="*70)
    
    loader = DatasetLoader(args.raw_dir, args.dataset_dir)
    X_train_cpu, y_train_cpu, _, _ = loader.load_full_dataset("train", use_clahe=args.clahe)
    analyzer = TorchAnalysis(device)
    
    best_k = 50 
    print(f"\n[MODELO] Pipeline: PCA(Raw) -> Standardize(Weights) -> Fisher -> Amplify")
    
    try:
        X_train_gpu = X_train_cpu.to(device)
        y_train_gpu = y_train_cpu.to(device)
        
        # 1. PCA sobre Píxeles Crudos (Centrados)
        X_train_weights, V, pixel_mean = analyzer.fit_pca_raw(X_train_gpu, best_k)
        
        # 2. Estandarizar Ponderantes
        X_train_std, w_mean, w_std = analyzer.standardize_weights(X_train_weights)
        
        # 3. Fisher Score
        J_weights = analyzer.fisher_score(X_train_std, y_train_gpu)
        
        # 4. Amplificación
        # Nota: Multiplicamos directamente J, o sqrt(J)
        # El asesor dijo "multiplicar por esa cosa [Ratio]". 
        # Usaremos sqrt(J) por robustez numérica, pero J directo también valdría.
        amplification_factor = torch.sqrt(J_weights) 
        X_train_final = X_train_std * amplification_factor
        
        del X_train_gpu, X_train_weights
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"Error OOM: {e}")
        return

    # Inferencia
    print("\n[INFERENCIA] Cargando Test Set...")
    X_test_cpu, y_test_cpu, _, _ = loader.load_full_dataset("test", use_clahe=args.clahe)
    
    try:
        X_test_gpu = X_test_cpu.to(device)
        
        # 1. Proyectar (Mismo PCA)
        X_test_centered = X_test_gpu - pixel_mean
        X_test_weights = torch.mm(X_test_centered, V)
        
        # 2. Estandarizar (Mismo Scaler)
        X_test_std = (X_test_weights - w_mean) / w_std
        
        # 3. Amplificar (Mismo Factor)
        X_test_final = X_test_std * amplification_factor
        
        # 4. Predecir
        y_pred_gpu = analyzer.knn_predict(X_train_final, y_train_gpu, X_test_final, k=5)
        y_pred_cpu = y_pred_gpu.cpu()
        
        acc = accuracy_score(y_test_cpu.numpy(), y_pred_cpu.numpy())
        print(f"\n>>> GLOBAL ACCURACY (STRICT): {acc*100:.2f}% <<<")
        
    except RuntimeError as e:
        print(f"Error en inferencia: {e}")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/full_warped_dataset")
    parser.add_argument("--raw-dir", default="data/dataset/COVID-19_Radiography_Dataset")
    parser.add_argument("--clahe", action="store_true", default=True)
    args = parser.parse_args()
    
    run_scientific_validation(args, GLOBAL_DEVICE)
