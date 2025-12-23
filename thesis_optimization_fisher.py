#!/usr/bin/env python3
"""
ROLE: Senior Computer Vision Research Engineer
PROJECT: Thesis - Geometric Validation of Lung Warping
STACK: PyTorch (GPU), Numpy

OBJETIVO:
Grid Search para optimizar parámetros de preprocesamiento (CLAHE).
Busca maximizar el accuracy de la clasificación binaria (Fisher-Linear).
"""

import numpy as np
import pandas as pd
import cv2
import torch
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import itertools
import gc

# Verificar GPU
GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DatasetLoader:
    def __init__(self, warped_root, image_size=224):
        self.warped_root = Path(warped_root)
        self.image_size = image_size
        self.cache = {} # Simple cache to avoid reloading raw images if logic permits

    def load_dataset_with_params(self, split, clip_limit, tile_size):
        """
        Carga y procesa on-the-fly.
        """
        csv_path = self.warped_root / split / "images.csv"
        if not csv_path.exists(): csv_path = self.warped_root / "images.csv"
        
        df = pd.read_csv(csv_path)
        if 'split' in df.columns: df = df[df['split'] == split]
        
        # Filtrar un subset para el Grid Search si es muy lento
        # df = df.sample(frac=0.2, random_state=42) 

        N = len(df)
        D = self.image_size * self.image_size
        X = np.zeros((N, D), dtype=np.float32)
        y = np.zeros(N, dtype=np.int32)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        print(f"  [LOADER] Procesando {split} | CLAHE: Limit={clip_limit}, Tile={tile_size}x{tile_size}")
        
        loaded_count = 0
        for idx, row in df.iterrows():
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
            
            # Optimización: Leer imagen raw una vez
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = clahe.apply(img)
            
            X[loaded_count] = img.flatten()
            y[loaded_count] = 0 if cat == 'Normal' else 1
            loaded_count += 1

        return torch.from_numpy(X[:loaded_count]), torch.from_numpy(y[:loaded_count])

class TorchAnalysis:
    def __init__(self, device):
        self.device = device

    def fit_pca_efficient(self, X, n_components):
        U, S, V = torch.pca_lowrank(X, q=n_components, center=True, niter=2)
        mean = torch.mean(X, dim=0)
        X_pca = torch.mm(X - mean, V)
        return X_pca, V, mean

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
        dist_matrix = torch.cdist(X_test, X_train, p=2)
        topk_vals, topk_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)
        topk_labels = y_train[topk_indices]
        mode_val, _ = torch.mode(topk_labels, dim=1)
        return mode_val

def run_optimization(args, device):
    print("="*60)
    print("GRID SEARCH: CLAHE Optimization")
    print("="*60)
    
    loader = DatasetLoader(args.dataset_dir)
    analyzer = TorchAnalysis(device)
    
    # Grid de parámetros
    clip_limits = [1.0, 2.0, 4.0]
    tile_sizes = [2, 4, 8]
    
    results = []
    
def evaluate_configuration(clip, tile, loader, analyzer, device):
    """
    Ejecuta una iteración aislada para garantizar limpieza de memoria al salir del scope.
    """
    try:
        print(f"\n>>> Probando: Clip={clip}, Tile={tile}")
        
        # 1. Cargar en CPU
        # Nota: Esto consume RAM, no VRAM.
        X_train, y_train = loader.load_dataset_with_params("train", clip, tile)
        X_test, y_test = loader.load_dataset_with_params("test", clip, tile)
        
        # 2. Mover a GPU (Entrenamiento)
        # Se mueve solo lo necesario y se libera CPU si es posible (aunque aquí lo necesitamos para liberar después)
        X_train_gpu = X_train.to(device)
        y_train_gpu = y_train.to(device)
        
        # Liberar memoria CPU inmediatamente si ya está en GPU
        del X_train
        
        # 3. Normalización Estricta (Pixel Z-Score)
        mean_px = X_train_gpu.mean(dim=0)
        std_px = X_train_gpu.std(dim=0) + 1e-8
        X_train_norm = (X_train_gpu - mean_px) / std_px
        
        # 4. PCA
        # X_train_gpu ya no se necesita crudo, podríamos liberarlo si PCA soporta in-place, pero torch.pca_lowrank no.
        # Pero X_train_norm reemplaza conceptualmente a X_train_gpu para el PCA.
        del X_train_gpu 
        
        X_train_pca, V, mean_pca = analyzer.fit_pca_efficient(X_train_norm, n_components=50)
        del X_train_norm # Liberar input del PCA
        
        # 5. Fisher Weighting
        J_weights = analyzer.fisher_score(X_train_pca, y_train_gpu)
        weights = torch.sqrt(J_weights)
        X_train_final = X_train_pca * weights
        
        # Liberar intermedios de Train
        del X_train_pca
        
        # 6. Proyección Test (Streaming/Batching si fuera necesario, pero aquí cabe si borramos Train)
        X_test_gpu = X_test.to(device)
        del X_test # Liberar CPU
        
        X_test_norm = (X_test_gpu - mean_px) / std_px
        del X_test_gpu
        
        X_test_centered = X_test_norm - mean_pca
        del X_test_norm
        
        X_test_pca = torch.mm(X_test_centered, V)
        del X_test_centered
        
        X_test_final = X_test_pca * weights
        del X_test_pca
        
        # 7. Predicción
        y_pred_gpu = analyzer.knn_predict(X_train_final, y_train_gpu, X_test_final, k=5)
        
        acc = accuracy_score(y_test.numpy(), y_pred_gpu.cpu().numpy())
        print(f"    --> Accuracy: {acc*100:.2f}%")
        
        # Retornar métricas
        return {
            "clip_limit": clip,
            "tile_size": tile,
            "accuracy": acc
        }
        
    except Exception as e:
        print(f"    Error crítico en config {clip}/{tile}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Limpieza paranoica
        gc.collect()
        torch.cuda.empty_cache()

def run_optimization(args, device):
    print("="*60)
    print("GRID SEARCH: CLAHE Optimization (Memory Optimized)")
    print("="*60)
    
    loader = DatasetLoader(args.dataset_dir)
    analyzer = TorchAnalysis(device)
    
    # Grid de parámetros
    clip_limits = [1.0, 2.0, 4.0]
    tile_sizes = [2, 4, 8]
    
    results = []
    
    for clip, tile in itertools.product(clip_limits, tile_sizes):
        res = evaluate_configuration(clip, tile, loader, analyzer, device)
        if res:
            results.append(res)
        
        # Forzar limpieza entre iteraciones del bucle principal
        gc.collect()
        torch.cuda.empty_cache()

    # Guardar resultados
    out_path = Path("results/grid_search_clahe.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResultados guardados en {out_path}")
    
    # Encontrar mejor
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nMEJOR CONFIGURACIÓN: Clip={best['clip_limit']}, Tile={best['tile_size']} -> Acc={best['accuracy']*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/full_warped_dataset")
    args = parser.parse_args()
    
    try:
        run_optimization(args, GLOBAL_DEVICE)
    except Exception as e:
        print(f"Error crítico: {e}")
