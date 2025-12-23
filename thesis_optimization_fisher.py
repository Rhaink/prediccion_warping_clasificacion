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
    LÓGICA STRICT (ASESOR): PCA sobre Raw -> Standardize Weights -> Fisher
    """
    try:
        print(f"\n>>> Probando: Clip={clip}, Tile={tile}")
        
        # 1. Cargar en CPU
        X_train, y_train = loader.load_dataset_with_params("train", clip, tile)
        X_test, y_test = loader.load_dataset_with_params("test", clip, tile)
        
        # 2. Mover a GPU
        X_train_gpu = X_train.to(device)
        y_train_gpu = y_train.to(device)
        del X_train
        
        # --- LÓGICA STRICT ---
        
        # 3. PCA sobre Píxeles Crudos (Solo Centrado, NO dividir por STD)
        mean_px = X_train_gpu.mean(dim=0)
        X_train_centered = X_train_gpu - mean_px
        
        # PCA Lowrank
        # Nota: fit_pca_efficient en TorchAnalysis original hacía (X-mean) internamente si center=True
        # Aquí lo hacemos manual para control total o usamos la función adaptada.
        # Vamos a adaptar la lógica inline para ser explícitos.
        
        U, S, V = torch.pca_lowrank(X_train_centered, q=50, center=False, niter=2)
        
        # Proyectar para obtener Ponderantes (Weights)
        X_train_weights = torch.mm(X_train_centered, V)
        
        del X_train_gpu, X_train_centered
        
        # 4. Estandarizar Ponderantes (Feature-wise Z-score)
        # "Poner a competir a los ponderantes en igualdad"
        w_mean = X_train_weights.mean(dim=0)
        w_std = X_train_weights.std(dim=0) + 1e-8
        X_train_std = (X_train_weights - w_mean) / w_std
        
        del X_train_weights
        
        # 5. Fisher Weighting sobre Ponderantes Estandarizados
        J_weights = analyzer.fisher_score(X_train_std, y_train_gpu)
        amplification_factor = torch.sqrt(J_weights)
        
        X_train_final = X_train_std * amplification_factor
        del X_train_std
        
        # 6. Proyección Test
        X_test_gpu = X_test.to(device)
        del X_test
        
        # Aplicar misma transformación
        X_test_centered = X_test_gpu - mean_px
        del X_test_gpu
        
        X_test_weights = torch.mm(X_test_centered, V)
        del X_test_centered
        
        X_test_std = (X_test_weights - w_mean) / w_std
        del X_test_weights
        
        X_test_final = X_test_std * amplification_factor
        del X_test_std
        
        # 7. Predicción
        y_pred_gpu = analyzer.knn_predict(X_train_final, y_train_gpu, X_test_final, k=5)
        
        acc = accuracy_score(y_test.numpy(), y_pred_gpu.cpu().numpy())
        print(f"    --> Accuracy (Strict): {acc*100:.2f}%")
        
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
