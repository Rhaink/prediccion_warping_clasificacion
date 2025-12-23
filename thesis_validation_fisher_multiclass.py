#!/usr/bin/env python3
"""
ROLE: Senior Computer Vision Research Engineer
PROJECT: Thesis - Geometric Validation of Lung Warping via Fisher Linear Analysis (Multiclass Extension)
STACK: PyTorch (GPU), Numpy, Matplotlib, Scikit-Image

OBJETIVO:
Extensión Multiclase (Normal, Neumonía, COVID) de la validación Fisher.
Usa F-Statistic (ANOVA) como generalización del Fisher Score para >2 clases.
"""

import numpy as np
import pandas as pd
import cv2
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from tqdm import tqdm
import warnings
import shutil

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

    def load_full_dataset(self, split="train", use_clahe=True, clip_limit=2.0, tile_size=(4,4)):
        csv_path = self.warped_root / split / "images.csv"
        if not csv_path.exists(): csv_path = self.warped_root / "images.csv"
        
        df = pd.read_csv(csv_path)
        if 'split' in df.columns: df = df[df['split'] == split]

        N = len(df)
        D = self.image_size * self.image_size
        X = np.zeros((N, D), dtype=np.float32)
        
        # Etiquetas: 0=Normal, 1=Pneumonia, 2=COVID
        label_map = {'Normal': 0, 'Viral Pneumonia': 1, 'COVID': 2}
        # Fallback para nombres variados en el CSV
        
        y = np.zeros(N, dtype=np.int64)
        names = []
        categories = []
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size) if use_clahe else None
        
        print(f"[LOADER] Cargando {N} imágenes '{split}' en memoria (CPU)...")
        loaded_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=N):
            name = row['image_name']
            cat = row['category']
            w_name = row.get('warped_filename', row.get('filename', f"{name}.png"))
            
            # Cargar WARPED
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
            
            # Mapeo de categorías robusto
            if cat == 'Normal':
                y[loaded_count] = 0
            elif 'Pneumonia' in cat: # 'Viral Pneumonia'
                y[loaded_count] = 1
            elif 'COVID' in cat:
                y[loaded_count] = 2
            else:
                y[loaded_count] = 1 # Fallback a neumonia genérica si hay algo raro
                
            names.append(name)
            categories.append(cat)
            loaded_count += 1

        return torch.from_numpy(X[:loaded_count]), torch.from_numpy(y[:loaded_count]), names, categories

class TorchAnalysisMulticlass:
    def __init__(self, device):
        self.device = device

    def fit_pca_efficient(self, X, n_components):
        # PCA estándar con Torch
        U, S, V = torch.pca_lowrank(X, q=n_components, center=True, niter=2)
        mean = torch.mean(X, dim=0)
        eigvals = S ** 2 / (X.shape[0] - 1)
        total_var = torch.var(X, dim=0, unbiased=True).sum()
        explained_variance_ratio = eigvals / total_var
        X_pca = torch.mm(X - mean, V)
        return X_pca, explained_variance_ratio, V, mean

    def fisher_score_multiclass(self, X_pca, y):
        """
        Calcula el F-Statistic (ANOVA) para cada componente principal.
        J = Var_between / Var_within
        """
        unique_classes = torch.unique(y)
        num_classes = len(unique_classes)
        
        global_mean = torch.mean(X_pca, dim=0)
        
        S_B = torch.zeros(X_pca.shape[1], device=self.device) # Between-class scatter (diag)
        S_W = torch.zeros(X_pca.shape[1], device=self.device) # Within-class scatter (diag) 
        
        for c in unique_classes:
            mask = (y == c)
            Xc = X_pca[mask]
            n_c = Xc.shape[0]
            
            mean_c = torch.mean(Xc, dim=0)
            var_c = torch.var(Xc, dim=0, unbiased=True)
            
            # Contribución a Between: n_c * (mean_c - global_mean)**2
            S_B += n_c * (mean_c - global_mean)**2
            
            # Contribución a Within: (n_c - 1) * var_c
            # Nota: Usamos var_c (unbiased) por lo que multiplicamos por (n_c - 1) para obtener sum of squares
            S_W += (n_c - 1) * var_c

        # Normalización por grados de libertad
        # MS_between = S_B / (K - 1)
        # MS_within = S_W / (N - K)
        N = X_pca.shape[0]
        K = num_classes
        
        MS_B = S_B / (K - 1)
        MS_W = S_W / (N - K)
        
        J = MS_B / (MS_W + 1e-9)
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

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusión Multiclase')
    plt.savefig(output_path)
    plt.close()

def run_multiclass_validation(args, device):
    print("\n" + "="*70)
    print("VALIDACIÓN MULTICLASE (Normal vs Neumonía vs COVID)")
    print("="*70)
    
    loader = DatasetLoader(args.raw_dir, args.dataset_dir)
    
    # 1. Cargar Train
    X_train_cpu, y_train_cpu, _, _ = loader.load_full_dataset("train", use_clahe=args.clahe)
    
    analyzer = TorchAnalysisMulticlass(device)
    
    best_k = 50 
    print(f"\n[MODELO] Ajustando Fisher-PCA (k={best_k}) en todo TRAIN...")
    
    try:
        X_train_gpu = X_train_cpu.to(device)
        y_train_gpu = y_train_cpu.to(device)
        
        # Pre-procesamiento Estricto (GPU)
        mean = X_train_gpu.mean(dim=0)
        std = X_train_gpu.std(dim=0) + 1e-8
        X_train_norm = (X_train_gpu - mean) / std
        
        # PCA
        X_train_pca, _, V, mean_pca = analyzer.fit_pca_efficient(X_train_norm, best_k)
        
        # Fisher Multiclase (F-Statistic)
        J_weights = analyzer.fisher_score_multiclass(X_train_pca, y_train_gpu)
        weights = torch.sqrt(J_weights) # Mantenemos la heurística de sqrt(J)
        
        X_train_final = X_train_pca * weights
        
        del X_train_gpu, X_train_norm
        torch.cuda.empty_cache()
        
    except RuntimeError:
        print("Error OOM entrenando modelo final.")
        return

    # 2. Cargar Test
    print("\n[INFERENCIA] Cargando Test Set...")
    X_test_cpu, y_test_cpu, names_test, cats_test = loader.load_full_dataset("test", use_clahe=args.clahe)
    
    try:
        X_test_gpu = X_test_cpu.to(device)
        
        # Proyección
        X_test_norm = (X_test_gpu - mean) / std
        X_test_centered = X_test_norm - mean_pca
        X_test_pca = torch.mm(X_test_centered, V)
        X_test_final = X_test_pca * weights
        
        # Predicción KNN
        y_pred_gpu = analyzer.knn_predict(X_train_final, y_train_gpu, X_test_final, k=5)
        y_pred_cpu = y_pred_gpu.cpu()
        
        # Métricas
        acc = accuracy_score(y_test_cpu.numpy(), y_pred_cpu.numpy())
        print(f"\n>>> GLOBAL ACCURACY (3-Class): {acc*100:.2f}% <<<")
        
        class_names = ['Normal', 'Neumonía', 'COVID']
        print("\nReporte de Clasificación:")
        print(classification_report(y_test_cpu.numpy(), y_pred_cpu.numpy(), target_names=class_names))
        
        # Guardar resultados
        output_dir = Path("results/multiclass_experiment")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_confusion_matrix(y_test_cpu.numpy(), y_pred_cpu.numpy(), class_names, output_dir / "confusion_matrix.png")
        
        # Guardar PCA components plot (2D projection) para ver separabilidad
        plt.figure(figsize=(10, 8))
        X_viz = X_test_final[:, :2].cpu().numpy()
        y_viz = y_test_cpu.numpy()
        for i, c in enumerate(class_names):
            plt.scatter(X_viz[y_viz==i, 0], X_viz[y_viz==i, 1], label=c, alpha=0.5, s=10)
        plt.legend()
        plt.title("Proyección Fisher-PCA (2 Componentes Principales)")
        plt.savefig(output_dir / "fisher_projection_2d.png")
        
    except RuntimeError as e:
        print(f"Error en inferencia: {e}")
        import traceback
        traceback.print_exc()
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/full_warped_dataset")
    parser.add_argument("--raw-dir", default="data/dataset/COVID-19_Radiography_Dataset")
    parser.add_argument("--clahe", action="store_true", default=True)
    args = parser.parse_args()
    
    try:
        run_multiclass_validation(args, GLOBAL_DEVICE)
    except Exception as e:
        print(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()
