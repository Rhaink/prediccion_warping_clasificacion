#!/usr/bin/env python3
"""
ROLE: Senior Computer Vision Research Engineer
PROJECT: Thesis - Geometric Validation of Lung Warping via Fisher Linear Analysis
STACK: Python, Numpy, Scikit-Learn, Matplotlib (WSL2 Env)

OBJETIVO:
Realizar una búsqueda de hiperparámetros (Grid Search) para encontrar la configuración óptima
que demuestre la separabilidad lineal de las imágenes Warped vs Raw.

CAMBIOS V2:
- Iteración sobre número de componentes PCA [10, 25, 50, 100, 150].
- Comparación de clasificadores: k-NN vs Regresión Logística (Separabilidad Lineal Pura).
- Preprocesamiento CLAHE opcional.
"""

import numpy as np
import pandas as pd
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')

class DatasetLoader:
    def __init__(self, raw_root, warped_root, image_size=224):
        self.raw_root = Path(raw_root)
        self.warped_root = Path(warped_root)
        self.image_size = image_size
        self._validate()

    def _validate(self):
        if not self.raw_root.exists():
            # Fallback comun
            self.raw_root = Path("data/dataset")
        if not self.warped_root.exists():
            # Fallback path relativo
            self.warped_root = Path.cwd() / self.warped_root

    def load_data(self, split="train", balance=False, use_clahe=False):
        csv_path = self.warped_root / split / "images.csv"
        if not csv_path.exists():
            csv_path = self.warped_root / "images.csv" # Fallback estructura plana
        
        df = pd.read_csv(csv_path)
        if 'split' in df.columns:
            df = df[df['split'] == split]

        # Balanceo
        if balance and split == 'train':
            df['label_temp'] = df['category'].apply(lambda x: 0 if x == 'Normal' else 1)
            g = df.groupby('label_temp')
            df = g.apply(lambda x: x.sample(g.size().min(), random_state=42)).reset_index(drop=True)
            print(f"[LOADER] Balanceado a {len(df)} imágenes.")

        X_raw, X_warped, y = [], [], []
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) if use_clahe else None

        print(f"Cargando {split} desde {self.warped_root}...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            name = row['image_name']
            cat = row['category']
            w_name = row.get('warped_filename', row.get('filename', f"{name}.png"))
            
            # Buscar RAW
            raw_p = None
            for p in [self.raw_root / cat / "images" / f"{name}.png", 
                      self.raw_root / cat / f"{name}.png",
                      self.raw_root / f"{name}.png"]:
                if p.exists(): raw_p = p; break
            
            # Buscar WARPED
            warp_p = None
            for p in [self.warped_root / split / cat / w_name,
                      self.warped_root / cat / w_name,
                      self.warped_root / w_name]:
                if p.exists(): warp_p = p; break
                
            if not raw_p or not warp_p: continue
            
            img_r = cv2.imread(str(raw_p), cv2.IMREAD_GRAYSCALE)
            img_w = cv2.imread(str(warp_p), cv2.IMREAD_GRAYSCALE)
            
            if img_r is None or img_w is None: continue
            
            img_r = cv2.resize(img_r, (self.image_size, self.image_size))
            img_w = cv2.resize(img_w, (self.image_size, self.image_size))
            
            if use_clahe:
                img_r = clahe.apply(img_r)
                img_w = clahe.apply(img_w)
                
            X_raw.append(img_r.flatten())
            X_warped.append(img_w.flatten())
            y.append(0 if cat == 'Normal' else 1)

        return np.array(X_raw), np.array(X_warped), np.array(y)

class FisherOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def get_fisher_weights(self, X, y, n_comp):
        # Calcular Fisher para CADA componente
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        
        # Vectorizado para velocidad
        mean_0 = np.mean(X_0, axis=0)
        mean_1 = np.mean(X_1, axis=0)
        var_0 = np.var(X_0, axis=0, ddof=1)
        var_1 = np.var(X_1, axis=0, ddof=1)
        
        # J = (m1 - m2)^2 / (v1 + v2)
        num = (mean_0 - mean_1)**2
        den = var_0 + var_1 + 1e-9
        J = num / den
        return np.sqrt(J) # Pesos son sqrt(J)

    def run_grid_search(self, X_train, y_train, X_test, y_test, components_list):
        results = []
        
        # 1. Escalar (Global)
        print("Escalando datos...")
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # PCA Global (ajustar al máximo n para no recalcular siempre)
        max_n = max(components_list)
        print(f"Ajustando PCA (max_n={max_n})...")
        pca_full = PCA(n_components=max_n)
        X_train_pca_full = pca_full.fit_transform(X_train_s)
        X_test_pca_full = pca_full.transform(X_test_s)
        
        explained_variance_full = np.cumsum(pca_full.explained_variance_ratio_)

        for n in components_list:
            print(f"  -> Evaluando n_components={n}...")
            
            # Slice de componentes
            X_tr = X_train_pca_full[:, :n]
            X_te = X_test_pca_full[:, :n]
            
            # Fisher Weighting
            weights = self.get_fisher_weights(X_tr, y_train, n)
            X_tr_w = X_tr * weights
            X_te_w = X_te * weights
            
            # Clasificadores
            models = {
                "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
                "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs'),
                "LinearSVM": LinearSVC(max_iter=1000, dual=False)
            }
            
            step_res = {
                "n_components": n,
                "explained_variance": explained_variance_full[n-1]
            }
            
            for name, model in models.items():
                model.fit(X_tr_w, y_train)
                pred = model.predict(X_te_w)
                acc = accuracy_score(y_test, pred)
                step_res[name] = acc
                
            results.append(step_res)
            
        return results

def plot_results(results_raw, results_warped, output_dir):
    output_dir = Path(output_dir)
    
    # Extraer datos
    comps = [r['n_components'] for r in results_raw]
    
    # Grafica 1: Varianza
    plt.figure(figsize=(10, 6))
    plt.plot(comps, [r['explained_variance'] for r in results_raw], 'r--o', label='RAW Variance')
    plt.plot(comps, [r['explained_variance'] for r in results_warped], 'b-o', label='WARPED Variance')
    plt.title("Capacidad de Compresión: Varianza Explicada vs Componentes")
    plt.xlabel("# Componentes")
    plt.ylabel("Varianza Explicada Acumulada")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "grid_variance.png")
    plt.close()
    
    # Grafica 2: Accuracy (Regresión Logística - Separabilidad Lineal)
    plt.figure(figsize=(10, 6))
    model = "LogisticRegression"
    plt.plot(comps, [r[model] for r in results_raw], 'r--s', label=f'RAW {model}')
    plt.plot(comps, [r[model] for r in results_warped], 'b-s', label=f'WARPED {model}')
    
    # Grafica 3: Accuracy (k-NN)
    model = "k-NN (k=5)"
    plt.plot(comps, [r[model] for r in results_raw], 'r--^', alpha=0.5, label=f'RAW {model}')
    plt.plot(comps, [r[model] for r in results_warped], 'b-^', alpha=0.5, label=f'WARPED {model}')
    
    plt.title("Separabilidad Lineal vs Complejidad (Regresión Logística vs k-NN)")
    plt.xlabel("# Componentes")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "grid_accuracy.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--raw-dir", default="data/dataset/COVID-19_Radiography_Dataset")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--balance", action="store_true")
    args = parser.parse_args()
    
    loader = DatasetLoader(args.raw_dir, args.dataset_dir)
    
    # Cargar datos una sola vez
    X_raw_tr, X_warp_tr, y_tr = loader.load_data("train", balance=args.balance, use_clahe=args.clahe)
    X_raw_te, X_warp_te, y_te = loader.load_data("test", balance=False, use_clahe=args.clahe)
    
    optimizer = FisherOptimizer()
    components = [10, 25, 50, 100, 150, 200]
    
    print("\n>>> GRID SEARCH: RAW DATASET <<<")
    res_raw = optimizer.run_grid_search(X_raw_tr, y_tr, X_raw_te, y_te, components)
    
    print("\n>>> GRID SEARCH: WARPED DATASET <<<")
    res_warp = optimizer.run_grid_search(X_warp_tr, y_tr, X_warp_te, y_te, components)
    
    # Imprimir tablas
    print("\nRESULTADOS RAW:")
    print(pd.DataFrame(res_raw))
    print("\nRESULTADOS WARPED:")
    print(pd.DataFrame(res_warp))
    
    # Guardar
    plot_results(res_raw, res_warp, "results")
    with open("results/grid_search.json", "w") as f:
        json.dump({"raw": res_raw, "warped": res_warp}, f, indent=2)

if __name__ == "__main__":
    main()
