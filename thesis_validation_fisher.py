#!/usr/bin/env python3
"""
ROLE: Senior Computer Vision Research Engineer
PROJECT: Thesis - Geometric Validation of Lung Warping via Fisher Linear Analysis
STACK: Python, Numpy, Scikit-Learn, Matplotlib (WSL2 Env)

OBJETIVO:
Validar la técnica de Warping mediante análisis clásico (PCA + Fisher).
Hipótesis: Si el warping es correcto, las imágenes deben ser linealmente separables.

METODOLOGÍA:
1. Cargar datasets RAW y WARPED
2. Reducir dimensionalidad con PCA (10 componentes)
3. Implementar Fisher Linear Discriminant Analysis manualmente
4. Ponderar componentes por su ratio de Fisher (J)
5. Clasificar con k-NN (k=5)
6. Comparar rendimiento RAW vs WARPED
7. Generar visualizaciones críticas para validación del asesor

DATASETS:
- DS_GroundTruth: ~957 imágenes de alta calidad
- DS_Massive: 15k+ imágenes (train/val/test)

AUTOR: Generado para Tesis de Grado
FECHA: Diciembre 2025
"""

import numpy as np
import pandas as pd
import cv2
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')


class FisherPCAAnalyzer:
    """
    Implementa el análisis de Fisher sobre componentes principales.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.fisher_weights = None
        self.fisher_ratios = None

    def fit_transform_pca(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Paso 1: Normalizar (media=0, std=1)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Paso 2: Aplicar PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        print(f"[PCA] Varianza explicada acumulada: {self.pca.explained_variance_ratio_.sum():.4f}")
        return X_train_pca, X_test_pca

    def compute_fisher_ratios(self, X_pca: np.ndarray, y: np.ndarray) -> np.ndarray:
        fisher_ratios = np.zeros(self.n_components)

        X_healthy = X_pca[y == 0]  # Sano (Normal)
        X_sick = X_pca[y == 1]     # Enfermo (COVID + Viral Pneumonia)

        print(f"\n[Fisher] Distribución de clases para cálculo:")
        print(f"  - Sanos: {len(X_healthy)} muestras")
        print(f"  - Enfermos: {len(X_sick)} muestras")

        for i in range(self.n_components):
            mu_healthy = np.mean(X_healthy[:, i])
            var_healthy = np.var(X_healthy[:, i], ddof=1)
            mu_sick = np.mean(X_sick[:, i])
            var_sick = np.var(X_sick[:, i], ddof=1)

            numerator = (mu_healthy - mu_sick) ** 2
            denominator = var_healthy + var_sick

            if denominator > 1e-10:
                fisher_ratios[i] = numerator / denominator
            else:
                fisher_ratios[i] = 0.0

        self.fisher_ratios = fisher_ratios
        print(f"[Fisher] Ratios calculados: {fisher_ratios}")
        print(f"[Fisher] Componente más discriminante: PC{np.argmax(fisher_ratios) + 1} (J={fisher_ratios.max():.4f})")
        return fisher_ratios

    def apply_fisher_weighting(self, X_pca: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        if self.fisher_ratios is None:
            if y_train is None:
                raise ValueError("Debe calcular Fisher ratios primero o proporcionar y_train")
            self.compute_fisher_ratios(X_pca, y_train)

        self.fisher_weights = np.sqrt(self.fisher_ratios)
        X_weighted = X_pca * self.fisher_weights
        print(f"[Fisher Weighting] Pesos aplicados: {self.fisher_weights}")
        return X_weighted


class DatasetLoader:
    """
    Carga y prepara datasets RAW y WARPED para comparación.
    """

    def __init__(self,
                 raw_root: str,
                 warped_root: str,
                 image_size: int = 224):
        self.raw_root = Path(raw_root)
        self.warped_root = Path(warped_root)
        self.image_size = image_size
        self._validate_paths()

    def _validate_paths(self):
        if not self.raw_root.exists():
            print(f"[WARNING] Raw dataset no encontrado: {self.raw_root}")
            print("[INFO] Intentando con data/dataset/ como alternativa...")
            self.raw_root = Path("data/dataset")
        
        if not self.warped_root.exists():
             # Intentar crear path relativo si no existe absoluto
            relative_path = Path.cwd() / self.warped_root
            if relative_path.exists():
                self.warped_root = relative_path
            else:
                raise FileNotFoundError(f"Warped dataset no encontrado: {self.warped_root}")

        print(f"[Dataset] RAW: {self.raw_root.resolve()}")
        print(f"[Dataset] WARPED: {self.warped_root.resolve()}")

    def load_split(self, split: str = "test", balance: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Cargar CSV de mapeo
        csv_path = self.warped_root / split / "images.csv"
        # Fallback para datasets que no tienen subcarpetas split (como warped_dataset original)
        if not csv_path.exists():
            csv_path = self.warped_root / "images.csv"
            
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV de mapeo no encontrado: {csv_path}")

        df = pd.read_csv(csv_path)
        
        # Filtrar split si el CSV es global (tiene columna 'split')
        if 'split' in df.columns:
            df = df[df['split'] == split]
            
        print(f"\n[{split.upper()}] Cargando {len(df)} imágenes desde {self.warped_root}...")

        # Balancing logic (Undersampling majority class)
        if balance and split == 'train':
            print("[BALANCE] Equilibrando clases 50/50...")
            # Asumimos que la categoria esta en 'category'
            df['label'] = df['category'].apply(lambda x: 0 if x == 'Normal' else 1)
            
            count_0 = len(df[df['label'] == 0])
            count_1 = len(df[df['label'] == 1])
            min_count = min(count_0, count_1)
            
            df_0 = df[df['label'] == 0].sample(n=min_count, random_state=42)
            df_1 = df[df['label'] == 1].sample(n=min_count, random_state=42)
            df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"[BALANCE] Dataset reducido a {len(df)} muestras ({min_count} por clase)")

        X_raw_list = []
        X_warped_list = []
        y_list = []
        image_names = []
        
        # Para verificación visual
        debug_pairs = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Cargando {split}"):
            image_name = row['image_name']
            category = row['category']
            
            # Manejar diferentes nombres de columna para archivo warped
            if 'warped_filename' in row:
                warped_filename = row['warped_filename']
            elif 'filename' in row:
                warped_filename = row['filename']
            else:
                warped_filename = f"{image_name}.png" # Asunción por defecto

            label = 0 if category == "Normal" else 1

            # Cargar RAW
            raw_path_candidates = [
                self.raw_root / category / "images" / f"{image_name}.png",
                self.raw_root / category / f"{image_name}.png",
                Path("data/dataset") / category / f"{image_name}.png",
                self.raw_root / f"{image_name}.png"
            ]

            raw_img = None
            for raw_path in raw_path_candidates:
                if raw_path.exists():
                    raw_img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
                    if raw_img is not None:
                        break

            if raw_img is None: continue

            # Cargar WARPED
            # Intentar estructura anidada (split/category) o plana
            warped_path_candidates = [
                self.warped_root / split / category / warped_filename,
                self.warped_root / category / warped_filename,
                self.warped_root / warped_filename
            ]
            
            warped_path = None
            for wp in warped_path_candidates:
                if wp.exists():
                    warped_path = wp
                    break
            
            if warped_path is None: continue

            warped_img = cv2.imread(str(warped_path), cv2.IMREAD_GRAYSCALE)
            if warped_img is None: continue

            # Redimensionar
            raw_img = cv2.resize(raw_img, (self.image_size, self.image_size))
            warped_img = cv2.resize(warped_img, (self.image_size, self.image_size))

            X_raw_list.append(raw_img.flatten())
            X_warped_list.append(warped_img.flatten())
            y_list.append(label)
            image_names.append(image_name)
            
            if len(debug_pairs) < 5:
                debug_pairs.append((raw_img, warped_img, image_name, category))

        X_raw = np.array(X_raw_list, dtype=np.float32)
        X_warped = np.array(X_warped_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        print(f"[{split.upper()}] Cargadas: {len(y)} imágenes. Sanos: {(y==0).sum()}, Enfermos: {(y==1).sum()}")
        
        return X_raw, X_warped, y, image_names, debug_pairs


class Visualizer:
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def verify_matching(self, debug_pairs):
        """Genera una imagen grid para verificar correspondencia."""
        fig, axes = plt.subplots(5, 2, figsize=(8, 15))
        for i, (raw, warped, name, cat) in enumerate(debug_pairs):
            if i >= 5: break
            axes[i, 0].imshow(raw, cmap='gray')
            axes[i, 0].set_title(f"RAW: {name}\n({cat})")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(warped, cmap='gray')
            axes[i, 1].set_title(f"WARPED: {name}\n({cat})")
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        save_path = self.output_dir / "audit_verification_grid.png"
        plt.savefig(save_path)
        print(f"\n[AUDIT] Imagen de verificación guardada en: {save_path}")
        print("Revisa esta imagen para asegurar que RAW y WARPED corresponden al mismo paciente.")
        plt.close()

    def plot_fisher_ratios(self, fisher_ratios: np.ndarray, filename: str = "fisher_ratios.png"):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(1, len(fisher_ratios) + 1)
        colors = plt.cm.viridis(fisher_ratios / fisher_ratios.max())
        bars = ax.bar(x, fisher_ratios, color=colors, edgecolor='black', linewidth=1.5)
        
        max_idx = np.argmax(fisher_ratios)
        bars[max_idx].set_color('red')
        bars[max_idx].set_edgecolor('darkred')
        bars[max_idx].set_linewidth(2.5)

        ax.set_xlabel('Componente Principal (PC)')
        ax.set_ylabel('Fisher Ratio (J)')
        ax.set_title('Discriminabilidad de Componentes Principales')
        ax.set_xticks(x)
        ax.set_xticklabels([f'PC{i}' for i in x])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()

    def plot_pca_comparison(self, X_pca_u, X_pca_w, y, fisher_ratios, filename="pca_comparison.png"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        colors = ['green', 'red']
        labels = ['Sano', 'Enfermo']

        # PC1 vs PC2
        for i, (c, l) in enumerate(zip(colors, labels)):
            mask = (y == i)
            axes[0].scatter(X_pca_u[mask, 0], X_pca_u[mask, 1], c=c, label=l, alpha=0.5, s=15)
        axes[0].set_title('PCA Estándar (PC1 vs PC2)')
        axes[0].legend()

        # Top Fisher
        top_2 = np.argsort(fisher_ratios)[-2:][::-1]
        for i, (c, l) in enumerate(zip(colors, labels)):
            mask = (y == i)
            axes[1].scatter(X_pca_w[mask, top_2[0]], X_pca_w[mask, top_2[1]], c=c, label=l, alpha=0.5, s=15)
        axes[1].set_title(f'Fisher PCA (PC{top_2[0]+1} vs PC{top_2[1]+1})')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()

    def plot_dominant_reconstruction(self, pca, scaler, fisher_ratios, size=224, filename="dom.png"):
        dom_idx = np.argmax(fisher_ratios)
        amp = np.sqrt(pca.explained_variance_[dom_idx])
        vec = np.zeros(len(fisher_ratios))
        vec[dom_idx] = amp
        
        rec = scaler.inverse_transform(pca.inverse_transform(vec.reshape(1, -1)))
        img = rec.reshape(size, size)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='jet')
        ax.set_title(f'Reconstrucción PC{dom_idx+1} (J={fisher_ratios[dom_idx]:.2f})')
        ax.axis('off')
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()


def run_experiment(args):
    print("\n" + "="*80)
    print(f"FISHER VALIDATION: {args.dataset_dir}")
    print(f"Balancing: {args.balance}")
    print("="*80)

    loader = DatasetLoader(
        raw_root=args.raw_dir,
        warped_root=args.dataset_dir,
        image_size=224
    )
    
    visualizer = Visualizer(output_dir="./results")

    # 1. Cargar Datos
    try:
        # Intentar cargar train/test splits estandar
        X_raw_train, X_warped_train, y_train, _, train_pairs = loader.load_split("train", balance=args.balance)
        X_raw_test, X_warped_test, y_test, _, test_pairs = loader.load_split("test", balance=False)
    except Exception as e:
        print(f"[INFO] Fallo cargando splits estándar ({e}). Intentando cargar todo el dataset como train/test split manual...")
        # Fallback para datasets sin estructura train/test (como warped_dataset original)
        X_raw, X_warped, y, _, pairs = loader.load_split("all", balance=args.balance)
        train_pairs = pairs
        
        # Split manual 80/20
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(y))
        idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
        
        X_raw_train, X_raw_test = X_raw[idx_train], X_raw[idx_test]
        X_warped_train, X_warped_test = X_warped[idx_train], X_warped[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

    # Verificar Integridad y Salir
    if args.verify_matching:
        print("\n[VERIFICACIÓN] Generando grid de integridad visual...")
        visualizer.verify_matching(train_pairs)
        return

    # 2. Análisis RAW
    print("\n--- ANALIZANDO RAW ---")
    analyzer_raw = FisherPCAAnalyzer(n_components=10)
    X_raw_train_pca, X_raw_test_pca = analyzer_raw.fit_transform_pca(X_raw_train, X_raw_test)
    analyzer_raw.compute_fisher_ratios(X_raw_train_pca, y_train)
    X_raw_train_w = analyzer_raw.apply_fisher_weighting(X_raw_train_pca)
    X_raw_test_w = X_raw_test_pca * analyzer_raw.fisher_weights

    knn_raw = KNeighborsClassifier(n_neighbors=5)
    knn_raw.fit(X_raw_train_w, y_train)
    acc_raw = knn_raw.score(X_raw_test_w, y_test)
    print(f"RAW Accuracy: {acc_raw:.4f}")

    # 3. Análisis WARPED
    print("\n--- ANALIZANDO WARPED ---")
    analyzer_warped = FisherPCAAnalyzer(n_components=10)
    X_warped_train_pca, X_warped_test_pca = analyzer_warped.fit_transform_pca(X_warped_train, X_warped_test)
    analyzer_warped.compute_fisher_ratios(X_warped_train_pca, y_train)
    X_warped_train_w = analyzer_warped.apply_fisher_weighting(X_warped_train_pca)
    X_warped_test_w = X_warped_test_pca * analyzer_warped.fisher_weights

    knn_warped = KNeighborsClassifier(n_neighbors=5)
    knn_warped.fit(X_warped_train_w, y_train)
    acc_warped = knn_warped.score(X_warped_test_w, y_test)
    print(f"WARPED Accuracy: {acc_warped:.4f}")

    # 4. Reporte
    print("\n" + "="*80)
    print(f"RESULTADO FINAL (Balanceado: {args.balance})")
    print(f"RAW Accuracy:    {acc_raw:.4f}")
    print(f"WARPED Accuracy: {acc_warped:.4f}")
    print(f"Diferencia:      {acc_warped - acc_raw:.4f}")
    
    # Guardar Visualizaciones
    visualizer.plot_fisher_ratios(analyzer_warped.fisher_ratios, "fisher_ratios_warped.png")
    visualizer.plot_pca_comparison(X_warped_test_pca, X_warped_test_w, y_test, analyzer_warped.fisher_ratios, "pca_comparison_warped.png")
    visualizer.plot_dominant_reconstruction(analyzer_warped.pca, analyzer_warped.scaler, analyzer_warped.fisher_ratios, 224, "dom_warped.png")

    # Guardar JSON
    results = {
        "dataset": str(args.dataset_dir),
        "balanced": args.balance,
        "raw_accuracy": acc_raw,
        "warped_accuracy": acc_warped,
        "raw_fisher_max": float(analyzer_raw.fisher_ratios.max()),
        "warped_fisher_max": float(analyzer_warped.fisher_ratios.max()),
        "raw_explained_variance": float(analyzer_raw.pca.explained_variance_ratio_.sum()),
        "warped_explained_variance": float(analyzer_warped.pca.explained_variance_ratio_.sum())
    }
    with open(visualizer.output_dir / "audit_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/full_warped_dataset", help="Path to warped dataset")
    parser.add_argument("--raw-dir", default="data/dataset/COVID-19_Radiography_Dataset", help="Path to raw dataset")
    parser.add_argument("--balance", action="store_true", help="Balance training classes 50/50")
    parser.add_argument("--verify-matching", action="store_true", help="Generate integrity verification image and exit")
    
    args = parser.parse_args()
    run_experiment(args)
