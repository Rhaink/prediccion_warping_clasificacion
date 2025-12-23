#!/usr/bin/env python3
"""
ROLE: Senior Computer Vision Research Engineer
PROJECT: Thesis - Geometric Validation of Lung Warping via Fisher Linear Analysis
STACK: PyTorch (GPU), Numpy, Matplotlib, Scikit-Image

OBJETIVO:
Validación científica con generación de evidencia forense detallada.
Guarda comparativas individuales RAW vs WARPED para análisis cualitativo.
"""

import argparse
import shutil
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# Verificar GPU
GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Usando dispositivo: {GLOBAL_DEVICE}")


class DatasetLoader:
    def __init__(self, raw_root, warped_root, image_size=224):
        self.raw_root = Path(raw_root)
        self.warped_root = Path(warped_root)
        self.image_size = image_size
        self._validate()

    def _validate(self):
        if not self.raw_root.exists():
            self.raw_root = Path("data/dataset")
        if not self.warped_root.exists():
            self.warped_root = Path.cwd() / self.warped_root

    def get_raw_path(self, image_name, category):
        """Busca la imagen RAW original."""
        candidates = [
            self.raw_root / category / "images" / f"{image_name}.png",
            self.raw_root / category / f"{image_name}.png",
            self.raw_root / f"{image_name}.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def load_full_dataset(self, split="train", use_clahe=True):
        csv_path = self.warped_root / split / "images.csv"
        if not csv_path.exists():
            csv_path = self.warped_root / "images.csv"

        df = pd.read_csv(csv_path)
        if "split" in df.columns:
            df = df[df["split"] == split]

        N = len(df)
        D = self.image_size * self.image_size
        X = np.zeros((N, D), dtype=np.float32)
        y = np.zeros(N, dtype=np.int32)
        names = []
        categories = []

        clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) if use_clahe else None
        )

        print(f"[LOADER] Cargando {N} imágenes '{split}' en memoria (CPU)...")
        loaded_count = 0

        for idx, row in tqdm(df.iterrows(), total=N):
            name = row["image_name"]
            cat = row["category"]
            w_name = row.get("warped_filename", row.get("filename", f"{name}.png"))

            # Cargar WARPED
            p_candidates = [
                self.warped_root / split / cat / w_name,
                self.warped_root / cat / w_name,
                self.warped_root / w_name,
            ]

            img_path = None
            for p in p_candidates:
                if p.exists():
                    img_path = p
                    break

            if not img_path:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (self.image_size, self.image_size))
            if use_clahe:
                img = clahe.apply(img)

            X[loaded_count] = img.flatten()
            y[loaded_count] = 0 if cat == "Normal" else 1
            names.append(name)
            categories.append(cat)
            loaded_count += 1

        return (
            torch.from_numpy(X[:loaded_count]),
            torch.from_numpy(y[:loaded_count]),
            names,
            categories,
        )


class TorchAnalysis:
    def __init__(self, device):
        self.device = device

    def fit_pca_efficient(self, X, n_components):
        U, S, V = torch.pca_lowrank(X, q=n_components, center=True, niter=2)
        mean = torch.mean(X, dim=0)
        eigvals = S**2 / (X.shape[0] - 1)
        total_var = torch.var(X, dim=0, unbiased=True).sum()
        explained_variance_ratio = eigvals / total_var
        X_pca = torch.mm(X - mean, V)
        return X_pca, explained_variance_ratio, V, mean

    def fisher_score(self, X_pca, y):
        unique_classes = torch.unique(y)
        means = []
        vars = []
        for c in unique_classes:
            mask = y == c
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
                batch = X_test[i : i + batch_size]
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

    def save_detailed_results(
        self,
        X_test_cpu,
        y_test_cpu,
        y_pred_cpu,
        names,
        categories,
        loader,
        output_dir,
        num_samples=20,
    ):
        """
        Guarda comparativas individuales RAW vs WARPED para inspección manual.
        Calcula SSIM/PSNR para cuantificar deformación.
        """
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        y_true = y_test_cpu.numpy()
        y_pred = y_pred_cpu.numpy()

        # Categorizar índices
        indices = {
            "TP": np.where((y_true == 1) & (y_pred == 1))[0],
            "TN": np.where((y_true == 0) & (y_pred == 0))[0],
            "FP": np.where((y_true == 0) & (y_pred == 1))[0],
            "FN": np.where((y_true == 1) & (y_pred == 0))[0],
        }

        print(
            f"\n[DETALLE] Generando {num_samples} reportes individuales en '{output_dir}'..."
        )

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        for cat_label, idx_list in indices.items():
            # Crear subcarpeta
            cat_dir = output_dir / cat_label
            cat_dir.mkdir(exist_ok=True)

            # Seleccionar muestras
            if len(idx_list) == 0:
                continue
            selected = np.random.choice(
                idx_list, min(num_samples, len(idx_list)), replace=False
            )

            for idx in selected:
                name = names[idx]
                category = categories[idx]

                # 1. Obtener imagen Warped (del tensor)
                warped_flat = X_test_cpu[idx].numpy()
                img_warped = warped_flat.reshape(224, 224).astype(np.uint8)

                # 2. Cargar imagen Raw Original (del disco)
                raw_path = loader.get_raw_path(name, category)
                if not raw_path:
                    continue

                img_raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
                img_raw = cv2.resize(img_raw, (224, 224))

                # Aplicar CLAHE a RAW también para comparación justa visual
                img_raw_clahe = clahe.apply(img_raw)

                # 3. Métricas de Deformación
                # Cuantifican cuánto cambió la imagen (Geometría + Intensidad)
                val_ssim = ssim(img_raw_clahe, img_warped, data_range=255)
                val_psnr = psnr(img_raw_clahe, img_warped, data_range=255)

                # 4. Plot
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                axes[0].imshow(img_raw_clahe, cmap="gray")
                axes[0].set_title(f"RAW (Input)\n{name}", fontsize=10)
                axes[0].axis("off")

                axes[1].imshow(img_warped, cmap="gray")
                axes[1].set_title(
                    f"WARPED (Processed)\nSSIM: {val_ssim:.2f} | PSNR: {val_psnr:.1f}",
                    fontsize=10,
                )
                axes[1].axis("off")

                real_lbl = "Enfermo" if y_true[idx] == 1 else "Sano"
                pred_lbl = "Enfermo" if y_pred[idx] == 1 else "Sano"
                color = "green" if real_lbl == pred_lbl else "red"

                plt.suptitle(
                    f"{cat_label}: Real={real_lbl} vs Pred={pred_lbl}",
                    color=color,
                    fontsize=14,
                    fontweight="bold",
                )

                save_path = cat_dir / f"{cat_label}_{name}.png"
                plt.savefig(save_path, bbox_inches="tight", dpi=100)
                plt.close()


def run_scientific_validation(args, device):
    print("\n" + "=" * 70)
    print("ANÁLISIS FORENSE DE CLASIFICACIÓN (GPU)")
    print("=" * 70)

    loader = DatasetLoader(args.raw_dir, args.dataset_dir)

    # 1. Cargar Train (solo para ajustar el modelo)
    X_train_cpu, y_train_cpu, _, _ = loader.load_full_dataset(
        "train", use_clahe=args.clahe
    )

    analyzer = TorchAnalysis(device)

    # Modelo Óptimo Detectado Previamente
    best_k = 50
    print(f"\n[MODELO] Ajustando Fisher-PCA (k={best_k}) en todo TRAIN...")

    try:
        X_train_gpu = X_train_cpu.to(device)
        y_train_gpu = y_train_cpu.to(device)

        # Pipeline
        mean = X_train_gpu.mean(dim=0)
        std = X_train_gpu.std(dim=0) + 1e-8
        X_train_norm = (X_train_gpu - mean) / std

        # PCA & Fisher
        X_train_pca, _, V, mean_pca = analyzer.fit_pca_efficient(X_train_norm, best_k)
        J_weights = analyzer.fisher_score(X_train_pca, y_train_gpu)
        weights = torch.sqrt(J_weights)
        X_train_final = X_train_pca * weights

        del X_train_gpu, X_train_norm
        torch.cuda.empty_cache()

    except RuntimeError:
        print("Error OOM entrenando modelo final.")
        return

    # 2. Cargar Test para Análisis
    print("\n[INFERENCIA] Cargando Test Set con metadatos...")
    X_test_cpu, y_test_cpu, names_test, cats_test = loader.load_full_dataset(
        "test", use_clahe=args.clahe
    )

    try:
        X_test_gpu = X_test_cpu.to(device)

        # Proyección
        X_test_norm = (X_test_gpu - mean) / std
        X_test_centered = X_test_norm - mean_pca
        X_test_pca = torch.mm(X_test_centered, V)
        X_test_final = X_test_pca * weights

        # Predicción
        y_pred_gpu = analyzer.knn_predict(X_train_final, y_train_gpu, X_test_final, k=5)
        y_pred_cpu = y_pred_gpu.cpu()

        acc = accuracy_score(y_test_cpu.numpy(), y_pred_cpu.numpy())
        print(f"\n>>> GLOBAL ACCURACY: {acc * 100:.2f}% <<<")

        # Generar Reporte Detallado
        analyzer.save_detailed_results(
            X_test_cpu,
            y_test_cpu,
            y_pred_cpu,
            names_test,
            cats_test,
            loader,
            "results/detailed_analysis",
            num_samples=50,
        )

    except RuntimeError as e:
        print(f"Error en inferencia: {e}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/full_warped_dataset")
    parser.add_argument(
        "--raw-dir", default="data/dataset/COVID-19_Radiography_Dataset"
    )
    parser.add_argument("--clahe", action="store_true", default=True)
    args = parser.parse_args()

    try:
        run_scientific_validation(args, GLOBAL_DEVICE)
    except Exception as e:
        print(f"Error crítico: {e}")
        import traceback

        traceback.print_exc()
