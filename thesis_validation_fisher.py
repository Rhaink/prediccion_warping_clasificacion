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
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FisherPCAAnalyzer:
    """
    Implementa el análisis de Fisher sobre componentes principales.

    Fisher Ratio (J): Mide la separabilidad entre clases para cada característica.
    Formula: J_i = (μ_sano - μ_enfermo)² / (σ²_sano + σ²_enfermo)

    Interpretación:
    - J alto: La característica discrimina bien entre clases
    - J bajo: La característica es dominada por varianza intra-clase
    """

    def __init__(self, n_components: int = 10):
        """
        Args:
            n_components: Número de componentes principales a extraer
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.fisher_weights = None
        self.fisher_ratios = None

    def fit_transform_pca(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta PCA sobre train y transforma train/test.

        Args:
            X_train: Imágenes de entrenamiento (N, H*W)
            X_test: Imágenes de prueba (M, H*W)

        Returns:
            X_train_pca: Coeficientes PCA de train (N, n_components)
            X_test_pca: Coeficientes PCA de test (M, n_components)
        """
        # Paso 1: Normalizar (media=0, std=1)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Paso 2: Aplicar PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        print(f"[PCA] Varianza explicada acumulada: {self.pca.explained_variance_ratio_.sum():.4f}")
        print(f"[PCA] Forma de salida: Train {X_train_pca.shape}, Test {X_test_pca.shape}")

        return X_train_pca, X_test_pca

    def compute_fisher_ratios(self, X_pca: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula el Fisher Ratio para cada componente principal.

        Formula: J_i = (μ_0 - μ_1)² / (σ²_0 + σ²_1)
        Donde:
            - μ_0, μ_1: Medias de clase 0 (Sano) y clase 1 (Enfermo)
            - σ²_0, σ²_1: Varianzas de clase 0 y clase 1

        Args:
            X_pca: Datos en espacio PCA (N, n_components)
            y: Etiquetas binarias (N,) [0=Sano, 1=Enfermo]

        Returns:
            fisher_ratios: Array con J para cada componente (n_components,)
        """
        fisher_ratios = np.zeros(self.n_components)

        # Separar por clase
        X_healthy = X_pca[y == 0]  # Sano (Normal)
        X_sick = X_pca[y == 1]     # Enfermo (COVID + Viral Pneumonia)

        print(f"\n[Fisher] Distribución de clases:")
        print(f"  - Sanos: {len(X_healthy)} muestras")
        print(f"  - Enfermos: {len(X_sick)} muestras")

        # Calcular J para cada componente
        for i in range(self.n_components):
            # Estadísticas de clase 0 (Sano)
            mu_healthy = np.mean(X_healthy[:, i])
            var_healthy = np.var(X_healthy[:, i], ddof=1)  # ddof=1 para varianza muestral

            # Estadísticas de clase 1 (Enfermo)
            mu_sick = np.mean(X_sick[:, i])
            var_sick = np.var(X_sick[:, i], ddof=1)

            # Fisher Ratio
            numerator = (mu_healthy - mu_sick) ** 2
            denominator = var_healthy + var_sick

            # Evitar división por cero
            if denominator > 1e-10:
                fisher_ratios[i] = numerator / denominator
            else:
                fisher_ratios[i] = 0.0

        self.fisher_ratios = fisher_ratios
        print(f"[Fisher] Ratios calculados: {fisher_ratios}")
        print(f"[Fisher] Componente más discriminante: PC{np.argmax(fisher_ratios) + 1} (J={fisher_ratios.max():.4f})")

        return fisher_ratios

    def apply_fisher_weighting(self, X_pca: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Pondera cada componente principal por su Fisher Ratio.

        Estrategia: Multiplicar cada columna por sqrt(J_i) para amplificar
        la distancia Euclidiana en las componentes discriminantes.

        Args:
            X_pca: Datos en espacio PCA (N, n_components)
            y_train: Etiquetas de entrenamiento (solo si no se calcularon ratios antes)

        Returns:
            X_weighted: Datos ponderados por Fisher (N, n_components)
        """
        if self.fisher_ratios is None:
            if y_train is None:
                raise ValueError("Debe calcular Fisher ratios primero o proporcionar y_train")
            self.compute_fisher_ratios(X_pca, y_train)

        # Aplicar raíz cuadrada de J como peso (para distancia Euclidiana)
        self.fisher_weights = np.sqrt(self.fisher_ratios)
        X_weighted = X_pca * self.fisher_weights

        print(f"[Fisher Weighting] Pesos aplicados: {self.fisher_weights}")

        return X_weighted


class DatasetLoader:
    """
    Carga y prepara datasets RAW y WARPED para comparación.
    """

    def __init__(self,
                 raw_root: str = "data/dataset/COVID-19_Radiography_Dataset",
                 warped_root: str = "outputs/full_warped_dataset",
                 image_size: int = 224):
        """
        Args:
            raw_root: Ruta al dataset original (RAW)
            warped_root: Ruta al dataset alineado (WARPED)
            image_size: Tamaño de imagen (224x224)
        """
        self.raw_root = Path(raw_root)
        self.warped_root = Path(warped_root)
        self.image_size = image_size

        # Validar existencia de datasets
        self._validate_paths()

    def _validate_paths(self):
        """Verifica que los directorios existan."""
        if not self.raw_root.exists():
            print(f"[WARNING] Raw dataset no encontrado: {self.raw_root}")
            print("[INFO] Intentando con data/dataset/ como alternativa...")
            self.raw_root = Path("data/dataset")

        if not self.warped_root.exists():
            raise FileNotFoundError(f"Warped dataset no encontrado: {self.warped_root}")

        print(f"[Dataset] RAW: {self.raw_root.resolve()}")
        print(f"[Dataset] WARPED (Source of Truth): {self.warped_root.resolve()}")

    def load_split(self, split: str = "test", use_massive: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga un split (train/val/test) con versiones RAW y WARPED.

        Args:
            split: 'train', 'val' o 'test'
            use_massive: Si True, usa DS_Massive (15k). Si False, solo DS_GroundTruth (999)

        Returns:
            X_raw: Imágenes RAW aplanadas (N, H*W)
            X_warped: Imágenes WARPED aplanadas (N, H*W)
            y: Etiquetas binarias (N,) [0=Sano, 1=Enfermo]
            image_names: Lista de nombres de archivo (N,)
        """
        # Cargar CSV de mapeo (warped dataset)
        csv_path = self.warped_root / split / "images.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV de mapeo no encontrado: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"\n[{split.upper()}] Cargando {len(df)} imágenes desde {self.warped_root}...")

        X_raw_list = []
        X_warped_list = []
        y_list = []
        image_names = []

        failed_raw = 0
        failed_warped = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Cargando {split}"):
            image_name = row['image_name']
            category = row['category']
            warped_filename = row['warped_filename']

            # Etiquetado binario: 0=Normal (Sano), 1=COVID/Viral_Pneumonia (Enfermo)
            if category == "Normal":
                label = 0
            else:  # COVID o Viral_Pneumonia
                label = 1

            # Cargar RAW
            # Intentar diferentes rutas posibles
            raw_path_candidates = [
                self.raw_root / category / "images" / f"{image_name}.png",
                self.raw_root / category / f"{image_name}.png",
                Path("data/dataset") / category / f"{image_name}.png",
            ]

            raw_img = None
            for raw_path in raw_path_candidates:
                if raw_path.exists():
                    raw_img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
                    if raw_img is not None:
                        break

            if raw_img is None:
                failed_raw += 1
                continue

            # Cargar WARPED
            warped_path = self.warped_root / split / category / warped_filename
            
            # AUDITORIA: Muestreo de rutas para verificar origen correcto
            if idx % 2000 == 0:
                print(f"  [AUDIT] Cargando: {warped_path}")

            if not warped_path.exists():
                failed_warped += 1
                continue

            warped_img = cv2.imread(str(warped_path), cv2.IMREAD_GRAYSCALE)
            if warped_img is None:
                failed_warped += 1
                continue

            # Redimensionar a tamaño fijo
            raw_img = cv2.resize(raw_img, (self.image_size, self.image_size))
            warped_img = cv2.resize(warped_img, (self.image_size, self.image_size))

            # Flatten a vector 1D
            X_raw_list.append(raw_img.flatten())
            X_warped_list.append(warped_img.flatten())
            y_list.append(label)
            image_names.append(image_name)

        # Convertir a numpy arrays
        X_raw = np.array(X_raw_list, dtype=np.float32)
        X_warped = np.array(X_warped_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        print(f"[{split.upper()}] Cargadas exitosamente: {len(y)} imágenes")
        print(f"[{split.upper()}] Fallos RAW: {failed_raw}, WARPED: {failed_warped}")
        print(f"[{split.upper()}] Distribución - Sanos: {(y==0).sum()}, Enfermos: {(y==1).sum()}")
        print(f"[{split.upper()}] Shape - X_raw: {X_raw.shape}, X_warped: {X_warped.shape}")

        return X_raw, X_warped, y, image_names


class Visualizer:
    """
    Genera las 3 visualizaciones críticas requeridas por el asesor.
    """

    def __init__(self, output_dir: str = "./results"):
        """
        Args:
            output_dir: Directorio para guardar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_fisher_ratios(self, fisher_ratios: np.ndarray, filename: str = "fisher_ratios.png"):
        """
        1. GRÁFICO DE BARRAS: Fisher Ratios para cada PC

        Objetivo: Identificar qué componentes principales son discriminantes.
        Interpretación: Barras altas = componentes que separan bien Sano vs Enfermo.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(1, len(fisher_ratios) + 1)
        colors = plt.cm.viridis(fisher_ratios / fisher_ratios.max())

        bars = ax.bar(x, fisher_ratios, color=colors, edgecolor='black', linewidth=1.5)

        # Destacar el componente máximo
        max_idx = np.argmax(fisher_ratios)
        bars[max_idx].set_color('red')
        bars[max_idx].set_edgecolor('darkred')
        bars[max_idx].set_linewidth(2.5)

        ax.set_xlabel('Componente Principal (PC)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Fisher Ratio (J)', fontsize=14, fontweight='bold')
        ax.set_title('Discriminabilidad de Componentes Principales\n(Análisis de Fisher)',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'PC{i}' for i in x])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Anotar el máximo
        ax.annotate(f'Máximo: PC{max_idx+1}\nJ={fisher_ratios[max_idx]:.4f}',
                   xy=(max_idx+1, fisher_ratios[max_idx]),
                   xytext=(max_idx+1+1, fisher_ratios[max_idx]*0.9),
                   fontsize=12, fontweight='bold', color='darkred',
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Viz] Fisher Ratios guardado: {save_path}")
        plt.close()

    def plot_pca_comparison(self,
                           X_pca_unweighted: np.ndarray,
                           X_pca_weighted: np.ndarray,
                           y: np.ndarray,
                           fisher_ratios: np.ndarray,
                           filename: str = "pca_comparison.png"):
        """
        2. SCATTER PLOT COMPARATIVO: PCA sin Fisher vs PCA con Fisher

        Objetivo: Demostrar que el weighting de Fisher mejora la separabilidad visual.
        Izquierda: PC1 vs PC2 (varianza máxima, sin discriminación)
        Derecha: Top 2 PCs por Fisher Ratio (discriminación óptima)
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Colores por clase
        colors = ['green', 'red']
        labels = ['Sano (Normal)', 'Enfermo (COVID + VP)']

        # Panel Izquierdo: PC1 vs PC2 (sin Fisher)
        ax1 = axes[0]
        for class_idx, (color, label) in enumerate(zip(colors, labels)):
            mask = (y == class_idx)
            ax1.scatter(X_pca_unweighted[mask, 0],
                       X_pca_unweighted[mask, 1],
                       c=color, label=label, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)

        ax1.set_xlabel('PC1 (Varianza Máxima)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax1.set_title('SIN Fisher Weighting\n(PCA Estándar)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(alpha=0.3, linestyle='--')

        # Panel Derecho: Top 2 PCs por Fisher (con weighting)
        ax2 = axes[1]

        # Obtener índices de las 2 componentes con mayor Fisher Ratio
        top_2_indices = np.argsort(fisher_ratios)[-2:][::-1]  # Descendente
        pc1_idx, pc2_idx = top_2_indices

        for class_idx, (color, label) in enumerate(zip(colors, labels)):
            mask = (y == class_idx)
            ax2.scatter(X_pca_weighted[mask, pc1_idx],
                       X_pca_weighted[mask, pc2_idx],
                       c=color, label=label, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)

        ax2.set_xlabel(f'PC{pc1_idx+1} (J={fisher_ratios[pc1_idx]:.3f})', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'PC{pc2_idx+1} (J={fisher_ratios[pc2_idx]:.3f})', fontsize=12, fontweight='bold')
        ax2.set_title('CON Fisher Weighting\n(Componentes Discriminantes)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', frameon=True, shadow=True)
        ax2.grid(alpha=0.3, linestyle='--')

        plt.suptitle('Comparación: PCA Estándar vs Fisher-Weighted PCA',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Viz] PCA Comparison guardado: {save_path}")
        plt.close()

    def plot_dominant_component_reconstruction(self,
                                              pca: PCA,
                                              scaler: StandardScaler,
                                              fisher_ratios: np.ndarray,
                                              image_size: int = 224,
                                              filename: str = "dominant_component.png"):
        """
        3. RECONSTRUCCIÓN VISUAL: Componente con Fisher Ratio máximo

        Objetivo: Validar que la componente más discriminante captura anatomía pulmonar.
        Método: Usar inverse_transform() de PCA para volver a espacio de imagen.
        Interpretación: Si se iluminan los pulmones, el warping preserva anatomía relevante.
        """
        # Obtener índice del componente dominante
        dominant_idx = np.argmax(fisher_ratios)
        dominant_j = fisher_ratios[dominant_idx]

        # Crear vector con solo ese componente activo
        # Usamos la raíz de la varianza explicada como amplitud
        amplitude = np.sqrt(pca.explained_variance_[dominant_idx])
        pca_vector = np.zeros(len(fisher_ratios))
        pca_vector[dominant_idx] = amplitude

        # Reconstruir en espacio original
        reconstructed_scaled = pca.inverse_transform(pca_vector.reshape(1, -1))
        reconstructed = scaler.inverse_transform(reconstructed_scaled)

        # Reshape a imagen
        reconstructed_img = reconstructed.reshape(image_size, image_size)

        # Normalizar para visualización
        reconstructed_img = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min() + 1e-8)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(reconstructed_img, cmap='jet', interpolation='bilinear')
        ax.set_title(f'Reconstrucción de PC{dominant_idx+1}\n(Fisher Ratio J={dominant_j:.4f})',
                    fontsize=16, fontweight='bold')
        ax.axis('off')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Intensidad Normalizada', fontsize=12, fontweight='bold')

        # Texto informativo
        textstr = f'Interpretación:\n- Regiones rojas/amarillas: Alta contribución\n- Regiones azules: Baja contribución\n- Buscar: ¿Se destacan los pulmones?'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Viz] Dominant Component guardado: {save_path}")
        plt.close()


def run_experiment(dataset_loader: DatasetLoader,
                  analyzer: FisherPCAAnalyzer,
                  visualizer: Visualizer,
                  use_train_for_test: bool = False):
    """
    Ejecuta el experimento completo: RAW vs WARPED con Fisher-weighted PCA.

    Pipeline:
    1. Cargar train/test splits
    2. PCA en ambos (RAW y WARPED)
    3. Calcular Fisher ratios
    4. Ponderar componentes
    5. Clasificar con k-NN
    6. Generar visualizaciones

    Args:
        dataset_loader: Cargador de datos
        analyzer: Analizador PCA + Fisher
        visualizer: Generador de gráficos
        use_train_for_test: Si True, usa train para test (modo debug)
    """
    print("\n" + "="*80)
    print("EXPERIMENTO: VALIDACIÓN DE WARPING VÍA FISHER LINEAR ANALYSIS")
    print("="*80)

    # =========================================================================
    # 1. CARGAR DATOS
    # =========================================================================
    print("\n[PASO 1/6] Cargando datasets...")

    X_raw_train, X_warped_train, y_train, _ = dataset_loader.load_split("train")

    if use_train_for_test:
        print("[DEBUG MODE] Usando train como test para verificación rápida")
        X_raw_test, X_warped_test, y_test, _ = X_raw_train, X_warped_train, y_train, _
    else:
        X_raw_test, X_warped_test, y_test, _ = dataset_loader.load_split("test")

    # =========================================================================
    # 2. EXPERIMENTO 1: RAW (Control)
    # =========================================================================
    print("\n" + "="*80)
    print("[EXPERIMENTO 1] IMÁGENES RAW (Control)")
    print("="*80)

    print("\n[PASO 2/6] Aplicando PCA a RAW...")
    analyzer_raw = FisherPCAAnalyzer(n_components=10)
    X_raw_train_pca, X_raw_test_pca = analyzer_raw.fit_transform_pca(X_raw_train, X_raw_test)

    print("\n[PASO 3/6] Calculando Fisher Ratios para RAW...")
    fisher_ratios_raw = analyzer_raw.compute_fisher_ratios(X_raw_train_pca, y_train)

    print("\n[PASO 4/6] Aplicando Fisher Weighting a RAW...")
    X_raw_train_weighted = analyzer_raw.apply_fisher_weighting(X_raw_train_pca)
    X_raw_test_weighted = X_raw_test_pca * analyzer_raw.fisher_weights

    print("\n[PASO 5/6] Clasificando con k-NN (k=5) - RAW...")
    knn_raw = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_raw.fit(X_raw_train_weighted, y_train)
    y_pred_raw = knn_raw.predict(X_raw_test_weighted)

    print("\n[RESULTADOS RAW]")
    print(classification_report(y_test, y_pred_raw, target_names=['Sano', 'Enfermo'], digits=4))
    acc_raw = accuracy_score(y_test, y_pred_raw)

    # =========================================================================
    # 3. EXPERIMENTO 2: WARPED (Target)
    # =========================================================================
    print("\n" + "="*80)
    print("[EXPERIMENTO 2] IMÁGENES WARPED (Target)")
    print("="*80)

    print("\n[PASO 2/6] Aplicando PCA a WARPED...")
    analyzer_warped = FisherPCAAnalyzer(n_components=10)
    X_warped_train_pca, X_warped_test_pca = analyzer_warped.fit_transform_pca(X_warped_train, X_warped_test)

    print("\n[PASO 3/6] Calculando Fisher Ratios para WARPED...")
    fisher_ratios_warped = analyzer_warped.compute_fisher_ratios(X_warped_train_pca, y_train)

    print("\n[PASO 4/6] Aplicando Fisher Weighting a WARPED...")
    X_warped_train_weighted = analyzer_warped.apply_fisher_weighting(X_warped_train_pca)
    X_warped_test_weighted = X_warped_test_pca * analyzer_warped.fisher_weights

    print("\n[PASO 5/6] Clasificando con k-NN (k=5) - WARPED...")
    knn_warped = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_warped.fit(X_warped_train_weighted, y_train)
    y_pred_warped = knn_warped.predict(X_warped_test_weighted)

    print("\n[RESULTADOS WARPED]")
    print(classification_report(y_test, y_pred_warped, target_names=['Sano', 'Enfermo'], digits=4))
    acc_warped = accuracy_score(y_test, y_pred_warped)

    # =========================================================================
    # 4. COMPARACIÓN FINAL
    # =========================================================================
    print("\n" + "="*80)
    print("COMPARACIÓN FINAL: RAW vs WARPED")
    print("="*80)
    print(f"Accuracy RAW:    {acc_raw:.4f}")
    print(f"Accuracy WARPED: {acc_warped:.4f}")
    print(f"Mejora:          {(acc_warped - acc_raw):.4f} ({((acc_warped - acc_raw) / acc_raw * 100):.2f}%)")

    if acc_warped > acc_raw:
        print("\n✅ HIPÓTESIS VALIDADA: El warping mejora la separabilidad lineal.")
    else:
        print("\n⚠️  ADVERTENCIA: El warping NO mejoró el rendimiento.")

    # =========================================================================
    # 5. VISUALIZACIONES
    # =========================================================================
    print("\n[PASO 6/6] Generando visualizaciones...")

    # Viz 1: Fisher Ratios (usar WARPED por ser el target)
    visualizer.plot_fisher_ratios(fisher_ratios_warped, "fisher_ratios_warped.png")
    visualizer.plot_fisher_ratios(fisher_ratios_raw, "fisher_ratios_raw.png")

    # Viz 2: PCA Comparison (WARPED)
    visualizer.plot_pca_comparison(
        X_warped_test_pca,
        X_warped_test_weighted,
        y_test,
        fisher_ratios_warped,
        "pca_comparison_warped.png"
    )

    visualizer.plot_pca_comparison(
        X_raw_test_pca,
        X_raw_test_weighted,
        y_test,
        fisher_ratios_raw,
        "pca_comparison_raw.png"
    )

    # Viz 3: Reconstrucción de componente dominante (WARPED)
    visualizer.plot_dominant_component_reconstruction(
        analyzer_warped.pca,
        analyzer_warped.scaler,
        fisher_ratios_warped,
        224,
        "dominant_component_warped.png"
    )

    visualizer.plot_dominant_component_reconstruction(
        analyzer_raw.pca,
        analyzer_raw.scaler,
        fisher_ratios_raw,
        224,
        "dominant_component_raw.png"
    )

    # =========================================================================
    # 6. GUARDAR RESULTADOS
    # =========================================================================
    results = {
        "experiment": "Fisher Linear Analysis - Warping Validation",
        "date": "2025-12-20",
        "dataset": {
            "train_size": len(y_train),
            "test_size": len(y_test),
            "class_distribution_train": {"Sano": int((y_train==0).sum()), "Enfermo": int((y_train==1).sum())},
            "class_distribution_test": {"Sano": int((y_test==0).sum()), "Enfermo": int((y_test==1).sum())}
        },
        "pca": {
            "n_components": 10,
            "explained_variance_raw": float(analyzer_raw.pca.explained_variance_ratio_.sum()),
            "explained_variance_warped": float(analyzer_warped.pca.explained_variance_ratio_.sum())
        },
        "fisher_ratios": {
            "raw": fisher_ratios_raw.tolist(),
            "warped": fisher_ratios_warped.tolist(),
            "dominant_pc_raw": int(np.argmax(fisher_ratios_raw)) + 1,
            "dominant_pc_warped": int(np.argmax(fisher_ratios_warped)) + 1
        },
        "classification": {
            "raw": {
                "accuracy": float(acc_raw),
                "confusion_matrix": confusion_matrix(y_test, y_pred_raw).tolist()
            },
            "warped": {
                "accuracy": float(acc_warped),
                "confusion_matrix": confusion_matrix(y_test, y_pred_warped).tolist()
            },
            "improvement": float(acc_warped - acc_raw),
            "improvement_percent": float((acc_warped - acc_raw) / acc_raw * 100)
        },
        "conclusion": "WARPED better than RAW" if acc_warped > acc_raw else "RAW better or equal to WARPED"
    }

    results_path = visualizer.output_dir / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Resultados] Guardados en: {results_path}")
    print("\n" + "="*80)
    print("EXPERIMENTO COMPLETADO")
    print("="*80)
    print(f"\nRevisa los resultados en: {visualizer.output_dir}")


def main():
    """
    Punto de entrada principal.
    """
    print("\n" + "="*80)
    print("THESIS VALIDATION: FISHER LINEAR ANALYSIS FOR WARPING")
    print("="*80)
    print("\nVerificando existencia de datasets...")

    # Configuración
    loader = DatasetLoader(
        raw_root="data/dataset/COVID-19_Radiography_Dataset",
        warped_root="outputs/full_warped_dataset",
        image_size=224
    )

    analyzer = FisherPCAAnalyzer(n_components=10)
    visualizer = Visualizer(output_dir="./results")

    # Ejecutar experimento
    run_experiment(loader, analyzer, visualizer, use_train_for_test=False)


if __name__ == "__main__":
    main()
