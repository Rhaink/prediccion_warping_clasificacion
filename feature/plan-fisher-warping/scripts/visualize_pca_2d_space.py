"""
Script: visualize_pca_2d_space.py
Propósito: Visualizar la separación de clases en el espacio PCA (PC1 vs PC2)
Input:
    - Imágenes originales (data/dataset/)
    - Imágenes warped (outputs/full_warped_dataset/)
Output:
    - results/figures/pca_explained/pca_2d_scatter_full_warped.png
    - results/figures/pca_explained/pca_2d_scatter_full_original.png
    - results/figures/pca_explained/pca_2d_scatter_comparison.png

Descripción:
    Calcula PCA sobre las imágenes del dataset y visualiza la proyección en PC1 vs PC2.
    Compara la separación de clases entre imágenes originales y warped.
    Agrega elipses de confianza (95%) para mostrar la distribución de cada clase.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas base
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FEATURE_DIR = PROJECT_ROOT / "feature" / "plan-fisher-warping"

# Input paths
ORIGINAL_DIR = PROJECT_ROOT / "data" / "dataset"
WARPED_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"

# Output paths
OUTPUT_DIR = FEATURE_DIR / "results" / "figures" / "pca_explained"
OUTPUT_WARPED = OUTPUT_DIR / "pca_2d_scatter_full_warped.png"
OUTPUT_ORIGINAL = OUTPUT_DIR / "pca_2d_scatter_full_original.png"
OUTPUT_COMPARISON = OUTPUT_DIR / "pca_2d_scatter_comparison.png"

# Configuración
IMAGE_SIZE = 224  # Tamaño para redimensionar
N_COMPONENTS = 2  # Solo PC1 y PC2
MAX_IMAGES_PER_CLASS = 300  # Limitar para performance


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_dataset_images(base_dir: Path, is_warped: bool = False) -> tuple:
    """
    Carga imágenes del dataset.

    Args:
        base_dir: Directorio base (data/dataset o outputs/full_warped_dataset)
        is_warped: True si son imágenes warped

    Returns:
        (images, labels, image_names)
    """
    images = []
    labels = []
    image_names = []

    # Clases a cargar (usamos solo binario: enfermos vs normales para claridad)
    if is_warped:
        splits = ['train', 'val', 'test']
        categories = ['COVID', 'Normal', 'Viral_Pneumonia']

        for split in splits:
            for category in categories:
                cat_dir = base_dir / split / category
                if not cat_dir.exists():
                    continue

                # Leer solo las primeras MAX_IMAGES_PER_CLASS
                image_files = sorted(cat_dir.glob("*_warped.png"))[:MAX_IMAGES_PER_CLASS]

                for img_path in image_files:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    # Las imágenes warped ya están en 224x224
                    images.append(img.flatten())

                    # Label binario: 0=Normal, 1=Enfermo (COVID o Viral Pneumonia)
                    label = 0 if category == "Normal" else 1
                    labels.append(label)
                    image_names.append(img_path.stem)

    else:
        categories = ['COVID', 'Normal', 'Viral_Pneumonia']

        for category in categories:
            cat_dir = base_dir / category
            if not cat_dir.exists():
                continue

            # Leer solo las primeras MAX_IMAGES_PER_CLASS
            image_files = sorted(cat_dir.glob("*.png"))[:MAX_IMAGES_PER_CLASS]

            for img_path in image_files:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Redimensionar a 224x224
                img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                images.append(img_resized.flatten())

                # Label binario: 0=Normal, 1=Enfermo
                label = 0 if category == "Normal" else 1
                labels.append(label)
                image_names.append(img_path.stem)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return images, labels, image_names


def compute_pca(images: np.ndarray, n_components: int = 2) -> tuple:
    """
    Calcula PCA sobre las imágenes.

    Args:
        images: Array de imágenes aplanadas (n_samples, n_features)
        n_components: Número de componentes principales

    Returns:
        (pca_model, projections)
    """
    print(f"   Calculando PCA con {n_components} componentes...")

    # Normalizar (centrar los datos)
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(images)

    print(f"   ✓ Varianza explicada PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"   ✓ Varianza explicada PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"   ✓ Varianza total: {pca.explained_variance_ratio_.sum():.2%}")

    return pca, projections


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Dibuja una elipse de confianza para los datos x, y.

    Args:
        x, y: Datos
        ax: Axes de matplotlib
        n_std: Número de desviaciones estándar (2.0 ≈ 95%)
        facecolor: Color de relleno
        **kwargs: Argumentos adicionales para Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Eigenvalores y eigenvectores
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Crear elipse
    ellipse = Ellipse((0, 0),
                     width=ell_radius_x * 2,
                     height=ell_radius_y * 2,
                     facecolor=facecolor,
                     **kwargs)

    # Calcular transformación
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def create_scatter_plot(
    projections: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    pca_model: PCA
):
    """
    Crea scatter plot de PC1 vs PC2 con elipses de confianza.

    Args:
        projections: Proyecciones PCA (n_samples, 2)
        labels: Labels binarios (0=Normal, 1=Enfermo)
        title: Título del gráfico
        output_path: Path donde guardar
        pca_model: Modelo PCA para obtener varianza explicada
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colores científicos
    colors = {0: '#1f77b4', 1: '#d62728'}  # Azul para Normal, Rojo para Enfermo
    labels_str = {0: 'Normal', 1: 'Enfermo (COVID + Viral Pneumonia)'}

    # Scatter plot para cada clase
    for label in [0, 1]:
        mask = labels == label
        pc1 = projections[mask, 0]
        pc2 = projections[mask, 1]

        # Puntos
        ax.scatter(pc1, pc2,
                  c=colors[label],
                  label=labels_str[label],
                  alpha=0.6,
                  s=20,
                  edgecolors='white',
                  linewidth=0.5)

        # Elipse de confianza (95%)
        confidence_ellipse(pc1, pc2, ax,
                          n_std=2.0,  # 2 std ≈ 95%
                          edgecolor=colors[label],
                          linewidth=2,
                          linestyle='--',
                          alpha=0.5)

    # Etiquetas de ejes con varianza explicada
    var1 = pca_model.explained_variance_ratio_[0]
    var2 = pca_model.explained_variance_ratio_[1]

    ax.set_xlabel(f'PC1 ({var1:.1%} varianza)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var2:.1%} varianza)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Leyenda
    ax.legend(loc='best', framealpha=0.95, fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Ajustar layout
    plt.tight_layout()

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"   ✓ Gráfico guardado: {output_path.name}")

    plt.close(fig)


def create_comparison_plot(
    proj_original: np.ndarray,
    proj_warped: np.ndarray,
    labels: np.ndarray,
    pca_original: PCA,
    pca_warped: PCA,
    output_path: Path
):
    """
    Crea comparación lado a lado de Original vs Warped.

    Args:
        proj_original: Proyecciones originales
        proj_warped: Proyecciones warped
        labels: Labels
        pca_original: Modelo PCA original
        pca_warped: Modelo PCA warped
        output_path: Path donde guardar
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {0: '#1f77b4', 1: '#d62728'}
    labels_str = {0: 'Normal', 1: 'Enfermo'}

    # Panel 1: Original
    ax = axes[0]
    for label in [0, 1]:
        mask = labels == label
        pc1 = proj_original[mask, 0]
        pc2 = proj_original[mask, 1]

        ax.scatter(pc1, pc2, c=colors[label], label=labels_str[label],
                  alpha=0.6, s=20, edgecolors='white', linewidth=0.5)

        confidence_ellipse(pc1, pc2, ax, n_std=2.0,
                          edgecolor=colors[label], linewidth=2,
                          linestyle='--', alpha=0.5)

    var1 = pca_original.explained_variance_ratio_[0]
    var2 = pca_original.explained_variance_ratio_[1]
    ax.set_xlabel(f'PC1 ({var1:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({var2:.1%})', fontsize=11)
    ax.set_title('(a) Imágenes Originales', fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 2: Warped
    ax = axes[1]
    for label in [0, 1]:
        mask = labels == label
        pc1 = proj_warped[mask, 0]
        pc2 = proj_warped[mask, 1]

        ax.scatter(pc1, pc2, c=colors[label], label=labels_str[label],
                  alpha=0.6, s=20, edgecolors='white', linewidth=0.5)

        confidence_ellipse(pc1, pc2, ax, n_std=2.0,
                          edgecolor=colors[label], linewidth=2,
                          linestyle='--', alpha=0.5)

    var1 = pca_warped.explained_variance_ratio_[0]
    var2 = pca_warped.explained_variance_ratio_[1]
    ax.set_xlabel(f'PC1 ({var1:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({var2:.1%})', fontsize=11)
    ax.set_title('(b) Imágenes Warped (Normalizadas)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Ajustar layout
    plt.tight_layout()

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"   ✓ Comparación guardada: {output_path.name}")

    plt.close(fig)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta el pipeline de visualización."""
    print("=" * 80)
    print("VISUALIZACIÓN DEL ESPACIO PCA (PC1 vs PC2)")
    print("=" * 80)

    # 1. Cargar imágenes originales
    print(f"\n1. Cargando imágenes originales...")
    images_orig, labels_orig, names_orig = load_dataset_images(ORIGINAL_DIR, is_warped=False)
    print(f"   ✓ {len(images_orig)} imágenes cargadas")
    print(f"   ✓ Normal: {(labels_orig == 0).sum()}, Enfermo: {(labels_orig == 1).sum()}")

    # 2. Cargar imágenes warped
    print(f"\n2. Cargando imágenes warped...")
    images_warped, labels_warped, names_warped = load_dataset_images(WARPED_DIR, is_warped=True)
    print(f"   ✓ {len(images_warped)} imágenes cargadas")
    print(f"   ✓ Normal: {(labels_warped == 0).sum()}, Enfermo: {(labels_warped == 1).sum()}")

    # 3. Calcular PCA para imágenes originales
    print(f"\n3. Calculando PCA para imágenes originales...")
    pca_orig, proj_orig = compute_pca(images_orig, N_COMPONENTS)

    # 4. Calcular PCA para imágenes warped
    print(f"\n4. Calculando PCA para imágenes warped...")
    pca_warped, proj_warped = compute_pca(images_warped, N_COMPONENTS)

    # 5. Crear scatter plot para originales
    print(f"\n5. Generando scatter plot para imágenes originales...")
    create_scatter_plot(
        proj_orig,
        labels_orig,
        "Espacio PCA - Imágenes Originales",
        OUTPUT_ORIGINAL,
        pca_orig
    )

    # 6. Crear scatter plot para warped
    print(f"\n6. Generando scatter plot para imágenes warped...")
    create_scatter_plot(
        proj_warped,
        labels_warped,
        "Espacio PCA - Imágenes Warped (Normalizadas Geométricamente)",
        OUTPUT_WARPED,
        pca_warped
    )

    # 7. Crear comparación lado a lado
    print(f"\n7. Generando comparación lado a lado...")
    # Usar subset común para comparación justa
    min_samples = min(len(labels_orig), len(labels_warped))
    create_comparison_plot(
        proj_orig[:min_samples],
        proj_warped[:min_samples],
        labels_orig[:min_samples],
        pca_orig,
        pca_warped,
        OUTPUT_COMPARISON
    )

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\nOutputs generados:")
    print(f"  1. {OUTPUT_ORIGINAL}")
    print(f"  2. {OUTPUT_WARPED}")
    print(f"  3. {OUTPUT_COMPARISON}")
    print(f"\nLos gráficos muestran:")
    print(f"  - Scatter plots de PC1 vs PC2")
    print(f"  - Elipses de confianza (95%) para cada clase")
    print(f"  - Comparación visual de separabilidad")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
