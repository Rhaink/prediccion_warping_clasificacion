"""
Script: visualize_pca_2d_space.py
Propósito: Visualizar la separación de clases en el espacio PCA (PC1 vs PC2)
Input:
    - Imágenes originales (data/dataset/COVID-19_Radiography_Dataset/)
    - Imágenes warped (outputs/full_warped_dataset/)
    - CSVs de splits (train/val/test images.csv)
Output:
    - results/figures/pca_explained/pca_2d_scatter_full_warped.png
    - results/figures/pca_explained/pca_2d_scatter_full_original.png
    - results/figures/pca_explained/pca_2d_scatter_comparison.png

Descripción:
    Usa la MISMA metodología que thesis_validation_fisher.py:
    - Entrena PCA en TRAIN completo (11,364 imágenes)
    - Visualiza separación en TEST (1,518 imágenes)
    - Compara original vs warped usando exactamente las mismas imágenes
    - Usa CLAHE como en el resto del proyecto
    - Binario: Normal=0, Enfermo (COVID+Viral)=1
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
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
ORIGINAL_DIR = PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset"
WARPED_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"

# Output paths
OUTPUT_DIR = FEATURE_DIR / "results" / "figures" / "pca_explained"
OUTPUT_WARPED = OUTPUT_DIR / "pca_2d_scatter_full_warped.png"
OUTPUT_ORIGINAL = OUTPUT_DIR / "pca_2d_scatter_full_original.png"
OUTPUT_COMPARISON = OUTPUT_DIR / "pca_2d_scatter_comparison.png"

# Configuración
IMAGE_SIZE = 224  # Tamaño estándar del proyecto
N_COMPONENTS = 2  # Solo PC1 y PC2 para visualización
USE_CLAHE = True  # Usar CLAHE como en thesis_validation_fisher.py


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_dataset_from_csv(warped_dir: Path, original_dir: Path, split: str, use_clahe: bool = True) -> tuple:
    """
    Carga dataset usando la MISMA metodología que thesis_validation_fisher.py:
    - Lee el CSV del split correspondiente
    - Carga las imágenes warped especificadas en el CSV
    - Carga las mismas imágenes en versión original
    - Aplica CLAHE si se especifica

    Args:
        warped_dir: Directorio de imágenes warped (outputs/full_warped_dataset)
        original_dir: Directorio de imágenes originales (data/dataset/COVID-19_Radiography_Dataset)
        split: Split a cargar ('train', 'val', o 'test')
        use_clahe: Si aplicar CLAHE (default: True, como en el proyecto)

    Returns:
        (images_original, images_warped, labels, image_names)
    """
    # Leer CSV del split
    csv_path = warped_dir / split / "images.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}")

    df = pd.read_csv(csv_path)
    N = len(df)

    print(f"\n[LOADER] Cargando {N} imágenes del split '{split}'...")

    # Preparar CLAHE si se usa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) if use_clahe else None

    images_original = []
    images_warped = []
    labels = []
    image_names = []

    loaded_count = 0
    missing_count = 0

    for idx, row in df.iterrows():
        name = row['image_name']
        category = row['category']
        warped_filename = row.get('warped_filename', f"{name}_warped.png")

        # Cargar imagen WARPED
        warped_path = warped_dir / split / category / warped_filename
        if not warped_path.exists():
            missing_count += 1
            continue

        img_warped = cv2.imread(str(warped_path), cv2.IMREAD_GRAYSCALE)
        if img_warped is None:
            missing_count += 1
            continue

        # Aplicar CLAHE a warped si se especifica
        if clahe is not None:
            img_warped = clahe.apply(img_warped)

        # Cargar imagen ORIGINAL
        # Mapear "Viral_Pneumonia" -> "Viral Pneumonia" para carpeta
        original_category = category.replace("_", " ")
        original_path = original_dir / original_category / "images" / f"{name}.png"

        if not original_path.exists():
            missing_count += 1
            continue

        img_original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            missing_count += 1
            continue

        # Redimensionar original a 224x224 (warped ya está en ese tamaño)
        img_original = cv2.resize(img_original, (IMAGE_SIZE, IMAGE_SIZE))

        # Aplicar CLAHE a original si se especifica
        if clahe is not None:
            img_original = clahe.apply(img_original)

        # Agregar a las listas
        images_original.append(img_original.flatten())
        images_warped.append(img_warped.flatten())

        # Label binario: 0=Normal, 1=Enfermo (COVID o Viral Pneumonia)
        label = 0 if category == "Normal" else 1
        labels.append(label)
        image_names.append(name)

        loaded_count += 1

        # Mostrar progreso cada 1000 imágenes
        if loaded_count % 1000 == 0:
            print(f"   Cargadas: {loaded_count}/{N}...")

    # Convertir a arrays numpy
    images_original = np.array(images_original, dtype=np.float32)
    images_warped = np.array(images_warped, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"   ✓ Cargadas: {loaded_count} imágenes")
    print(f"   ✓ Faltantes: {missing_count}")
    print(f"   ✓ Normal: {(labels == 0).sum()}, Enfermo: {(labels == 1).sum()}")
    print(f"   ✓ CLAHE: {'Activado' if use_clahe else 'Desactivado'}")

    # Verificación crítica
    assert len(images_original) == len(images_warped), \
        f"ERROR: Diferentes números de imágenes ({len(images_original)} vs {len(images_warped)})"

    return images_original, images_warped, labels, image_names


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
    """
    Función principal que ejecuta el pipeline de visualización.

    Metodología (consistente con thesis_validation_fisher.py):
    1. Cargar TRAIN completo (original y warped)
    2. Entrenar PCA en TRAIN
    3. Cargar TEST completo (original y warped)
    4. Proyectar TEST con los PCA entrenados
    5. Visualizar separación de clases en TEST
    """
    print("=" * 80)
    print("VISUALIZACIÓN DEL ESPACIO PCA (PC1 vs PC2)")
    print("=" * 80)
    print("\nMetodología (consistente con thesis_validation_fisher.py):")
    print("  - Entrena PCA en TRAIN (11,364 imágenes)")
    print("  - Visualiza separación en TEST (1,518 imágenes)")
    print("  - CLAHE activado")
    print("  - Binario: Normal=0, Enfermo (COVID+Viral)=1")
    print("=" * 80)

    # 1. Cargar TRAIN para entrenar PCA
    print(f"\n[1/6] Cargando TRAIN para entrenar PCA...")
    train_orig, train_warped, y_train, _ = load_dataset_from_csv(
        WARPED_DIR,
        ORIGINAL_DIR,
        split="train",
        use_clahe=USE_CLAHE
    )

    # 2. Entrenar PCA en TRAIN
    print(f"\n[2/6] Entrenando PCA en TRAIN (Original)...")
    pca_orig, _ = compute_pca(train_orig, N_COMPONENTS)

    print(f"\n[3/6] Entrenando PCA en TRAIN (Warped)...")
    pca_warped, _ = compute_pca(train_warped, N_COMPONENTS)

    # Liberar memoria
    del train_orig, train_warped, y_train

    # 3. Cargar TEST para visualización
    print(f"\n[4/6] Cargando TEST para visualización...")
    test_orig, test_warped, y_test, names_test = load_dataset_from_csv(
        WARPED_DIR,
        ORIGINAL_DIR,
        split="test",
        use_clahe=USE_CLAHE
    )

    # 4. Proyectar TEST con PCA entrenado
    print(f"\n[5/6] Proyectando TEST con PCA entrenado...")
    print("   Proyectando originales...")
    proj_orig = pca_orig.transform(test_orig)
    print("   Proyectando warped...")
    proj_warped = pca_warped.transform(test_warped)

    # 5. Crear visualizaciones
    print(f"\n[6/6] Generando visualizaciones...")

    print("   (a) Scatter plot - Originales...")
    create_scatter_plot(
        proj_orig,
        y_test,
        "Espacio PCA - Imágenes Originales (TEST Set)",
        OUTPUT_ORIGINAL,
        pca_orig
    )

    print("   (b) Scatter plot - Warped...")
    create_scatter_plot(
        proj_warped,
        y_test,
        "Espacio PCA - Imágenes Warped (TEST Set)",
        OUTPUT_WARPED,
        pca_warped
    )

    print("   (c) Comparación lado a lado...")
    create_comparison_plot(
        proj_orig,
        proj_warped,
        y_test,
        pca_orig,
        pca_warped,
        OUTPUT_COMPARISON
    )

    # 6. Resumen final
    print("\n" + "=" * 80)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"   - Split visualizado: TEST")
    print(f"   - Total imágenes: {len(test_orig)}")
    print(f"   - Normal: {(y_test == 0).sum()}")
    print(f"   - Enfermo (COVID + Viral Pneumonia): {(y_test == 1).sum()}")
    print(f"\n✅ VALIDACIÓN METODOLÓGICA:")
    print(f"   - PCA entrenado en TRAIN (11,364 imágenes)")
    print(f"   - Visualización en TEST (1,518 imágenes)")
    print(f"   - CLAHE activado (como en thesis_validation_fisher.py)")
    print(f"   - Mismas imágenes comparadas (original vs warped)")
    print(f"   - Labels binarios consistentes con el proyecto")
    print(f"\n📁 Outputs generados:")
    print(f"   1. {OUTPUT_ORIGINAL}")
    print(f"   2. {OUTPUT_WARPED}")
    print(f"   3. {OUTPUT_COMPARISON}")
    print(f"\n📈 Los gráficos muestran:")
    print(f"   - Scatter plots de PC1 vs PC2 en TEST")
    print(f"   - Elipses de confianza (95%) para cada clase")
    print(f"   - Comparación de separabilidad (original vs warped)")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
