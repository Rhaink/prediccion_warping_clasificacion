import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def visualize_pca_real():
    # 1. Cargar Datos Reales (Subconjunto)
    print("Cargando imágenes reales...")
    dataset_path = Path("outputs/warped_dataset")
    csv_path = dataset_path / "train" / "images.csv"
    
    if not csv_path.exists():
        print(f"No encontré {csv_path}. Verifica la ruta.")
        return

    df = pd.read_csv(csv_path).head(100) # Usamos 100 imágenes
    images = []
    
    for _, row in df.iterrows():
        # Construir ruta
        p = dataset_path / "train" / row['category'] / row.get('warped_filename', row['image_name'])
        if not p.exists(): continue
        
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (128, 128)) # Tamaño manejable
        images.append(img.flatten())
    
    X = np.array(images, dtype=np.float32)
    print(f"Matriz de Datos: {X.shape}")

    # 2. PCA INCORRECTO (Sin Centrar)
    # SVD directo sobre raw pixels
    print("Calculando PCA sin centrar...")
    U, S, Vt_bad = np.linalg.svd(X, full_matrices=False)
    # El primer componente (Eigenface 1)
    eigenface_bad = Vt_bad[0].reshape(128, 128)

    # 3. PCA CORRECTO (Centrado)
    print("Calculando PCA centrado...")
    mean_img = np.mean(X, axis=0)
    X_centered = X - mean_img
    U, S, Vt_good = np.linalg.svd(X_centered, full_matrices=False)
    # El primer componente (Eigenface 1)
    eigenface_good = Vt_good[0].reshape(128, 128)

    # 4. Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Promedio
    axes[0].imshow(mean_img.reshape(128, 128), cmap='gray')
    axes[0].set_title("Imagen Promedio (La Media)")
    axes[0].axis('off')
    
    # Malo
    axes[1].imshow(eigenface_bad, cmap='jet') # Usamos color para ver intensidad
    axes[1].set_title("MAL (Sin Centrar)\nComponente 1 ≈ Promedio")
    axes[1].axis('off')

    # Bueno
    axes[2].imshow(eigenface_good, cmap='jet')
    axes[2].set_title("BIEN (Centrado)\nComponente 1 = Variación (Forma)")
    axes[2].axis('off')

    plt.tight_layout()
    output_path = Path("results/figures/pca_real_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\n✅ IMAGEN GENERADA: {output_path}")
    print("Ábrela para ver la diferencia radical.")

if __name__ == "__main__":
    visualize_pca_real()
