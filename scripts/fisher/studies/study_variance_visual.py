import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def visualize_ghosting_effect():
    print("Comparando nitidez del promedio (RAW vs WARPED)...")
    dataset_path = Path("outputs/full_warped_dataset")
    # Si no existe full, intentamos con warped normal
    if not dataset_path.exists():
        dataset_path = Path("outputs/warped_dataset")

    csv_path = dataset_path / "train" / "images.csv"
    raw_root = Path("data/dataset/COVID-19_Radiography_Dataset") # Ajusta si tu raw está en otro lado
    
    if not csv_path.exists():
        print("No encontré CSV.")
        return

    df = pd.read_csv(csv_path).head(100)
    
    raw_accum = None
    warped_accum = None
    count = 0
    
    for _, row in df.iterrows():
        # Cargar Warped
        w_p = dataset_path / "train" / row['category'] / row.get('warped_filename', row['image_name'])
        if not w_p.exists(): continue
        img_w = cv2.imread(str(w_p), cv2.IMREAD_GRAYSCALE)
        img_w = cv2.resize(img_w, (256, 256))
        
        # Cargar Raw
        # Intentamos adivinar la ruta raw
        r_candidates = [
            raw_root / row['category'] / "images" / f"{row['image_name']}.png",
            raw_root / row['category'] / f"{row['image_name']}.png"
        ]
        img_r = None
        for r_p in r_candidates:
            if r_p.exists():
                img_r = cv2.imread(str(r_p), cv2.IMREAD_GRAYSCALE)
                break
        
        if img_r is None: continue
        img_r = cv2.resize(img_r, (256, 256))
        
        # Acumular
        if raw_accum is None:
            raw_accum = np.zeros_like(img_r, dtype=np.float32)
            warped_accum = np.zeros_like(img_w, dtype=np.float32)
            
        raw_accum += img_r
        warped_accum += img_w
        count += 1

    # Promediar
    mean_raw = raw_accum / count
    mean_warped = warped_accum / count
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(mean_raw, cmap='gray')
    axes[0].set_title(f"Promedio RAW (n={count})\nBorrosa = Alta Entropía")
    axes[0].axis('off')
    
    axes[1].imshow(mean_warped, cmap='gray')
    axes[1].set_title(f"Promedio WARPED (n={count})\nNítida = Baja Entropía\n(PCA funciona mejor aquí)")
    axes[1].axis('off')
    
    plt.tight_layout()
    output_path = Path("results/figures/variance_ghosting_proof.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Evidencia generada: {output_path}")

if __name__ == "__main__":
    visualize_ghosting_effect()
