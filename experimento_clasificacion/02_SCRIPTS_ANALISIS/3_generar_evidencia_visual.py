#!/usr/bin/env python3
"""
PROYECTO: Validación Geométrica de Warping Pulmonar
SCRIPT: 3_generar_evidencia_visual.py
OBJETIVO: Generar láminas comparativas RAW vs WARPED para inspección cualitativa.

Este script satisface la solicitud del asesor:
ASESOR: "...que por cierto a mí me gustaría ver esas imágenes, ¿verdad?"
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Configuración
BASE_DIR = Path(__file__).parent.parent / "01_DATOS_ENTRADA"
OUTPUT_DIR = Path(__file__).parent.parent / "03_EVIDENCIA_RESULTADOS" / "comparativa_visual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
WARPED_DIR = BASE_DIR / "full_warped_dataset"
RAW_DIR = BASE_DIR / "COVID-19_Radiography_Dataset"

def encontrar_raw(filename, category):
    """Busca la imagen original correspondiente."""
    # Mapeo de categorías Warped -> Raw
    cat_map = {
        'Normal': 'Normal',
        'COVID': 'COVID',
        'Viral Pneumonia': 'Viral Pneumonia'
    }
    raw_cat = cat_map.get(category, category)
    
    candidates = [
        RAW_DIR / raw_cat / "images" / filename,
        RAW_DIR / raw_cat / filename,
        RAW_DIR / filename
    ]
    for p in candidates:
        if p.exists(): return p
    return None

def generar_comparativa(n_muestras=20):
    print(f"Generando {n_muestras} láminas comparativas...")
    
    # Cargar lista de imágenes de TEST
    df = pd.read_csv(WARPED_DIR / "test" / "images.csv")
    
    # Seleccionar aleatoriamente
    muestras = df.sample(n=n_muestras, random_state=42)
    
    for idx, row in muestras.iterrows():
        name = row['image_name']
        cat = row['category']
        w_name = row.get('warped_filename', f"{name}.png")
        
        # 1. Cargar Warped
        path_warped = WARPED_DIR / "test" / cat / w_name
        if not path_warped.exists(): continue
        img_warped = cv2.imread(str(path_warped), cv2.IMREAD_GRAYSCALE)
        img_warped = cv2.resize(img_warped, (224, 224))
        
        # 2. Cargar Raw
        path_raw = encontrar_raw(f"{name}.png", cat)
        if not path_raw: continue
        img_raw = cv2.imread(str(path_raw), cv2.IMREAD_GRAYSCALE)
        img_raw = cv2.resize(img_raw, (224, 224))
        
        # 3. Calcular SSIM (Diferencia estructural)
        score, diff = ssim(img_raw, img_warped, full=True)
        
        # 4. Plotear
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Raw
        axes[0].imshow(img_raw, cmap='gray')
        axes[0].set_title(f"RAW (Original)\n{cat}", fontsize=10)
        axes[0].axis('off')
        
        # Warped
        axes[1].imshow(img_warped, cmap='gray')
        axes[1].set_title(f"WARPED (Alineada)\nSSIM: {score:.3f}", fontsize=10)
        axes[1].axis('off')
        
        # Diferencia (Heatmap)
        im = axes[2].imshow(diff, cmap='inferno')
        axes[2].set_title("Mapa de Diferencia\n(Cambio Geométrico)", fontsize=10)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"Paciente: {name}", fontsize=14)
        plt.tight_layout()
        
        save_path = OUTPUT_DIR / f"comparativa_{cat}_{name}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  -> Guardada: {save_path.name}")

if __name__ == "__main__":
    generar_comparativa()
