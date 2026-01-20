"""Tests de exportación a PDF."""

import pytest
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np
import pandas as pd


def test_export_to_pdf():
    """Verifica que export_to_pdf genera archivo válido."""
    from src_v2.gui.visualizer import export_to_pdf

    # Crear datos dummy
    original = Image.new('L', (224, 224))
    landmarks_img = Image.new('RGB', (224, 224))
    warped = Image.new('L', (224, 224))
    gradcam = Image.new('RGB', (224, 224))

    # Crear diccionario de imágenes
    images = {
        'original': original,
        'landmarks': landmarks_img,
        'warped': warped,
        'gradcam': gradcam
    }

    metrics_df = pd.DataFrame({
        'Landmark': [f'L{i}' for i in range(1, 16)],
        'X': np.random.uniform(0, 224, 15),
        'Y': np.random.uniform(0, 224, 15)
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_result.pdf"

        # Metadata con clasificación
        metadata = {
            'classification': {'COVID-19': 0.8, 'Normal': 0.15, 'Neumonía Viral': 0.05},
            'predicted_class': 'COVID-19'
        }

        # La función no retorna nada, solo crea el PDF
        export_to_pdf(
            images,
            metrics_df,
            str(output_path),
            metadata
        )

        # Verificar que el PDF se creó
        assert output_path.exists(), "PDF no fue creado"
        assert output_path.stat().st_size > 1000, f"PDF muy pequeño: {output_path.stat().st_size} bytes"
