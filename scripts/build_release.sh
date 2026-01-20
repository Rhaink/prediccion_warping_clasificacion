#!/bin/bash
set -e

VERSION="1.0.0"
RELEASE_NAME="covid-demo-v${VERSION}"
BUILD_DIR="build/${RELEASE_NAME}"

echo "Building release ${RELEASE_NAME}..."

# Clean previous build
rm -rf build/
mkdir -p "$BUILD_DIR"

# Copy source code
echo "Copying source code..."
cp -r src_v2/ "$BUILD_DIR/"
cp -r configs/ "$BUILD_DIR/"
mkdir -p "$BUILD_DIR/scripts"
cp scripts/run_demo.py "$BUILD_DIR/scripts/"

# Copy examples if they exist
if [ -d "examples" ]; then
    echo "Copying examples..."
    cp -r examples/ "$BUILD_DIR/"
fi

# Copy documentation
mkdir -p "$BUILD_DIR/docs"
if [ -f "docs/REPRO_FULL_PIPELINE.md" ]; then
    cp docs/REPRO_FULL_PIPELINE.md "$BUILD_DIR/docs/"
fi

# Copy models (flatten structure)
echo "Copying models..."
mkdir -p "$BUILD_DIR/models/landmarks"
mkdir -p "$BUILD_DIR/models/classifier"
mkdir -p "$BUILD_DIR/models/shape_analysis"

# Check if models exist and copy them
if [ -f "checkpoints/session10/ensemble/seed123/final_model.pt" ]; then
    cp checkpoints/session10/ensemble/seed123/final_model.pt \
       "$BUILD_DIR/models/landmarks/seed123_final.pt"
else
    echo "Warning: checkpoints/session10/ensemble/seed123/final_model.pt not found"
fi

if [ -f "checkpoints/session13/seed321/final_model.pt" ]; then
    cp checkpoints/session13/seed321/final_model.pt \
       "$BUILD_DIR/models/landmarks/seed321_final.pt"
else
    echo "Warning: checkpoints/session13/seed321/final_model.pt not found"
fi

if [ -f "checkpoints/repro_split111/session14/seed111/final_model.pt" ]; then
    cp checkpoints/repro_split111/session14/seed111/final_model.pt \
       "$BUILD_DIR/models/landmarks/seed111_final.pt"
else
    echo "Warning: checkpoints/repro_split111/session14/seed111/final_model.pt not found"
fi

if [ -f "checkpoints/repro_split666/session16/seed666/final_model.pt" ]; then
    cp checkpoints/repro_split666/session16/seed666/final_model.pt \
       "$BUILD_DIR/models/landmarks/seed666_final.pt"
else
    echo "Warning: checkpoints/repro_split666/session16/seed666/final_model.pt not found"
fi

if [ -f "outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt" ]; then
    cp outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt \
       "$BUILD_DIR/models/classifier/best_classifier.pt"
else
    echo "Warning: outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt not found"
fi

if [ -f "outputs/shape_analysis/canonical_shape_gpa.json" ]; then
    cp outputs/shape_analysis/canonical_shape_gpa.json \
       "$BUILD_DIR/models/shape_analysis/"
else
    echo "Warning: outputs/shape_analysis/canonical_shape_gpa.json not found"
fi

if [ -f "outputs/shape_analysis/canonical_delaunay_triangles.json" ]; then
    cp outputs/shape_analysis/canonical_delaunay_triangles.json \
       "$BUILD_DIR/models/shape_analysis/"
else
    echo "Warning: outputs/shape_analysis/canonical_delaunay_triangles.json not found"
fi

# Copy installers
echo "Copying installer scripts..."
cp install.sh "$BUILD_DIR/"
cp install.bat "$BUILD_DIR/"
cp run_demo.sh "$BUILD_DIR/"
cp run_demo.bat "$BUILD_DIR/"

# Make scripts executable
chmod +x "$BUILD_DIR/install.sh"
chmod +x "$BUILD_DIR/run_demo.sh"

# Copy metadata
cp requirements.txt "$BUILD_DIR/"
cp pyproject.toml "$BUILD_DIR/" 2>/dev/null || echo "pyproject.toml not found, skipping"
cp GROUND_TRUTH.json "$BUILD_DIR/" 2>/dev/null || echo "GROUND_TRUTH.json not found, skipping"

# Create LICENSE if it doesn't exist
if [ -f "LICENSE" ]; then
    cp LICENSE "$BUILD_DIR/"
else
    echo "MIT License" > "$BUILD_DIR/LICENSE"
    echo "Warning: LICENSE file created with placeholder text"
fi

# Create README for end users
cat > "$BUILD_DIR/README.md" << 'EOFREADME'
# COVID-19 Detection Demo - Thesis Defense Version

Sistema de detección de COVID-19 mediante landmarks anatómicos y normalización geométrica.

## Resultados Validados
- **Error de Landmarks**: 3.61 ± 2.48 px (ensemble 4 modelos)
- **Accuracy de Clasificación**: 98.05%
- **F1-Score**: 97.12% (macro), 98.04% (weighted)

## Requisitos del Sistema

- **Sistema Operativo**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.9 o superior
- **RAM**: 4 GB mínimo, 8 GB recomendado
- **GPU**: Opcional (NVIDIA con CUDA para inferencia rápida)
- **Espacio en Disco**: 2 GB (incluye PyTorch)

## Instalación Rápida

### Linux / macOS

```bash
# 1. Extraer el paquete
unzip covid-demo-v1.0.0.zip
cd covid-demo-v1.0.0

# 2. Ejecutar instalador
bash install.sh

# 3. Lanzar demo
bash run_demo.sh
```

### Windows

```batch
REM 1. Extraer el paquete (doble click en .zip)
REM 2. Abrir carpeta en terminal
REM 3. Ejecutar instalador
install.bat

REM 4. Lanzar demo
run_demo.bat
```

La interfaz se abrirá automáticamente en http://localhost:7860

## Uso de la Interfaz

1. **Cargar Imagen**: Arrastra una radiografía de tórax o usa ejemplos precargados
2. **Procesar**: Click en "Procesar Imagen"
3. **Resultados**: Visualiza las 4 etapas del pipeline
   - Imagen Original
   - Landmarks Detectados (15 puntos)
   - Imagen Normalizada (Warped)
   - GradCAM (Explicabilidad)

## Troubleshooting

**Error: Python no encontrado**
```bash
# Linux/Mac
sudo apt install python3.9  # Ubuntu/Debian
brew install python@3.9     # macOS

# Windows: Descargar de https://www.python.org/downloads/
```

**Error: GPU sin memoria**
- La interfaz automáticamente usará CPU
- Tiempo de inferencia: ~1-2 segundos en CPU vs. ~0.5s en GPU

**Puerto 7860 ocupado**
```bash
run_demo.sh --port 8080
```

## Arquitectura

```
Input Image (224×224)
    ↓
Landmark Detection (Ensemble 4× ResNet-18 + Coordinate Attention)
    ↓ (15 landmarks)
Piecewise Affine Warping (Delaunay triangulation, 18 triángulos)
    ↓ (Normalized image)
Classification (ResNet-18) + GradCAM
    ↓
Diagnosis: COVID-19 / Normal / Viral Pneumonia
```

## Versión

v1.0.0 - Enero 2026

## Licencia

MIT License

## Contacto

Para más información, consulte la documentación incluida en `docs/`.

EOFREADME

# Create checksum
echo "Creating checksums..."
cd build/
find "$RELEASE_NAME" -type f -exec sha256sum {} \; > "${RELEASE_NAME}_checksums.txt"

# Create archive
echo "Creating archive..."
zip -r "${RELEASE_NAME}.zip" "$RELEASE_NAME" -q

echo ""
echo "=========================================="
echo "✓ Release built successfully!"
echo "=========================================="
echo ""
echo "Archive: build/${RELEASE_NAME}.zip"
echo "Size: $(du -h build/${RELEASE_NAME}.zip | cut -f1)"
echo ""
echo "To test:"
echo "  cd build"
echo "  unzip ${RELEASE_NAME}.zip"
echo "  cd ${RELEASE_NAME}"
echo "  bash install.sh"
echo "  bash run_demo.sh"
echo ""
