# Plan de generacion de figuras para Capitulo 4 (Secciones 4.1 a 4.5)

## 1. Resumen del sistema y programas clave (contexto tecnico)
- Flujo global: imagen RX -> preprocesamiento (CLAHE + resize 224) -> modelo de landmarks (ResNet-18 + Coordinate Attention + head con GroupNorm) -> forma canonica por GPA -> warping piecewise affine -> clasificador CNN.
- Datos y transforms: `src_v2/data/dataset.py` y `src_v2/data/transforms.py` (CLAHE, flip con pares simetricos, rotacion, normalizacion ImageNet).
- Modelo landmarks: `src_v2/models/resnet_landmark.py` (backbone ResNet-18, Coordinate Attention, head profunda con GroupNorm).
- Forma canonica y triangulacion: `src_v2/processing/gpa.py` + CLI `python -m src_v2 compute-canonical`.
- Warping: `src_v2/processing/warp.py` + CLI `python -m src_v2 generate-dataset`.
- Clasificador y augmentations: `src_v2/models/classifier.py` (`get_classifier_transforms`).
- Scripts de referencia para figuras: `scripts/visualization/` (bloques de presentacion), `scripts/visualize_gpa_methodology*.py`, `scripts/verify_canonical_delaunay.py`, `scripts/predict_landmarks_dataset.py`.

## 2. Entradas y artefactos necesarios
- Dataset original (no versionado): `data/dataset/<COVID|Normal|Viral_Pneumonia>/`.
- Coordenadas GT: `data/coordenadas/coordenadas_maestro.csv`.
- Checkpoints ensemble landmarks (segun `configs/ensemble_best.json`).
- Forma canonica y triangulos:
  - `outputs/shape_analysis/canonical_shape_gpa.json`
  - `outputs/shape_analysis/canonical_delaunay_triangles.json`
  - `outputs/shape_analysis/aligned_shapes.npz`
- Cache landmarks (opcional para speed): `outputs/landmark_predictions/.../predictions.npz`.
- Warped dataset: `outputs/warped_lung_best/session_warping/` (train/val/test).
- Carpeta final de figuras: `docs/Tesis/Figures/`.

## 3. Preparacion base (una vez)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
1) Generar landmarks agregados:
```bash
python scripts/generate_all_landmarks_npz.py \
  --csv data/coordenadas/coordenadas_maestro.csv \
  --output outputs/predictions/all_landmarks.npz
```
2) Calcular forma canonica y triangulos:
```bash
python -m src_v2 compute-canonical \
  data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize
```
3) Cache de landmarks (si no existe):
```bash
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/session_warping/predictions.npz \
  --ensemble-config configs/ensemble_best.json \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4
```
4) Dataset warped:
```bash
python -m src_v2 generate-dataset --config configs/warping_best.json
```

## 4. Convencion de salida de figuras
- Usar `docs/Tesis/Figures/` con nombres estables:
  - `f4_1_fases_sistema.png`, `f4_2_flujo_operacion.png`, `f4_2a_landmarks.png`,
    `f4_2b_herramienta_etiquetado.png`, `f4_3_clahe.png`, `f4_4_arquitectura.png`,
    `f4_5_wing_loss.png`, `f4_6_gpa_proceso.png`, `f4_7_delaunay.png`,
    `f4_8_warping_comparacion.png`, `f4_9_margin_scale.png`,
    `f4_10_pipeline_normalizacion.png`, `f4_11_augmentation_clasificador.png`.
- Exportar a PNG 300 dpi (o PDF si es vectorial). Mantener aspecto 4:3 o 16:9 segun figura.

## 5. Plan por figura (scripts y pasos)

### F4.1 Diagrama de fases del sistema (fig:fases_sistema)
- Script recomendado: nuevo `scripts/thesis_figures/fig_f4_1_fases.py`.
- Logica: diagrama con dos bandas (Preparacion offline y Operacion runtime) usando `matplotlib.patches.FancyBboxPatch`.
- Contenido:
  - Preparacion: anotacion manual -> entrenamiento -> GPA.
  - Operacion: imagen nueva -> preproc -> landmarks -> warping -> clasificacion.
- Salida: `docs/Tesis/Figures/f4_1_fases_sistema.png`.

### F4.2 Diagrama de bloques del flujo de operacion (fig:flujo_general)
- Script base: `scripts/visualization/generate_pipeline_visualizations.py` (adaptar).
- Rehacer con cajas y flechas, agregando dimensiones:
  - 224x224x3 -> 15x2 -> 224x224x3 -> 3.
- Salida: `docs/Tesis/Figures/f4_2_flujo_operacion.png`.

### F4.2a Landmarks anatomicos (fig:landmarks_anatomicos)
- Script base: `scripts/visualization/generate_bloque1_assets.py` (funcion `create_slide4_landmarks_anatomia`).
- Ajustes:
  - Seleccionar una imagen limpia (Normal) con `coordenadas_maestro.csv`.
  - Dibujar puntos L1-L15 con colores por grupo y conexiones desde `scripts/landmark_connections.py`.
- Salida: `docs/Tesis/Figures/f4_2a_landmarks.png`.

### F4.2b Interfaz herramienta de etiquetado (fig:herramienta_etiquetado)
- Si existe la herramienta real: abrirla con un ejemplo y capturar pantalla.
- Si no existe: recrear una "captura" estatica con un script nuevo:
  - Cargar una RX, dibujar linea central azul, puntos verdes numerados, lineas rojas del contorno.
  - Agregar bloque de texto con atajos (WASD o teclas reales).
- Salida: `docs/Tesis/Figures/f4_2b_herramienta_etiquetado.png`.

### F4.3 Comparacion CLAHE (fig:clahe_comparison)
- Script base: `scripts/visualization/generate_bloque3_preprocesamiento.py` (funcion `apply_clahe`).
- Pasos:
  - Elegir una imagen con bajo contraste.
  - Panel 1: original, Panel 2: CLAHE clip=2.0 tile=4.
- Salida: `docs/Tesis/Figures/f4_3_clahe.png`.

### F4.4 Arquitectura ResNet-18 + Coordinate Attention (fig:arquitectura_modelo)
- Script base: `scripts/visualization/generate_architecture_diagrams.py` (funcion `generate_model_architecture_diagram`).
- Ajustar texto para coincidir con la descripcion de 4.3 (entrada 224x224x3, CA, head, salida 30).
- Salida: `docs/Tesis/Figures/f4_4_arquitectura.png`.

### F4.5 Wing Loss comparacion (fig:wing_loss)
- Script base: `scripts/visualization/generate_bloque4_arquitectura.py` (asset `wing_loss_comparacion.png`)
  o `scripts/visualization/generate_detailed_diagrams.py` (wing loss).
- Generar grafica Wing vs L1 vs L2 en rango [-25, 25], con cambio en w=10 px.
- Salida: `docs/Tesis/Figures/f4_5_wing_loss.png`.

### F4.6 Proceso GPA (fig:gpa_proceso)
- Script base: `scripts/visualize_gpa_methodology_fixed.py` (produce paneles paso a paso).
- Requerimientos:
  - `outputs/predictions/all_landmarks.npz`
  - `outputs/shape_analysis/aligned_shapes.npz`
- Extraer/combinar 4 paneles: (a) original, (b) centrado+escalado, (c) rotado, (d) canonica.
- Salida: `docs/Tesis/Figures/f4_6_gpa_proceso.png`.

### F4.7 Triangulacion de Delaunay (fig:triangulacion_delaunay)
- Script base: `scripts/verify_canonical_delaunay.py` o nuevo script minimal.
- Cargar `canonical_shape_gpa.json` y trazar triangulos sobre landmarks.
- Salida: `docs/Tesis/Figures/f4_7_delaunay.png`.

### F4.8 Original vs warped (sin vs con full coverage) (fig:warping_comparison)
- Nuevo script recomendado: `scripts/thesis_figures/fig_f4_8_warping.py`.
- Pasos:
  - Elegir una imagen del dataset original.
  - Obtener landmarks predichos (usar cache o ensemble con TTA).
  - Warpear con `piecewise_affine_warp(..., use_full_coverage=False)` y luego `True`.
  - Calcular `compute_fill_rate` y anotar en subfiguras.
- Salida: `docs/Tesis/Figures/f4_8_warping_comparacion.png`.

### F4.9 Efecto margin_scale (fig:margin_scale_effect)
- Nuevo script recomendado: `scripts/thesis_figures/fig_f4_9_margin.py`.
- Pasos:
  - Usar la misma imagen/landmarks de F4.8.
  - Aplicar `scale_landmarks_from_centroid` con 1.00, 1.05, 1.25.
  - Warpear con `use_full_coverage=True`.
- Salida: `docs/Tesis/Figures/f4_9_margin_scale.png`.

### F4.10 Pipeline completo de normalizacion (fig:proceso_normalizacion)
- Script nuevo: `scripts/thesis_figures/fig_f4_10_pipeline_norm.py`.
- Diagrama de flujo: prediccion -> margin_scale -> puntos borde -> Delaunay -> warp -> imagen normalizada.
- Incluir dimensiones y notas (224x224, 15 o 23 puntos).
- Salida: `docs/Tesis/Figures/f4_10_pipeline_normalizacion.png`.

### F4.11 Data augmentation del clasificador (fig:augmentation_clasificador)
- Script nuevo: `scripts/thesis_figures/fig_f4_11_augmentation.py`.
- Usar `src_v2/models/classifier.get_classifier_transforms(train=True)`:
  - `GrayscaleToRGB`, resize 224, flip, rotacion, affine, normalizacion.
- Para reproducibilidad: fijar `torch.manual_seed(42)` y aplicar transforms manuales o
  usar `torchvision.transforms.functional` con parametros fijos.
- Imagen base: una RX ya warpeada de `outputs/warped_lung_best/session_warping/train/Normal/`.
- Salida: `docs/Tesis/Figures/f4_11_augmentation_clasificador.png`.

## 6. Checklist de QA antes de insertar en LaTeX
- Verificar que todas las figuras estan en `docs/Tesis/Figures/` y con DPI >= 300.
- Confirmar que las figuras usan la misma imagen base cuando se comparan (F4.8/F4.9/F4.11).
- Validar que los parametros mostrados coinciden con `constants.py` y `configs/*.json`.
- Actualizar `docs/Tesis/capitulo4/*.tex` con `\\includegraphics` reales.

## 7. Sugerencia de estructura de scripts nuevos
- `scripts/thesis_figures/fig_f4_1_fases.py`
- `scripts/thesis_figures/fig_f4_2_flujo.py`
- `scripts/thesis_figures/fig_f4_8_warping.py`
- `scripts/thesis_figures/fig_f4_9_margin.py`
- `scripts/thesis_figures/fig_f4_10_pipeline_norm.py`
- `scripts/thesis_figures/fig_f4_11_augmentation.py`

Cada script debe aceptar `--input`, `--output`, y registrar el seed usado.
