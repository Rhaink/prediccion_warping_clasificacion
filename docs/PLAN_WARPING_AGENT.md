# Plan detallado: dejar warping listo (para agente de IA)

**Nota**: Este plan ya fue implementado. El flujo actual esta en
`docs/QUICKSTART_WARPING.md` y `docs/REPRO_FULL_PIPELINE.md`.

Objetivo: preparar la parte de **warping** para que sea reproducible desde cero,
usando el **ensemble best 3.61 px**, y dejar dataset warpeado listo para la fase
de clasificación. Este plan termina **después** de generar el dataset warpeado,
antes de entrenar clasificadores.

## Contexto (solo lo necesario)
- **warped_96**: warping con full-coverage + grayscale + CLAHE.
  Fill rate ~96%. Es el recomendado por balance de accuracy y robustez.

## Qué existe hoy (verificación rápida)
- `outputs/warped_dataset/`:
  - `dataset_summary.json` muestra fill_rate ~0.47.
  - `warping_config.json` indica **Ground Truth landmarks** (no modelo).
- `outputs/full_warped_dataset/`:
  - `dataset_summary.json` también tiene fill_rate ~0.47.
  - Fue generado con un ensemble simplificado (2 modelos), sin TTA.

Estos datasets **no** usan el ensemble best actual (3.61).

## Qué significa TTA en warping
TTA solo afecta **la predicción de landmarks**. Se hace una predicción normal
y otra con flip horizontal; luego se corrige el flip y se promedian las
coordenadas. Beneficio: landmarks más estables (menos ruido) -> warping más
consistente. Costo: ~2x tiempo de inferencia.

Para este plan, **siempre usar TTA** y el ensemble best (3.61).

## Plan de acción (paso a paso, sin ambigüedades)

### Paso 0: Confirmar data y best ensemble
1) Dataset original debe existir en:
   - `data/COVID-19_Radiography_Dataset/`
   - Clases esperadas: `COVID`, `Normal`, `Viral Pneumonia`
2) Ensemble best actual en config:
   - `configs/ensemble_best.json` (best 3.61 px)

### Paso 1: Predecir landmarks una sola vez (cache)
Motivo: el dataset `data/COVID-19_Radiography_Dataset` no tiene GT. Si se
reprocesa cada vez, se repite el costo de inferencia. Solucion: predecir
landmarks **una sola vez** con el ensemble best + TTA y guardar resultados.

Requisitos del cache:
- Un archivo por dataset con:
  - `image_path` o `image_name`
  - `category`
  - `landmarks` en escala 224 (pixeles)
  - metadata: modelos, TTA, CLAHE, seed, timestamp

Implementacion sugerida:
- Agregar un script tipo `scripts/predict_landmarks_dataset.py`
  que recorra el dataset y guarde un JSON/NPZ con predicciones.

### Paso 2: Warping desde predicciones guardadas
Modificar o extender `generate-dataset` (CLI `python -m src_v2 generate-dataset`)
para aceptar:
- `--predictions` (archivo JSON/NPZ con landmarks)
  - Nota: esta opcion **no existe aun**; debe implementarse.

Si `--predictions` esta presente:
- No correr inferencia.
- Usar los landmarks guardados para cada imagen.
- Loggear en `dataset_summary.json` la ruta del archivo de predicciones.

### Paso 3: Crear config para warping best (requiere soporte de config)
Crear `configs/warping_best.json` con:
- `input_dir`: `data/COVID-19_Radiography_Dataset`
- `output_dir`: `outputs/warped_96_best/sessionXX`
- `predictions`: `outputs/landmark_predictions/sessionXX/predictions.npz`
- `canonical`: `outputs/shape_analysis/canonical_shape_gpa.json`
- `triangles`: `outputs/shape_analysis/canonical_delaunay_triangles.json`
- `margin`: 1.05
- `splits`: `0.75,0.125,0.125`
- `seed`: 42
- `clahe`: true
- `clahe_clip`: 2.0
- `clahe_tile`: 4
- `use_full_coverage`: true
- `tta`: true

Nota: `generate-dataset` **no soporta configs** hoy; el agente debe agregar
`--config` o una utilidad equivalente.

### Paso 4: Crear quickstart de warping
Crear `scripts/quickstart_warping.sh`:
1) Verificar/crear canonical shape (una sola vez):
   ```bash
   python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
     --output-dir outputs/shape_analysis --visualize
   ```
2) Predecir landmarks para todo el dataset (cache):
   ```bash
   python scripts/predict_landmarks_dataset.py \
     --input-dir data/COVID-19_Radiography_Dataset \
     --output outputs/landmark_predictions/sessionXX/predictions.npz \
     --ensemble-config configs/ensemble_best.json \
     --tta --clahe --clahe-clip 2.0 --clahe-tile 4
   ```
3) Ejecutar generate-dataset con `configs/warping_best.json` (una vez agregado soporte de config):
   ```bash
   python -m src_v2 generate-dataset \
     data/COVID-19_Radiography_Dataset \
     outputs/warped_96_best/sessionXX \
     --canonical outputs/shape_analysis/canonical_shape_gpa.json \
     --triangles outputs/shape_analysis/canonical_delaunay_triangles.json \
     --margin 1.05 \
     --splits 0.75,0.125,0.125 \
     --seed 42 \
     --clahe --clahe-clip 2.0 --clahe-tile 4 \
     --use-full-coverage \
     --predictions outputs/landmark_predictions/sessionXX/predictions.npz
   ```

### Paso 5: Verificación posterior (obligatoria)
1) Revisar `outputs/warped_96_best/sessionXX/dataset_summary.json`
2) Validar:
   - `fill_rate_mean` ~0.96
   - conteos por split y clase
3) Guardar logs:
   - `outputs/warping_quickstart.log`
4) Actualizar documentación:
   - `docs/EXPERIMENTS.md`
   - `docs/README.md`
   - `scripts/README.md`
   - `README.md`
   - `GROUND_TRUTH.json` (si se quiere fijar fill rate oficial)
   - `CHANGELOG.md`

## Resultado esperado
Un dataset warpeado reproducible (warped_96) generado con el **ensemble best 3.61**,
con metadata clara y comandos únicos para repetirlo en futuras sesiones.
