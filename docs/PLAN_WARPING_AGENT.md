# Plan detallado: dejar warping listo (para agente de IA)

Objetivo: preparar la parte de **warping** para que sea reproducible desde cero,
usando el **ensemble best 3.61 px**, y dejar dataset warpeado listo para la fase
de clasificación. Este plan termina **después** de generar el dataset warpeado,
antes de entrenar clasificadores.

## Contexto y definiciones (evitar confusiones)
- **warped_47**: warping solo área pulmonar (18 triángulos, sin full-coverage).
  Fill rate ~47%. Este es el dataset histórico en `outputs/warped_dataset`.
- **warped_96**: warping con full-coverage + grayscale + CLAHE.
  Fill rate ~96%. Es el recomendado por balance de accuracy y robustez.
- **warped_99**: warping con full-coverage + RGB+CLAHE (LAB).
  Fill rate ~99%. Es legacy y menos robusto.

## Qué existe hoy (verificación rápida)
- `outputs/warped_dataset/`:
  - `dataset_summary.json` muestra fill_rate ~0.47.
  - `warping_config.json` indica **Ground Truth landmarks** (no modelo).
  - Por tanto **NO** usa ensemble.
- `outputs/full_warped_dataset/`:
  - `dataset_summary.json` también tiene fill_rate ~0.47.
  - Fue generado por `scripts/generate_full_warped_dataset.py`, que usa
    `scripts/predict.EnsemblePredictor` con **solo 2 modelos** (seed123/seed456).
  - **No** usa el ensemble 4-model (3.71) ni el best actual (3.61).

## Qué significa TTA en warping
TTA solo afecta **la predicción de landmarks**. Se hace una predicción normal
y otra con flip horizontal; luego se corrige el flip y se promedian las
coordenadas. Beneficio: landmarks más estables (menos ruido) -> warping más
consistente. Costo: ~2x tiempo de inferencia.

El comando actual `generate-dataset` **no usa TTA**, así que si se quiere TTA,
hay que implementarlo en el pipeline de warping.

## Plan de acción (paso a paso, sin ambigüedades)

### Paso 0: Confirmar data y best ensemble
1) Dataset original debe existir en:
   - `data/COVID-19_Radiography_Dataset/`
   - Clases esperadas: `COVID`, `Normal`, `Viral Pneumonia`
2) Ensemble best actual en config:
   - `configs/ensemble_best.json` (best 3.61 px)

### Paso 1: Habilitar ensemble + TTA en generate-dataset
Modificar `src_v2/cli.py` en el comando `generate-dataset`:
1) Agregar opciones:
   - `--ensemble-config` (JSON con lista de modelos)
   - `--ensemble-checkpoints` (lista explícita de checkpoints)
   - `--tta/--no-tta` (default: true o false, documentar)
2) Validar:
   - Si `--ensemble-config` o `--ensemble-checkpoints` están presentes,
     entonces `--checkpoint` ya no es requerido.
3) Implementar predicción de landmarks:
   - Cargar todos los modelos.
   - Si `--tta`:
     - Predicción normal + flip horizontal.
     - Corregir flip y pares simétricos (igual que evaluate-ensemble).
   - Promediar predicciones de todos los modelos.
4) Registrar metadata en `dataset_summary.json`:
   - `models` usados (lista)
   - `tta` (true/false)
   - `clahe` (true/false)
   - `clahe_clip`, `clahe_tile`
   - `margin`, `use_full_coverage`, `seed`, `splits`

### Paso 2: Crear config para warping best
Crear `configs/warping_best.json` con:
- `input_dir`: `data/COVID-19_Radiography_Dataset`
- `output_dir`: `outputs/warped_96_best/sessionXX`
- `ensemble_config`: `configs/ensemble_best.json`
- `canonical`: `outputs/shape_analysis/canonical_shape_gpa.json`
- `triangles`: `outputs/shape_analysis/canonical_delaunay_triangles.json`
- `margin`: 1.05
- `splits`: `0.75,0.125,0.125`
- `seed`: 42
- `clahe`: true
- `clahe_clip`: 2.0
- `clahe_tile`: 4
- `use_full_coverage`: true
- `tta`: true (si se decide usar)

### Paso 3: Crear quickstart de warping
Crear `scripts/quickstart_warping.sh`:
1) Verificar/crear canonical shape:
   ```bash
   python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
     --output-dir outputs/shape_analysis --visualize
   ```
2) Ejecutar generate-dataset con `configs/warping_best.json`:
   ```bash
   python -m src_v2 generate-dataset \
     --ensemble-config configs/ensemble_best.json \
     --input-dir data/COVID-19_Radiography_Dataset \
     --output-dir outputs/warped_96_best/sessionXX \
     --canonical outputs/shape_analysis/canonical_shape_gpa.json \
     --triangles outputs/shape_analysis/canonical_delaunay_triangles.json \
     --margin 1.05 \
     --splits 0.75,0.125,0.125 \
     --seed 42 \
     --clahe --clahe-clip 2.0 --clahe-tile 4 \
     --use-full-coverage \
     --tta
   ```

### Paso 4: Verificación posterior (obligatoria)
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
