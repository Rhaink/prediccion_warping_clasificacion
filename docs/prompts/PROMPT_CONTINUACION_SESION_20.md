# Prompt para Sesion 20 - Implementacion de Comandos CLI Faltantes

Copia y pega este prompt para continuar:

---

## CONTEXTO DE SESION 20

Continuacion del proyecto COVID-19 Landmark Detection.

**Estado actual:**
- Session 19 completada: Validacion de comandos CLI exitosa
- Analisis de gaps realizado: 5 comandos criticos faltantes
- CLI tiene 12 comandos funcionales, 244 tests pasando
- Documentacion en:
  - `docs/ANALISIS_GAPS_CLI.md` - Gaps identificados
  - `docs/PLAN_IMPLEMENTACION_CLI_v2.md` - Plan de implementacion
  - `docs/sesiones/SESION_19_VALIDACION_CLI.md` - Resultados de validacion

**Comandos CLI actuales (12):**
1. train, evaluate, predict, warp, evaluate-ensemble (landmarks)
2. classify, train-classifier, evaluate-classifier (clasificacion)
3. cross-evaluate, evaluate-external, test-robustness (investigacion)
4. version

**Gaps criticos identificados:**
1. `generate-dataset` - Crear datasets warped completos con splits
2. `compute-canonical` - Calcular forma canonica con GPA
3. Arquitecturas faltantes: AlexNet, ResNet-50, MobileNetV2, VGG-16

## OBJETIVO SESION 20

Implementar los dos comandos CLI mas criticos para completar el pipeline:

### TAREA 1: Comando `generate-dataset`

**Proposito:** Generar dataset warped completo con train/val/test splits

**Uso esperado:**
```bash
python -m src_v2 generate-dataset \
    --input data/COVID-19_Radiography_Dataset \
    --output outputs/new_warped_dataset \
    --checkpoint checkpoints_v2/final_model.pt \
    --margin 1.05 \
    --splits 0.75,0.125,0.125 \
    --seed 42
```

**Funcionalidades requeridas:**
- Cargar imagenes de dataset original (estructura COVID/Normal/Viral_Pneumonia)
- Predecir landmarks con modelo o ensemble
- Aplicar warping con margen configurable
- Crear splits train/val/test manteniendo distribucion de clases
- Guardar metadata (landmarks.json, images.csv, dataset_summary.json)
- Progreso con tqdm
- Soporte opcional para TTA en prediccion de landmarks
- Soporte opcional para ensemble de modelos

**Referencia:** `scripts/generate_full_warped_dataset.py`

### TAREA 2: Comando `compute-canonical`

**Proposito:** Calcular forma canonica de landmarks usando GPA

**Uso esperado:**
```bash
python -m src_v2 compute-canonical \
    --landmarks-csv data/coordenadas/coordenadas_maestro.csv \
    --output-dir outputs/shape_analysis \
    --visualize
```

**Funcionalidades requeridas:**
- Cargar coordenadas de landmarks desde CSV
- Implementar Generalized Procrustes Analysis (GPA)
- Calcular forma canonica (mean shape alineada)
- Generar triangulacion de Delaunay
- Guardar canonical_shape_gpa.json
- Guardar canonical_delaunay_triangles.json
- Visualizacion opcional de forma canonica

**Referencia:** `scripts/gpa_analysis.py`

### TAREA 3 (Si hay tiempo): Agregar Arquitecturas

**Archivo:** `src_v2/models/classifier.py`

Agregar soporte para:
- AlexNet
- ResNet-50
- MobileNetV2
- VGG-16

### ARCHIVOS CLAVE:

- `src_v2/cli.py` - Agregar nuevos comandos (~linea 3155+)
- `src_v2/models/classifier.py` - Agregar arquitecturas
- `scripts/generate_full_warped_dataset.py` - Referencia para generate-dataset
- `scripts/gpa_analysis.py` - Referencia para compute-canonical
- `outputs/shape_analysis/` - Archivos canonicos existentes como ejemplo

### RESTRICCIONES:

- Mantener consistencia con estilo de comandos existentes
- Agregar tests para nuevos comandos
- Documentar parametros con docstrings
- Usar logging consistente con resto del CLI
- No romper funcionalidad existente (244 tests deben seguir pasando)

### ENTREGABLES:

1. Comando `generate-dataset` funcional
2. Comando `compute-canonical` funcional
3. Tests para ambos comandos
4. Arquitecturas adicionales (si hay tiempo)
5. Documentacion de Session 20

### DATOS DISPONIBLES:

- Dataset original: `data/dataset/COVID-19_Radiography_Dataset/`
- Coordenadas: `data/coordenadas/coordenadas_maestro.csv`
- Modelo de landmarks: `checkpoints_v2/final_model.pt`
- Forma canonica existente: `outputs/shape_analysis/canonical_shape_gpa.json`
- Triangulacion existente: `outputs/shape_analysis/canonical_delaunay_triangles.json`

---

**Usa ultrathink para disenar la arquitectura de los comandos antes de implementar.**
