# Prompt de Continuacion - Sesion 18

## Contexto del Proyecto

Este es un proyecto de tesis de maestria sobre clasificacion de COVID-19 mediante normalizacion geometrica (warping) usando landmarks anatomicos.

### Hipotesis Principal (YA DEMOSTRADA)
> "La normalizacion geometrica mediante landmarks mejora la generalizacion y robustez DENTRO de un dominio, aunque NO resuelve domain shift entre datasets diferentes."

### Resultados Clave Establecidos
- **Cross-evaluation**: Warped generaliza **11x mejor** (gap 2.24% vs 25.36%)
- **Robustez**: **30x mejor** a JPEG, **3x mejor** a blur
- **Validacion externa**: ~55% ambos en FedCOVIDx (domain shift domina)
- **Modelo recomendado**: DenseNet-121 Warped margin 1.05

## Estado Actual del CLI (9 comandos)

### Implementados
1. `train` - Entrenar modelo de landmarks
2. `evaluate` - Evaluar modelo individual
3. `predict` - Predecir landmarks en imagen
4. `warp` - Aplicar warping a dataset
5. `version` - Version del paquete
6. `evaluate-ensemble` - Evaluar ensemble de modelos
7. `classify` - Clasificar imagenes (pipeline E2E)
8. `train-classifier` - Entrenar clasificador CNN
9. `evaluate-classifier` - Evaluar clasificador

### NO Implementados (Prioridad)

#### Alta Prioridad
1. **`cross-evaluate`** - Evaluar modelo en dominio diferente al entrenado
   - Original→Warped, Warped→Original
   - Demuestra la hipotesis principal (11x mejor)
   - Script original: `session30_cross_evaluation.py`

2. **`evaluate-external`** - Validacion en dataset externo
   - Mapeo de clases 3→2
   - Script original: `evaluate_external_baseline.py`

#### Media Prioridad
3. **`test-robustness`** - Pruebas de perturbaciones (JPEG, blur, ruido)
4. **`gradcam`** - Visualizacion de atencion del modelo
5. **Soporte DenseNet-121** en train-classifier

## Sesion 17 Completada

- Pipeline E2E con EfficientNet-B0: **93.78% accuracy**
- Consistente con experimentos originales (Session 15: 95.45%)
- Degradacion ~4% es esperada (landmarks predichos vs GT)
- Documentacion actualizada en `docs/sesiones/SESION_17_PIPELINE_E2E.md`

## Documentacion Creada

1. `docs/sesiones/SESION_17_PIPELINE_E2E.md` - Resultados pipeline E2E
2. `docs/REFERENCIA_EXPERIMENTOS_ORIGINALES.md` - Guia completa de experimentos originales vs CLI

## Archivos Clave

### Scripts Originales a Adaptar
- `scripts/session30_cross_evaluation.py` - Cross-evaluation
- `scripts/evaluate_external_baseline.py` - Validacion externa
- `scripts/test_robustness_artifacts.py` - Robustez

### Modelos Disponibles
- Landmarks: `checkpoints/session10/ensemble/seed*/` y `checkpoints/session13/seed*/`
- Clasificadores: `outputs/classifier_full/`, `outputs/classifier_efficientnet/`
- Multi-arq: `outputs/classifier_comparison/`

### Datasets
- Original: `data/dataset/COVID-19_Radiography_Dataset/` (3 clases, NO Lung_Opacity)
- Warped: `outputs/full_warped_dataset/`
- Externo: `outputs/external_validation/dataset3/`

## Tareas Pendientes para Sesion 18

1. [ ] Implementar comando `cross-evaluate`
2. [ ] Implementar comando `evaluate-external`
3. [ ] Agregar soporte DenseNet-121 a train-classifier
4. [ ] Verificar que CLI reproduce resultados de Session 30
5. [ ] Opcional: comando `test-robustness`

## Advertencias

1. **NO usar Lung_Opacity** - Es la 4ta clase del dataset, no entrenada
2. **NO inventar experimentos** - CLI debe reproducir experimentos originales
3. **Degradacion ~4% en E2E es NORMAL** - Documentada en Session 15
4. **Domain shift es el limite** - Warping no resuelve cross-dataset

## Comando para Iniciar

```bash
cd /home/donrobot/Projects/prediccion_warping_clasificacion
source .venv/bin/activate
```

---

## PROMPT PARA CONTINUAR

```
Continuacion de la Sesion 17 del proyecto COVID-19 Landmark Detection.

CONTEXTO:
- CLI tiene 9 comandos implementados (train, evaluate, predict, warp, classify, etc.)
- Session 17 completada: Pipeline E2E con EfficientNet logra 93.78% accuracy
- Documentacion actualizada en docs/sesiones/SESION_17_PIPELINE_E2E.md
- Referencia completa en docs/REFERENCIA_EXPERIMENTOS_ORIGINALES.md

OBJETIVO SESION 18:
Implementar funcionalidades faltantes en CLI para reproducir experimentos originales:

1. PRIORIDAD ALTA:
   - Comando `cross-evaluate`: Evaluar modelo en dominio diferente
     * Original→Warped y Warped→Original
     * Debe reproducir resultados de Session 30 (11x mejor generalizacion)
     * Script referencia: scripts/session30_cross_evaluation.py

   - Comando `evaluate-external`: Validacion en dataset externo (FedCOVIDx)
     * Mapeo de clases 3→2 (COVID vs Non-COVID)
     * Script referencia: scripts/evaluate_external_baseline.py

2. PRIORIDAD MEDIA:
   - Agregar DenseNet-121 como backbone en train-classifier
   - Comando test-robustness (JPEG, blur, ruido)

RESTRICCIONES:
- NO inventar experimentos nuevos
- NO usar Lung_Opacity (4ta clase no entrenada)
- Mantener consistencia con resultados originales
- Seguir patron de codigo existente en src_v2/cli.py

ARCHIVOS CLAVE:
- src_v2/cli.py (comandos actuales)
- scripts/session30_cross_evaluation.py (referencia cross-eval)
- scripts/evaluate_external_baseline.py (referencia external)
- docs/REFERENCIA_EXPERIMENTOS_ORIGINALES.md (documentacion completa)

Tests: 224 tests pasando actualmente
```
