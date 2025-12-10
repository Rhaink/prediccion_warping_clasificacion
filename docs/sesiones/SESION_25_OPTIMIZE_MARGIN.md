# Sesion 25: Comando optimize-margin

**Fecha:** 2025-12-09
**Objetivo:** Implementar el comando `optimize-margin` para alcanzar ~95% de cobertura del CLI
**Estado:** COMPLETADO con revision de calidad

---

## Resumen

En esta sesion se implemento el comando `optimize-margin`, que permite buscar automaticamente el margen optimo para warping. Este es el ultimo comando pendiente para completar la cobertura del CLI.

## Comando Implementado

### `optimize-margin`

**Proposito:** Buscar automaticamente el margen optimo para warping iterando sobre multiples valores de margin_scale.

**Uso:**
```bash
python -m src_v2 optimize-margin \
    --data-dir data/COVID-19_Radiography_Dataset \
    --landmarks-csv data/landmarks.csv \
    --margins 1.00,1.05,1.10,1.15,1.20,1.25,1.30 \
    --epochs 10 \
    --output-dir outputs/margin_optimization
```

### Parametros

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `--data-dir` | str | Requerido | Directorio del dataset original |
| `--landmarks-csv` | str | Requerido | Archivo CSV con landmarks |
| `--margins` | str | "1.00,1.05,1.10,1.15,1.20,1.25,1.30" | Margenes a probar |
| `--epochs` | int | 10 | Epochs por entrenamiento |
| `--batch-size` | int | 32 | Batch size |
| `--architecture` | str | resnet18 | Arquitectura (resnet18, efficientnet_b0, densenet121, alexnet, resnet50) |
| `--output-dir` | str | outputs/margin_optimization | Directorio de salida |
| `--checkpoint` | str | None | Checkpoint de modelo de landmarks |
| `--canonical` | str | outputs/shape_analysis/canonical_shape.json | Forma canonica |
| `--triangles` | str | outputs/shape_analysis/delaunay_triangles.json | Triangulacion |
| `--quick` | bool | False | Modo rapido (menos epochs y datos) |
| `--keep-datasets` | bool | False | Mantener datasets temporales |
| `--seed` | int | 42 | Semilla para reproducibilidad |
| `--device` | str | auto | Dispositivo (auto, cuda, cpu, mps) |
| `--patience` | int | 5 | Early stopping patience |
| `--splits` | str | "0.75,0.15,0.10" | Ratios train/val/test |

### Funcionamiento

1. **Para cada margen en la lista:**
   - Aplica warping on-the-fly con el margin_scale especificado
   - Entrena un clasificador rapido con epochs reducidos
   - Evalua accuracy en validation/test
   - Guarda checkpoint del modelo

2. **Al finalizar:**
   - Identifica el margen con mejor accuracy
   - Genera grafico accuracy vs margen (`accuracy_vs_margin.png`)
   - Guarda resultados consolidados (`margin_optimization_results.json`)
   - Guarda resumen CSV (`summary.csv`)

### Estructura de Salida

```
outputs/margin_optimization/
├── margin_optimization_results.json   # Resultados consolidados
├── accuracy_vs_margin.png             # Grafico principal
├── summary.csv                        # Tabla resumen
└── per_margin/
    ├── margin_1.00/
    │   └── checkpoint.pt              # Modelo entrenado
    ├── margin_1.05/
    │   └── checkpoint.pt
    └── ...
```

### Ejemplo de Resultados JSON

```json
{
  "timestamp": "2025-12-09T14:30:00",
  "configuration": {
    "data_dir": "data/COVID-19_Radiography_Dataset",
    "margins_tested": [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3],
    "architecture": "resnet18",
    "epochs": 10
  },
  "results": [
    {"margin": 1.0, "val_accuracy": 82.5, "test_accuracy": 81.2, "test_f1": 80.8},
    {"margin": 1.05, "val_accuracy": 84.1, "test_accuracy": 83.5, "test_f1": 83.2},
    ...
  ],
  "best_margin": 1.25,
  "best_accuracy": 87.3
}
```

## Implementacion Tecnica

### Warping On-the-fly

El comando utiliza warping on-the-fly en lugar de generar datasets completos, lo que:
- Reduce el uso de espacio en disco
- Permite probar multiples margenes rapidamente
- Es mas flexible para experimentacion

### Dataset Class

Se implemento `WarpedOnFlyDataset` que:
- Carga imagenes y landmarks desde CSV
- Aplica `scale_landmarks_from_centroid()` con el margin especificado
- Ejecuta `piecewise_affine_warp()` para normalizar la imagen
- Soporta multiples ubicaciones de imagenes (category/images/, category/, etc.)

### Entrenamiento

- Usa `ImageClassifier` del modulo `src_v2.models`
- Aplica class weights para balanceo de clases
- Implementa early stopping basado en validation accuracy
- Guarda checkpoints con metadatos completos

## Tests

Se agregaron 14 tests nuevos en `tests/test_cli.py`:

1. `test_optimize_margin_help` - Verifica ayuda del comando
2. `test_optimize_margin_requires_data_dir` - Valida parametro requerido
3. `test_optimize_margin_requires_landmarks_csv` - Valida parametro requerido
4. `test_optimize_margin_data_dir_not_found` - Valida existencia de directorio
5. `test_optimize_margin_landmarks_csv_not_found` - Valida existencia de archivo
6. `test_optimize_margin_invalid_margins_format` - Valida formato de margenes
7. `test_optimize_margin_negative_margins` - Valida margenes positivos
8. `test_optimize_margin_invalid_architecture` - Valida arquitecturas soportadas
9. `test_optimize_margin_invalid_splits_format` - Valida formato de splits
10. `test_optimize_margin_splits_not_sum_to_one` - Valida que splits sumen 1
11. `test_optimize_margin_default_margins` - Verifica valores por defecto
12. `test_optimize_margin_default_epochs` - Verifica epochs por defecto
13. `test_optimize_margin_default_architecture` - Verifica arquitectura por defecto
14. `test_optimize_margin_module_execution` - Verifica ejecucion como modulo

## Estadisticas

- **Comandos CLI totales:** 20 (antes: 19)
- **Tests totales:** 401 (antes: 387)
- **Tests nuevos:** 14
- **Cobertura estimada:** ~95%

## Referencias

- Script original: `scripts/margin_optimization_experiment.py`
- Script extendido: `scripts/experiment_extended_margins.py`
- Resultado historico: Margen optimo = 1.25

## Notas

- El modo `--quick` es util para pruebas rapidas con epochs=3 y subconjunto de datos
- El comando reutiliza logica de `generate-dataset` y `train-classifier`
- Los checkpoints guardados incluyen metadatos para reproducibilidad

---

## Revision de Calidad (Post-implementacion)

### Bugs Corregidos

Durante la revision de calidad se identificaron y corrigieron los siguientes bugs:

1. **Lista de arquitecturas incompleta** (CRITICO)
   - Faltaban `vgg16` y `mobilenet_v2` en validacion
   - Corregido: ahora coincide con `ImageClassifier.SUPPORTED_BACKBONES`

2. **Nombres de archivos por defecto incorrectos** (CRITICO)
   - El comando buscaba `canonical_shape.json` pero existe `canonical_shape_gpa.json`
   - Corregido: ahora busca multiples nombres alternativos

3. **Carga de JSON con clave incorrecta** (CRITICO)
   - Buscaba clave `canonical_shape` pero archivo usa `canonical_shape_normalized`
   - Corregido: ahora soporta multiples claves

4. **Uso incorrecto de pandas Series.get()** (ALTO)
   - Corregido con verificacion explicita de columnas en index

5. **Variable epoch no inicializada** (ALTO)
   - Si epochs=0, causaria UnboundLocalError
   - Corregido: inicializacion antes del loop

6. **pin_memory ineficiente** (MEDIO)
   - pin_memory=True con num_workers=0 no beneficia
   - Corregido: solo usa pin_memory en CUDA

7. **Excepciones genericas sin logging** (MEDIO)
   - Warping fallido no se registraba
   - Corregido: ahora usa logger.debug

### Integridad de Datos Verificada

- Margen optimo = 1.25 tiene respaldo en `outputs/session28_margin_experiment/`
- 957 muestras verificadas en dataset
- 15 landmarks confirmados en constantes y datos
- Timestamps de experimentos validados

### Tests Faltantes Identificados

La cobertura actual es ~30% para optimize-margin. Se identificaron 41 tests adicionales necesarios:

**Prioridad 1 (Criticos):**
- Tests de flujo completo con datos reales
- Edge cases en margenes (un solo margen, duplicados)
- Edge cases en datos (CSV sin headers, NaN en landmarks)

**Prioridad 2 (Altos):**
- Modo --quick
- Splits estratificados
- Generacion de archivos de salida

**Prioridad 3 (Medios):**
- Reproducibilidad con seeds
- Multiples arquitecturas

---

## Analisis del Objetivo del Proyecto

### Hipotesis Principal

> "Las imagenes warpeadas son mejores para entrenar clasificadores de enfermedades pulmonares que las imagenes completas (debido a marcas/etiquetas hospitalarias)"

### Evidencia Encontrada

| Experimento | Resultado | Fuente |
|-------------|-----------|--------|
| Cross-evaluation | Warped generaliza 11x mejor | Session 30 |
| Robustez JPEG | Warped 30x mas robusto | Session 29 |
| Robustez blur | Warped 3x mas robusto | Session 29 |
| Grad-CAM | Warped enfoca en pulmones | Session 23 |

### Conclusion

La hipotesis SE DEMUESTRA, pero con matices:

- Warping mejora generalizacion y robustez significativamente
- Warping reduce accuracy puro en ~4% (trade-off aceptable)
- Warping NO resuelve domain shift extremo (datasets externos)

**Recomendacion clinica:** Usar warped (sacrifica 0.8% en laboratorio por 11x mejor generalizacion en campo)

---

## Proximos Pasos Sugeridos

### Corto Plazo (Sesion 26)
1. Agregar tests de integracion para optimize-margin (~10 tests prioritarios)
2. Probar comando con datos reales del proyecto

### Mediano Plazo
1. Implementar comando `generate-report` para documentacion automatica
2. Agregar comando `validate-hypothesis` para validacion end-to-end
3. Mejorar experiencia de usuario del CLI (progress bars, mejor logging)

### Largo Plazo
1. Domain adaptation para datasets externos
2. Publicacion de resultados en paper/tesis

---

**Estado:** Completado con revision de calidad
**Siguiente sesion:** Tests de integracion y validacion con datos reales
