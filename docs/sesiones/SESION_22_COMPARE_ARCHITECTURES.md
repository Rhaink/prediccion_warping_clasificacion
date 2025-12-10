# Sesion 22: Comando compare-architectures

## Fecha: 2025-12-09

## Objetivo
Implementar el comando `compare-architectures` para automatizar la comparacion sistematica de multiples arquitecturas CNN en el pipeline de clasificacion COVID-19, incluyendo generacion de reportes y visualizaciones.

## Resumen Ejecutivo

| Aspecto | Detalle |
|---------|---------|
| **Comando implementado** | `compare-architectures` |
| **Lineas de codigo** | ~850 lineas nuevas |
| **Tests nuevos** | 15 tests (de 312 a 327) |
| **Bugs corregidos** | 9 bugs (5 criticos/altos, 4 medios/bajos) |
| **Arquitecturas soportadas** | 7 (resnet18, resnet50, efficientnet_b0, densenet121, alexnet, vgg16, mobilenet_v2) |
| **Validacion** | Probado con datos reales (ResNet-18: 83.33% acc, 84% F1) |

## Archivos Modificados

### src_v2/cli.py
- **Lineas agregadas:** 3862-4770 (~910 lineas)
- **Funciones nuevas:**
  - `SUPPORTED_ARCHITECTURES` - Lista de arquitecturas
  - `ARCHITECTURE_DISPLAY_NAMES` - Nombres para visualizacion
  - `_train_single_architecture()` - Entrenar una arquitectura
  - `_generate_comparison_figures()` - Generar graficos
  - `_generate_comparison_reports()` - Generar JSON/CSV
  - `compare_architectures()` - Comando principal

### tests/test_cli.py
- **Tests agregados:** 15 nuevos tests en clase `TestCompareArchitectures`
- **Actualizacion:** Conteo de comandos de 14 a 15

## Comando Implementado

### Uso
```bash
# Comparar todas las arquitecturas
python -m src_v2 compare-architectures outputs/warped_dataset \
    --epochs 30 --seed 42

# Comparar arquitecturas especificas
python -m src_v2 compare-architectures outputs/warped_dataset \
    --architectures resnet18,efficientnet_b0,densenet121

# Modo rapido para pruebas
python -m src_v2 compare-architectures outputs/warped_dataset --quick

# Comparar warped vs original
python -m src_v2 compare-architectures outputs/warped_dataset \
    --original-data-dir data/COVID-19_Radiography_Dataset
```

### Parametros

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `DATA_DIR` | Path | Requerido | Dataset warped con train/val/test |
| `--output-dir` | Path | `outputs/arch_comparison` | Directorio de salida |
| `--architectures` | str | Todas | Lista separada por coma |
| `--original-data-dir` | Path | None | Para comparar warped vs original |
| `--epochs` | int | 30 | Epocas por arquitectura |
| `--batch-size` | int | 32 | Tamano de batch |
| `--lr` | float | 1e-4 | Learning rate |
| `--patience` | int | 10 | Paciencia para early stopping |
| `--seed` | int | 42 | Semilla aleatoria |
| `--device` | str | auto | Dispositivo (auto/cuda/cpu/mps) |
| `--quick` | bool | False | Modo rapido (5 epochs) |

### Estructura de Salida
```
outputs/arch_comparison/
├── comparison_results.json      # Metricas completas
├── comparison_results.csv       # Tabla resumen
├── training_logs/
│   ├── resnet18_warped_log.json
│   └── ...
├── checkpoints/
│   ├── resnet18_warped_best.pt
│   └── ...
└── figures_warped/
    ├── accuracy_comparison.png
    ├── f1_comparison.png
    ├── confusion_matrices.png
    └── training_curves.png
```

## Bugs Corregidos

### Criticos (2)

#### 1. Memory Leak en GPU
**Ubicacion:** `_train_single_architecture()` linea 4083
**Problema:** Tensores no liberados entre entrenamientos de arquitecturas
**Solucion:**
```python
# Antes del return
del model
if torch_device.type == 'cuda':
    torch.cuda.empty_cache()
```

#### 2. Retorno de modelo incorrecto
**Ubicacion:** `_train_single_architecture()` linea 4083
**Problema:** Retornaba ultimo estado del modelo, no el mejor
**Solucion:**
```python
if best_model_state is None:
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
return results, best_model_state
```

### Altos (3)

#### 3. Figuras sobrescritas
**Ubicacion:** `_generate_comparison_figures()` linea 4121
**Problema:** Figuras de warped y original se guardaban en mismo directorio
**Solucion:** Cambio a subdirectorios `figures_warped/` y `figures_original/`

#### 4. Bug de axes en matplotlib
**Ubicacion:** `_generate_comparison_figures()` linea 4187
**Problema:** IndexError cuando n_cols=1 y n_rows>1
**Solucion:**
```python
elif n_cols == 1:
    axes = axes.reshape(-1, 1)
```

#### 5. Sin validacion de loaders vacios
**Ubicacion:** `_train_single_architecture()` linea 3925
**Problema:** ZeroDivisionError si dataset vacio
**Solucion:**
```python
if len(train_loader) == 0:
    raise ValueError("train_loader está vacío")
```

### Medios (2)

#### 6. Clases hardcodeadas
**Ubicacion:** `OriginalDataset` linea 4579
**Problema:** Clases COVID/Normal/Viral_Pneumonia hardcodeadas
**Solucion:** Refactorizado para recibir `class_names` como parametro

#### 7. Sin validacion de CSV
**Ubicacion:** `OriginalDataset` linea 4584
**Problema:** No validaba columnas requeridas
**Solucion:** Agregada validacion de `image_name` y `category`

### Bajos (2)

#### 8. Imagenes no encontradas silenciosas
**Solucion:** Warning con conteo de imagenes faltantes

#### 9. Solo extension .png
**Solucion:** Ahora busca .png, .jpg, .jpeg

## Tests Agregados (15 nuevos)

```python
class TestCompareArchitectures:
    # Tests basicos
    test_compare_architectures_help
    test_compare_architectures_requires_data_dir
    test_compare_architectures_missing_data_dir
    test_compare_architectures_invalid_architecture
    test_compare_architectures_mixed_valid_invalid  # NUEVO
    test_compare_architectures_valid_architecture_names
    test_compare_architectures_default_epochs
    test_compare_architectures_quick_mode
    test_compare_architectures_original_data_option
    test_compare_architectures_module_execution
    test_architecture_display_names

    # Tests de sincronizacion
    test_cli_architectures_match_classifier  # NUEVO
    test_all_architectures_have_display_names  # NUEVO
    test_display_names_no_extra_keys  # NUEVO
    test_all_architectures_can_instantiate  # NUEVO
```

## Verificaciones Ejecutadas

### Tests
```bash
$ .venv/bin/python -m pytest tests/ -v --tb=short
====================== 327 passed, 23 warnings in 15.50s =======================
```

### Comando con datos reales
```bash
$ .venv/bin/python -m src_v2 compare-architectures outputs/warped_dataset \
    --architectures resnet18 --quick --device cpu

[1/1] Entrenando ResNet-18...
    Epoch   5: Val Acc=0.8681, F1=0.8669
    Test Accuracy: 83.33%, F1: 84.00%, Tiempo: 3.3 min

Mejor (warped): ResNet-18 con F1=84.00%
Resultados guardados en: /tmp/test_compare
```

### Archivos generados
```
/tmp/test_compare/
├── comparison_results.json (2.7 KB)
├── comparison_results.csv (145 B)
├── checkpoints/resnet18_warped_best.pt (44.8 MB)
├── training_logs/resnet18_warped_log.json (2.5 KB)
└── figures_warped/
    ├── accuracy_comparison.png (43 KB)
    ├── f1_comparison.png (42 KB)
    ├── confusion_matrices.png (40 KB)
    └── training_curves.png (160 KB)
```

## Criterios de Exito

| Criterio | Estado |
|----------|--------|
| Comando `--help` muestra documentacion | CUMPLIDO |
| Entrena arquitecturas secuencialmente | CUMPLIDO |
| Genera `comparison_results.json` | CUMPLIDO |
| Genera `comparison_results.csv` | CUMPLIDO |
| Genera figuras de comparacion | CUMPLIDO |
| Tests nuevos pasan (15 tests) | CUMPLIDO |
| 312+ tests siguen pasando (327) | CUMPLIDO |
| Documentacion creada | CUMPLIDO |

## Estado Final del Proyecto

| Metrica | Valor |
|---------|-------|
| Comandos CLI | 15 |
| Tests totales | 327 |
| Arquitecturas soportadas | 7 |
| Cobertura experimentos | ~80% |

## Gaps Restantes

| Funcionalidad | Prioridad | Descripcion |
|---------------|-----------|-------------|
| `gradcam` | Media | Visualizacion Grad-CAM para explicabilidad |
| `analyze-errors` | Media | Analisis de errores de clasificacion |
| `optimize-margin` | Baja | Busqueda automatica de margen optimo |

## Notas para Proxima Sesion

1. **Proximos comandos a implementar:**
   - `gradcam` - Critico para explicabilidad clinica
   - `analyze-errors` - Util para entender fallos del modelo

2. **Consideraciones:**
   - El comando `compare-architectures` puede tardar varias horas con 7 arquitecturas
   - Usar `--quick` para pruebas rapidas
   - Las figuras generadas estan listas para inclusion en tesis

3. **Arquitectura validada:**
   - ResNet-18 funciona correctamente como baseline
   - Early stopping y schedulers funcionan
   - Checkpoints compatibles con `evaluate-classifier`
