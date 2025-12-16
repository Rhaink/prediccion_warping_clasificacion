# Prompt de Continuacion - Sesion 22: Comando compare-architectures

**Fecha:** 2025-12-08
**Sesion anterior:** 21 (Validacion Funcional y Verificacion de Integridad)
**Estado previo:** 14 comandos CLI, 312 tests, 3 bugs corregidos, proyecto verificado

## Resumen de Sesiones Anteriores

### Sesion 20 - Comandos de Procesamiento
- Implementados: `compute-canonical` y `generate-dataset`
- Nuevo modulo: `src_v2/processing/` (gpa.py, warp.py)
- 4 arquitecturas nuevas: ResNet-50, AlexNet, VGG-16, MobileNetV2

### Sesion 21 - Validacion y Correccion de Bugs
- **3 bugs corregidos:**
  1. Division por cero en `scale_canonical_to_image` (gpa.py:265-269)
  2. NameError si `max_iterations=0` en GPA (gpa.py:183)
  3. Calculo incorrecto de area de triangulo (warp.py:263-274)
- **19 tests nuevos** para funciones helper y edge cases
- **312 tests pasando** (vs 293 anteriores)
- Forma canonica validada identica a referencia (diff < 0.000001 px)

## Objetivo de Sesion 22

Implementar el comando `compare-architectures` para automatizar la comparacion sistematica de multiples arquitecturas CNN en el pipeline de clasificacion COVID-19.

**Prioridad:** ALTA
**Complejidad estimada:** ~300 lineas de codigo
**Referencia:** `scripts/train_all_architectures.py`, `scripts/compare_classifiers.py`

## Contexto Tecnico

### Arquitecturas Disponibles (7 total)
El clasificador en `src_v2/models/classifier.py` ya soporta:
```python
SUPPORTED_BACKBONES = [
    "resnet18", "resnet50", "efficientnet_b0", "densenet121",
    "alexnet", "vgg16", "mobilenet_v2"
]
```

### Comando Existente de Entrenamiento
```bash
python -m src_v2 train-classifier outputs/warped_dataset \
    --backbone resnet18 --epochs 50 --batch-size 32
```

## Tarea Principal: Comando compare-architectures

### Uso Esperado
```bash
# Comparar todas las arquitecturas
python -m src_v2 compare-architectures \
    --data-dir outputs/full_warped_dataset \
    --output-dir outputs/arch_comparison \
    --epochs 30 \
    --seed 42

# Comparar arquitecturas especificas
python -m src_v2 compare-architectures \
    --data-dir outputs/full_warped_dataset \
    --architectures resnet18,efficientnet_b0,densenet121 \
    --output-dir outputs/arch_comparison_subset \
    --epochs 30

# Con dataset original para comparacion warped vs original
python -m src_v2 compare-architectures \
    --data-dir outputs/full_warped_dataset \
    --original-data-dir data/COVID-19_Radiography_Dataset \
    --output-dir outputs/full_comparison \
    --epochs 30
```

### Funcionalidades Requeridas

1. **Entrenamiento secuencial** de multiples arquitecturas
2. **Evaluacion consistente** en mismo split de test
3. **Generacion de reporte comparativo** (CSV y JSON)
4. **Visualizaciones:**
   - Grafico de barras comparando accuracy/F1
   - Matrices de confusion por arquitectura
   - Curva de entrenamiento por arquitectura
5. **Soporte opcional** para comparar warped vs original
6. **Tiempo de entrenamiento** por arquitectura

### Estructura de Salida Esperada
```
outputs/arch_comparison/
├── comparison_results.json      # Metricas de todas las arquitecturas
├── comparison_results.csv       # Tabla para facil analisis
├── training_logs/
│   ├── resnet18_log.json
│   ├── efficientnet_b0_log.json
│   └── ...
├── checkpoints/
│   ├── resnet18_best.pt
│   ├── efficientnet_b0_best.pt
│   └── ...
└── figures/
    ├── accuracy_comparison.png
    ├── f1_comparison.png
    ├── confusion_matrices.png
    └── training_curves.png
```

### Metricas a Reportar por Arquitectura
- Accuracy (test)
- F1 Macro (test)
- F1 Weighted (test)
- Precision/Recall/F1 por clase
- Tiempo de entrenamiento (segundos)
- Numero de parametros
- Tamano del modelo (MB)

## Parametros CLI

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `--data-dir` | Path | Requerido | Dataset warped con splits |
| `--output-dir` | Path | `outputs/arch_comparison` | Directorio de salida |
| `--architectures` | str | Todas | Lista separada por comas |
| `--original-data-dir` | Path | None | Para comparar warped vs original |
| `--epochs` | int | 30 | Epocas por arquitectura |
| `--batch-size` | int | 32 | Tamano de batch |
| `--seed` | int | 42 | Seed para reproducibilidad |
| `--device` | str | auto | Dispositivo (auto/cuda/cpu) |

## Implementacion Sugerida

### Paso 1: Agregar comando en cli.py
```python
@app.command()
def compare_architectures(
    data_dir: Path = typer.Argument(..., help="Dataset warped con splits"),
    output_dir: Path = typer.Option(Path("outputs/arch_comparison")),
    architectures: str = typer.Option(None, help="Arquitecturas separadas por coma"),
    original_data_dir: Optional[Path] = typer.Option(None),
    epochs: int = typer.Option(30),
    batch_size: int = typer.Option(32),
    seed: int = typer.Option(42),
    device: str = typer.Option("auto"),
):
    """Comparar multiples arquitecturas CNN para clasificacion."""
    ...
```

### Paso 2: Reutilizar logica existente
- Usar `ImageClassifier` de `src_v2/models/classifier.py`
- Usar funciones de entrenamiento de `train-classifier`
- Usar evaluacion de `evaluate-classifier`

### Paso 3: Agregar generacion de reportes y visualizaciones

## Tests Requeridos

Agregar en `tests/test_processing.py` o nuevo archivo `tests/test_compare_architectures.py`:

```python
class TestCompareArchitecturesCommand:
    def test_compare_architectures_help(self):
        result = runner.invoke(app, ['compare-architectures', '--help'])
        assert result.exit_code == 0
        assert '--architectures' in result.stdout
        assert '--epochs' in result.stdout

    def test_compare_architectures_missing_data_dir(self):
        result = runner.invoke(app, ['compare-architectures'])
        assert result.exit_code != 0

    def test_compare_architectures_invalid_architecture(self):
        result = runner.invoke(app, [
            'compare-architectures',
            '/some/data',
            '--architectures', 'invalid_arch'
        ])
        assert result.exit_code != 0
```

## Archivos Clave a Consultar

```
scripts/train_all_architectures.py          # Logica original de comparacion
scripts/compare_classifiers.py              # Comparacion warped vs original
src_v2/models/classifier.py                 # ImageClassifier con 7 backbones
src_v2/cli.py                               # CLI existente (train-classifier)
docs/sesiones/SESION_21_VALIDACION_FUNCIONAL.md  # Estado actual del proyecto
docs/ANALISIS_GAPS_CLI.md                   # Lista de gaps restantes
```

## Criterios de Exito

1. [ ] Comando `compare-architectures --help` muestra documentacion
2. [ ] Entrena al menos 2 arquitecturas secuencialmente sin errores
3. [ ] Genera `comparison_results.json` con metricas correctas
4. [ ] Genera `comparison_results.csv` para analisis
5. [ ] Genera al menos 1 figura de comparacion
6. [ ] Tests nuevos pasan (minimo 3 tests)
7. [ ] 312+ tests siguen pasando
8. [ ] Documentacion de sesion creada

## Gaps Restantes Despues de Sesion 22

| Funcionalidad | Prioridad | Sesion Estimada |
|---------------|-----------|-----------------|
| `gradcam` | Media | 23-24 |
| `analyze-errors` | Media | 23 |
| `optimize-margin` | Baja | 24 |

## Notas Adicionales

- Priorizar robustez sobre velocidad
- El entrenamiento puede tardar (7 arquitecturas x 30 epochs)
- Considerar agregar `--quick` para pruebas rapidas (5 epochs)
- Mantener compatibilidad con checkpoints existentes
- El script original usa epochs=50, pero 30 es suficiente para comparacion

---

**Para comenzar la sesion:**
1. Lee este prompt completo
2. Revisa `scripts/train_all_architectures.py` para entender la logica
3. Implementa el comando siguiendo la estructura sugerida
4. Agrega tests antes de probar con datos reales

**Usa ultrathink para disenar la arquitectura del comando antes de implementar.**
