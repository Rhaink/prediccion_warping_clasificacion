# Prompt de Continuacion - Sesion 35

## Contexto Rapido

Proyecto de clasificacion COVID-19 con normalizacion geometrica (warping).
**Hipotesis CONFIRMADA:** Warped mejora generalizacion 11x y robustez 30x.

## Estado Post-Sesion 34

### Logros Sesion 34
1. **16 figuras GradCAM** para defensa de tesis (outputs/thesis_figures/)
2. **Datos verificados** - 15/15 tests matematicos pasaron
3. **Bugs corregidos**: N1 (CLASSIFIER_CLASSES), N3 (Image.open RGB)
4. **Analisis exhaustivo** de tests, Hydra, git

### Metricas Actuales
- 551+ tests pasando
- 21 comandos CLI funcionando
- 35% cobertura de tests de integracion CLI
- 18,512 lineas de codigo sin commit (3 dias)

## Problemas Identificados

### P1: Commits Pendientes (CRITICO)
- 3 dias sin commits
- 18,512 lineas de codigo nuevo
- Recomendacion: 7 commits tematicos

### P2: Tests de Integracion CLI (65% gap)
Comandos sin tests de integracion:
1. classify, train-classifier, evaluate-classifier (ALTA)
2. gradcam, analyze-errors (ALTA)
3. cross-evaluate, evaluate-external, test-robustness (MEDIA)
4. compute-canonical, generate-dataset (MEDIA)
5. compare-architectures, evaluate-ensemble, version (BAJA)

### P3: Hydra Codigo Muerto
- setup_hydra_config() no se usa efectivamente
- Solo extrae 2 valores de toda la config
- Recomendacion: REMOVER

## Objetivos Sesion 35

### Opcion A: Validacion Estadistica
- Calcular p-values para diferencias original vs warped
- Intervalos de confianza 95%
- Tests de hipotesis formal (t-test, Wilcoxon)

### Opcion B: Commits + Tests Criticos
- Hacer los 7 commits tematicos pendientes
- Agregar tests de integracion para classify, gradcam, analyze-errors
- Remover codigo Hydra muerto

### Opcion C: Ambos (recomendado)
1. Primero: Commits (organizar el trabajo de 3 dias)
2. Luego: Tests criticos o validacion estadistica

## Archivos Clave

### Codigo Modificado (sin commit)
```
src_v2/cli.py              # +5,666 lineas
src_v2/models/classifier.py # Nuevo modulo
src_v2/processing/         # GPA + warping
src_v2/visualization/      # GradCAM + errors
tests/test_*.py            # +7,000 lineas tests
```

### Datos Verificados
```
outputs/session30_analysis/consolidated_results.json  # Generalizacion
outputs/session29_robustness/artifact_robustness_results.json  # Robustez
```

### Figuras de Tesis
```
outputs/thesis_figures/combined_figures/
  comparison_*.png    # 6 lado-a-lado
  crossdomain_*.png   # 6 cross-domain
  matrix_*.png        # 3 matrices
  summary_metrics.png # Resumen metricas
```

## Comandos Utiles

```bash
# Ver estado git
git status
git diff --stat

# Correr tests
.venv/bin/python -m pytest tests/ -v --tb=short

# Verificar CLI
.venv/bin/python -m src_v2 --help

# Regenerar figuras
.venv/bin/python scripts/create_thesis_figures.py
```

## Datos de Hipotesis (VERIFICADOS)

| Metrica | Original | Warped | Mejora |
|---------|----------|--------|--------|
| Gap generalizacion | 25.36% | 2.24% | 11.32x |
| Degradacion JPEG Q50 | 16.14% | 0.53% | 30.62x |

## Decision del Usuario

Al comenzar Sesion 35, elegir:
1. Hacer commits primero (organizar trabajo pendiente)
2. Agregar validacion estadistica (p-values)
3. Agregar tests de integracion criticos
4. Limpiar codigo Hydra

Recomendacion: Hacer commits PRIMERO para no perder trabajo.
