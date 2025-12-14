# Reorganización del Proyecto - 2025-12-13

**Proyecto:** Predicción de Warping con Clasificación de Radiografías de Tórax
**Auditor:** Claude (Consultor Senior de Arquitectura de Software)
**Versión:** 1.0

---

## 1. Resumen Ejecutivo

### Hallazgos Principales

El proyecto presenta una **estructura funcional pero con acumulación significativa de deuda técnica documental y organizacional**. Tras analizar 358,912 archivos (362 Python útiles, 202 Markdown, 142 JSON), se identifican los siguientes problemas críticos:

1. **31 archivos sueltos en raíz** - Prompts de sesión, reportes de verificación y documentos que deberían estar organizados
2. **4 directorios de checkpoints redundantes** - `checkpoints/`, `checkpoints_v2/`, `checkpoints_v2_correct/`, `checkpoints_v2_full/` sin documentación de cuál es el canónico
3. **Bifurcación documental** - 19 archivos `PROMPT_CONTINUACION_SESION_*.md` duplican contenido de sesiones
4. **Scripts con duplicación funcional** - 3-4 pares de scripts casi idénticos diferenciados solo por parámetros
5. **Ausencia total de CI/CD** - Sin GitHub Actions, GitLab CI, ni automatización de tests

### Recomendaciones Críticas

| Prioridad | Acción | Impacto |
|-----------|--------|---------|
| 1 | Consolidar directorios de checkpoints | ALTO - Claridad de cuál modelo usar |
| 2 | Mover archivos de raíz a directorios apropiados | ALTO - Navegabilidad |
| 3 | Eliminar/consolidar documentación redundante | MEDIO - Mantenibilidad |
| 4 | Implementar CI/CD básico | MEDIO - Calidad continua |
| 5 | Fusionar scripts duplicados | BAJO - Reducir confusión |

### Siguiente Paso Inmediato

**Ejecutar limpieza de raíz**: Mover los 15 archivos `Prompt para Sesion *.txt` a `docs/prompts/historicos/` y los 6 archivos `REPORTE_VERIFICACION_*.md` a `docs/legacy/`.

---

## 2. Estado Actual

### 2.1 Diagrama de Estructura Actual

```
prediccion_warping_clasificacion/
├── src_v2/                          # CÓDIGO FUENTE PRINCIPAL (27 archivos)
│   ├── cli.py                       # CLI con Typer (21+ comandos)
│   ├── constants.py                 # Constantes centralizadas
│   ├── data/                        # dataset.py, transforms.py, utils.py
│   ├── models/                      # classifier.py, hierarchical.py, losses.py, resnet_landmark.py
│   ├── processing/                  # gpa.py, warp.py
│   ├── training/                    # trainer.py, callbacks.py
│   ├── evaluation/                  # metrics.py
│   ├── visualization/               # gradcam.py, error_analysis.py, pfs_analysis.py
│   └── utils/                       # geometry.py
│
├── scripts/                         # SCRIPTS DE EXPERIMENTOS (84 archivos)
│   ├── train*.py                    # 8 variantes de entrenamiento
│   ├── validation_session*.py       # 9 scripts de validación
│   ├── generate_*.py                # 7 generadores de datasets
│   ├── gradcam_*.py                 # 3 análisis GradCAM
│   ├── verify_*.py                  # 6 verificaciones
│   └── visualization/               # 20 scripts de figuras para tesis
│
├── tests/                           # SUITE DE TESTS (22 archivos, 12K líneas)
│   ├── conftest.py                  # 507 líneas de fixtures
│   ├── test_cli*.py                 # 3,890 líneas (CLI e integración)
│   └── test_*.py                    # 18 archivos más
│
├── data/                            # DATOS (~350K imágenes PNG/JPG)
│   ├── dataset/                     # Múltiples subdatasets
│   │   ├── COVID-19_Radiography_Dataset/
│   │   ├── dataset1/, dataset2/, dataset3/
│   │   └── COVID/, Normal/, Viral_Pneumonia/
│   └── coordenadas/                 # Ground truth landmarks
│
├── checkpoints/                     # MODELOS (51 subdirectorios)
│   ├── phase1/, phase2/             # Baseline
│   ├── session9/, session10/        # Experimentos
│   ├── session13/, session14/       # Hierarchical
│   └── exp_clahe_*/                 # Variantes CLAHE
├── checkpoints_v2/                  # DUPLICADO? (3 archivos)
├── checkpoints_v2_correct/          # DUPLICADO? (3 archivos)
├── checkpoints_v2_full/             # DUPLICADO? (3 archivos)
│
├── outputs/                         # RESULTADOS (51 subdirectorios)
│   ├── classifier*/                 # Evaluaciones
│   ├── gradcam*/                    # Mapas de atención
│   └── session*/                    # Por sesión
├── outputs_v2/                      # OUTPUTS ALTERNATIVO
│
├── audit/                           # AUDITORÍA EN PROGRESO
│   ├── sessions/                    # 8+ sesiones de auditoría
│   ├── findings/                    # Hallazgos consolidados
│   └── MASTER_PLAN.md
│
├── docs/                            # DOCUMENTACIÓN
│   ├── sesiones/                    # 52 sesiones técnicas
│   ├── PROMPT_CONTINUACION_*.md     # 19 archivos (REDUNDANTES)
│   ├── INTROSPECCION_*.md           # 4 archivos (REDUNDANTES)
│   └── *.md                         # 12 documentos adicionales
│
├── documentación/                   # DOCUMENTACIÓN LATEX
│   └── apendices/                   # Vacío
│
├── configs/                         # CONFIGURACIÓN
│   └── final_config.json            # Único archivo
│
├── .claude/                         # CONFIGURACIÓN CLAUDE CODE
│   ├── settings.local.json
│   └── ESTADO_PROYECTO.md
│
└── [31 ARCHIVOS EN RAÍZ]            # PROBLEMA PRINCIPAL
    ├── Prompt para Sesion *.txt     # 15 archivos
    ├── REPORTE_VERIFICACION_*.md    # 6 archivos
    ├── RESUMEN_*.md                 # 4 archivos
    ├── README.md, CHANGELOG.md      # Documentación legítima
    ├── pyproject.toml, requirements.txt
    └── tempfile (14MB)              # ELIMINAR
```

### 2.2 Métricas

| Categoría | Cantidad | Observación |
|-----------|----------|-------------|
| **Archivos Python** | 133 | 27 en src_v2, 84 en scripts, 22 en tests |
| **Archivos Markdown** | 139 | Excesivo, con duplicación |
| **Archivos JSON** | 142 | Mayormente resultados |
| **Modelos PyTorch (.pt)** | 1,328 | En 4 directorios diferentes |
| **Imágenes** | ~323K | PNG y JPG de datasets |
| **Total archivos** | 358,912 | Incluyendo .venv |

### 2.3 Problemas Identificados (Lista Priorizada)

| # | Severidad | Problema | Archivos Afectados |
|---|-----------|----------|-------------------|
| P1 | ALTA | Archivos sueltos en raíz | 31 archivos |
| P2 | ALTA | Múltiples directorios checkpoints sin documentar cuál es canónico | 4 directorios |
| P3 | ALTA | Referencias obsoletas en `configs/final_config.json` | 1 archivo |
| P4 | MEDIA | Bifurcación documental (PROMPT_CONTINUACION duplica sesiones) | 19 archivos |
| P5 | MEDIA | Scripts con duplicación funcional | 6-8 scripts |
| P6 | MEDIA | Ausencia de CI/CD | N/A |
| P7 | BAJA | `tempfile` de 14MB en raíz | 1 archivo |
| P8 | BAJA | Documentación LaTeX con directorios vacíos | 1 directorio |
| P9 | BAJA | Tests CLI muy grandes (2,748 + 1,142 líneas) | 2 archivos |

---

## 3. Estructura Propuesta

### 3.1 Diagrama de Estructura Objetivo

```
prediccion_warping_clasificacion/
├── src_v2/                          # SIN CAMBIOS - Bien organizado
│
├── scripts/                         # CONSOLIDAR variantes
│   ├── training/                    # train.py + variantes consolidadas
│   ├── evaluation/                  # evaluate*.py, validate*.py
│   ├── generation/                  # generate_*.py consolidados
│   ├── analysis/                    # gradcam*.py, gpa*.py, analyze*.py
│   ├── visualization/               # SIN CAMBIOS (20 scripts)
│   └── utils/                       # piecewise_affine_warp.py, gpa_analysis.py, predict.py
│
├── tests/                           # CONSIDERAR dividir tests grandes
│
├── data/                            # SIN CAMBIOS
│
├── checkpoints/                     # CONSOLIDAR - Una única fuente de verdad
│   ├── production/                  # Modelos finales para uso
│   │   ├── ensemble/                # 4 seeds del ensemble final
│   │   └── classifier/              # Clasificador warped
│   ├── experiments/                 # Modelos de experimentos
│   │   ├── session9/, session10/
│   │   ├── session13/, session14/
│   │   └── clahe_variants/
│   └── archive/                     # Versiones antiguas (v2, v2_correct, v2_full)
│
├── outputs/                         # MANTENER estructura actual
│
├── docs/                            # REORGANIZAR
│   ├── sesiones/                    # Sesiones consolidadas
│   ├── reference/                   # Documentos de referencia actuales
│   │   ├── RESULTADOS_EXPERIMENTALES_v2.md
│   │   ├── REFERENCIA_EXPERIMENTOS_ORIGINALES.md
│   │   └── REPRODUCIBILITY.md       # Mover desde raíz
│   ├── guides/                      # Guías de uso
│   │   └── CONTRIBUTING.md          # Mover desde raíz
│   ├── legacy/                      # Documentación histórica
│   │   ├── REPORTE_VERIFICACION_*.md
│   │   ├── RESUMEN_VERIFICACION_*.md
│   │   └── PROMPT_CONTINUACION_*.md
│   └── prompts/                     # Prompts de sesiones (históricos)
│       └── historicos/
│
├── audit/                           # SIN CAMBIOS - Rama activa
│
├── configs/                         # EXPANDIR
│   ├── final_config.json            # Actualizar referencias
│   └── experiments/                 # Configs de experimentos
│
├── .github/                         # NUEVO - CI/CD
│   └── workflows/
│       └── tests.yml
│
└── [RAÍZ LIMPIA]
    ├── README.md
    ├── CHANGELOG.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── .gitignore
    └── GROUND_TRUTH.json
```

### 3.2 Justificación de Cambios Arquitectónicos

| Cambio | Justificación |
|--------|---------------|
| **Consolidar checkpoints/** | Los 4 directorios (checkpoints, _v2, _v2_correct, _v2_full) generan confusión sobre qué modelos usar. Estructura jerárquica con `production/`, `experiments/` y `archive/` clarifica propósito. |
| **Mover archivos de raíz** | 31 archivos sueltos dificultan navegación. README y configs principales deben ser únicos en raíz. |
| **Crear docs/legacy/** | Preserva historia sin contaminar documentación activa. PROMPT_CONTINUACION y REPORTE_VERIFICACION son históricos valiosos pero no actuales. |
| **Subdividir scripts/** | 84 scripts planos dificultan encontrar funcionalidad. Agrupación por propósito mejora descubribilidad. |
| **Agregar CI/CD** | Tests existentes (12K líneas) no se ejecutan automáticamente. GitHub Actions básico previene regresiones. |

---

## 4. Plan de Acción

### 4.1 Acciones de Alta Prioridad

| # | Acción | Archivos Afectados | Riesgo | Comando/Proceso |
|---|--------|-------------------|--------|-----------------|
| 1 | Eliminar `tempfile` | 1 | BAJO | `rm tempfile` |
| 2 | Crear estructura de directorios | 0 | BAJO | `mkdir -p docs/{legacy,reference,guides,prompts/historicos}` |
| 3 | Mover prompts de sesión | 15 | BAJO | `mv "Prompt para Sesion*.txt" docs/prompts/historicos/` |
| 4 | Mover reportes de verificación | 6 | BAJO | `mv REPORTE_VERIFICACION_*.md docs/legacy/` |
| 5 | Mover resúmenes de verificación | 4 | BAJO | `mv RESUMEN_*.md docs/legacy/` |
| 6 | Documentar checkpoints canónicos | 1 | BAJO | Crear `checkpoints/README.md` |

### 4.2 Acciones de Media Prioridad

| # | Acción | Archivos Afectados | Riesgo | Estado |
|---|--------|-------------------|--------|--------|
| 7 | Actualizar referencias en `final_config.json` | 1 | MEDIO | Pendiente - Verificar paths |
| 8 | Consolidar PROMPT_CONTINUACION en sesiones | 19 | MEDIO | Pendiente - Revisar contenido único |
| 9 | Fusionar `train_classifier.py` + `train_classifier_original.py` | 2 | MEDIO | Pendiente |
| 10 | Fusionar `generate_warped_dataset*.py` (3 variantes) | 3 | MEDIO | Pendiente |
| 11 | Implementar GitHub Actions básico | 1 | BAJO | Pendiente |

### 4.3 Acciones de Baja Prioridad

| # | Acción | Archivos Afectados | Riesgo | Estado |
|---|--------|-------------------|--------|--------|
| 12 | Dividir `test_cli_integration.py` | 1 | BAJO | Opcional |
| 13 | Deprecar scripts con versiones mejoradas | 4 | BAJO | Pendiente - REVISAR primero |
| 14 | Limpiar directorio `documentación/apendices/` vacío | 1 | BAJO | Pendiente |
| 15 | Consolidar gradcam_*.py en script único | 3 | BAJO | Opcional |

---

## 5. Archivos para Revisión Manual

Los siguientes archivos requieren decisión humana antes de actuar:

### 5.1 Checkpoints - ¿Cuál es el canónico?

| Directorio | Contenido | Pregunta |
|------------|-----------|----------|
| `checkpoints/` | 51 subdirs, experimentos completos | ¿Este es el principal? |
| `checkpoints_v2/` | 3 archivos: final_model.pt, phase1/, phase2/ | ¿Versión mejorada? |
| `checkpoints_v2_correct/` | 3 archivos idénticos | ¿Corrección de bug? |
| `checkpoints_v2_full/` | 3 archivos idénticos | ¿Dataset completo? |

**Acción requerida**: Documentar en README.md cuál usar para reproducir resultados publicados.

### 5.2 Scripts con Versiones Múltiples

| Script Original | Script Mejorado | Pregunta |
|-----------------|-----------------|----------|
| `validation_session26.py` | `validation_session26_v2.py` | ¿Deprecar v1? |
| `visualize_gpa_methodology.py` | `visualize_gpa_methodology_fixed.py` | ¿Deprecar original? |
| `generate_bloque1_profesional.py` | `generate_bloque1_v2_profesional.py` | ¿Deprecar v1? |

**Acción requerida**: Confirmar que v2/fixed funciona correctamente antes de deprecar.

### 5.3 Documentación Contradictoria

| Documento 1 | Documento 2 | Discrepancia |
|-------------|-------------|--------------|
| `INTROSPECCION_SESION_29.md` | `PROMPT_CONTINUACION_SESION_29.md` | 482 vs 494 tests reportados |
| `SESION_38_VALIDACION_EXPERIMENTAL.md` | `SESION_38_INTROSPECCION_PROFUNDA.md` | Dos versiones de sesión 38 |

**Acción requerida**: Determinar cuál es el documento autorizado.

### 5.4 Referencias Potencialmente Obsoletas

| Archivo | Referencia | Estado |
|---------|------------|--------|
| `configs/final_config.json:250` | `checkpoints/session10/` | VERIFICAR - ¿Existe? |
| `configs/final_config.json:183` | `checkpoints/session13/` | VERIFICAR - ¿Existe? |

---

## 6. Notas para Continuidad

### 6.1 Decisiones Tomadas y Razonamiento

| Decisión | Razonamiento |
|----------|--------------|
| No eliminar PROMPT_CONTINUACION inmediatamente | Pueden contener contexto único no presente en sesiones. Mover a legacy primero. |
| Mantener estructura de `scripts/visualization/` | Los 20 scripts de bloques están bien organizados y son para figuras de tesis. |
| No fusionar todos los scripts de train | Algunos tienen propósitos específicos (hierarchical, expanded_dataset). Solo fusionar duplicados claros. |
| Crear docs/legacy/ en lugar de eliminar | Preserva historia del proyecto para referencia futura. |

### 6.2 Áreas que Requieren Más Investigación

1. **Validar modelos en checkpoints_v2_***:
   - Ejecutar evaluación con cada uno
   - Determinar cuál reproduce los 3.71 px reportados

2. **Verificar integridad de `final_config.json`**:
   - Confirmar que todos los paths referenciados existen
   - Actualizar o documentar discrepancias

3. **Auditoría de rama `audit/main`**:
   - Está en progreso con 8 sesiones completadas
   - Considerar merge a main cuando complete

### 6.3 Contexto Necesario para Siguiente Sesión

```
Estado del proyecto:
- Versión: 2.0.0 (src_v2)
- Python: 3.12
- Framework: PyTorch + Typer CLI
- Tests: 22 archivos, 12K+ líneas, pytest
- Error ensemble: 3.71 px (GROUND_TRUTH.json)
- Clasificación: 98.73% accuracy

Rama actual: audit/main
Commits pendientes: Ver git status

Archivos modificados (sin commit):
- audit/sessions/session_07a_gradcam.md
- audit/sessions/session_07b_error_analysis.md

Archivos sin seguimiento:
- audit/prompts/prompt_session_04b.md
- audit/prompts/prompt_session_05a.md
- audit/prompts/prompt_session_05b.md
- audit/prompts/prompt_session_07b.md
- audit/prompts/prompt_session_07c.md
```

---

## Anexo A: Inventario de Archivos Obsoletos/Redundantes

### A.1 Para ELIMINAR (sin pérdida de información)

| Archivo | Razón | Tamaño |
|---------|-------|--------|
| `tempfile` | Archivo temporal sin uso | 14 MB |

### A.2 Para MOVER a docs/legacy/

| Archivo | Razón | Destino |
|---------|-------|---------|
| `REPORTE_VERIFICACION_01_analisis_exploratorio.md` | Histórico, contenido en sesiones | `docs/legacy/` |
| `REPORTE_VERIFICACION_03_funciones_perdida.md` | Histórico | `docs/legacy/` |
| `REPORTE_VERIFICACION_06_optimizacion_arquitectura.md` | Histórico | `docs/legacy/` |
| `REPORTE_VERIFICACION_DESCUBRIMIENTOS_GEOMETRICOS.md` | Histórico | `docs/legacy/` |
| `REPORTE_VERIFICACION_DOCS_16_17.md` | Histórico | `docs/legacy/` |
| `REPORTE_REVISION_ENSEMBLE_TTA.md` | Histórico | `docs/legacy/` |
| `RESUMEN_VERIFICACION_03.md` | Histórico | `docs/legacy/` |
| `RESUMEN_VERIFICACION_06.md` | Histórico | `docs/legacy/` |
| `RESUMEN_VERIFICACION.md` | Histórico | `docs/legacy/` |
| `RESUMEN_REVISION_ENSEMBLE_TTA.md` | Histórico | `docs/legacy/` |

### A.3 Para MOVER a docs/prompts/historicos/

| Archivo | Razón |
|---------|-------|
| `Prompt para Sesion 19.txt` | Prompt histórico |
| `Prompt para Sesion 36.txt` - `Prompt para Sesion 50.txt` | Prompts históricos (15 archivos) |

### A.4 Scripts para DEPRECAR (después de verificación)

| Script | Reemplazado por | Verificar antes |
|--------|-----------------|-----------------|
| `validation_session26.py` | `validation_session26_v2.py` | Ejecutar v2 |
| `visualize_gpa_methodology.py` | `visualize_gpa_methodology_fixed.py` | Ejecutar fixed |
| `generate_bloque1_profesional.py` | `generate_bloque1_v2_profesional.py` | Ejecutar v2 |
| `generate_bloque2_metodologia_datos.py` | `generate_bloque2_v2_mejorado.py` | Ejecutar v2 |

---

## Anexo B: Script de Reorganización Sugerido

```bash
#!/bin/bash
# reorganizar-proyecto.sh
# Ejecutar desde raíz del proyecto

# 1. Crear estructura de directorios
mkdir -p docs/{legacy,reference,guides,prompts/historicos}
mkdir -p checkpoints/{production/ensemble,production/classifier,experiments,archive}

# 2. Eliminar archivos temporales
rm -f tempfile

# 3. Mover prompts históricos
mv "Prompt para Sesion"*.txt docs/prompts/historicos/ 2>/dev/null

# 4. Mover reportes a legacy
mv REPORTE_VERIFICACION_*.md docs/legacy/ 2>/dev/null
mv RESUMEN_*.md docs/legacy/ 2>/dev/null

# 5. Mover documentación de referencia
mv REPRODUCIBILITY.md docs/reference/ 2>/dev/null
mv CONTRIBUTING.md docs/guides/ 2>/dev/null

# 6. Documentar checkpoints
cat > checkpoints/README.md << 'EOF'
# Checkpoints

## Estructura

- `production/` - Modelos finales para reproducir resultados publicados
- `experiments/` - Modelos de experimentos individuales
- `archive/` - Versiones antiguas

## Modelo Canónico

Para reproducir el error de 3.71 px del ensemble:
```
checkpoints/session10/ensemble/seed*/final_model.pt
```
EOF

echo "Reorganización completada. Verificar con: git status"
```

---

*Documento generado automáticamente. Última actualización: 2025-12-13*
