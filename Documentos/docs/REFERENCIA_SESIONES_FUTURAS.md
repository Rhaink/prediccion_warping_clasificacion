# REFERENCIA DEFINITIVA PARA SESIONES FUTURAS

> **IMPORTANTE:** Este documento es la referencia maestra del proyecto. Antes de cualquier sesión de trabajo,
> consulta este archivo para mantener el contexto y NO desviarte de los objetivos.

**Fecha de creación:** 2025-12-10
**Última actualización:** Sesión 43
**Basado en:** Análisis exhaustivo de 43 sesiones + toda la documentación

---

## 1. OBJETIVO PRINCIPAL DEL PROYECTO

### Hipótesis Central (REFORMULADA - Sesión 39)

> **"La normalización geométrica mediante landmarks anatómicos proporciona:**
> 1. **30x mejor robustez a compresión JPEG**
> 2. **2.4x mejor robustez a blur**
> 3. **2.4x mejor generalización cross-dataset**
>
> **El mecanismo es:**
> - ~75% por reducción de información (regularización implícita)
> - ~25% adicional por normalización geométrica"

### Lo que SÍ está demostrado:
- Error de predicción de landmarks: **3.71 px** (ensemble 4 modelos + TTA)
- Clasificación en dataset warped: **98.73% accuracy**
- Robustez JPEG Q50: **30x superior** (0.53% vs 16.14% degradación)
- Robustez Blur: **2.4x superior** (6.06% vs 14.43% degradación)
- Generalización cross-dataset: **2.4x mejor** (gap 3.17% vs 7.70%)

### Lo que NO está demostrado:
- ~~"El warping elimina marcas hospitalarias"~~ → Solo las excluye/recorta
- ~~"Generaliza 11x mejor"~~ → Solo **2.4x** (corregido Sesión 39)
- ~~"Fuerza atención pulmonar"~~ → PFS ~0.49 ≈ chance (invalidado Sesión 39)
- ~~"Resuelve domain shift externo"~~ → FedCOVIDx ~55% (NO resuelve)

---

## 2. ARQUITECTURA DEL SISTEMA

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO DE CLASIFICACIÓN                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ETAPA 1: Predicción de Landmarks                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Input: Radiografía 299x299 (grayscale)                            │   │
│  │ Preprocessing: CLAHE (clip=2.0, tile=4) + resize 224x224          │   │
│  │ Modelo: ResNet-18 + Coordinate Attention + Deep Head (768 dim)    │   │
│  │ Output: 15 landmarks (30 coordenadas normalizadas [0,1])          │   │
│  │ Error: 3.71 px (ensemble 4 modelos + TTA)                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                           │
│  ETAPA 2: Normalización Geométrica (Warping)                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Método: Piecewise Affine Warping (Delaunay)                       │   │
│  │ Forma canónica: GPA (Generalized Procrustes Analysis)             │   │
│  │ Puntos: 15 landmarks + 8 puntos de borde = 23 puntos              │   │
│  │ Margin scale: 1.05 (óptimo)                                       │   │
│  │ Fill rate: 99% (con use_full_coverage=True)                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                           │
│  ETAPA 3: Clasificación COVID-19                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Backbone recomendado: DenseNet-121 (pretrained ImageNet)          │   │
│  │ Clases: COVID (24%), Normal (67%), Viral_Pneumonia (9%)           │   │
│  │ Accuracy: 98.73% (warped 99% fill)                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. ESTRUCTURA DEL CÓDIGO

```
prediccion_warping_clasificacion/
├── src_v2/                          # Código fuente principal (12,891 líneas)
│   ├── constants.py                 # Constantes centralizadas
│   ├── cli.py                       # CLI con Typer (6,665 líneas, 20 comandos)
│   ├── data/                        # Módulo de datos
│   │   ├── dataset.py               # Dataset loading
│   │   ├── transforms.py            # Augmentaciones
│   │   └── utils.py                 # Utilidades
│   ├── models/                      # Modelos neurales
│   │   ├── resnet_landmark.py       # ResNet-18 + Coordinate Attention
│   │   ├── hierarchical.py          # Modelo jerárquico
│   │   ├── classifier.py            # Clasificador (7 backbones)
│   │   └── losses.py                # Wing Loss
│   ├── training/                    # Entrenamiento
│   │   ├── trainer.py               # Lógica principal
│   │   └── callbacks.py             # Early stopping, etc.
│   ├── evaluation/                  # Evaluación
│   │   └── metrics.py               # Métricas
│   ├── processing/                  # Procesamiento geométrico
│   │   ├── warp.py                  # Piecewise Affine Warping
│   │   └── gpa.py                   # Generalized Procrustes Analysis
│   └── visualization/               # Visualización
│       ├── gradcam.py               # Grad-CAM
│       └── pfs_analysis.py          # Pulmonary Focus Score
│
├── tests/                           # Tests (10,781 líneas, 553+ tests)
│   ├── conftest.py                  # Fixtures compartidos
│   ├── test_cli_integration.py      # Tests de integración CLI
│   └── ...
│
├── scripts/                         # Scripts de análisis (~50 archivos)
│   ├── calculate_pfs_warped.py      # PFS con máscaras warped
│   ├── filter_dataset_3_classes.py  # Filtrar dataset
│   └── ...
│
├── docs/                            # Documentación
│   ├── sesiones/                    # 39 sesiones documentadas
│   ├── RESULTADOS_EXPERIMENTALES_v2.md  # Resultados validados
│   ├── REFERENCIA_SESIONES_FUTURAS.md   # ← ESTE ARCHIVO
│   └── ...
│
├── data/                            # Datasets
│   └── COVID-19_Radiography_Dataset/
│
├── checkpoints/                     # Modelos entrenados (~145 GB)
│   ├── session10/ensemble/          # Ensemble original
│   └── session13/                   # Modelos adicionales
│
└── outputs/                         # Resultados experimentales
    ├── full_coverage_warped_dataset/  # INVALIDADO; ver reportes/RESUMEN_DEPURACION_OUTPUTS.md
    ├── original_3_classes/
    └── original_cropped_47/
```

---

## 4. COMANDOS CLI DISPONIBLES (20 comandos)

### Landmarks
```bash
python -m src_v2 train                    # Entrenar modelo de landmarks
python -m src_v2 evaluate                 # Evaluar modelo individual
python -m src_v2 evaluate-ensemble        # Evaluar ensemble (3.71 px)
python -m src_v2 predict                  # Predecir en imagen individual
python -m src_v2 warp                     # Aplicar warping a imagen
```

### Clasificación
```bash
python -m src_v2 classify                 # Clasificar imagen
python -m src_v2 train-classifier         # Entrenar clasificador
python -m src_v2 evaluate-classifier      # Evaluar clasificador
python -m src_v2 cross-evaluate           # Cross-evaluation A↔B
python -m src_v2 evaluate-external        # Evaluación externa (binaria)
python -m src_v2 test-robustness          # Test de robustez (JPEG, blur)
python -m src_v2 compare-architectures    # Comparar arquitecturas
```

### Procesamiento
```bash
python -m src_v2 compute-canonical        # Calcular forma canónica (GPA)
python -m src_v2 generate-dataset         # Generar dataset warped
python -m src_v2 generate-lung-masks      # Generar máscaras pulmonares aproximadas
python -m src_v2 optimize-margin          # Buscar margen óptimo para warping
```

### Visualización y Análisis
```bash
python -m src_v2 gradcam                  # Generar visualizaciones Grad-CAM
python -m src_v2 analyze-errors           # Analizar errores de clasificación
python -m src_v2 pfs-analysis             # Analizar Pulmonary Focus Score (PFS)
```

### Utilidad
```bash
python -m src_v2 version                  # Mostrar versión
```

**Instalación como paquete:** `pip install -e .` → `covid-landmarks --help`

---

## 5. RESULTADOS EXPERIMENTALES VALIDADOS

### 5.1 Predicción de Landmarks (VALIDADO)

| Métrica | Valor | Configuración |
|---------|-------|---------------|
| **Error ensemble (MEJOR)** | **3.71 px** | 4 modelos + TTA |
| Error std | 2.42 px | - |
| Error mediano | 3.17 px | - |
| Mejor individual | 4.04 px | seed=456 + TTA |

**Modelos del Ensemble:**
```
checkpoints/session10/ensemble/seed123/final_model.pt
checkpoints/session10/ensemble/seed456/final_model.pt
checkpoints/session13/seed321/final_model.pt
checkpoints/session13/seed789/final_model.pt
```

### 5.2 Clasificación (VALIDADO)

| Dataset | Accuracy | F1-Score | Fill Rate |
|---------|----------|----------|-----------|
| Original 100% | 98.84% | 98.16% | 100% |
| Original Cropped 47% | 98.89% | 98.25% | 47% |
| Warped 47% | 98.02% | - | 47% |
| Warped 99% | 98.73% | 97.95% | 99% |

### 5.3 Robustez a Perturbaciones (VALIDADO - Sesión 39)

| Modelo | Fill Rate | JPEG Q50 | JPEG Q30 | Blur σ1 |
|--------|-----------|----------|----------|---------|
| Original 100% | 100% | 16.14% | 29.97% | 14.43% |
| Original Cropped 47% | 47% | 2.11% | 7.65% | 7.65% |
| **Warped 47%** | 47% | **0.53%** | **1.32%** | **6.06%** |
| Warped 99% | 99% | 7.34% | 16.73% | 11.35% |

**Mecanismo de Robustez (Experimento de Control Sesión 39):**
- **~75%** por reducción de información (Original Cropped es 7.6x más robusto)
- **~25% adicional** por normalización geométrica (Warped es 4x más robusto que Cropped)

### 5.4 Cross-Evaluation (VALIDADO - Sesión 39)

| Modelo | En Original | En Warped 99% | Gap |
|--------|-------------|---------------|-----|
| Original | 98.84% | 91.13% | **7.70%** |
| Warped | 95.57% | 98.73% | **3.17%** |

**Ratio: 2.4x** - El modelo warped generaliza **2.4x mejor**.

### 5.5 PFS (VALIDADO - Sesión 39)

| Métrica | Valor |
|---------|-------|
| Mean PFS | 0.487 ± 0.091 |

**Conclusión:** PFS ~0.49 ≈ chance. **NO** hay evidencia de que warping fuerce atención pulmonar.

---

## 6. CLAIMS CIENTÍFICOS (ESTADO FINAL)

### ✅ CLAIMS VÁLIDOS (usar en tesis):

1. **Predicción de landmarks:** Error **3.71 px** con ensemble 4 modelos + TTA
2. **Robustez JPEG Q50:** **30x superior** (0.53% vs 16.14%)
3. **Robustez JPEG Q30:** **23x superior** (1.32% vs 29.97%)
4. **Robustez Blur:** **2.4x superior** (6.06% vs 14.43%)
5. **Generalización cross-dataset:** **2.4x mejor** (gap 3.17% vs 7.70%)
6. **Mecanismo causal:** ~75% reducción información + ~25% normalización geométrica
7. **Estructura paramétrica landmarks:** t = 0.25, 0.50, 0.75 (100% verificado)
8. **Error mínimo teórico:** ~1.3 px (ruido de anotación)

### ❌ CLAIMS INVALIDADOS (NO usar):

1. ~~"Generaliza 11x mejor"~~ → Solo **2.4x** (corregido Sesión 39)
2. ~~"Fuerza atención pulmonar"~~ → PFS ~0.49 = chance (invalidado)
3. ~~"Elimina marcas hospitalarias"~~ → Solo las excluye/recorta
4. ~~"Resuelve domain shift externo"~~ → FedCOVIDx ~55% (NO resuelve)

### ⚠️ CLAIMS REFORMULADOS:

1. "Normalización mejora robustez" → **"Contribuye ~25% adicional a robustez"**
2. "Mejora generalización" → **"Reduce gap de generalización 2.4x"**

---

## 7. EVOLUCIÓN HISTÓRICA DEL PROYECTO

### Fase 1: Fundamentos (Sesiones 0-10)
- Reestructuración de código (constantes, módulos)
- CLI básico con Typer
- Pipeline de entrenamiento landmarks
- 169 tests creados

### Fase 2: Ensemble y Clasificador (Sesiones 11-17)
- Ensemble de 4 modelos: **3.71 px**
- Clasificador integrado: **97.76%** accuracy
- Pipeline E2E funcional

### Fase 3: Validación y CLI (Sesiones 18-27)
- CLI con múltiples comandos implementados
- Cross-evaluate, test-robustness, compare-architectures
- Tests de integración

### Fase 4: Testing y Visualización (Sesiones 28-34)
- Galería visual GradCAM
- Tests de integración adicionales
- Documentación de resultados

### Fase 5: Introspección Crítica (Sesiones 35-39) ⚠️
**AQUÍ SE IDENTIFICARON LOS PROBLEMAS:**
- Sesión 35: Identificación de sesgo metodológico en cross-evaluation
- Sesión 36: Análisis de validez de hipótesis
- Sesión 37: Implementación warp_mask()
- Sesión 38: Hipótesis de robustez cuestionada (99% fill pierde robustez)
- Sesión 39: **EXPERIMENTO DE CONTROL DEFINITIVO**
  - Reveló mecanismo causal: 75% info + 25% geo
  - Corrigió claim de generalización: 11x → 2.4x
  - Invalidó PFS (≈ chance)

### Fase 6: Production Ready (Sesiones 40-43) ✅
- Sesión 40: Documentación de claims corregidos y referencia maestra
- Sesión 41: Actualización de documentación con claims validados
- Sesión 42: Debugging y producción (pyproject.toml, CHANGELOG, seeds, 9 comandos CLI nuevos)
- Sesión 43: **CONSOLIDACIÓN FINAL Y TAG v2.0.0**
  - 553 tests pasando
  - 20 comandos CLI
  - Paquete pip instalable
  - Tag v2.0.0 production-ready

---

## 8. PROBLEMAS RESUELTOS POR SESIÓN

| Sesión | Problema | Solución |
|--------|----------|----------|
| 0-10 | Constantes duplicadas | Centralización en constants.py |
| 11-13 | Error alto (4.50 px) | Ensemble 4 modelos (3.71 px) |
| 14-17 | Pipeline incompleto | CLI con 15 comandos |
| 21-22 | Bugs en warping | Corrección de GPA y warp |
| 35 | Cross-eval inválido | Filtrar a 3 clases consistentes |
| 37 | Máscaras no warped | Implementar warp_mask() |
| 38 | Robustez desaparece 99% | Identificar mecanismo causal |
| 39 | Claims incorrectos | Experimento de control |
| 40-41 | Documentación desactualizada | Actualizar claims validados |
| 42 | Preparación producción | pyproject.toml, seeds, 9 comandos CLI nuevos |
| 43 | Consolidación final | Tag v2.0.0, 553 tests, paquete pip instalable |

---

## 9. CHECKLIST PARA NUEVAS SESIONES

### Antes de Empezar:
- [ ] Leer este documento completo
- [ ] Verificar rama Git: `feature/restructure-production`
- [ ] Revisar último estado: `git status`
- [ ] Ejecutar tests: `python -m pytest tests/ -v`

### Durante la Sesión:
- [ ] Documentar cada cambio en `docs/sesiones/SESION_XX_*.md`
- [ ] Actualizar tests si se modifica código
- [ ] NO inventar datos - verificar experimentalmente
- [ ] NO hacer claims sin evidencia

### Al Finalizar:
- [ ] Actualizar este documento si hay nuevos hallazgos
- [ ] Actualizar `RESULTADOS_EXPERIMENTALES_v2.md`
- [ ] Hacer commit con mensaje descriptivo
- [ ] Verificar que tests pasen

---

## 10. TRABAJO PENDIENTE (PRIORIZADO)

### ✅ COMPLETADO (Sesiones 40-43):
- [x] Actualizar README.md con claims corregidos
- [x] Commit final con tag de versión (v2.0.0)
- [x] pyproject.toml funcional (`pip install -e .`)
- [x] Implementar comando `gradcam` para explicabilidad
- [x] Implementar comando `analyze-errors`
- [x] Implementar comando `pfs-analysis`
- [x] Implementar comando `generate-lung-masks`
- [x] Implementar comando `optimize-margin`
- [x] 553 tests pasando
- [x] 20 comandos CLI

### Media Prioridad (Antes de Defensa):
- [ ] Revisión final de documentación LaTeX
- [ ] Evaluar en datasets externos (Montgomery, Shenzhen)

### Baja Prioridad (Post-Defensa):
- [ ] API REST con FastAPI
- [ ] Domain adaptation para cross-dataset
- [ ] Modelo de 4 clases (incluir Lung_Opacity)

---

## 11. ERRORES COMUNES A EVITAR

### ❌ NO hacer:
1. **NO usar claim "11x mejor generalización"** - Solo es 2.4x
2. **NO decir "fuerza atención pulmonar"** - PFS ~0.49 = chance
3. **NO decir "elimina marcas"** - Solo las excluye/recorta
4. **NO comparar datasets con clases diferentes** - Invalida cross-eval
5. **NO asumir que robustez viene de normalización** - 75% es por reducción de info

### ✅ SÍ hacer:
1. Usar claims validados experimentalmente
2. Citar sesión donde se validó cada claim
3. Verificar datos con scripts antes de reportar
4. Documentar cada experimento en sesiones
5. Ejecutar tests antes y después de cambios

---

## 12. CONTACTO Y RECURSOS

### Archivos Clave:
- **Resultados validados:** `docs/RESULTADOS_EXPERIMENTALES_v2.md`
- **Sesiones:** `docs/sesiones/SESION_XX_*.md`
- **Tests:** `tests/`
- **CLI:** `src_v2/cli.py`

### Datasets:
- **Principal:** `data/COVID-19_Radiography_Dataset/`
- **Warped 99%:** `outputs/full_coverage_warped_dataset/` (INVALIDADO; ver reportes/RESUMEN_DEPURACION_OUTPUTS.md)
- **Original 3 clases:** `outputs/original_3_classes/`
- **Control (cropped 47%):** `outputs/original_cropped_47/`

### Checkpoints:
- **Ensemble landmarks:** `checkpoints/session10/ensemble/`, `checkpoints/session13/`
- **Clasificadores:** `outputs/classifier_*/`

---

## 13. RESUMEN EJECUTIVO

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ESTADO DEL PROYECTO                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Progreso:        100% completo (v2.0.0 production-ready)               │
│  Tests:           553 pasando                                           │
│  CLI:             20 comandos funcionales                               │
│  Instalación:     pip install -e . → covid-landmarks                    │
│  Tag:             v2.0.0                                                │
│                                                                          │
│  RESULTADO PRINCIPAL:                                                    │
│  ├─ Error landmarks: 3.71 px (ensemble 4 modelos + TTA)                 │
│  ├─ Accuracy clasificación: 98.73%                                      │
│  ├─ Robustez JPEG: 30x superior                                         │
│  ├─ Generalización: 2.4x mejor                                          │
│  └─ Mecanismo: 75% reducción info + 25% normalización geo               │
│                                                                          │
│  CLAIMS CORREGIDOS (Sesión 39):                                         │
│  ├─ "11x generalización" → 2.4x                                         │
│  ├─ "Fuerza atención pulmonar" → NO (PFS ~0.49 = chance)                │
│  └─ "Robustez por normalización" → 75% info + 25% geo                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**FIN DEL DOCUMENTO DE REFERENCIA**

*Última actualización: Sesión 43 (2025-12-11) - Tag v2.0.0 Production Ready*
