# Lista de Entregables - Estancia INAOE

**Fecha:** 2026-01-28
**Proyecto:** Normalización y alineación automática de landmarks pulmonares
**Estudiante:** Rafael Alejandro Cruz Ovando

---

## Según Plan de Trabajo Firmado

El plan (inaoe.pdf, página 3) especifica 4 entregables:

1. ✅ Prototipo funcional del modelo con código documentado
2. ✅ Checkpoints y configuraciones de entrenamiento de cada fase experimental
3. ✅ Reporte de resultados con análisis cuantitativo y visualizaciones representativas
4. ✅ Documentación técnica

---

## 1. Prototipo Funcional (Código Fuente)

### Directorio completo: `src_v2/`
**Tamaño:** 2.1 MB
**Contenido:** Sistema completo de landmarks + clasificación

```
src_v2/
├── models/
│   ├── resnet_landmark.py      # ResNet-18 + Coordinate Attention
│   ├── losses.py               # Wing Loss, Symmetry, Alignment
│   ├── classifier.py           # Clasificador COVID-19
│   └── hierarchical.py         # (experimental)
├── training/
│   ├── trainer.py              # Entrenamiento 2 fases
│   └── callbacks.py            # Early stopping, checkpointing
├── data/
│   ├── dataset.py              # LandmarkDataset
│   ├── transforms.py           # CLAHE, TTA, aumentación
│   └── utils.py                # Splits, normalización
├── processing/
│   ├── gpa.py                  # Generalized Procrustes Analysis
│   └── warp.py                 # Warping afín por partes
├── evaluation/
│   ├── metrics.py              # Métricas de landmarks
│   └── ensemble.py             # Evaluación ensemble
├── visualization/              # (opcional, 11 archivos)
├── gui/                        # (opcional, 6 archivos)
├── utils/                      # Geometría
├── cli.py                      # Interfaz CLI principal
├── constants.py                # Constantes del proyecto
└── __main__.py                 # Entry point
```

**Archivos críticos (pueden entregarse solos si el directorio es muy grande):**
- `models/resnet_landmark.py` (implementación arquitectura)
- `models/losses.py` (pérdidas geométricas)
- `training/trainer.py` (entrenamiento 2 fases)
- `processing/gpa.py` (GPA)
- `processing/warp.py` (warping)
- `evaluation/metrics.py` (métricas)
- `cli.py` (interfaz)

---

## 2. Checkpoints y Configuraciones

### A. Checkpoints del Ensemble (4 modelos)
**Tamaño total:** 184 MB (46 MB cada uno)

| Archivo | Tamaño | Semilla |
|---------|--------|---------|
| `checkpoints/session10/ensemble/seed123/final_model.pt` | 46 MB | 123 |
| `checkpoints/session13/seed321/final_model.pt` | 46 MB | 321 |
| `checkpoints/repro_split111/session14/seed111/final_model.pt` | 46 MB | 111 |
| `checkpoints/repro_split666/session16/seed666/final_model.pt` | 46 MB | 666 |

**Nota:** Estos 4 modelos son los que conforman el ensemble reportado (3.61 px error).

### B. Configuraciones JSON
**Tamaño total:** ~10 KB

| Archivo | Tamaño | Descripción |
|---------|--------|-------------|
| `configs/ensemble_best.json` | 336 B | Config ensemble (TTA+CLAHE) |
| `configs/landmarks_train_base.json` | 439 B | Hiperparámetros entrenamiento |
| `configs/warping_best.json` | 544 B | Parámetros warping óptimos |
| `configs/classifier_warped_base.json` | 255 B | Config clasificador |

---

## 3. Reporte de Resultados

### A. Reporte LaTeX (fuente)
**Archivo:** `docs/estancia/REPORTE_ESTANCIA_INAOE.tex`
**Tamaño:** 33 KB
**Contenido:** 1,021 líneas con análisis completo

### B. Reporte PDF (compilado)
**Archivo:** `REPORTE_ESTANCIA_INAOE.pdf` (generar con pdflatex)
**Tamaño estimado:** ~350 KB
**Secciones:**
- Datos generales
- Resumen ejecutivo (métricas principales)
- Introducción (contexto, landmarks)
- Metodología implementada (datos, modelo, entrenamiento, warping)
- Resultados y análisis (métricas por landmark/categoría, CV)
- Entregables producidos
- Cronograma ejecutado
- Conclusiones

**Comando para compilar:**
```bash
cd docs/estancia
pdflatex REPORTE_ESTANCIA_INAOE.tex
pdflatex REPORTE_ESTANCIA_INAOE.tex  # 2 veces para referencias
```

---

## 4. Documentación Técnica

### A. Archivos principales

| Archivo | Tamaño | Descripción |
|---------|--------|-------------|
| `CLAUDE.md` | 11 KB | Guía principal del proyecto |
| `GROUND_TRUTH.json` | 17 KB | Valores validados experimentalmente |
| `docs/estancia/AUDITORIA_REPORTE_INAOE.md` | ~10 KB | Auditoría del reporte |

### B. Documentación extendida (opcional)

```
docs/
├── manual/
│   └── README.md               # Manual de uso
└── Tesis/anexos/
    └── inaoe.pdf               # Plan firmado (referencia)
```

---

## Resumen de Tamaños

| Entregable | Tamaño | Archivos |
|-----------|--------|----------|
| 1. Código (`src_v2/`) | 2.1 MB | ~40 archivos Python |
| 2A. Checkpoints (4 modelos) | 184 MB | 4 archivos .pt |
| 2B. Configs | 10 KB | 4 archivos .json |
| 3. Reporte PDF | ~350 KB | 1 archivo PDF |
| 4. Documentación | 38 KB | 3 archivos principales |
| **TOTAL** | **~186.5 MB** | |

---

## Opciones de Entrega

### Opción 1: Entrega Completa (Recomendada)
**Directorio:** `entregables_inaoe/`
**Contenido:**
```
entregables_inaoe/
├── 01_codigo/
│   └── src_v2/                 # 2.1 MB
├── 02_modelos/
│   ├── seed123_final_model.pt  # 46 MB
│   ├── seed321_final_model.pt  # 46 MB
│   ├── seed111_final_model.pt  # 46 MB
│   └── seed666_final_model.pt  # 46 MB
├── 03_configs/
│   ├── ensemble_best.json
│   ├── landmarks_train_base.json
│   ├── warping_best.json
│   └── classifier_warped_base.json
├── 04_reporte/
│   ├── REPORTE_ESTANCIA_INAOE.pdf  # 350 KB
│   └── REPORTE_ESTANCIA_INAOE.tex  # 33 KB (fuente)
├── 05_documentacion/
│   ├── CLAUDE.md
│   ├── GROUND_TRUTH.json
│   └── AUDITORIA_REPORTE_INAOE.md
└── README.md                   # Instrucciones de uso
```

**Archivo comprimido:** `entregables_inaoe.tar.gz` (~150 MB comprimido)

### Opción 2: Entrega Ligera (Sin checkpoints grandes)
Si los checkpoints son muy pesados para enviar:
- Entregar solo configs + rutas de descarga
- Checkpoints disponibles en repositorio Git LFS o Drive
- **Total:** ~2.5 MB

---

## Comandos para Preparar Entrega

```bash
# 1. Crear estructura
mkdir -p entregables_inaoe/{01_codigo,02_modelos,03_configs,04_reporte,05_documentacion}

# 2. Copiar código
cp -r src_v2/ entregables_inaoe/01_codigo/

# 3. Copiar checkpoints (renombrados)
cp checkpoints/session10/ensemble/seed123/final_model.pt entregables_inaoe/02_modelos/seed123_final_model.pt
cp checkpoints/session13/seed321/final_model.pt entregables_inaoe/02_modelos/seed321_final_model.pt
cp checkpoints/repro_split111/session14/seed111/final_model.pt entregables_inaoe/02_modelos/seed111_final_model.pt
cp checkpoints/repro_split666/session16/seed666/final_model.pt entregables_inaoe/02_modelos/seed666_final_model.pt

# 4. Copiar configs
cp configs/ensemble_best.json entregables_inaoe/03_configs/
cp configs/landmarks_train_base.json entregables_inaoe/03_configs/
cp configs/warping_best.json entregables_inaoe/03_configs/
cp configs/classifier_warped_base.json entregables_inaoe/03_configs/

# 5. Compilar y copiar reporte
cd docs/estancia
pdflatex REPORTE_ESTANCIA_INAOE.tex
pdflatex REPORTE_ESTANCIA_INAOE.tex
cd ../..
cp docs/estancia/REPORTE_ESTANCIA_INAOE.pdf entregables_inaoe/04_reporte/
cp docs/estancia/REPORTE_ESTANCIA_INAOE.tex entregables_inaoe/04_reporte/

# 6. Copiar documentación
cp CLAUDE.md entregables_inaoe/05_documentacion/
cp GROUND_TRUTH.json entregables_inaoe/05_documentacion/
cp docs/estancia/AUDITORIA_REPORTE_INAOE.md entregables_inaoe/05_documentacion/

# 7. Comprimir
tar -czf entregables_inaoe.tar.gz entregables_inaoe/
```

---

## Verificación Final

Antes de entregar, verificar:

- [ ] Código `src_v2/` completo y documentado
- [ ] 4 checkpoints del ensemble presentes
- [ ] 4 configuraciones JSON incluidas
- [ ] PDF del reporte compilado correctamente
- [ ] CLAUDE.md y GROUND_TRUTH.json incluidos
- [ ] README con instrucciones de uso
- [ ] Tamaño total razonable (~150 MB comprimido)

---

**Preparado por:** Claude Code
**Fecha:** 2026-01-28
