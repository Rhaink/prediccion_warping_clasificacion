# Prompt de Continuación - Sesión 55

## Objetivo: Validación Científica Externa Rigurosa

**Rama de trabajo:** `feature/external-validation`
**Estado previo:** v2.1.0 - APROBADO PARA DEFENSA
**Prioridad:** ALTA - Fortalecer claims científicos

---

## CONTEXTO CRÍTICO

### Resumen de Sesiones Anteriores

| Sesión | Hallazgo Clave |
|--------|----------------|
| 35-36 | Identificación de errores metodológicos (comparación 4 vs 3 clases) |
| 39 | Experimento de control: 75% robustez = reducción información, 25% = normalización |
| 52 | Bug CLI corregido, warped_96 alcanza 99.10% accuracy |
| 53 | Trade-off fill rate documentado, warped_96 es óptimo |
| 54 | Preparación final defensa, 655 tests pasando |

### Estado Actual de Validación Externa

**YA REALIZADO (Sesión 36-37):**
- Dataset3 (FedCOVIDx): 8,482 imágenes evaluadas
- Mejor accuracy externa: **57.5%** (vs 98.84% interno)
- Gap: **40+ puntos** - Domain shift significativo
- Conclusión documentada: "NO resuelve domain shift externo"

**PENDIENTE:**
- warped_96 (clasificador recomendado) NO evaluado externamente
- Resultados NO están en GROUND_TRUTH.json
- Documentación formal incompleta

---

## ANÁLISIS PREVIO (Resumen de 4 Agentes)

### 1. Dataset Actual

```
COVID-19 Radiography Dataset:
├── Total: 15,153 imágenes (3 clases)
├── COVID: 3,616 (23.9%)
├── Normal: 10,192 (67.3%)
├── Viral_Pneumonia: 1,345 (8.9%)
├── Formato: PNG 299x299 grayscale
├── Landmarks anotados: 957 imágenes
└── Splits: 75/15/10 estratificado, seed=42

Preprocesamiento:
├── CLAHE: clip_limit=2.0, tile_size=4
├── Resize: 224x224 (Bilinear)
├── Normalización: ImageNet (mean/std)
└── Espacio color: Grayscale + CLAHE (warped_96)
```

### 2. Dataset Externo (FedCOVIDx)

```
Dataset3 (FedCOVIDx):
├── Total test: 8,482 imágenes
├── Clases: 2 binarias (Positive/Negative)
├── Distribución: 50/50 balanceado
├── Fuentes: BIMCV (~95%), RICORD, RSNA
├── Resolución: Variable (300-3000 px)
├── Landmarks: NO disponibles (requiere predicción)
└── Ubicación: outputs/external_validation/dataset3/

Mapeo de clases (3→2):
├── P(positive) = P(COVID)
└── P(negative) = P(Normal) + P(Viral_Pneumonia)
```

### 3. Comandos CLI Disponibles

```bash
# Evaluar en dataset externo binario
python -m src_v2 evaluate-external \
    outputs/classifier_replication_v2/best_classifier.pt \
    --external-data outputs/external_validation/dataset3 \
    --output results.json

# Test de robustez
python -m src_v2 test-robustness \
    outputs/classifier_replication_v2/best_classifier.pt \
    --data-dir outputs/warped_replication_v2

# Cross-evaluation
python -m src_v2 cross-evaluate \
    model_a.pt model_b.pt \
    --data-a dataset_a --data-b dataset_b
```

### 4. Metodología Científica (Lecciones Aprendidas)

**Errores históricos corregidos:**
- Sesión 35: Comparación 4 vs 3 clases → INVÁLIDO
- Sesión 36: Claims sin evidencia (PFS ~0.49 = chance)
- Sesión 39: "11x generalización" → Solo 2.4x

**Principios de comparación justa:**
1. Mismo preprocesamiento entre datasets
2. Mismas métricas (Accuracy + F1 + Matriz confusión)
3. Documentar diferencias inevitables
4. Usar experimentos de control
5. NO hardcodear valores en tests

---

## TAREAS PARA SESIÓN 55

### ALTA PRIORIDAD

#### Tarea 1: Evaluar warped_96 en Dataset3 (FedCOVIDx)
```bash
# Crear rama
git checkout -b feature/external-validation

# Evaluar clasificador recomendado
python -m src_v2 evaluate-external \
    outputs/classifier_replication_v2/best_classifier.pt \
    --external-data outputs/external_validation/dataset3 \
    --output outputs/external_validation/warped_96_on_dataset3.json
```

**Resultado esperado:** ~55-60% accuracy (similar a otros modelos)

#### Tarea 2: Documentar en GROUND_TRUTH.json
```json
{
  "external_validation": {
    "dataset3_fedcovidx": {
      "n_samples": 8482,
      "best_internal_model": {
        "name": "warped_96",
        "internal_accuracy": 99.10,
        "external_accuracy": "PENDIENTE",
        "gap": "PENDIENTE"
      },
      "conclusion": "Domain shift > methodology improvement",
      "recommendation": "Requires domain adaptation for cross-dataset use"
    }
  }
}
```

#### Tarea 3: Actualizar RESULTADOS_EXPERIMENTALES_v2.md
Agregar sección:
```markdown
## 5. Validación Externa (Sesión 55)

### 5.1 Dataset3 (FedCOVIDx)
| Modelo | Accuracy Interna | Accuracy Externa | Gap |
|--------|------------------|------------------|-----|
| Original (ResNet-18) | 98.84% | 57.50% | 41.3% |
| Warped 99% | 98.73% | ~55% | ~43% |
| **Warped 96%** | **99.10%** | **PENDIENTE** | **PENDIENTE** |

### 5.2 Conclusión
La normalización geométrica:
- **SÍ mejora**: Generalización within-domain
- **NO resuelve**: Domain shift between-domain
```

### MEDIA PRIORIDAD

#### Tarea 4: Análisis de Causas del Domain Shift
Documentar:
1. Diferencias de equipos/protocolos
2. Diferencias de población
3. Calidad de landmarks predichos en datos externos
4. Impacto del mapeo 3→2 clases

#### Tarea 5: Actualizar README.md
Agregar sección "Limitations and External Validation":
```markdown
### External Validation Results

| Dataset | Type | Accuracy | Notes |
|---------|------|----------|-------|
| Internal (COVID-19 Radiography) | 3 classes | 99.10% | Recommended model |
| External (FedCOVIDx) | Binary | ~55% | Domain shift |

> **Important:** The geometric normalization improves within-domain
> generalization but does not resolve cross-domain shift. Domain
> adaptation techniques are required for deployment on new datasets.
```

### BAJA PRIORIDAD (Si hay tiempo)

#### Tarea 6: Investigar Otros Datasets
- Montgomery County TB (~138 imágenes)
- Shenzhen Hospital TB
- COVIDx (clases idénticas)

---

## CHECKLIST DE VALIDACIÓN CIENTÍFICA

Antes de reportar cualquier resultado:

```
DATOS:
[ ] Verificar n_samples coincide con documentación
[ ] Reproducibilidad con seed=42
[ ] No valores hardcodeados
[ ] Tests fallan si data ausente (no defaults)

METODOLOGÍA:
[ ] Mismo preprocesamiento (CLAHE, resize, normalización)
[ ] Mapeo de clases documentado (3→2)
[ ] Métricas completas (Accuracy + F1 + Matriz)
[ ] Limitaciones explícitas

REPORTING:
[ ] Claims contrastados con evidencia
[ ] Domain shift documentado honestamente
[ ] NO decir "modelo falla" - decir "requiere adaptation"
```

---

## ARCHIVOS RELEVANTES

### Para Modificar
| Archivo | Cambio |
|---------|--------|
| `GROUND_TRUTH.json` | Agregar sección external_validation |
| `docs/RESULTADOS_EXPERIMENTALES_v2.md` | Agregar sección 5 |
| `README.md` | Agregar limitaciones de generalización |

### Para Consultar
| Archivo | Contenido |
|---------|-----------|
| `outputs/external_validation/baseline_results.json` | Resultados previos (12 modelos) |
| `outputs/external_validation/dataset3/` | Dataset FedCOVIDx procesado |
| `scripts/evaluate_external_baseline.py` | Script de evaluación |
| `src_v2/cli.py` (L2685-2942) | Comando evaluate-external |

### Nuevos a Crear
| Archivo | Propósito |
|---------|-----------|
| `outputs/external_validation/warped_96_results.json` | Resultados warped_96 |
| `docs/sesiones/SESION_55_VALIDACION_EXTERNA.md` | Documentación sesión |

---

## RESULTADOS PREVIOS (Referencia)

### Evaluación FedCOVIDx (Sesión 36-37)

| Modelo | Arquitectura | Dataset | Accuracy | Gap |
|--------|--------------|---------|----------|-----|
| ResNet-18 | ResNet-18 | Original | **57.50%** | 38.3% |
| ResNet-50 | ResNet-50 | Original | 56.50% | 37.3% |
| DenseNet-121 | DenseNet | Warped | 56.26% | 33.3% |
| VGG-16 | VGG | Warped | 56.43% | 34.2% |
| ResNet-18 | ResNet-18 | Warped | 50.21% | 35.2% |

**Observación:** Todos los modelos ~55% (cercano a random para binario).

---

## COMANDO DE INICIO

```
Sesión 55: Validación Científica Externa

CONTEXTO:
- El proyecto está APROBADO PARA DEFENSA (v2.1.0)
- El clasificador warped_96 (99.10% accuracy) NO ha sido evaluado externamente
- Ya existe evaluación previa en FedCOVIDx (~55% accuracy para otros modelos)
- El objetivo es documentar formalmente las limitaciones de generalización

TAREAS (en orden):
1. Crear rama feature/external-validation
2. Evaluar warped_96 en Dataset3 (FedCOVIDx)
3. Documentar resultados en GROUND_TRUTH.json
4. Actualizar RESULTADOS_EXPERIMENTALES_v2.md con sección de validación externa
5. Actualizar README.md con limitaciones honestas
6. Crear documentación de sesión

METODOLOGÍA:
- Usar EXACTAMENTE el mismo preprocesamiento
- Mapeo 3→2 clases: P(pos)=P(COVID), P(neg)=P(Normal)+P(Viral)
- Reportar: Accuracy, F1, AUC-ROC, Matriz de confusión
- Documentar domain shift como hallazgo válido (no como fracaso)

ARCHIVOS CLAVE:
- Clasificador: outputs/classifier_replication_v2/best_classifier.pt
- Dataset externo: outputs/external_validation/dataset3/
- Resultados previos: outputs/external_validation/baseline_results.json
- CLI: python -m src_v2 evaluate-external

EXPECTATIVA:
- Accuracy ~55-60% (similar a modelos previos)
- Esto es un HALLAZGO VÁLIDO que documenta limitaciones
- NO es un fracaso - es honestidad científica
```

---

## NOTAS FINALES

### Por qué es importante esta validación

1. **Honestidad científica:** Documentar limitaciones fortalece la tesis
2. **Expectativas realistas:** El modelo NO generaliza sin adaptation
3. **Contribución clara:** "Mejora within-domain, no cross-domain"
4. **Trabajo futuro:** Abre camino para domain adaptation

### Qué NO hacer

- NO esperar que warped_96 mejore significativamente en FedCOVIDx
- NO interpretar ~55% como "fracaso" - es domain shift esperado
- NO modificar preprocesamiento para "mejorar" resultados externos
- NO omitir estos resultados de la documentación

### Qué SÍ hacer

- Documentar resultados honestamente
- Explicar causas del domain shift
- Proponer soluciones (domain adaptation) como trabajo futuro
- Usar esto como evidencia de rigor metodológico

---

**Fin del Prompt de Continuación - Sesión 55**
