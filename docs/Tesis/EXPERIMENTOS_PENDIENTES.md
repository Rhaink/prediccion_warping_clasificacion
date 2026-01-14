# EXPERIMENTOS PENDIENTES PARA LA TESIS

**Fecha de creación:** 16 Diciembre 2025
**Estado:** Pendientes de ejecución antes de defensa

Este documento registra los experimentos que deben ejecutarse antes de finalizar la tesis.

---

## EXPERIMENTO 1: Clasificación Binaria Neumonía vs Normal

### Descripción

Evaluar el rendimiento del sistema con una configuración binaria donde se agrupan COVID-19 y Neumonía Viral como una sola clase "Neumonía" vs la clase "Normal".

### Justificación

1. **Validez médica:** COVID-19 es técnicamente una neumonía viral causada por SARS-CoV-2
2. **Pregunta científica:** ¿El sistema distingue patología pulmonar de normalidad, o solo distingue entre tipos específicos de patología?
3. **Simplificación:** Reduce la complejidad de clasificación (2 clases vs 3 clases)
4. **Alineación con título:** El título menciona "detección de neumonía y COVID-19", lo cual se satisface con esta configuración

### Configuración Propuesta

```
Mapeo de clases:
- Clase "Neumonía" = COVID-19 + Viral_Pneumonia
- Clase "Normal" = Normal

Distribución esperada del dataset:
- Neumonía: 324 (COVID) + 200 (Viral) = 524 imágenes (52.4%)
- Normal: 475 imágenes (47.6%)
- Total: 999 imágenes
```

### Métricas a Reportar

1. **Clasificación:**
   - Accuracy
   - F1-Score
   - Precisión
   - Sensibilidad (Recall)
   - Especificidad
   - AUC-ROC

2. **Robustez:**
   - Degradación bajo JPEG Q50
   - Degradación bajo blur sigma=1

3. **Comparación:**
   - Rendimiento vs configuración de 3 clases
   - Impacto del warping en configuración binaria

### Implementación Sugerida

```bash
# Opción A: Modificar el script de entrenamiento para aceptar mapeo de clases
# Opción B: Crear dataset con estructura de 2 clases y reentrenar

# Estructura de dataset binario:
# data/dataset_binary/
# ├── train/
# │   ├── Neumonia/    # COVID + Viral_Pneumonia
# │   └── Normal/
# ├── val/
# └── test/
```

### Estado

- [x] Crear dataset con estructura binaria ✅ 16-Dic-2025
- [x] Entrenar clasificador en dataset warped (2 clases) ✅ 16-Dic-2025
- [x] Evaluar métricas de clasificación ✅ 16-Dic-2025
- [x] Evaluar robustez ✅ 16-Dic-2025
- [x] Comparar con resultados de 3 clases ✅ 16-Dic-2025
- [x] Documentar resultados ✅ 16-Dic-2025

### RESULTADOS OBTENIDOS (16-Dic-2025)

**Configuración:**
- Dataset: outputs/binary_experiment (symlinks a warped_replication_v2; INVALIDADO)
- Modelo: ResNet-18 preentrenado
- Epochs: 41 (early stopping, patience=15)
- Mejor modelo: Epoch 26

**Métricas en Test:**
| Métrica | Valor |
|---------|-------|
| Accuracy | **99.05%** |
| F1 Macro | 98.92% |
| F1 Weighted | 99.05% |

**Métricas por Clase:**
| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| Neumonía | 98.39% | 98.71% | 98.55% | 621 |
| Normal | 99.37% | 99.22% | 99.29% | 1,274 |

**Matriz de Confusión:**
```
              Pred Neumonia  Pred Normal
Neumonia          613            8
Normal             10         1264
```

**Errores:** 18 total (8 FN + 10 FP)
- 8 neumonías clasificadas como normales (1.3% de neumonías)
- 10 normales clasificados como neumonía (0.8% de normales)

**Comparación con 3 clases:**
| Configuración | Accuracy | F1 Macro |
|---------------|----------|----------|
| 3 clases (warped_96) | 99.10% | 98.45% |
| 2 clases (Neumonía vs Normal) | 99.05% | 98.92% |

**Conclusión (Clasificación):** El modelo binario logra rendimiento similar al de 3 clases, confirmando que el sistema distingue efectivamente patología pulmonar de normalidad.

### RESULTADOS DE ROBUSTEZ (16-Dic-2025)

| Perturbación | Accuracy | Error | Degradación |
|--------------|----------|-------|-------------|
| Original | 99.05% | 0.95% | --- |
| JPEG Q50 | 92.61% | 7.39% | +6.44% |
| JPEG Q30 | 84.96% | 15.04% | +14.09% |
| Blur σ=1 | 94.93% | 5.07% | +4.12% |
| Blur σ=2 | 63.69% | 36.31% | +35.36% |
| Noise σ=0.05 | 39.95% | 60.05% | +59.10% |
| Noise σ=0.10 | 35.51% | 64.49% | +63.54% |

**Comparación de Robustez con 3 clases (warped_96):**

| Perturbación | 3 clases | 2 clases | Diferencia |
|--------------|----------|----------|------------|
| JPEG Q50 | 3.06% | 6.44% | +3.38% |
| Blur σ=1 | 2.43% | 4.12% | +1.69% |

**Conclusión (Robustez):** El modelo de 3 clases es ligeramente más robusto que el binario. La diferencia sugiere que la tarea de 3 clases puede estar aprendiendo representaciones más generalizables que ayudan bajo perturbaciones.

---

## EXPERIMENTO 2: Validación Externa con Configuración Binaria Neumonía vs Normal

### Descripción

Si el experimento 1 muestra resultados prometedores, evaluar en dataset externo (si está disponible un dataset con estructura Normal vs Neumonía).

### Datasets Candidatos

1. **Montgomery County TB Dataset** - Tiene Normal vs Anormal (TB)
2. **Shenzhen Hospital TB Dataset** - Similar estructura
3. **RSNA Pneumonia Detection** - Normal vs Pneumonia

### Estado

- [ ] Identificar dataset externo apropiado
- [ ] Preparar datos
- [ ] Evaluar modelo entrenado
- [ ] Documentar resultados

---

## PRIORIDAD

| Experimento | Prioridad | Razón |
|-------------|-----------|-------|
| **Exp. 1: Neumonía vs Normal** | 🔴 ALTA | Pregunta fundamental, afecta interpretación de resultados |
| Exp. 2: Validación externa | 🟡 MEDIA | Dependiente de Exp. 1 y disponibilidad de datos |

---

## IMPACTO EN LA TESIS

Si los resultados de la clasificación binaria Neumonía vs Normal son:

### Caso A: Mejor rendimiento que 3 clases
- **Implicación:** El modelo distingue bien patología de normalidad
- **Acción:** Discutir como hallazgo positivo adicional
- **Sección afectada:** 5.2 Resultados de clasificación

### Caso B: Rendimiento similar o peor
- **Implicación:** El sistema está optimizado para distinguir COVID específicamente
- **Acción:** Discutir como limitación o característica del diseño
- **Sección afectada:** 5.5 Discusión general, 6.3 Limitaciones

---

## NOTAS

- Estos experimentos deben completarse ANTES de finalizar la redacción del Capítulo 5 (Resultados)
- Los resultados afectarán la discusión y conclusiones
- Estimar tiempo de ejecución: ~2-4 horas para experimento 1

---

*Última actualización: 16 Diciembre 2025*
