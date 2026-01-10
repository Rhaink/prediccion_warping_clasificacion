# Plan de Trabajo Futuro - Post Sesión 54

**Fecha de creación:** 2025-12-14
**Última actualización:** 2025-12-14 (Análisis multi-agente)
**Estado del proyecto:** v2.1.0 - APROBADO PARA DEFENSA
**Última sesión completada:** 54

---

## Resumen Ejecutivo

| Prioridad | Tarea | Tiempo Est. | Impacto |
|-----------|-------|-------------|---------|
| **1 (ALTA)** | Validación Externa | 8-12h | Fortalece claims científicos |
| **2 (MEDIA)** | GUI con Gradio | 12-16h | Demo visual para defensa |
| **3 (BAJA)** | Mejorar Landmarks | 20+h | Mejora técnica opcional |

---

## Prioridades de Trabajo (En Orden)

### PRIORIDAD 1: Validación Científica Externa (Opción B)
**Estado:** PENDIENTE - Próxima sesión
**Rama:** `feature/external-validation`

**Objetivo:** Fortalecer claims científicos mediante evaluación en datasets externos.

**Justificación:**
- Los resultados actuales son válidos pero limitados al dataset COVID-19 Radiography
- La comunidad científica espera validación en múltiples datasets
- Ya hay evaluación preliminar en FedCOVIDx (~55% accuracy) que debe documentarse formalmente
- El clasificador recomendado (warped_96) NO ha sido evaluado externamente

**Tareas:**
1. Evaluar warped_96 en Dataset3 (FedCOVIDx)
2. Investigar y preparar datasets adicionales (Montgomery, Shenzhen, COVIDx)
3. Asegurar preprocesamiento comparable entre datasets
4. Documentar limitaciones de generalización honestamente
5. Actualizar GROUND_TRUTH.json con resultados externos

**Metodología requerida:**
- Comparación justa: mismo preprocesamiento, mismas métricas
- Documentar diferencias entre datasets (resolución, origen, clases)
- Análisis de causas de domain shift
- Método científico estricto

---

### PRIORIDAD 2: Interfaz Gráfica (GUI)
**Estado:** PLANIFICADO - Después de validación externa
**Rama:** `feature/gradio-gui`

**Objetivo:** Crear interfaz visual para demostración y uso del sistema.

**Framework recomendado:** Gradio (12-16 horas estimadas)

**Funcionalidades planificadas:**
1. **Predicción de landmarks:**
   - Cargar imagen → ver landmarks superpuestos
   - Comparar con ground truth si disponible

2. **Visualización de warping:**
   - Before/after de normalización geométrica
   - Mostrar triangulación Delaunay

3. **Clasificación:**
   - Probabilidades por clase (COVID, Normal, Viral Pneumonia)
   - Indicador de confianza

4. **Explicabilidad:**
   - Grad-CAM visualization
   - PFS (Pulmonary Focus Score)

**Prerequisitos:**
- Verificar que CLI funciona correctamente
- Documentar API interna
- Tests de integración para GUI

---

### PRIORIDAD 3: Mejorar Precisión de Landmarks
**Estado:** OPCIONAL - Post-defensa
**Rama:** `feature/improved-landmarks`

**Objetivo:** Reducir error de landmarks de 3.71 px hacia el límite teórico de 1.3 px.

**Estado actual:**
- Error: 3.71 px (ensemble 4 modelos + TTA)
- Límite teórico: 1.3 px (ruido de anotación)
- Margen de mejora: 65%

**Posibles mejoras:**
1. Backbone más grande (ResNet-50, EfficientNet-B4)
2. Mayor resolución de entrada (448x448 en lugar de 224x224)
3. Más datos de entrenamiento
4. Arquitecturas especializadas (HRNet, Hourglass)

**Impacto esperado:**
- Menor mejora en clasificación (ya es 99.10%)
- Mayor precisión en warping
- Mejor calidad visual de normalización

---

## Cronograma Sugerido

| Sesión | Tarea | Prioridad |
|--------|-------|-----------|
| 55-57 | Validación externa completa | ALTA |
| 58-60 | GUI con Gradio | MEDIA |
| 61+ | Mejora de landmarks (opcional) | BAJA |

---

## Notas Importantes

### Sobre validación externa:
- **NO** esperar que el modelo generalice perfectamente a datasets externos
- El objetivo es **documentar honestamente** las limitaciones
- FedCOVIDx ya mostró ~55% accuracy - esto es un hallazgo válido
- El claim "NO resuelve domain shift externo" ya está documentado

### Sobre GUI:
- Debe ser complemento, no reemplazo del CLI
- Enfocarse en demostración visual para defensa de tesis
- No sobre-engineerear - MVP funcional es suficiente

### Sobre mejora de landmarks:
- Solo perseguir si hay tiempo después de defensa
- El modelo actual (3.71 px) ya es suficiente para clasificación
- Mayor impacto en aplicaciones que requieran alta precisión geométrica

---

## Referencias

- Estado actual: `docs/sesiones/SESION_54_PREPARACION_DEFENSA.md`
- Resultados validados: `GROUND_TRUTH.json` v2.1.0
- Evaluación externa previa: `outputs/external_validation/`
- CLI disponible: `python -m src_v2 --help`

---

**Próximo paso:** Crear rama `feature/external-validation` y comenzar Sesión 55.
