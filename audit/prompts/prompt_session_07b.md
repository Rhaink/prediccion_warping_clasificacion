# Prompt para Sesion 7b: Visualizacion - Error Analysis

Estoy realizando una auditoria academica de mi proyecto de tesis de maestria (clasificacion de radiografias de torax mediante deep learning). El proyecto esta en /home/donrobot/Projects/prediccion_warping_clasificacion/.

IMPORTANTE: Lee primero referencia_auditoria.md en la raiz del proyecto - contiene el protocolo COMPLETO que debes seguir A RAJA TABLA.

## ESTADO ACTUAL DE LA AUDITORIA

### Sesiones Completadas

| Sesion | Modulo                               | Estado        | Hallazgos                          |
|--------|--------------------------------------|---------------|-----------------------------------|
| 0      | Mapeo del proyecto                   | Completada    | 0🔴, 4🟠, 5🟡, 4⚪                 |
| 1      | Configuracion y utilidades           | APROBADO      | 0🔴, 0🟠, 1🟡, 4⚪                 |
| 2      | Gestion de datos (data/)             | APROBADO      | 0🔴, 2🟠 resueltos, 5🟡, 8⚪       |
| 3a     | Funciones de perdida (losses.py)     | APROBADO      | 0🔴, 1🟠 resuelto, 4🟡, 10⚪       |
| 3b     | ResNet Landmark (resnet_landmark.py) | APROBADO      | 0🔴, 0🟠, 2🟡, 15⚪                |
| 3c     | Clasificador (classifier.py)         | APROBADO      | 0🔴, 1🟠 resuelto, 2🟡, 15⚪       |
| 3d     | Jerarquico (hierarchical.py)         | APROBADO      | 0🔴, 0🟠, 2🟡, 20⚪ (experimental) |
| 4a     | Trainer (trainer.py)                 | APROBADO      | 0🔴, 0🟠, 5🟡, 18⚪                |
| 4b     | Callbacks (callbacks.py)             | APROBADO      | 0🔴, 0🟠, 1🟡, 18⚪                |
| 5a     | GPA (gpa.py)                         | APROBADO      | 0🔴, 1🟠 resuelto, 1🟡, 23⚪       |
| 5b     | Warping (warp.py)                    | APROBADO      | 0🔴, 0🟠, 0🟡, 26⚪                |
| 6      | Metricas (metrics.py)                | APROBADO      | 0🔴, 0🟠, 0🟡, 29⚪                |
| 7a     | Grad-CAM (gradcam.py)                | APROBADO      | 0🔴, 0🟠, 0🟡, 36⚪                |

### Hallazgos 🟠 Mayores PENDIENTES (de Sesion 0)

1. **M1:** Remover claim incorrecto sobre PFS en README.md
2. **M3:** Anadir seccion de sesgos y disclaimer medico
3. **M4:** Documentar justificacion del margen optimo 1.05

### Archivos de Referencia

- Protocolo: referencia_auditoria.md
- Plan maestro: audit/MASTER_PLAN.md
- Sesion anterior: audit/sessions/session_07a_gradcam.md
- Hallazgos: audit/findings/consolidated_issues.md

---

## SESION 7b: VISUALIZACION - ERROR ANALYSIS

### NOTA: Division del Modulo visualization/

El modulo visualization/ tiene **1529 lineas totales**, excediendo el limite de 500 lineas por sesion (§4.3). Se divide en sub-sesiones:

| Sesion | Archivo | Lineas | Estado |
|--------|---------|--------|--------|
| 7a | gradcam.py + __init__.py | 376 + 45 = 421 | APROBADO |
| 7b | error_analysis.py | 478 | **ESTA SESION** |
| 7c | pfs_analysis.py | 630 | Pendiente |

### Archivos a Auditar

```
src_v2/visualization/
└── error_analysis.py      (478 lineas) ← ESTA SESION
Total: 478 lineas
```

### Contexto Tecnico de error_analysis.py

Este archivo implementa **analisis de errores de clasificacion** para investigar patrones de fallo del modelo:

**Componentes principales:**

| Componente | Lineas | Descripcion |
|------------|--------|-------------|
| `ErrorDetail` dataclass | 27-37 | Detalles de un error individual |
| `ErrorSummary` dataclass | 39-50 | Estadisticas resumen de errores |
| `ErrorAnalyzer` class | 53-337 | Clase principal para analizar errores |
| `analyze_classification_errors()` | 340-376 | Funcion de alto nivel para analisis |
| `create_error_visualizations()` | 379-478 | Crear visualizaciones de errores |

**Clase ErrorAnalyzer:**
- Metodos: `add_prediction()`, `add_batch()`, `get_summary()`, `get_top_errors()`, `get_errors_by_pair()`, `save_reports()`
- Genera reportes JSON y CSV
- Calcula matriz de confusion
- Trackea confianza de errores vs correctos

**Importancia academica:** Error analysis es fundamental para:
- Entender patrones de fallo del modelo
- Identificar clases problematicas (COVID vs Viral_Pneumonia)
- Documentar analisis de errores en la tesis
- Generar visualizaciones para el jurado

### TESTS EXISTENTES

| Archivo | Lineas | Descripcion |
|---------|--------|-------------|
| tests/test_visualization.py | 166-309 | TestErrorAnalysisModule con ~15 tests |

Tests especificos:
- test_error_analyzer_initialization
- test_add_prediction_correct
- test_add_prediction_error
- test_add_batch
- test_get_summary
- test_get_top_errors
- test_save_reports_creates_files
- test_confusion_matrix_correct

### Dependencias

- **Usa:** torch, numpy, PIL, matplotlib, json, csv, logging, dataclasses
- **Es usado por:** CLI (analyze-errors), pfs_analysis.py (integracion)
- **Impacto:** Errores aqui afectarian todo el analisis de errores del modelo

---

## LECCIONES APRENDIDAS DE SESIONES ANTERIORES (CUMPLIR ESTRICTAMENTE)

### De Sesion 7a (verificacion con multiples agentes):

1. **Verificacion exhaustiva:** 6 agentes confirmaron cumplimiento al 100%
2. **Conteo verificado:** 6+12+8+10 = 36⚪ verificado manualmente
3. **Sin desviaciones:** Ninguna desviacion identificada

### De Sesion 5b:

1. **Regla §5.1 CRITICA:** Si la solucion propuesta dice "Opcional", el hallazgo es ⚪ (Nota), NO 🟡 (Menor)
2. **Desviaciones:** Documentar CUALQUIER desviacion detectada y corregida

### De Sesiones anteriores:

1. **§4.4 paso 1 OBLIGATORIO:** Incluir seccion "Contexto de Sesion Anterior"
2. **Conteo MANUAL OBLIGATORIO:** Contar CADA hallazgo antes de reportar totales
3. **Protocolo §7.2 OBLIGATORIO:** Solicitud de validacion antes de ejecutar comandos
4. **Orden de Auditores §3.2 (ESTRICTO):**
   1. Arquitecto de Software
   2. Revisor de Codigo
   3. Especialista en Documentacion
   4. Ingeniero de Validacion
   5. Auditor Maestro (con TABLA de veredicto)

---

## AREAS DE ENFOQUE ESPECIAL

Dado que error analysis es central para entender el comportamiento del modelo:

1. **Dataclasses ErrorDetail y ErrorSummary:**
   - Campos correctamente tipados
   - Uso de field(default_factory=...) para mutables
   - Conversion a dict para serializacion

2. **ErrorAnalyzer:**
   - Validacion de labels en rango valido
   - Manejo de tensores 1D y 2D en add_prediction()
   - Matriz de confusion actualizada correctamente
   - Division por cero manejada en error_rate

3. **Reportes (JSON/CSV):**
   - Encoding UTF-8 para caracteres especiales
   - Paths manejados con pathlib
   - mkdir con parents=True, exist_ok=True

4. **Visualizaciones:**
   - plt.close() despues de savefig (previene memory leaks)
   - Manejo de casos vacios (sin errores/sin correctos)
   - Colores apropiados para clases

5. **Logging:**
   - Uso de logger.info() para reportar archivos guardados
   - Logger del modulo configurado correctamente

---

## INSTRUCCIONES

1. Lee referencia_auditoria.md completo
2. Lee audit/sessions/session_07a_gradcam.md para contexto y formato
3. Sigue el flujo §4.4 paso a paso
4. **INCLUYE seccion "Contexto de Sesion Anterior"**
5. Aplica perspectiva de los 5 auditores EN ORDEN §3.2
6. **ANTES de clasificar:** Si la solucion es "Opcional" → usar ⚪, no 🟡
7. **ANTES de reportar conteo:** Contar manualmente cada severidad en las tablas
8. En ⚪: Listar CADA hallazgo separadamente (no combinar)
9. Documenta hallazgos con severidad calibrada segun §5.1
10. Veredicto en formato TABLA
11. Solicita validacion con protocolo §7.2 ANTES de ejecutar tests
12. Crea documento audit/sessions/session_07b_error_analysis.md
13. **ANTES del commit:** Verificar que conteo coincide con tablas
14. Incluye seccion "Registro de Commit" y "Desviaciones de Protocolo"
15. Haz commit: `audit(session-7b): auditoria error_analysis.py`

---

## CHECKLIST PRE-COMMIT (OBLIGATORIO)

Antes de hacer commit, verificar:
- [ ] Seccion "Contexto de Sesion Anterior" incluida
- [ ] Plantilla §6 cumple 14+ secciones
- [ ] Clasificacion §5.1 correcta (no "Opcional" en 🟡)
- [ ] Conteo manual coincide con hallazgos listados en tablas
- [ ] Cada ⚪ listado separadamente (no combinados)
- [ ] Flujo §4.4 completo (9/9 pasos)
- [ ] Orden de auditores §3.2 respetado (5/5 en orden)
- [ ] Protocolo §7.2 aplicado en validaciones
- [ ] Seccion "Registro de Commit" incluida
- [ ] Seccion "Desviaciones de Protocolo" incluida

---

## PROGRESO GLOBAL

| Metrica | Valor |
|---------|-------|
| Modulos completados | 11/12 |
| Modulo models/ | COMPLETADO (4/4) |
| Modulo training/ | COMPLETADO (2/2) |
| Modulo processing/ | COMPLETADO (2/2) |
| Modulo evaluation/ | COMPLETADO (1/1) |
| Modulo visualization/ | 1/3 (gradcam.py aprobado) |
| Hallazgos totales | 🔴:0 \| 🟠:9 (6 resueltos, 3 pendientes) \| 🟡:28 \| ⚪:226 |
| Objetivo | Completar visualization/ (esta sesion: error_analysis.py) |

---

## AL FINALIZAR ESTA SESION

Con error_analysis.py completado:
- Modulo visualization/ estara 2/3 completado
- Proxima sesion: 7c (visualization/ - pfs_analysis.py)
- Progreso: 11/12 modulos (visualization parcial)

---

## REFERENCIA: ESTRUCTURA DE ErrorAnalyzer

```python
class ErrorAnalyzer:
    def __init__(self, class_names: List[str])
    def add_prediction(output, label, image_path) -> bool
    def add_batch(outputs, labels, image_paths) -> int
    def get_summary() -> ErrorSummary
    def get_top_errors(k, by, descending) -> List[ErrorDetail]
    def get_errors_by_pair(true_class, predicted_class) -> List[ErrorDetail]
    def save_reports(output_dir, save_json, save_csv) -> Dict[str, Path]
```

Verificar que cada metodo:
- Tiene docstring completo
- Tiene type hints
- Maneja edge cases
- Esta cubierto por tests

---

¿Listo para comenzar con la Sesion 7b?
