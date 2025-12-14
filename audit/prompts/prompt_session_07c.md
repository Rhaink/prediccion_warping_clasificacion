# Prompt para Sesion 7c: Visualizacion - PFS Analysis

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
| 7b     | Error Analysis (error_analysis.py)   | APROBADO      | 0🔴, 0🟠, 0🟡, 42⚪                |

### Hallazgos 🟠 Mayores PENDIENTES (de Sesion 0)

1. **M1:** Remover claim incorrecto sobre PFS en README.md
2. **M3:** Anadir seccion de sesgos y disclaimer medico
3. **M4:** Documentar justificacion del margen optimo 1.05

### Archivos de Referencia

- Protocolo: referencia_auditoria.md
- Plan maestro: audit/MASTER_PLAN.md
- Sesion anterior: audit/sessions/session_07b_error_analysis.md
- Hallazgos: audit/findings/consolidated_issues.md

---

## SESION 7c: VISUALIZACION - PFS ANALYSIS

### NOTA: Division del Modulo visualization/

El modulo visualization/ tiene **1529 lineas totales**, excediendo el limite de 500 lineas por sesion (§4.3). Se divide en sub-sesiones:

| Sesion | Archivo | Lineas | Estado |
|--------|---------|--------|--------|
| 7a | gradcam.py + __init__.py | 376 + 45 = 421 | APROBADO |
| 7b | error_analysis.py | 478 | APROBADO |
| 7c | pfs_analysis.py | 631 | **ESTA SESION** |

**NOTA:** pfs_analysis.py excede 500 lineas (631), pero es el ultimo archivo del modulo visualization/ y mantiene cohesion funcional. Se auditara completo en una sola sesion.

### Archivos a Auditar

```
src_v2/visualization/
└── pfs_analysis.py      (631 lineas) ← ESTA SESION
Total: 631 lineas
```

### Contexto Tecnico de pfs_analysis.py

Este archivo implementa **analisis de Pulmonary Focus Score (PFS)** para evaluar si el modelo enfoca su atencion en regiones pulmonares:

**Formula PFS:**
```
PFS = sum(heatmap * mask) / sum(heatmap)
- 1.0 = Modelo enfoca completamente en pulmones (ideal)
- 0.5 = Atencion igual en pulmones y no-pulmones
- <0.5 = Modelo enfoca mas en areas no-pulmonares (preocupante)
```

**Componentes principales:**

| Componente | Lineas | Descripcion |
|------------|--------|-------------|
| Docstring modulo | 1-10 | Formula PFS y significado de valores |
| Imports | 12-24 | json, logging, dataclasses, pathlib, typing, numpy, torch, PIL |
| `PFSResult` dataclass | 29-42 | Resultado PFS para una imagen |
| `PFSSummary` dataclass | 45-63 | Estadisticas resumen de analisis PFS |
| `PFSAnalyzer` class | 66-232 | Clase principal para analisis PFS |
| `load_lung_mask()` | 235-254 | Cargar mascara pulmonar desde archivo |
| `find_mask_for_image()` | 257-301 | Buscar mascara correspondiente a imagen |
| `generate_approximate_mask()` | 304-332 | Generar mascara rectangular aproximada |
| `run_pfs_analysis()` | 335-443 | Ejecutar analisis PFS en dataset |
| `create_pfs_visualizations()` | 446-573 | Crear visualizaciones de analisis PFS |
| `save_low_pfs_gradcam_samples()` | 576-630 | Guardar GradCAM de muestras con bajo PFS |

**Clase PFSAnalyzer:**
- Metodos: `add_result()`, `add_results()`, `get_summary()`, `get_low_pfs_results()`, `save_reports()`
- Genera reportes JSON y CSV (summary, details, by_class, low_pfs_samples)
- Calcula estadisticas por clase y correct vs incorrect
- Trackea muestras con PFS bajo threshold

**Importancia academica:** PFS analysis es central para:
- Validar que el modelo aprende caracteristicas pulmonares relevantes
- Identificar casos donde el modelo "hace trampa" (Clever Hans effect)
- Generar evidencia visual para el jurado (heatmaps con bajo PFS)
- Documentar explicabilidad del modelo en la tesis

### TESTS EXISTENTES

| Archivo | Lineas | Descripcion |
|---------|--------|-------------|
| tests/test_visualization.py | 311-560+ | TestPFSAnalysisModule con ~15 tests |

Tests especificos identificados:
- test_pfs_result_creation
- test_pfs_result_to_dict
- test_pfs_analyzer_initialization
- test_pfs_analyzer_empty_class_names_raises
- test_pfs_analyzer_invalid_threshold_raises
- test_pfs_analyzer_add_result
- test_pfs_analyzer_get_summary
- test_pfs_analyzer_get_summary_no_results_raises
- test_pfs_analyzer_get_low_pfs_results
- test_pfs_analyzer_save_reports
- test_pfs_summary_by_class
- test_pfs_summary_correct_vs_incorrect

### Dependencias

- **Usa:** torch, numpy, PIL, matplotlib, cv2, json, csv, logging, dataclasses
- **Importa de proyecto:** `src_v2.visualization.gradcam` (GradCAM, get_target_layer, calculate_pfs, overlay_heatmap)
- **Es usado por:** CLI (pfs-analysis command)
- **Impacto:** Errores aqui afectarian toda la explicabilidad del modelo

---

## LECCIONES APRENDIDAS DE SESIONES ANTERIORES (CUMPLIR ESTRICTAMENTE)

### De Sesion 7b (verificacion con multiples agentes):

1. **Verificacion exhaustiva con 6 agentes:** Todos confirmaron cumplimiento al 100%
2. **Conteo verificado:** 6+14+12+10 = 42⚪ verificado manualmente
3. **Sin desviaciones:** Ninguna desviacion identificada

### De Sesion 7a:

1. **Verificacion exhaustiva:** 6 agentes confirmaron cumplimiento al 100%
2. **Conteo verificado:** 6+12+8+10 = 36⚪ verificado manualmente
3. **Sin desviaciones:** Ninguna desviacion identificada

### De Sesiones anteriores:

1. **Regla §5.1 CRITICA:** Si la solucion propuesta dice "Opcional", el hallazgo es ⚪ (Nota), NO 🟡 (Menor)
2. **Desviaciones:** Documentar CUALQUIER desviacion detectada y corregida
3. **§4.4 paso 1 OBLIGATORIO:** Incluir seccion "Contexto de Sesion Anterior"
4. **Conteo MANUAL OBLIGATORIO:** Contar CADA hallazgo antes de reportar totales
5. **Protocolo §7.2 OBLIGATORIO:** Solicitud de validacion antes de ejecutar comandos
6. **Orden de Auditores §3.2 (ESTRICTO):**
   1. Arquitecto de Software
   2. Revisor de Codigo
   3. Especialista en Documentacion
   4. Ingeniero de Validacion
   5. Auditor Maestro (con TABLA de veredicto)

---

## AREAS DE ENFOQUE ESPECIAL

Dado que PFS analysis es central para explicabilidad del modelo:

1. **Dataclasses PFSResult y PFSSummary:**
   - Campos correctamente tipados
   - Uso de field(default_factory=...) para mutables (Dict)
   - Metodo to_dict() para serializacion

2. **PFSAnalyzer:**
   - Validacion de class_names no vacio
   - Validacion de threshold en [0, 1]
   - Manejo de caso sin resultados en get_summary()
   - Division por cero manejada en low_pfs_rate

3. **Funciones de mascara:**
   - load_lung_mask(): Conversion RGB a grayscale, normalizacion a [0,1]
   - find_mask_for_image(): Multiples paths de busqueda, manejo de '_warped' suffix
   - generate_approximate_mask(): Validacion de margin en [0, 0.5)

4. **run_pfs_analysis():**
   - Context manager para GradCAM
   - Limite de muestras por clase
   - Manejo de errores con try/except y logging
   - Soporte para mascaras reales o aproximadas

5. **Visualizaciones:**
   - plt.close() despues de savefig (previene memory leaks)
   - 4 figuras: distribucion, by_class, vs_confidence, correct_vs_incorrect
   - Colores semanticos (verde=correcto/bueno, rojo=incorrecto/malo)

6. **save_low_pfs_gradcam_samples():**
   - Importacion de cv2 solo cuando se necesita
   - Manejo de errores por muestra individual
   - Anotaciones con texto en imagen

---

## INSTRUCCIONES

1. Lee referencia_auditoria.md completo
2. Lee audit/sessions/session_07b_error_analysis.md para contexto y formato
3. Sigue el flujo §4.4 paso a paso
4. **INCLUYE seccion "Contexto de Sesion Anterior"**
5. Aplica perspectiva de los 5 auditores EN ORDEN §3.2
6. **ANTES de clasificar:** Si la solucion es "Opcional" → usar ⚪, no 🟡
7. **ANTES de reportar conteo:** Contar manualmente cada severidad en las tablas
8. En ⚪: Listar CADA hallazgo separadamente (no combinar)
9. Documenta hallazgos con severidad calibrada segun §5.1
10. Veredicto en formato TABLA
11. Solicita validacion con protocolo §7.2 ANTES de ejecutar tests
12. Crea documento audit/sessions/session_07c_pfs_analysis.md
13. **ANTES del commit:** Verificar que conteo coincide con tablas
14. Incluye seccion "Registro de Commit" y "Desviaciones de Protocolo"
15. Haz commit: `audit(session-7c): auditoria pfs_analysis.py`

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
| Modulo visualization/ | 2/3 (gradcam.py + error_analysis.py aprobados) |
| Hallazgos totales | 🔴:0 \| 🟠:9 (6 resueltos, 3 pendientes) \| 🟡:28 \| ⚪:268 |
| Objetivo | Completar visualization/ (esta sesion: pfs_analysis.py - ULTIMO ARCHIVO) |

---

## AL FINALIZAR ESTA SESION

Con pfs_analysis.py completado:
- Modulo visualization/ estara **3/3 COMPLETADO**
- **TODOS LOS MODULOS DEL CODIGO FUENTE AUDITADOS**
- Proxima fase: Consolidacion final y resumen ejecutivo
- Solo quedan 3🟠 pendientes de documentacion (M1, M3, M4)

---

## REFERENCIA: ESTRUCTURA DE PFSAnalyzer

```python
class PFSAnalyzer:
    def __init__(self, class_names: List[str], threshold: float = 0.5)
    def add_result(result: PFSResult) -> None
    def add_results(results: List[PFSResult]) -> None
    def get_summary() -> PFSSummary
    def get_low_pfs_results() -> List[PFSResult]
    def save_reports(output_dir) -> Dict[str, Path]
```

**Funciones auxiliares:**
```python
def load_lung_mask(mask_path) -> np.ndarray
def find_mask_for_image(image_path, mask_dir, class_name) -> Optional[Path]
def generate_approximate_mask(image_shape, margin) -> np.ndarray
def run_pfs_analysis(model, dataloader, class_names, device, ...) -> Tuple[PFSAnalyzer, List[Dict]]
def create_pfs_visualizations(detailed_results, output_dir, summary) -> Dict[str, Path]
def save_low_pfs_gradcam_samples(detailed_results, output_dir, threshold, max_samples) -> int
```

Verificar que cada funcion/metodo:
- Tiene docstring completo
- Tiene type hints
- Maneja edge cases
- Esta cubierto por tests

---

## NOTA FINAL: HITO IMPORTANTE

Esta sesion marca el **final de la auditoria de codigo fuente**. Al completar pfs_analysis.py:
- 12/12 modulos de codigo auditados
- visualization/ sera el ultimo modulo completado
- Solo restara la sesion de consolidacion final

¿Listo para comenzar con la Sesion 7c?
