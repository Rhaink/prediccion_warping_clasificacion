# Prompt para Sesion 7a: Visualizacion - Grad-CAM

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

### Hallazgos 🟠 Mayores PENDIENTES (de Sesion 0)

1. **M1:** Remover claim incorrecto sobre PFS en README.md
2. **M3:** Anadir seccion de sesgos y disclaimer medico
3. **M4:** Documentar justificacion del margen optimo 1.05

### Archivos de Referencia

- Protocolo: referencia_auditoria.md
- Plan maestro: audit/MASTER_PLAN.md
- Sesion anterior: audit/sessions/session_06_metrics.md
- Hallazgos: audit/findings/consolidated_issues.md

---

## SESION 7a: VISUALIZACION - GRAD-CAM

### NOTA: Division del Modulo visualization/

El modulo visualization/ tiene **1529 lineas totales**, excediendo el limite de 500 lineas por sesion (§4.3). Se divide en sub-sesiones:

| Sesion | Archivo | Lineas | Estado |
|--------|---------|--------|--------|
| 7a | gradcam.py + __init__.py | 376 + 45 = 421 | **ESTA SESION** |
| 7b | error_analysis.py | 478 | Pendiente |
| 7c | pfs_analysis.py | 630 | Pendiente |

### Archivos a Auditar

```
src_v2/visualization/
├── gradcam.py      (376 lineas) ← ESTA SESION
└── __init__.py     (45 lineas)  ← Revisar brevemente
Total: 421 lineas
```

### Contexto Tecnico de gradcam.py

Este archivo implementa **Grad-CAM (Gradient-weighted Class Activation Mapping)** para interpretabilidad del clasificador COVID-19:

**Componentes principales:**

| Componente | Lineas | Descripcion |
|------------|--------|-------------|
| `TARGET_LAYER_MAP` | 20-28 | Mapeo de capas objetivo por arquitectura |
| `get_target_layer()` | 31-57 | Obtener capa para GradCAM segun backbone |
| `_get_layer_by_name()` | 60-84 | Navegacion por path de capa (privada) |
| `GradCAM` class | 87-235 | Clase principal para generar heatmaps |
| `calculate_pfs()` | 238-284 | Calculo de Pulmonary Focus Score |
| `overlay_heatmap()` | 287-330 | Superponer heatmap en imagen |
| `create_gradcam_visualization()` | 333-377 | Crear visualizacion con anotaciones |

**Clase GradCAM:**
- Usa hooks de PyTorch para capturar activaciones y gradientes
- Implementa context manager (`with GradCAM(...) as cam:`)
- Metodo `remove_hooks()` para prevenir memory leaks
- Validacion de dimensiones de entrada

**Importancia academica:** Grad-CAM es fundamental para la tesis porque:
- Provee explicabilidad del modelo (requerido en IA medica)
- Permite calcular PFS (Pulmonary Focus Score) - metrica de la tesis
- El jurado evaluara si la implementacion sigue el paper original
- La formula PFS = sum(heatmap * mask) / sum(heatmap) debe ser correcta

### TESTS EXISTENTES

| Archivo | Lineas | Descripcion |
|---------|--------|-------------|
| tests/test_visualization.py | 554 | Tests para gradcam, overlay, PFS |

### Dependencias

- **Usa:** torch, numpy, PIL, matplotlib.cm, cv2
- **Es usado por:** CLI (visualize-gradcam), pfs_analysis.py
- **Impacto:** Errores aqui afectarian toda la explicabilidad del modelo

---

## LECCIONES APRENDIDAS DE SESIONES ANTERIORES (CUMPLIR ESTRICTAMENTE)

### De Sesion 6 (verificacion con multiples agentes):

1. **Verificacion exhaustiva:** Usar multiples agentes post-auditoria confirmo cumplimiento total
2. **Conteo correcto:** 5+11+6+7 = 29⚪ verificado manualmente
3. **Sin desviaciones:** Ningunoa desviacion identificada

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

Dado que Grad-CAM es central para la explicabilidad del modelo:

1. **Implementacion de Grad-CAM:**
   - Hooks de forward/backward correctamente registrados
   - Global Average Pooling de gradientes: `weights = gradients.mean(dim=(2,3))`
   - Combinacion ponderada: `cam = sum(weights * activations)`
   - ReLU para mantener contribuciones positivas
   - Normalizacion a [0,1]

2. **Manejo de memoria:**
   - `remove_hooks()` llamado para prevenir memory leaks
   - Context manager implementado correctamente
   - Limpieza de gradientes y activaciones

3. **Pulmonary Focus Score (PFS):**
   - Formula: `PFS = sum(heatmap * mask) / sum(heatmap)`
   - Manejo de mascaras RGB vs grayscale
   - Redimensionamiento de mascara si es necesario
   - Division por cero manejada

4. **Soporte multi-arquitectura:**
   - TARGET_LAYER_MAP cubre: resnet18, resnet50, densenet121, efficientnet_b0, vgg16, alexnet, mobilenet_v2
   - Navegacion por path de capas con puntos (backbone.layer4)

5. **Validaciones de entrada:**
   - Dimension 4D requerida (B, C, H, W)
   - Batch size = 1 requerido
   - target_class en rango valido

---

## INSTRUCCIONES

1. Lee referencia_auditoria.md completo
2. Lee audit/sessions/session_06_metrics.md para contexto y formato
3. Sigue el flujo §4.4 paso a paso
4. **INCLUYE seccion "Contexto de Sesion Anterior"**
5. Aplica perspectiva de los 5 auditores EN ORDEN §3.2
6. **ANTES de clasificar:** Si la solucion es "Opcional" → usar ⚪, no 🟡
7. **ANTES de reportar conteo:** Contar manualmente cada severidad en las tablas
8. En ⚪: Listar CADA hallazgo separadamente (no combinar)
9. Documenta hallazgos con severidad calibrada segun §5.1
10. Veredicto en formato TABLA
11. Solicita validacion con protocolo §7.2 ANTES de ejecutar tests
12. Crea documento audit/sessions/session_07a_gradcam.md
13. **ANTES del commit:** Verificar que conteo coincide con tablas
14. Incluye seccion "Registro de Commit" y "Desviaciones de Protocolo"
15. Haz commit: `audit(session-7a): auditoria gradcam.py`

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
| Modulo visualization/ | 0/3 (gradcam.py pendiente) |
| Hallazgos totales | 🔴:0 \| 🟠:9 (6 resueltos, 3 pendientes) \| 🟡:28 \| ⚪:190 |
| Objetivo | Completar visualization/ (esta sesion: gradcam.py) |

---

## AL FINALIZAR ESTA SESION

Con gradcam.py completado:
- Modulo visualization/ estara 1/3 completado
- Proxima sesion: 7b (visualization/ - error_analysis.py)
- Progreso: 11/12 modulos (visualization parcial)

---

## NOTA SOBRE PFS Y EL CLAIM EN README

Recordar que hay un hallazgo 🟠 pendiente (M1) sobre un claim incorrecto de PFS en README.md. Durante esta auditoria de gradcam.py:

1. Verificar que la implementacion de `calculate_pfs()` es correcta
2. Documentar que hace la funcion (para poder corregir el claim del README)
3. NO confundir la correctitud de la implementacion con los claims sobre resultados experimentales

La funcion `calculate_pfs()` en si misma puede ser correcta aunque el claim sobre mejora de PFS en el README sea incorrecto o no verificado.

---

## REFERENCIA: FORMULA GRAD-CAM

Segun el paper original (Selvaraju et al., 2017):

```
L_c^{Grad-CAM} = ReLU(sum_k(alpha_k^c * A^k))

donde:
- alpha_k^c = (1/Z) * sum_i sum_j (dY^c / dA_{ij}^k)  [Global Average Pooling de gradientes]
- A^k = activaciones del feature map k
- Y^c = score de clase c (antes de softmax)
```

Verificar que la implementacion sigue esta formula.

---

¿Listo para comenzar con la Sesion 7a?
