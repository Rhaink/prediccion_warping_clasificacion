# Prompt para Sesion 6: Metricas de Evaluacion

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

### Hallazgos 🟠 Mayores PENDIENTES (de Sesion 0)

1. **M1:** Remover claim incorrecto sobre PFS en README.md
2. **M3:** Anadir seccion de sesgos y disclaimer medico
3. **M4:** Documentar justificacion del margen optimo 1.05

### Archivos de Referencia

- Protocolo: referencia_auditoria.md
- Plan maestro: audit/MASTER_PLAN.md
- Sesion anterior: audit/sessions/session_05b_warp.md
- Hallazgos: audit/findings/consolidated_issues.md

---

## SESION 6: METRICAS DE EVALUACION

### Archivo a Auditar

```
src_v2/evaluation/
├── metrics.py          (437 lineas) ← ESTA SESION
└── __init__.py         (19 lineas)  ← Revisar brevemente
Total modulo: ~456 lineas
```

### Contexto Tecnico de metrics.py

Este archivo implementa **metricas de evaluacion para prediccion de landmarks**, incluyendo:
- Calculo de error euclidiano en pixeles
- Evaluacion por landmark individual y por categoria anatomica
- Generacion de reportes de evaluacion
- Test-Time Augmentation (TTA) con flip horizontal

**Funciones principales:**
| Funcion | Lineas | Descripcion |
|---------|--------|-------------|
| `compute_pixel_error()` | 23-45 | Error euclidiano en pixeles |
| `compute_error_per_landmark()` | 47-63 | Error por landmark individual |
| `evaluate_model()` | 65-160 | Evaluacion completa de modelo |
| `compute_error_per_category()` | 162-196 | Error por categoria anatomica |
| `generate_evaluation_report()` | 198-250 | Generar reporte textual |
| `compute_success_rate()` | 252-273 | Tasa de exito bajo umbral |
| `_flip_landmarks_horizontal()` | 275-299 | Flip horizontal para TTA (privada) |
| `predict_with_tta()` | 301-340 | Prediccion con TTA |
| `evaluate_model_with_tta()` | 342-437 | Evaluacion completa con TTA |

**Importancia academica:** Las metricas son fundamentales para validar la hipotesis de la tesis. El jurado evaluara:
- Correctitud matematica de las metricas
- Interpretabilidad de los resultados
- Uso apropiado de TTA
- Documentacion de formulas

### TESTS EXISTENTES

| Archivo | Lineas | Descripcion |
|---------|--------|-------------|
| tests/test_evaluation_metrics.py | 410 | Tests exhaustivos para metricas |

### Dependencias

- **Usa:** torch, numpy, constants.py (LANDMARK_NAMES, SYMMETRIC_PAIRS)
- **Es usado por:** CLI (evaluate), training (validation loop)
- **Impacto:** Errores aqui afectarian la validacion de todos los experimentos

---

## LECCIONES APRENDIDAS DE SESIONES ANTERIORES (CUMPLIR ESTRICTAMENTE)

### De Sesion 5b (verificacion con multiples agentes):

1. **Regla §5.1 CRITICA:** Si la solucion propuesta dice "Opcional", el hallazgo es ⚪ (Nota), NO 🟡 (Menor)
2. **Verificacion post-auditoria:** Usar multiples agentes para verificar cumplimiento si hay dudas
3. **Desviaciones:** Documentar CUALQUIER desviacion detectada y corregida

### De Sesion 5a:

1. **Conteo correcto:** El conteo manual debe coincidir con hallazgos listados
2. **Clasificacion correcta:** D01 como 🟠 (jurado notara) y C01 como 🟡 (mejora recomendada)

### De Sesiones 4a y 4b:

1. **§4.4 paso 1 OBLIGATORIO:** Incluir seccion "Contexto de Sesion Anterior"
2. **Conteo MANUAL OBLIGATORIO:** Contar CADA hallazgo antes de reportar totales
3. **V04 y V05 separados:** No combinar hallazgos en una sola linea

### De Sesiones 1-3:

1. **Protocolo §7.2 OBLIGATORIO:** Solicitud de validacion antes de ejecutar comandos
2. **Limite §4.3:** Maximo 500 lineas por sesion. metrics.py (437 lineas) cumple
3. **Orden de Auditores §3.2 (ESTRICTO):**
   1. Arquitecto de Software
   2. Revisor de Codigo
   3. Especialista en Documentacion
   4. Ingeniero de Validacion
   5. Auditor Maestro (con TABLA de veredicto)

4. **Veredicto en formato TABLA**

---

## AREAS DE ENFOQUE ESPECIAL

Dado que las metricas determinan la validez de los resultados experimentales:

1. **Correctitud matematica:**
   - Formula de error euclidiano correcta
   - Desnormalizacion de coordenadas [0,1] → pixeles
   - Promediado correcto (por batch, por landmark, global)

2. **Test-Time Augmentation (TTA):**
   - `_flip_landmarks_horizontal()` debe intercambiar pares simetricos correctamente
   - SYMMETRIC_PAIRS usado correctamente
   - Promedio de predicciones original + flipped

3. **Categorizacion anatomica:**
   - `compute_error_per_category()` agrupa landmarks correctamente
   - Categorias: apices, claviculas, diafragma, silueta cardiaca, etc.

4. **Reproducibilidad:**
   - evaluate_model() debe ser determinista (no random)
   - Documentar si hay stochasticity

5. **Documentacion academica:**
   - Referencias a literatura de metricas si aplica
   - Formulas documentadas en docstrings

---

## INSTRUCCIONES

1. Lee referencia_auditoria.md completo
2. Lee audit/sessions/session_05b_warp.md para contexto y formato
3. Sigue el flujo §4.4 paso a paso
4. **INCLUYE seccion "Contexto de Sesion Anterior"**
5. Aplica perspectiva de los 5 auditores EN ORDEN §3.2
6. **ANTES de clasificar:** Si la solucion es "Opcional" → usar ⚪, no 🟡
7. **ANTES de reportar conteo:** Contar manualmente cada severidad en las tablas
8. En ⚪: Listar CADA hallazgo separadamente (no combinar)
9. Documenta hallazgos con severidad calibrada segun §5.1
10. Veredicto en formato TABLA
11. Solicita validacion con protocolo §7.2 ANTES de ejecutar tests
12. Crea documento audit/sessions/session_06_metrics.md
13. **ANTES del commit:** Verificar que conteo coincide con tablas
14. Incluye seccion "Registro de Commit" y "Desviaciones de Protocolo"
15. Haz commit: `audit(session-6): auditoria metrics.py`

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
| Modulos completados | 10/12 |
| Modulo models/ | COMPLETADO (4/4) |
| Modulo training/ | COMPLETADO (2/2) |
| Modulo processing/ | COMPLETADO (2/2) |
| Modulo evaluation/ | 0/1 (metrics.py pendiente) |
| Hallazgos totales | 🔴:0 \| 🟠:9 (6 resueltos, 3 pendientes) \| 🟡:28 \| ⚪:161 |
| Objetivo | Completar evaluation/ (esta sesion: metrics.py) |

---

## AL FINALIZAR ESTA SESION

Con metrics.py completado:
- Modulo evaluation/ estara 1/1 COMPLETADO
- Proxima sesion: 7a (visualization/ - gradcam.py)
- Progreso: 11/12 modulos completados

---

## NOTA SOBRE TTA Y SYMMETRIC_PAIRS

La funcion `_flip_landmarks_horizontal()` usa SYMMETRIC_PAIRS de constants.py para intercambiar landmarks simetricos al hacer flip horizontal. Verificar:

1. Que SYMMETRIC_PAIRS esta correctamente definido en constants.py
2. Que el intercambio de indices es correcto
3. Que el flip de coordenadas X es correcto (x' = 1 - x para coords normalizadas)

Esto es critico para que TTA funcione correctamente.

---

¿Listo para comenzar con la Sesion 6?
