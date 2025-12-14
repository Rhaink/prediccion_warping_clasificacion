# Prompt para Sesión 4b: Callbacks de Entrenamiento

Estoy realizando una auditoría académica de mi proyecto de tesis de maestría (clasificación de radiografías de tórax mediante deep learning). El proyecto está en /home/donrobot/Projects/prediccion_warping_clasificacion/.

IMPORTANTE: Lee primero referencia_auditoria.md en la raíz del proyecto - contiene el protocolo COMPLETO que debes seguir A RAJA TABLA.

## ESTADO ACTUAL DE LA AUDITORÍA

### Sesiones Completadas

| Sesión | Módulo                               | Estado        | Hallazgos                         |
|--------|--------------------------------------|---------------|-----------------------------------|
| 0      | Mapeo del proyecto                   | ✅ Completada | 0🔴, 4🟠, 5🟡, 4⚪                |
| 1      | Configuración y utilidades           | ✅ APROBADO   | 0🔴, 0🟠, 1🟡, 4⚪                |
| 2      | Gestión de datos (data/)             | ✅ APROBADO   | 0🔴, 2🟠 resueltos, 5🟡, 8⚪      |
| 3a     | Funciones de pérdida (losses.py)     | ✅ APROBADO   | 0🔴, 1🟠 resuelto, 4🟡, 10⚪      |
| 3b     | ResNet Landmark (resnet_landmark.py) | ✅ APROBADO   | 0🔴, 0🟠, 2🟡, 15⚪               |
| 3c     | Clasificador (classifier.py)         | ✅ APROBADO   | 0🔴, 1🟠 resuelto, 2🟡, 15⚪      |
| 3d     | Jerárquico (hierarchical.py)         | ✅ APROBADO   | 0🔴, 0🟠, 2🟡, 20⚪ (experimental)|
| 4a     | Trainer (trainer.py)                 | ✅ APROBADO   | 0🔴, 0🟠, 5🟡, 18⚪               |

### Hallazgos 🟠 Mayores PENDIENTES (de Sesión 0)

1. **M1:** Remover claim incorrecto sobre PFS en README.md
2. **M3:** Añadir sección de sesgos y disclaimer médico
3. **M4:** Documentar justificación del margen óptimo 1.05

### Archivos de Referencia

- Protocolo: referencia_auditoria.md
- Plan maestro: audit/MASTER_PLAN.md
- Sesión anterior: audit/sessions/session_04a_trainer.md
- Hallazgos: audit/findings/consolidated_issues.md

---

## SESIÓN 4b: CALLBACKS DE ENTRENAMIENTO (callbacks.py)

### Archivo a Auditar

```
src_v2/training/
├── trainer.py          (433 líneas) ← Sesión 4a ✅ APROBADO
├── callbacks.py        (240 líneas) ← ESTA SESIÓN
└── __init__.py         (13 líneas)
Total módulo: ~685 líneas (completado con esta sesión)
```

### Contexto Técnico de callbacks.py

Este archivo implementa los callbacks de entrenamiento usados por LandmarkTrainer:
- Clase `EarlyStopping`: Detiene entrenamiento si no hay mejora
- Clase `ModelCheckpoint`: Guarda mejores modelos durante entrenamiento
- Clase `LRSchedulerCallback`: Wrapper para learning rate schedulers

### TESTS EXISTENTES

| Archivo | Líneas | Descripción |
|---------|--------|-------------|
| tests/test_callbacks.py | 276 | Tests dedicados para los 3 callbacks |

### Dependencias con trainer.py (Sesión 4a)

callbacks.py es usado directamente por trainer.py en:
- `train_phase1()`: EarlyStopping, ModelCheckpoint
- `train_phase2()`: EarlyStopping, ModelCheckpoint, LRSchedulerCallback

---

## LECCIONES APRENDIDAS DE SESIONES ANTERIORES (CUMPLIR ESTRICTAMENTE)

### De Sesión 4a (verificación con 3 agentes):

1. **Conteo de ⚪:** Distinguir entre fortalezas (observaciones positivas) y observaciones opcionales
2. **Plantilla §6:** Son 14 secciones, NO 9 puntos
3. **Verificación exhaustiva:** Usar 3 agentes en paralelo para verificar cumplimiento antes del commit
4. **Desviaciones:** Documentar TODAS las desviaciones detectadas y corregidas

### De Sesión 3d:

1. **§4.4 paso 1 OBLIGATORIO:** Incluir sección "Contexto de Sesión Anterior" con referencia explícita
2. **§5.1 CRÍTICO:** Si la solución dice "Opcional", el hallazgo es ⚪ (Nota), NO 🟡 (Menor)
3. **Conteo manual:** Contar CADA hallazgo antes de reportar totales

### De Sesiones 1-3c:

1. **Protocolo §7.2 OBLIGATORIO:** Antes de ejecutar CUALQUIER comando:
   ```
   📋 SOLICITUD DE VALIDACIÓN
   - Comando a ejecutar: [comando]
   - Resultado esperado: [descripción]
   - Importancia: [por qué]
   - Criterio de éxito: [cómo saber si pasó]

   ¿Procedo? [Esperar mi confirmación]
   ```

2. **Límite §4.3:** Máximo 500 líneas por sesión. ✅ callbacks.py (240 líneas) cumple.

3. **Orden de Auditores §3.2 (ESTRICTO):**
   1. Arquitecto de Software
   2. Revisor de Código
   3. Especialista en Documentación
   4. Ingeniero de Validación
   5. Auditor Maestro (con TABLA de veredicto)

4. **Clasificación §5.1:**
   - 🔴 Crítico: Bloquea aprobación
   - 🟠 Mayor: Jurado notará
   - 🟡 Menor: Mejora recomendada (NO "Opcional")
   - ⚪ Nota: Fortalezas y observaciones opcionales

5. **Veredicto en formato TABLA:**
   ```
   | Métrica           | Valor              |
   |-------------------|--------------------|
   | Estado del módulo | ✅ APROBADO        |
   | Conteo            | 0🔴, X🟠, Y🟡, Z⚪ |
   | ...               |                    |
   ```

---

## INSTRUCCIONES

1. Lee referencia_auditoria.md completo
2. Lee audit/sessions/session_04a_trainer.md para contexto y formato correcto
3. Usa ultrathinking y múltiples agentes para análisis exhaustivo
4. Sigue el flujo §4.4 paso a paso
5. **INCLUYE sección "Contexto de Sesión Anterior"** (lección de 3d)
6. Aplica perspectiva de los 5 auditores EN ORDEN §3.2
7. ANTES de clasificar: Si la solución es "Opcional" → usar ⚪, no 🟡
8. ANTES de reportar conteo: Contar manualmente cada severidad
9. En ⚪: Distinguir fortalezas (10) de observaciones opcionales (8) si aplica
10. Documenta hallazgos con severidad calibrada según §5.1
11. Veredicto en formato TABLA
12. Solicita validación con protocolo §7.2 ANTES de ejecutar tests
13. Crea documento audit/sessions/session_04b_callbacks.md
14. ANTES del commit: Usa 3 agentes en paralelo para verificar cumplimiento del protocolo
15. Incluye sección "Registro de Commit" y "Desviaciones de Protocolo"
16. Haz commit: `audit(session-4b): auditoria callbacks.py`

---

## CHECKLIST PRE-COMMIT (OBLIGATORIO - Verificar con 3 agentes)

Antes de hacer commit, verificar:
- [ ] Sección "Contexto de Sesión Anterior" incluida
- [ ] Plantilla §6 cumple 14/14 secciones
- [ ] Clasificación §5.1 correcta (no "Opcional" en 🟡)
- [ ] Conteo manual coincide con hallazgos listados
- [ ] En ⚪: Desglose fortalezas vs observaciones opcionales
- [ ] Flujo §4.4 completo (9/9 pasos)
- [ ] Orden de auditores §3.2 respetado (5/5 en orden)
- [ ] Protocolo §7.2 aplicado en validaciones
- [ ] Sección "Registro de Commit" incluida
- [ ] Sección "Desviaciones de Protocolo" incluida

---

## PROGRESO GLOBAL

| Métrica | Valor |
|---------|-------|
| Módulos completados | 7/12 |
| Módulo models/ | ✅ COMPLETADO (4/4) |
| Módulo training/ | 1/2 (trainer.py ✅, callbacks.py pendiente) |
| Hallazgos totales | 🔴:0 \| 🟠:8 (5 resueltos, 3 pendientes) \| 🟡:26 \| ⚪:94 |
| Objetivo | Completar training/ (esta sesión finaliza el módulo) |

---

## AL FINALIZAR ESTA SESIÓN

Con callbacks.py completado:
- Módulo training/ estará 100% auditado (2/2 archivos)
- Próximos módulos: inference/, cli/, scripts/
- Progreso: 8/12 módulos completados

---

¿Listo para comenzar con la Sesión 4b?
