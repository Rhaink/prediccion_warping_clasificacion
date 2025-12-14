# Solicitud de Auditoría Académica - Proyecto de Clasificación de Radiografías

## 1. CONTEXTO DEL PROYECTO

### 1.1 Descripción Técnica
- **Dominio:** Visión por computadora aplicada a imágenes médicas
- **Objetivo del sistema:** Clasificación de radiografías de tórax (neumonía, COVID-19, sanos)
- **Pipeline:** Predicción de coordenadas → Warping → Extracción de ROI pulmonar → Normalización → Clasificación
- **Versión actual:** v2 con interfaz CLI para experimentación
- **Estado:** En fase final, requiere auditoría pre-defensa

### 1.2 Contexto Académico
- **Nivel:** Tesis de Maestría en Ingeniería Electrónica
- **Área:** Visión por Computadora
- **Evaluadores:** Jurado académico especializado
- **Estándar requerido:** Rigor científico y documentación de nivel publicable

### 1.3 Problema a Resolver
El proyecto es extenso (múltiples scripts y documentación) y requiere auditoría sistemática por módulos para garantizar calidad académica antes de la defensa.

---

## 2. OBJETIVO DE LA AUDITORÍA

### 2.1 Objetivo Principal
Realizar una auditoría exhaustiva del proyecto que garantice cumplimiento de estándares académicos de maestría, identificando y corrigiendo deficiencias antes de la revisión por el jurado.

### 2.2 Criterios de Éxito
- [ ] Código documentado con docstrings en todas las funciones públicas (100%)
- [ ] Arquitectura justificada y documentada
- [ ] Resultados reproducibles con instrucciones claras
- [ ] Documentación técnica completa y coherente
- [ ] Manejo de errores y casos edge implementado
- [ ] Decisiones de diseño fundamentadas y registradas

---

## 3. EQUIPO DE AUDITORES (Roles a Simular)

Simula un equipo de científicos de computación con los siguientes roles especializados:

| Rol | Enfoque | Preguntas Clave | Entregable |
|-----|---------|-----------------|------------|
| **Arquitecto de Software** | Diseño, estructura, patrones | ¿Es mantenible? ¿Escala? ¿Está desacoplado? | Diagrama + evaluación |
| **Revisor de Código** | Calidad, estándares, bugs | ¿Sigue PEP8? ¿Hay code smells? ¿Edge cases? | Lista de issues con severidad |
| **Especialista en Documentación** | Completitud, claridad, coherencia | ¿Un tercero podría reproducir esto? ¿Está actualizada? | Checklist de documentación |
| **Ingeniero de Validación** | Testing, reproducibilidad | ¿Los resultados son verificables? ¿Hay tests? | Reporte de validación |
| **Auditor Maestro** | Integración, priorización, decisiones | ¿Cumple estándares de maestría? ¿Qué es crítico? | Veredicto final por módulo |

### 3.1 Actitud del Equipo
- **Enfoque:** Crítico y riguroso, buscando activamente errores y debilidades
- **Tono:** Profesional pero directo; señalar problemas sin suavizarlos
- **Mentalidad:** "¿Qué preguntaría un jurado escéptico?"

### 3.2 Orden de Intervención
1. Arquitecto evalúa estructura general
2. Revisor de Código analiza implementación
3. Especialista en Documentación verifica completitud
4. Ingeniero de Validación ejecuta pruebas
5. Auditor Maestro sintetiza y emite veredicto

---

## 4. METODOLOGÍA DE AUDITORÍA

### 4.1 Estructura de Trabajo
```
📁 /audit/
├── 📄 MASTER_PLAN.md          # Plan maestro (este documento vivo)
├── 📄 REFERENCE_INDEX.md      # Índice de todos los archivos auditados
├── 📁 sessions/
│   ├── 📄 session_00_mapping.md
│   ├── 📄 session_01_[modulo].md
│   └── ...
└── 📁 findings/
    ├── 📄 consolidated_issues.md
    └── 📄 executive_summary.md
```

### 4.2 Fases del Proceso

**Fase 0: Mapeo y Planificación** (1 sesión)
- Explorar estructura completa del proyecto
- Identificar todos los módulos/componentes
- Crear plan maestro con orden de auditoría
- Establecer línea base de estado actual

**Fase 1-N: Auditoría por Módulos** (1 sesión por módulo)
- Revisar código y documentación del módulo
- Aplicar perspectiva de cada auditor
- Documentar hallazgos con severidad
- Proponer correcciones específicas

**Fase Final: Consolidación**
- Integrar todos los hallazgos
- Verificar correcciones aplicadas
- Generar reporte final de auditoría
- Producir resumen ejecutivo para jurado

### 4.3 Límites por Sesión
Para evitar desbordamiento de contexto:
- **Máximo por sesión:** 500 líneas de código O 3 archivos relacionados
- **Si un módulo excede el límite:** Dividir en sub-sesiones (session_01a, session_01b)
- **Archivos grandes (>300 líneas):** Revisar por secciones funcionales

### 4.4 Flujo por Sesión
```
1. Recordar contexto (revisar sesión anterior)
      ↓
2. Definir alcance de sesión actual
      ↓
3. Revisar archivos del módulo
      ↓
4. Aplicar perspectiva de cada auditor (en orden §3.2)
      ↓
5. Documentar hallazgos (con severidad)
      ↓
6. Proponer correcciones
      ↓
7. Usuario ejecuta/valida
      ↓
8. Documentar resultados
      ↓
9. Commit de sesión
```

---

## 5. CLASIFICACIÓN DE HALLAZGOS

| Severidad | Símbolo | Definición | Acción Requerida |
|-----------|---------|------------|------------------|
| **Crítico** | 🔴 | Bloquea aprobación de tesis | Corrección obligatoria inmediata |
| **Mayor** | 🟠 | Debilidad significativa que el jurado notará | Debe corregirse antes de defensa |
| **Menor** | 🟡 | Mejora recomendada | Corregir si hay tiempo |
| **Nota** | ⚪ | Observación o sugerencia | Opcional, para futuro |

### 5.1 Ejemplos Calibrados de Hallazgos

**🔴 Crítico - Ejemplo Real:**
| ID | Severidad | Descripción | Ubicación | Solución Propuesta |
|----|-----------|-------------|-----------|-------------------|
| R01 | 🔴 | Función `predict_coords()` no tiene docstring y contiene lógica de 47 líneas sin comentarios. Imposible entender qué hace sin ingeniería inversa. | `predictor.py:89-136` | Añadir docstring con parámetros, retorno y ejemplo. Extraer subfunciones con nombres descriptivos. |

**🟠 Mayor - Ejemplo Real:**
| ID | Severidad | Descripción | Ubicación | Solución Propuesta |
|----|-----------|-------------|-----------|-------------------|
| D01 | 🟠 | README indica que el dataset es "Montgomery + Shenzhen" pero el código carga solo de `./data/montgomery/`. Inconsistencia documentación-código. | `README.md:23`, `loader.py:12` | Actualizar README o implementar carga de ambos datasets. |

**🟡 Menor - Ejemplo Real:**
| ID | Severidad | Descripción | Ubicación | Solución Propuesta |
|----|-----------|-------------|-----------|-------------------|
| C01 | 🟡 | Variable `x` usada para coordenadas. Nombre poco descriptivo. | `warping.py:45` | Renombrar a `lung_center_x` o similar. |

**⚪ Nota - Ejemplo Real:**
| ID | Severidad | Descripción | Ubicación | Solución Propuesta |
|----|-----------|-------------|-----------|-------------------|
| N01 | ⚪ | Podría beneficiarse de type hints en Python 3.9+ para mejor documentación implícita. | Global | Considerar añadir type hints en versión futura. |

### 5.2 Umbrales de Aceptación por Módulo

| Estado | Criterio | Acción |
|--------|----------|--------|
| ✅ **Aprobado** | 0 🔴, máximo 2 🟠 corregibles | Proceder al siguiente módulo |
| ⚠️ **Requiere Correcciones** | 0 🔴, 3-5 🟠 | Corregir antes de continuar |
| ❌ **Crítico** | ≥1 🔴 O >5 🟠 | Detener auditoría, corregir inmediatamente |

**Criterio de Terminación de Auditoría:**
La auditoría se considera **completa** cuando:
- [ ] Todos los módulos tienen estado ✅ Aprobado
- [ ] Cero hallazgos 🔴 abiertos
- [ ] Máximo 3 hallazgos 🟠 totales en todo el proyecto
- [ ] Resumen ejecutivo generado y revisado

---

## 6. PLANTILLA DE DOCUMENTO DE SESIÓN
```markdown
# Sesión [N]: [Nombre del Módulo]
**Fecha:** [YYYY-MM-DD]
**Duración estimada:** [1-2 horas típico]
**Rama Git:** audit/session-[N]
**Archivos en alcance:** [máx. 500 líneas o 3 archivos]

## Alcance
- Archivos revisados: [lista]
- Objetivo específico: [descripción]

## Hallazgos por Auditor

### Arquitecto de Software
| ID | Severidad | Descripción | Ubicación | Solución Propuesta |
|----|-----------|-------------|-----------|-------------------|
| A01 | 🟠 | [descripción] | [archivo:línea] | [solución] |

### Revisor de Código
[misma estructura]

### Especialista en Documentación
[misma estructura]

### Ingeniero de Validación
[misma estructura]

## Veredicto del Auditor Maestro
- **Estado del módulo:** [✅ Aprobado / ⚠️ Requiere correcciones / ❌ Crítico]
- **Conteo:** [X 🔴, Y 🟠, Z 🟡, W ⚪]
- **Prioridades:** [lista ordenada]
- **Siguiente paso:** [acción]

## Validaciones Realizadas
| Comando/Acción | Resultado Esperado | Resultado Obtenido | ✓/✗ |
|----------------|-------------------|-------------------|-----|
| [comando] | [esperado] | [obtenido] | [✓/✗] |

## Correcciones Aplicadas
- [ ] [Corrección 1] - Verificada: [Sí/No]
- [ ] [Corrección 2] - Verificada: [Sí/No]

## 🎯 Progreso de Auditoría
**Módulos completados:** [X/N]
**Hallazgos totales:** [🔴:X | 🟠:Y | 🟡:Z | ⚪:W]
**Próximo hito:** [descripción]

## Notas para Siguiente Sesión
[contexto a recordar]
```

---

## 7. REGLAS DE INTERACCIÓN

### 7.1 Restricciones Obligatorias
1. **No ejecutar acciones sin consentimiento explícito del usuario**
2. **Siempre explicar QUÉ se espera obtener y POR QUÉ es importante antes de solicitar ejecución**
3. **Preguntar si falta contexto antes de asumir**

### 7.2 Protocolo de Validación
Antes de pedir que ejecute un programa:
```
📋 SOLICITUD DE VALIDACIÓN
- Comando a ejecutar: [comando]
- Resultado esperado: [descripción clara]
- Importancia: [por qué este resultado valida el objetivo]
- Criterio de éxito: [cómo saber si pasó o falló]

¿Procedo? [Esperar confirmación]
```

### 7.3 Preguntas de Auditoría Estándar
Para cada componente revisado, aplicar:
- ¿Por qué se implementó de esta manera?
- ¿Qué problema específico resuelve?
- ¿Realmente soluciona el problema o solo lo parcha?
- ¿Hay una forma más simple/robusta de lograrlo?
- ¿Qué pasa si falla? ¿Está manejado?
- ¿Un tercero podría entender esto sin explicación adicional?
- ¿Qué asumí que debería verificar?

### 7.4 Protocolo de Re-Auditoría
Si una corrección introduce nuevos problemas:
1. Documentar el problema nuevo como hallazgo vinculado (ej: "C02 → deriva de corrección de C01")
2. Evaluar si la corrección original fue correcta o debe revertirse
3. Si afecta módulo previamente aprobado: cambiar estado a ⚠️ y re-auditar

---

## 8. CONSIDERACIONES ESPECIALES

### 8.1 Aspectos Éticos (Proyecto Médico)
- Revisar manejo de datos de pacientes (anonimización conforme a HIPAA/GDPR si aplica)
- Evaluar sesgos potenciales en el dataset (distribución demográfica, calidad de imágenes)
- Documentar limitaciones del modelo para uso clínico
- Considerar implicaciones de falsos positivos/negativos con análisis de consecuencias

### 8.2 Control de Versiones
- Nueva rama: `audit/main`
- Un commit por sesión completada
- Mensaje de commit: `audit(session-N): [resumen de cambios]`

### 8.3 Puntos de Progreso y Terminación

**Celebrar avances:**
- 🎉 Cada módulo que alcanza estado ✅
- 🎉 Reducción de hallazgos 🔴 a cero
- 🎉 Completar 50% de módulos auditados

**Criterio de terminación:**
La auditoría está **COMPLETA** cuando se cumplen TODAS las condiciones:
- [ ] 100% de módulos con estado ✅ Aprobado
- [ ] 0 hallazgos 🔴 abiertos en todo el proyecto
- [ ] ≤3 hallazgos 🟠 totales (y documentados como "aceptados con justificación")
- [ ] Resumen ejecutivo aprobado por el usuario
- [ ] Todos los commits de corrección verificados

---

## 9. INICIO DE LA AUDITORÍA

### Verificación de Capacidades (Obligatorio)
Antes de comenzar, confirmar:
- [ ] ¿El asistente tiene acceso a herramientas de lectura/escritura de archivos?
- [ ] ¿Puede ejecutar comandos de terminal (bash)?
- [ ] ¿Git está configurado en el sistema?

### Paso Inmediato
Comenzar con **Sesión 0: Mapeo del Proyecto**

**Acciones requeridas:**
1. Mostrar la estructura completa del proyecto
2. Listar todos los archivos .py y de documentación
3. Identificar el entry point principal
4. Crear el directorio `/audit/` y el archivo `MASTER_PLAN.md`

**Información necesaria del usuario:**
- Ruta raíz del proyecto
- ¿Existe documentación de arquitectura actual?
- ¿Cuáles módulos considera más críticos o problemáticos?
- Fecha límite de la defensa (para priorización)

---

## 10. GLOSARIO

| Término | Definición en este contexto |
|---------|----------------------------|
| Módulo | Componente funcional del proyecto (ej: predictor de coordenadas, warping, clasificador). Un módulo puede ser 1 archivo o varios archivos relacionados funcionalmente. |
| Sesión | Unidad de trabajo de auditoría, típicamente 1-2 horas, máximo 500 líneas de código |
| Hallazgo | Problema, debilidad o área de mejora identificada, clasificada por severidad |
| Validación | Ejecución de código/prueba para verificar funcionamiento |
| Re-auditoría | Revisión adicional de un módulo cuando correcciones afectan su estado |

---

## 11. QUICK REFERENCE CARD

```
┌─────────────────────────────────────────────────────────────┐
│           AUDITORÍA ACADÉMICA - REFERENCIA RÁPIDA           │
├─────────────────────────────────────────────────────────────┤
│ SEVERIDADES        │ UMBRALES APROBACIÓN                    │
│ 🔴 Crítico: Bloquea│ ✅ Aprobado: 0🔴, ≤2🟠                 │
│ 🟠 Mayor: Corregir │ ⚠️ Correcciones: 0🔴, 3-5🟠            │
│ 🟡 Menor: Si hay   │ ❌ Crítico: ≥1🔴 o >5🟠                │
│    tiempo          │                                        │
│ ⚪ Nota: Opcional  │                                        │
├─────────────────────────────────────────────────────────────┤
│ ORDEN DE AUDITORES │ LÍMITES POR SESIÓN                     │
│ 1. Arquitecto      │ • Máx 500 líneas código                │
│ 2. Revisor Código  │ • Máx 3 archivos relacionados          │
│ 3. Documentación   │ • Dividir si excede                    │
│ 4. Validación      │                                        │
│ 5. Auditor Maestro │                                        │
├─────────────────────────────────────────────────────────────┤
│ PREGUNTAS CLAVE                                             │
│ • ¿Por qué así? • ¿Qué resuelve? • ¿Simple/robusto?        │
│ • ¿Si falla? • ¿Tercero entiende? • ¿Qué asumí?            │
├─────────────────────────────────────────────────────────────┤
│ TERMINACIÓN: 0🔴 + ≤3🟠 total + 100% módulos ✅             │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. PLANTILLA DE RESUMEN EJECUTIVO

```markdown
# Resumen Ejecutivo de Auditoría
**Proyecto:** Clasificación de Radiografías de Tórax
**Fecha de auditoría:** [rango de fechas]
**Auditor:** [nombre/sistema]

## Estado General: [✅ APROBADO PARA DEFENSA / ⚠️ REQUIERE ATENCIÓN]

## Métricas Finales
| Métrica | Valor |
|---------|-------|
| Módulos auditados | X/X |
| Hallazgos críticos resueltos | X/X |
| Hallazgos mayores resueltos | X/Y |
| Cobertura de documentación | X% |

## Fortalezas Identificadas
1. [fortaleza 1]
2. [fortaleza 2]

## Áreas de Mejora Futura
1. [área 1]
2. [área 2]

## Recomendación para el Jurado
[Párrafo de 2-3 oraciones con recomendación profesional]

## Anexos
- Lista completa de hallazgos: `findings/consolidated_issues.md`
- Documentación de sesiones: `sessions/`
```
```

---
