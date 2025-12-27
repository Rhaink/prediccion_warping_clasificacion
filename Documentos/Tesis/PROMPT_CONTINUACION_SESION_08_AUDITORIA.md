# PROMPT DE CONTINUACIÓN - SESIÓN 08: AUDITORÍA DE CUMPLIMIENTO

## INSTRUCCIONES CRÍTICAS PARA CLAUDE

**CONTEXTO:** En sesiones anteriores (01-06) se ignoraron instrucciones explícitas de `Documentos/Tesis/prompts/prompt_tesis.md`. El error más grave fue usar `\documentclass{article}` cuando se solicitó claramente una "plantilla popular para tesis de posgrado".

**MODO DE TRABAJO:** Esta sesión es de AUDITORÍA Y CORRECCIÓN CONTROLADA.

### REGLAS OBLIGATORIAS:

1. **NO hacer cambios masivos.** Cada cambio debe ser:
   - Explicado claramente antes de aplicarse
   - Mostrado al usuario (el antes y el después)
   - APROBADO explícitamente por el usuario antes de ejecutarse

2. **Revisar línea por línea** el archivo `Documentos/Tesis/prompts/prompt_tesis.md` y verificar cumplimiento.

3. **Documentar cada hallazgo** antes de proponer corrección.

4. **Preguntar en caso de duda.** No asumir ni inferir.

---

## ESTADO ACTUAL DEL PROYECTO

### Correcciones ya aplicadas (Sesión 07):

| Problema | Corrección | Estado |
|----------|------------|--------|
| `\documentclass{article}` | Cambiado a `\documentclass{report}` | ✅ Aplicado |
| Objetivos antes del índice | Reordenado: índice → objetivos | ✅ Aplicado |
| Estilo bibliográfico `apalike` | Cambiado a `IEEEtran` con `[numbers]` | ✅ Aplicado |
| Numeración por sección | Cambiado a numeración por capítulo | ✅ Aplicado |
| Algoritmos mal formateados | Corregidos comandos algpseudocode | ✅ Aplicado |
| Referencias faltantes | Agregadas 35 referencias a .bib | ✅ Aplicado |
| BibTeX deshabilitado | Habilitado con `\bibliography{references}` | ✅ Aplicado |

### Archivos clave:

| Archivo | Descripción |
|---------|-------------|
| `Documentos/Tesis/prompts/prompt_tesis.md` | Instrucciones originales - AUDITAR CONTRA ESTO |
| `Documentos/Tesis/main.tex` | Archivo principal LaTeX |
| `Documentos/Tesis/references.bib` | Referencias bibliográficas (35 refs) |
| `Documentos/Tesis/capitulo4/*.tex` | 6 secciones del Capítulo 4 |
| `Documentos/Tesis/ESTRUCTURA_TESIS.md` | Estructura aprobada |
| `Documentos/Tesis/DECISIONES_FASE_1.md` | Decisiones de fase 1 |

---

## TAREA: AUDITORÍA DE Documentos/Tesis/prompts/prompt_tesis.md

### Proceso paso a paso:

1. **Leer `Documentos/Tesis/prompts/prompt_tesis.md` completo**
2. **Para cada requisito, verificar:**
   - ¿Está implementado correctamente?
   - Si NO: documentar el problema
   - Proponer corrección específica
   - ESPERAR APROBACIÓN del usuario
   - Solo entonces aplicar el cambio

### Checklist de requisitos a auditar:

#### Sección: ESPECIFICACIONES TÉCNICAS (líneas 39-64)

| Línea | Requisito | Verificar |
|-------|-----------|-----------|
| 42-45 | Institución BUAP, Maestría Ing. Electrónica | ¿Portada correcta? |
| 48 | Páginas totales: 80-120 | ¿Extensión actual? |
| 49 | Formato: LaTeX | ✅ Cumplido |
| 50 | **Plantilla: popular para tesis posgrado** | ⚠️ Corregido a `report` |
| 51 | Idioma: Español latino formal | ¿Revisar redacción? |
| 54 | Estilo referencias: IEEE | ✅ Corregido en Sesión 07 |
| 55 | Cantidad mínima: 50 referencias | ❓ Actualmente 35 |
| 56 | Recencia: 60% últimos 4 años | ❓ Por verificar |
| 57 | Calidad: journals indexados | ❓ Por verificar |
| 60 | Figuras: vectorial o PNG ≥300 DPI | ❓ Figuras pendientes |
| 61 | Tablas: numeradas con caption | ❓ Por verificar |
| 63 | Figuras/tablas referenciadas en texto | ❓ Por verificar |

#### Sección: GUÍA DE ESTILO (líneas 339-368)

| Requisito | Verificar |
|-----------|-----------|
| Voz activa preferida | ¿Revisar redacción? |
| Sin pronombres personales | ¿Revisar capítulo 4? |
| Conectores lógicos | ¿Revisar transiciones? |
| Tiempo verbal consistente | ¿Revisar capítulo 4? |
| Referencias estilo IEEE en texto | ✅ Configurado |

#### Sección: CONSIDERACIONES ÉTICAS (líneas 371-400)

| Requisito | Verificar |
|-----------|-----------|
| Disclaimer médico obligatorio | ❓ ¿Incluido? |
| Sección de limitaciones | ❓ ¿Documentado? |
| Privacidad de datos | ❓ ¿Mencionado? |
| Implicaciones falsos +/- | ❓ ¿Discutido? |

---

## FORMATO DE TRABAJO

Para cada problema encontrado, presentar así:

```
### HALLAZGO #N: [Título breve]

**Requisito (Documentos/Tesis/prompts/prompt_tesis.md línea X):**
> [Cita textual del requisito]

**Estado actual:**
[Descripción de cómo está actualmente]

**Problema:**
[Explicación del incumplimiento]

**Corrección propuesta:**
[Descripción específica del cambio]

**Vista previa del cambio:**
```
ANTES:
[código/texto actual]

DESPUÉS:
[código/texto propuesto]
```

**¿Aprobar este cambio? (Sí/No/Modificar)**
```

---

## COMANDO INICIAL

```
Por favor:
1. Lee el archivo `Documentos/Tesis/prompts/prompt_tesis.md`
2. Lee el estado actual de `main.tex` y los archivos del capítulo 4
3. Comienza la auditoría línea por línea
4. Presenta el PRIMER hallazgo en el formato especificado
5. ESPERA mi aprobación antes de hacer cualquier cambio
```

---

## RECORDATORIOS

- **NO aplicar cambios sin aprobación explícita**
- **Mostrar siempre ANTES y DESPUÉS**
- **Un hallazgo a la vez**
- **Preguntar si hay ambigüedad**

---

*Prompt generado: 16 Diciembre 2025 - Sesión 07*
*Objetivo: Auditoría controlada de cumplimiento con Documentos/Tesis/prompts/prompt_tesis.md*
