# Prompt Optimizado para Sesión de Redacción 02

## Instrucciones de Uso
Copiar todo el contenido entre las líneas `---INICIO PROMPT---` y `---FIN PROMPT---` y pegarlo en una nueva conversación de Claude Code.

---INICIO PROMPT---

## CONTEXTO DEL PROYECTO

Estoy trabajando en mi tesis de Maestría en Ingeniería Electrónica (área Visión por Computadora) sobre **normalización geométrica mediante landmarks anatómicos para detección de COVID-19 en radiografías de tórax**.

**Directorio de trabajo:** `/home/donrobot/Projects/prediccion_warping_clasificacion/Tesis/`

**Estado actual:**
- 12 páginas compiladas (4 originales + 5 nuevas en borrador)
- Archivos creados: `8-Introduccion.tex`, `9-Hipotesis.tex`, `10-Justificacion.tex`, `11-MarcoTeorico.tex`
- Documentación de sesión anterior: `SESION_REDACCION_01.md`
- Plan de redacción: `/home/donrobot/.claude/plans/velvet-frolicking-penguin.md`

## TAREA PRINCIPAL

Refinar las 5 páginas nuevas (páginas 5-9) para que alcancen calidad profesional de publicación académica. El objetivo es que parezcan escritas por un ghostwriter profesional con 30 años de experiencia en redacción de tesis de ingeniería.

## METODOLOGÍA REQUERIDA

### Paso 1: Análisis Profundo (usar ultrathink)
Antes de hacer cualquier edición, realiza un análisis exhaustivo usando `ultrathink` y **3 agentes especializados en paralelo**:

**Agente 1 - Auditor de Estilo:**
- Leer `/home/donrobot/Projects/prediccion_warping_clasificacion/documentación/Tesis___Rafael_Alejandro_Cruz_Ovando/` (tesis de referencia)
- Extraer: densidad de párrafos, frecuencia de citas, uso de ecuaciones numeradas, estructura de subsecciones
- Comparar con archivos actuales en `/Tesis/`

**Agente 2 - Verificador de Contenido Técnico:**
- Leer `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/CLAIMS_TESIS.md`
- Leer `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/RESUMEN_DEFENSA.md`
- Verificar que todos los claims validados estén reflejados en el texto

**Agente 3 - Revisor de Debilidades:**
- Leer `SESION_REDACCION_01.md` sección 2 (debilidades identificadas)
- Crear checklist de correcciones pendientes
- Priorizar por impacto en la calidad del documento

### Paso 2: Correcciones Estructurales
Aplicar estas correcciones en orden:

1. **Numeración de secciones:** Cambiar `\section*{}` a `\section{}` con labels
2. **Ecuaciones numeradas:** Todas las ecuaciones deben tener `\label{eq:nombre}`
3. **Referencias cruzadas:** Agregar `\ref{}` entre secciones (mínimo 3 por sección)

### Paso 3: Enriquecimiento de Contenido

**Para cada sección, agregar:**

| Sección | Datos Obligatorios | Fórmulas Pendientes |
|---------|-------------------|---------------------|
| Introducción | COVID-19 mención explícita, 2B radiografías/año | - |
| Hipótesis | Error ensemble 3.71px, warped_96 99.10% | Euclidiano ya existe |
| Justificación | FedCOVIDx 8,482 muestras, domain shift | - |
| Marco Teórico | ResNet-18 11.7M params | Wing Loss, Procrustes |

**Fórmulas a incluir en Marco Teórico:**
```latex
% Wing Loss (agregar después de CNNs)
\begin{equation}
L_{\text{wing}}(y, \hat{y}) =
\begin{cases}
w \ln(1 + |y-\hat{y}|/\epsilon) & \text{si } |y-\hat{y}| < w \\
|y-\hat{y}| - C & \text{en otro caso}
\end{cases}
\label{eq:wing-loss}
\end{equation}

% Procrustes (agregar nueva subsección)
\begin{equation}
R^* = V U^T \quad \text{donde } \text{SVD}(X^T Y) = U \Sigma V^T
\label{eq:procrustes}
\end{equation}

% Fill Rate (definir en metodología o marco)
\begin{equation}
\text{fill\_rate} = 1 - \frac{\text{píxeles\_negros}}{\text{total\_píxeles}}
\label{eq:fill-rate}
\end{equation}
```

### Paso 4: Aumento de Densidad
- Párrafos actuales: 5-12 líneas → Meta: 10-15 líneas
- Agregar detalles técnicos sin ser redundante
- Incluir justificaciones de decisiones de diseño

### Paso 5: Citas Bibliográficas
- Meta: 2-3 citas por sección
- Usar formato `\cite{}` (preparar para BibTeX)
- Citas obligatorias: He et al. (ResNet), Feng et al. (Wing Loss), COVID-Net papers

### Paso 6: Conexiones Narrativas
Agregar transiciones explícitas entre secciones:
- Introducción → Hipótesis: "El problema de variabilidad geométrica nos lleva a formular..."
- Hipótesis → Justificación: "Estas predicciones son relevantes porque..."
- Justificación → Marco Teórico: "Para implementar esta solución, necesitamos fundamentos de..."

## CRITERIOS DE CALIDAD (AUDITORÍA)

Después de cada edición, verificar:
- [ ] ¿La ecuación tiene número y label?
- [ ] ¿El párrafo tiene al menos una cita o referencia interna?
- [ ] ¿La afirmación tiene evidencia cuantitativa?
- [ ] ¿La sección conecta con la anterior y siguiente?
- [ ] ¿El estilo es consistente con la tesis de referencia?

## DOCUMENTACIÓN

Al finalizar, actualizar `SESION_REDACCION_01.md` o crear `SESION_REDACCION_02.md` con:
- Cambios realizados por archivo
- Debilidades restantes
- Verificación de compilación
- Próximos pasos

## ARCHIVOS DE REFERENCIA

```
Tesis actual:
├── main.tex                    # Documento principal
├── 8-Introduccion.tex          # Página 5
├── 9-Hipotesis.tex             # Página 6
├── 10-Justificacion.tex        # Página 7
├── 11-MarcoTeorico.tex         # Páginas 8-9
├── SESION_REDACCION_01.md      # Documentación sesión 1
└── PROMPT_SESION_02.md         # Este archivo

Proyecto completo:
├── docs/CLAIMS_TESIS.md        # Claims validados
├── docs/RESUMEN_DEFENSA.md     # Resumen para defensa
├── docs/sesiones/SESION_39_*.md # Mecanismo causal
├── docs/sesiones/SESION_53_*.md # Análisis robustez
├── docs/sesiones/SESION_55_*.md # Validación externa

Referencia de estilo:
└── documentación/Tesis___Rafael_Alejandro_Cruz_Ovando/
```

## RESTRICCIONES

1. **NO crear archivos nuevos** excepto documentación `.md`
2. **NO agregar emojis** al contenido LaTeX
3. **NO hacer cambios que rompan compilación** - verificar después de cada edición mayor
4. **NO inventar datos** - usar solo valores del proyecto documentado
5. **Mantener honestidad científica** - incluir limitaciones y hallazgos negativos

## COMANDO DE VERIFICACIÓN

Después de ediciones, ejecutar:
```bash
cd /home/donrobot/Projects/prediccion_warping_clasificacion/Tesis && pdflatex main.tex
```

---FIN PROMPT---

## Notas Adicionales

### Por qué este prompt es efectivo:

1. **Contexto completo**: Incluye rutas exactas y estado actual
2. **Metodología estructurada**: Pasos claros y ordenados
3. **Uso de agentes**: Divide trabajo complejo en tareas paralelas especializadas
4. **Criterios de calidad**: Checklist verificable para auditoría
5. **Datos concretos**: Tablas con valores específicos a incluir
6. **Restricciones claras**: Evita errores comunes
7. **Documentación integrada**: Mantiene historial del proceso

### Mejoras respecto a sesión anterior:

- Más específico en las correcciones requeridas
- Incluye fórmulas LaTeX listas para copiar
- Define métricas de éxito claras
- Prioriza correcciones por impacto
- Mantiene referencia a documentación existente
