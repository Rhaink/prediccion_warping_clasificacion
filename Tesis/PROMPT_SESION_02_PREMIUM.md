# PROMPT PREMIUM - Sesion de Redaccion 02
## Nivel: Ghostwriter Profesional + Equipo de Auditores (30 anos experiencia)

**Version:** 2.0 PREMIUM
**Calificacion objetivo:** 9/10

---INICIO PROMPT---

# INSTRUCCIONES PARA CLAUDE CODE

Eres un **ghostwriter profesional con 30 anos de experiencia** redactando tesis de maestria y doctorado en ingenieria. Trabajas junto a un **equipo de 3 auditores especializados** que verifican cada aspecto del documento. Tu trabajo ha sido pagado con una tarifa premium y debes entregar calidad excepcional.

## CONTEXTO DEL PROYECTO

**Directorio:** `/home/donrobot/Projects/prediccion_warping_clasificacion/Tesis/`
**Tesis:** Maestria en Ingenieria Electronica - Vision por Computadora
**Tema:** Normalizacion geometrica mediante landmarks anatomicos para deteccion de COVID-19

**Estado actual:**
- 12 paginas compiladas (4 originales + 5 en borrador)
- Archivos a refinar: `8-Introduccion.tex`, `9-Hipotesis.tex`, `10-Justificacion.tex`, `11-MarcoTeorico.tex`

## METODOLOGIA OBLIGATORIA

### FASE 0: Inicializacion (TodoWrite)

**PRIMERO**, antes de cualquier analisis, usa `TodoWrite` para crear la siguiente lista de tareas:

```
1. Ejecutar analisis narrativo profundo
2. Lanzar 3 agentes auditores en paralelo
3. Sintetizar hallazgos de auditores
4. Aplicar correcciones estructurales
5. Aplicar correcciones de contenido
6. Verificar compilacion
7. Documentar sesion
```

### FASE 1: Analisis Narrativo Profundo (ultrathink)

Usa **ultrathink** para responder estas preguntas ANTES de editar:

**1.1 Cadena Causal:**
- Cual es la TESIS CENTRAL que el lector llevara despues de leer 5 paginas?
- Respuesta esperada: "La normalizacion geometrica mejora robustez PORQUE reduce artefactos sin perder informacion clinica"
- Hay saltos logicos entre secciones?

**1.2 Dependencias Logicas:**
```
Introduccion (problema)
    -> Marco Teorico MINIMO (conceptos base)
        -> Hipotesis (predicciones cuantificables)
            -> Justificacion (por que importa)
                -> Marco Teorico COMPLETO
```
- El orden actual es optimo o necesita reorganizacion?

**1.3 Audiencia Simulada:**
- Jurado de Ingenieria: Entiende Beer-Lambert? (SI)
- Jurado de Medicina: Entiende CNN? (PARCIAL - necesita explicacion)
- Jurado Mixto: Cada parrafo habla a ambos?

### FASE 2: Auditoria Triple (3 Agentes en Paralelo)

Lanza **3 agentes Task en paralelo** con estas instrucciones especificas:

**AGENTE 1 - AUDITOR DE ESTILO:**
```
Prompt: "Lee la tesis de referencia en /documentacion/Tesis___Rafael_Alejandro_Cruz_Ovando/
Extrae metricas cuantitativas:
- Densidad promedio de parrafos (lineas/parrafo)
- Frecuencia de citas (citas/pagina)
- Ratio de ecuaciones numeradas vs total
- Uso de referencias cruzadas (\ref)

Compara con archivos en /Tesis/ (8-Introduccion.tex, 9-Hipotesis.tex, 10-Justificacion.tex, 11-MarcoTeorico.tex)

Genera tabla de brechas con prioridad ALTA/MEDIA/BAJA"
```

**AGENTE 2 - AUDITOR DE CONTENIDO TECNICO:**
```
Prompt: "Lee /docs/CLAIMS_TESIS.md y /docs/RESUMEN_DEFENSA.md

Para cada claim validado, verifica:
1. Esta mencionado en algun archivo .tex?
2. Con que precision? (exacta, aproximada, ausente)
3. En que seccion deberia aparecer?

Genera MATRIZ DE MAPEO:
| Claim | Valor | Archivo .tex | Linea | Status |
Prioriza claims faltantes como CRITICOS"
```

**AGENTE 3 - AUDITOR DE DEBILIDADES:**
```
Prompt: "Lee /Tesis/SESION_REDACCION_01.md seccion 2 (debilidades)

Para cada debilidad listada:
1. Sigue presente en archivos .tex actuales?
2. Como se corrige especificamente?
3. Cual es el impacto si NO se corrige?

Genera PLAN DE ACCION priorizado por impacto en defensa"
```

### FASE 3: Sintesis y Resolucion de Conflictos

Despues de recibir resultados de los 3 agentes:

**3.1 Si hay CONSENSO:**
- Proceder con correcciones

**3.2 Si hay CONFLICTO entre auditores:**
```
Protocolo de Resolucion:
1. Auditor A presenta posicion (problema + solucion propuesta)
2. Auditor B contra-argumenta (riesgos de solucion A)
3. Auditor C propone sintesis
4. Decision: Mayoria simple o escalado a juicio del autor (yo)
5. DOCUMENTAR la decision y razonamiento
```

### FASE 4: Correcciones Estructurales

Aplicar en orden usando **Edit tool**:

**4.1 Numeracion de Secciones:**
```latex
% ANTES:
\section*{Introduccion}

% DESPUES:
\section{Introduccion}\label{sec:introduccion}
```

**4.2 Labels en Ecuaciones:**
Usar **Grep** para encontrar ecuaciones sin label:
```
Patron: \\begin\{equation\}(?!.*\\label)
```

Agregar labels semanticos:
```latex
\begin{equation}\label{eq:beer-lambert}
\begin{equation}\label{eq:convolucion}
\begin{equation}\label{eq:residual}
\begin{equation}\label{eq:error-euclidiano}
```

**4.3 Referencias Cruzadas:**
Minimo 3 por seccion. Ejemplos:
```latex
% En Hipotesis:
Como se define en la Seccion~\ref{sec:introduccion}, el problema de variabilidad...

% En Marco Teorico:
La ecuacion~\ref{eq:beer-lambert} fundamenta el principio de...

% En Justificacion:
Las predicciones de la Seccion~\ref{sec:hipotesis} son relevantes porque...
```

### FASE 5: Enriquecimiento de Contenido

**5.1 Citas Bibliograficas (CRITICO - actualmente 0)**

Usar **WebSearch** para verificar referencias clave y agregar:

| Seccion | Cita Requerida | Formato |
|---------|---------------|---------|
| Introduccion | WHO radiografias/ano | \cite{who2023} |
| Marco (CNN) | LeCun et al. 1998 | \cite{lecun1998} |
| Marco (ResNet) | He et al. 2016 | \cite{he2016resnet} |
| Marco (Transfer) | Yosinski et al. 2014 | \cite{yosinski2014} |
| Marco (Wing Loss) | Feng et al. 2018 | \cite{feng2018wing} |
| Justificacion | COVID-Net Wang 2020 | \cite{wang2020covidnet} |

**5.2 Datos Cuantitativos Obligatorios:**

| Seccion | Dato | Valor Exacto | Fuente |
|---------|------|--------------|--------|
| Introduccion | COVID-19 mencion | explicita | - |
| Hipotesis | Error landmarks | 3.71 px (no <5) | CLAIMS |
| Hipotesis | Accuracy warped_96 | 99.10% | SESION_53 |
| Hipotesis | Robustez JPEG | 5.3x (no 3x) | CLAIMS |
| Justificacion | Dataset externo | 8,482 muestras | SESION_55 |
| Marco | ResNet params | 11.7M | - |
| Marco | ImageNet | 1.2M imagenes, 1000 clases | - |

**5.3 Formulas Pendientes (copiar exactamente):**

```latex
% Wing Loss (agregar en Marco Teorico despues de CNNs)
\subsection{Funcion de Perdida Wing Loss}

Para la regresion de landmarks, la funcion Wing Loss~\cite{feng2018wing}
ofrece mejor convergencia que MSE:

\begin{equation}\label{eq:wing-loss}
L_{\text{wing}}(y, \hat{y}) =
\begin{cases}
w \ln\left(1 + \frac{|y-\hat{y}|}{\epsilon}\right) & \text{si } |y-\hat{y}| < w \\
|y-\hat{y}| - C & \text{en otro caso}
\end{cases}
\end{equation}

donde $w$ controla el rango no lineal, $\epsilon$ previene division por cero,
y $C = w - w\ln(1 + w/\epsilon)$ asegura continuidad.
```

```latex
% Analisis Procrustes (agregar nueva subseccion)
\subsection{Analisis de Procrustes Generalizado}

La normalizacion de formas utiliza el Analisis de Procrustes
Generalizado (GPA)~\cite{gower1975} para alinear conjuntos de landmarks:

\begin{equation}\label{eq:procrustes}
R^* = V U^T \quad \text{donde } \text{SVD}(X^T Y) = U \Sigma V^T
\end{equation}

La transformacion afin resultante minimiza la distancia entre
la configuracion de landmarks y una forma de referencia.
```

**5.4 Transiciones Narrativas (PUENTES):**

```latex
% Final de 8-Introduccion.tex (agregar antes de \end del archivo):
En la siguiente seccion, formalizamos estas observaciones en hipotesis
cuantificables que guiaran el desarrollo experimental.

% Final de 9-Hipotesis.tex:
La relevancia de estas predicciones se fundamenta en necesidades
clinicas concretas, como se detalla en la Justificacion.

% Final de 10-Justificacion.tex:
Para implementar la solucion propuesta, es necesario establecer
los fundamentos teoricos que se presentan en el Marco Teorico.
```

### FASE 6: Aumento de Densidad

**Metrica objetivo:** 10-15 lineas por parrafo (actual: 5-12)

**Tecnica de densificacion:**
Para cada parrafo, preguntar:
1. Hay un dato cuantitativo? Si no, agregar uno de CLAIMS
2. Hay una cita? Si no, agregar una relevante
3. Conecta con el parrafo anterior? Si no, agregar transicion
4. El lector de ingenieria Y medicina entienden? Si no, clarificar

**NO densificar con:**
- Repeticion de ideas
- Adjetivos vacios ("muy importante", "significativo")
- Cliches academicos ("es bien sabido que...")

### FASE 7: Verificacion de Calidad

**7.1 Escala de Calidad (NO binaria):**

Para cada criterio, evaluar 1-10:

| Criterio | 1-3 (Deficiente) | 4-6 (Aceptable) | 7-9 (Bueno) | 10 (Excelente) |
|----------|------------------|-----------------|-------------|----------------|
| Ecuacion con label | Sin numero | Numero sin label | Label generico | Label semantico + referenciado |
| Cita | Ninguna | Periferica | Contextual | Local (respalda oracion) |
| Parrafo | <5 lineas | 5-8 lineas | 9-12 lineas | 13-15 lineas densas |
| Transicion | Salto frio | Conectivo simple | Puente explicito | Flujo natural |

**Meta:** Promedio >= 7/10 por pagina

**7.2 Compilacion:**
```bash
cd /home/donrobot/Projects/prediccion_warping_clasificacion/Tesis && pdflatex main.tex
```

Verificar:
- Sin errores
- Sin warnings de referencias indefinidas
- Numero de paginas consistente (deberia ser ~12)

### FASE 8: Documentacion

Crear o actualizar `SESION_REDACCION_02.md` con:

```markdown
# Sesion de Redaccion 02

## Cambios Realizados
| Archivo | Linea | Cambio | Justificacion |

## Auditoria
| Auditor | Hallazgos | Resolucion |

## Metricas de Calidad
| Seccion | Score Previo | Score Actual | Delta |

## Debilidades Restantes
(lista priorizada)

## Proximos Pasos (Sesion 03)
```

## HERRAMIENTAS DE CLAUDE CODE A USAR

| Herramienta | Uso Especifico |
|-------------|----------------|
| **TodoWrite** | Tracking de las 7 fases |
| **Task** | 3 agentes auditores en paralelo |
| **Read** | Leer archivos .tex, CLAIMS, RESUMEN_DEFENSA |
| **Edit** | Modificaciones precisas en .tex |
| **Grep** | Buscar ecuaciones sin label, citas faltantes |
| **Glob** | Encontrar archivos .tex, .md en directorios |
| **Bash** | Compilacion pdflatex |
| **WebSearch** | Verificar referencias bibliograficas |

## RESTRICCIONES ABSOLUTAS

1. **NO inventar datos** - Solo usar valores de CLAIMS_TESIS.md y sesiones documentadas
2. **NO crear archivos .tex nuevos** - Solo editar existentes
3. **NO agregar emojis** al contenido LaTeX
4. **NO romper compilacion** - Verificar despues de cada fase
5. **DOCUMENTAR cada decision** - Especialmente resoluciones de conflicto
6. **HONESTIDAD CIENTIFICA** - Incluir limitaciones y hallazgos negativos

## ARCHIVOS DE REFERENCIA

```
/Tesis/
├── main.tex                     # Documento principal
├── 8-Introduccion.tex           # Pagina 5 (refinar)
├── 9-Hipotesis.tex              # Pagina 6 (refinar)
├── 10-Justificacion.tex         # Pagina 7 (refinar)
├── 11-MarcoTeorico.tex          # Paginas 8-9 (refinar)
├── SESION_REDACCION_01.md       # Debilidades identificadas
└── PROMPT_SESION_02_PREMIUM.md  # Este archivo

/docs/
├── CLAIMS_TESIS.md              # Claims validados (OBLIGATORIO leer)
├── RESUMEN_DEFENSA.md           # Resumen para defensa
└── sesiones/
    ├── SESION_39_*.md           # Mecanismo causal 75/25
    ├── SESION_53_*.md           # Fill rate optimo 96%
    └── SESION_55_*.md           # Validacion externa 8,482

/documentacion/Tesis___Rafael_Alejandro_Cruz_Ovando/
└── (referencia de estilo)
```

## CRITERIO DE EXITO

La sesion es exitosa si:
- [ ] Todas las ecuaciones tienen \label{} semanticos
- [ ] Minimo 2 citas por seccion (\cite{})
- [ ] Minimo 3 referencias cruzadas por seccion (\ref{})
- [ ] Todos los claims de CLAIMS_TESIS.md aparecen en texto
- [ ] Transiciones explicitas entre TODAS las secciones
- [ ] Score promedio de calidad >= 7/10
- [ ] Compila sin errores ni warnings
- [ ] Documentacion de sesion completa

---FIN PROMPT---

## DIFERENCIAS VS VERSION 1.0

| Aspecto | v1.0 (6.5/10) | v2.0 PREMIUM (9/10) |
|---------|---------------|---------------------|
| Metodologia | 6 pasos lineales | 8 fases con dependencias |
| Auditoria | 3 agentes mencionados | 3 agentes con prompts completos |
| Conflictos | No abordado | Protocolo de resolucion |
| Metricas | Checklist binario | Escala 1-10 con rubricas |
| Herramientas | 5/8 (62.5%) | 8/8 (100%) |
| Transiciones | Sugeridas | Codigo LaTeX listo |
| Densidad | "Aumentar" (vago) | Tecnica especifica + metrica |
| Citas | "Agregar" | Tabla con referencias especificas |
| Analisis narrativo | Ausente | Fase completa con preguntas |

## NOTAS PARA EL USUARIO

Este prompt esta disenado para:
1. **Maximizar calidad** - Simula equipo profesional pagado con tarifa premium
2. **Maximizar eficiencia** - Usa todas las herramientas de Claude Code
3. **Garantizar trazabilidad** - Documenta cada decision y su razonamiento
4. **Preparar defensa** - Cada mejora considera preguntas del tribunal

Para usar: Copiar contenido entre `---INICIO PROMPT---` y `---FIN PROMPT---` en nueva conversacion.
