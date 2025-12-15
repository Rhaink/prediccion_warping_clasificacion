# Sesión de Redacción 01 - Análisis y Primeras 5 Páginas

**Fecha:** 2025-12-15
**Estado:** Primera iteración completada, pendiente refinamiento

---

## 1. RESUMEN DE LO TRABAJADO

### Archivos Creados
| Archivo | Contenido | Estado |
|---------|-----------|--------|
| `8-Introduccion.tex` | Contexto, problema, pregunta, enfoque | Borrador inicial |
| `9-Hipotesis.tex` | Hipótesis, variables, predicciones | Borrador inicial |
| `10-Justificacion.tex` | Relevancia, contribuciones, limitaciones | Borrador inicial |
| `11-MarcoTeorico.tex` | Beer-Lambert, landmarks, CNNs, ResNet | Borrador inicial |
| `main.tex` | Actualizado con nuevas secciones | Funcional |

### Compilación
- **Páginas totales:** 12 (de 9 planeadas)
- **Estado:** Compila sin errores
- **PDF generado:** `main.pdf` (421 KB)

---

## 2. DEBILIDADES IDENTIFICADAS POR SECCIÓN

### Introducción (8-Introduccion.tex)
- [ ] Falta evidencia cuantitativa del problema (degradación sin normalización)
- [ ] No menciona COVID-19 explícitamente
- [ ] Warping afín no justificado (por qué esta técnica)
- [ ] Falta conexión con los 15 landmarks específicos

### Hipótesis (9-Hipotesis.tex)
- [ ] Hipótesis principal muy compleja (mezcla 3 conceptos)
- [ ] Thresholds no justificados (<5 px, 3×, 2×)
- [ ] Fill rate no definido (qué es 47% vs 96%)
- [ ] Métricas de accuracy/robustez sin formalismo

### Justificación (10-Justificacion.tex)
- [ ] Contribución 2 (mecanismo) es vaga
- [ ] Dataset externo (8,482) no caracterizado
- [ ] Impacto potencial especulativo
- [ ] Limitaciones segregadas al final

### Marco Teórico (11-MarcoTeorico.tex)
- [ ] No conecta Beer-Lambert con warping
- [ ] Landmarks asumen propiedades sin referencia a dataset
- [ ] Falta teoría de regresión de landmarks
- [ ] Falta teoría de warping afín
- [ ] Wing Loss nunca mencionado
- [ ] Transfer learning sin estrategia local

---

## 3. DIFERENCIAS DE ESTILO CON TESIS ORIGINAL

| Aspecto | Tesis Original | Nuevo Contenido | Acción |
|---------|----------------|-----------------|--------|
| Densidad párrafos | Alta (10-15 líneas) | Media (5-12) | Aumentar |
| Ecuaciones numeradas | 100% | <50% | Numerar todas |
| Referencias internas | Frecuentes | Pocas | Agregar 3-4/sección |
| Citas bibliográficas | 2-3/párrafo | Escasas | Duplicar |
| Estructura | Numerada | Sin números | Usar numeración |

---

## 4. DATOS OBLIGATORIOS PARA INCLUIR

### Predicción de Landmarks
- Error ensemble: **3.71 px** (±2.42)
- Mejor individual: 4.04 px
- Por categoría: Normal 3.42, COVID 3.77, Viral 4.40
- Límite teórico: ~1.3 px

### Clasificación
- warped_96: **99.10%** accuracy
- Robustez JPEG Q50: **5.3x mejor** que original
- Robustez JPEG Q30: 5.7x mejor
- Robustez Blur: 5.9x mejor

### Mecanismo Causal (Sesión 39)
- 75% por reducción de información
- 25% por normalización geométrica
- Experimento control: Original Cropped 47%

### Validación Externa (Sesión 55)
- FedCOVIDx: 8,482 muestras
- Accuracy: 53-55% (~random)
- Domain shift es problema FUNDAMENTAL

### Hallazgos Negativos
- PFS ~0.49 (no fuerza atención pulmonar)
- Generalización 2.4x (no 11x como se pensaba)

---

## 5. FÓRMULAS PENDIENTES DE AGREGAR

```latex
% Wing Loss
L(y, ŷ) = w·ln(1 + |y-ŷ|/ε)  si |y-ŷ| < w
          |y-ŷ| - C           si |y-ŷ| ≥ w

% Procrustes
R = V·U^T  donde SVD(X^T·Y) = U·S·V^T

% Fill Rate
fill_rate = 1 - (píxeles_negros / total_píxeles)

% PFS
PFS = sum(H·M) / sum(H)

% Degradación
degradation = (acc_original - acc_perturbed) × 100
```

---

## 6. CONEXIONES NARRATIVAS REQUERIDAS

1. **Introducción → Hipótesis:** "El problema de variabilidad geométrica nos lleva a formular..."
2. **Hipótesis → Justificación:** "Estas predicciones son relevantes porque..."
3. **Justificación → Marco Teórico:** "Para implementar esta solución, necesitamos fundamentos de..."
4. **Marco Teórico → Metodología:** "Aplicando estos conceptos, diseñamos..."

---

## 7. PRÓXIMOS PASOS (SESIÓN 02)

1. **Refinar estructura:** Usar numeración `\section{}` en lugar de `\section*{}`
2. **Agregar ecuaciones:** Numerar todas con `\label{eq:...}`
3. **Aumentar densidad:** Párrafos más largos con más detalles
4. **Agregar citas:** Mínimo 2-3 por sección
5. **Conectar secciones:** Referencias cruzadas explícitas
6. **Incluir datos:** Valores cuantitativos del proyecto
7. **Documentar limitaciones:** Integradas, no segregadas

---

## 8. ARCHIVOS DE REFERENCIA PARA SIGUIENTE SESIÓN

- Plan: `/home/donrobot/.claude/plans/velvet-frolicking-penguin.md`
- Claims validados: `/docs/CLAIMS_TESIS.md`
- Resumen defensa: `/docs/RESUMEN_DEFENSA.md`
- Sesiones clave: `/docs/sesiones/SESION_39_*.md`, `SESION_53_*.md`, `SESION_55_*.md`
- Tesis original (estilo): `/documentación/Tesis___Rafael_Alejandro_Cruz_Ovando/`
