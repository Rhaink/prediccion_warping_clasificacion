# Documentación del Proyecto Fisher-Warping

Este directorio contiene toda la documentación del proyecto de clasificación de radiografías de tórax usando warping geométrico + PCA (Eigenfaces) + Fisher LDA + KNN.

---

## Índice de Documentos

### 📋 Documentos Principales

| Documento | Propósito | Cuándo leerlo |
|-----------|-----------|---------------|
| **00_OBJETIVOS.md** | Objetivos del proyecto y requisitos del asesor | Al inicio del proyecto |
| **01_MATEMATICAS.md** | Fundamentos matemáticos del pipeline | Al implementar cada fase |
| **02_PIPELINE.md** | Descripción completa del pipeline paso a paso | Durante implementación |
| **03_ASESOR_CHECKLIST.md** | Verificación de cumplimiento de requisitos | Antes de reuniones con asesor |
| **DOCUMENTO_FINAL.md** | Documento final consolidado (si existe) | Al finalizar proyecto |

### 🚨 Documentos Post-Error Crítico (2026-01-07)

| Documento | Propósito | Cuándo usarlo |
|-----------|-----------|---------------|
| **POST_MORTEM_ERROR_CRITICO.md** | Análisis completo del error crítico del 2026-01-07 | Leer para entender qué salió mal |
| **VERIFICATION_CHECKLIST.md** | Checklist obligatorio por fase | **DESPUÉS DE CADA FASE** |
| **CORRECTION_PLAN.md** | Plan detallado para corregir experimentos | En la próxima sesión de corrección |

---

## Flujo de Trabajo Recomendado

### Durante Implementación de una Fase

1. **ANTES de ejecutar código:**
   - Leer la sección correspondiente en `01_MATEMATICAS.md`
   - Leer la sección correspondiente en `02_PIPELINE.md`
   - Revisar `VERIFICATION_CHECKLIST.md` para saber qué verificar

2. **DURANTE la ejecución:**
   - Seguir el pipeline en `02_PIPELINE.md`
   - Documentar decisiones y problemas encontrados

3. **DESPUÉS de ejecutar:**
   - **OBLIGATORIO:** Completar checklist en `VERIFICATION_CHECKLIST.md`
   - Verificar coherencia con fase anterior
   - Documentar resultados

### Antes de Reunión con Asesor

1. Revisar `03_ASESOR_CHECKLIST.md` - Verificar todos los requisitos cumplidos
2. Revisar `00_OBJETIVOS.md` - Recordar objetivos principales
3. Verificar números clave en notebooks coinciden con `summary.json`
4. Preparar respuestas a preguntas típicas del asesor

### Si Encuentras un Error

1. **DETENER** - No continuar hasta entender el error
2. Documentar el error (qué, cuándo, cómo se detectó)
3. Investigar causa raíz (no solo síntomas)
4. Usar `POST_MORTEM_ERROR_CRITICO.md` como template
5. Implementar salvaguardas para prevenir recurrencia

---

## ¿Qué Pasó el 2026-01-07?

### Resumen del Error Crítico

Los experimentos de clasificación de 2 clases usaron el CSV incorrecto durante 3 días:
- **CSV usado:** `01_full_balanced_3class_warped.csv` (680 test)
- **CSV correcto:** `02_full_balanced_2class_warped.csv` (1,245 test)
- **Impacto:** 3 días de trabajo, resultados subóptimos, ratio de clases invertido

### Lecciones Aprendidas

1. **NUNCA asumir** - Siempre verificar explícitamente
2. **Validaciones automáticas** son obligatorias, no opcionales
3. **Documentación prescriptiva** (reglas), no solo descriptiva (hechos)
4. **Checklists obligatorios** después de cada fase crítica
5. **Verificar coherencia** input→output antes de continuar

### Salvaguardas Implementadas

Para asegurar que esto **NUNCA** vuelva a pasar:

| Salvaguarda | Documento | Obligatorio |
|-------------|-----------|-------------|
| Checklist por fase | `VERIFICATION_CHECKLIST.md` | ✅ SÍ |
| Reglas prescriptivas CSVs | `config/SPLIT_PROTOCOL.md` | ✅ SÍ |
| Validaciones en código | Asserts en `generate_features.py` | ✅ SÍ |
| Logging explícito | Prints en cada script | ✅ SÍ |
| Verificación pre-reunión | `03_ASESOR_CHECKLIST.md` | ✅ SÍ |

---

## Guía Rápida: ¿Qué Documento Leer?

### "Quiero entender el proyecto"
→ `00_OBJETIVOS.md` + `02_PIPELINE.md`

### "Estoy implementando la fase X"
→ `01_MATEMATICAS.md` (sección X) + `02_PIPELINE.md` (sección X)

### "Terminé una fase, ¿qué verificar?"
→ `VERIFICATION_CHECKLIST.md` (sección correspondiente)

### "Voy a reunión con asesor"
→ `03_ASESOR_CHECKLIST.md` + revisar notebooks 01-08

### "Encontré un error / algo no cuadra"
→ `VERIFICATION_CHECKLIST.md` para diagnosticar
→ `POST_MORTEM_ERROR_CRITICO.md` como referencia

### "Voy a corregir el error del CSV"
→ `CORRECTION_PLAN.md` paso a paso

### "¿Qué CSV usar para mi experimento?"
→ `config/SPLIT_PROTOCOL.md` (REGLAS DE USO)

---

## Estructura del Proyecto

```
prediccion_warping_clasificacion/
├── config/
│   ├── SPLIT_PROTOCOL.md          ← Reglas de CSVs (ACTUALIZADO 2026-01-07)
│   └── ...
├── docs/                           ← ESTÁS AQUÍ
│   ├── README.md                   ← Este archivo
│   ├── 00_OBJETIVOS.md
│   ├── 01_MATEMATICAS.md
│   ├── 02_PIPELINE.md              ← ACTUALIZADO 2026-01-07
│   ├── 03_ASESOR_CHECKLIST.md      ← ACTUALIZADO 2026-01-07
│   ├── POST_MORTEM_ERROR_CRITICO.md  ← NUEVO 2026-01-07
│   ├── VERIFICATION_CHECKLIST.md     ← NUEVO 2026-01-07
│   ├── CORRECTION_PLAN.md            ← NUEVO 2026-01-07
│   └── DOCUMENTO_FINAL.md
├── notebooks/
│   ├── 01_Intro_Contexto.ipynb
│   ├── 02_Fase1_PCA_Eigenfaces.ipynb
│   ├── 03_Fase2_Visualizacion_2D.ipynb
│   ├── 04_Fase3_Fisher.ipynb
│   ├── 05_Fase4_Amplificacion.ipynb
│   ├── 06_Fase5_KNN.ipynb
│   ├── 07_Fase6_ErrorAnalysis.ipynb
│   ├── 08_Hallazgos_Resultados.ipynb
│   └── 09_Fase7_3Clases.ipynb
├── src/
│   ├── generate_features.py        ← CORREGIR en próxima sesión
│   ├── generate_fisher.py
│   ├── generate_classification.py
│   └── ...
└── results/
    ├── metrics/
    │   ├── 01_full_balanced_3class_*.csv  (para 3 clases)
    │   ├── 02_full_balanced_2class_*.csv  (para 2 clases) ← USAR ESTE
    │   ├── phase4_features/
    │   ├── phase5_fisher/
    │   ├── phase6_classification/
    │   └── phase7_comparison/
    └── figures/
        └── ...
```

---

## Reglas de Oro (NUNCA ROMPER)

### 1. Verificación Explícita Siempre
```python
# ❌ MAL: Asumir que funciona
df = pd.read_csv(csv_path)

# ✅ BIEN: Verificar explícitamente
df = pd.read_csv(csv_path)
print(f"[INFO] CSV cargado: {csv_path}")
print(f"[INFO] Test size: {len(df[df['split']=='test'])}")
assert len(df[df['split']=='test']) == expected_size
```

### 2. Documentación Prescriptiva
```markdown
❌ MAL (descriptivo):
"Existen dos CSVs: 01_* y 02_*"

✅ BIEN (prescriptivo):
"SI experimento=2class ENTONCES usar 02_*, NUNCA 01_*"
```

### 3. Checklist Obligatorio
Después de cada fase:
- [ ] Completar checklist en `VERIFICATION_CHECKLIST.md`
- [ ] Verificar coherencia con fase anterior
- [ ] Documentar en bitácora

### 4. Coherencia Input→Output
Antes de continuar a siguiente fase:
- Verificar que output de fase N tiene el tamaño esperado
- Verificar que se usa como input correcto en fase N+1

### 5. No Hardcodear Sin Comentarios
```python
# ❌ MAL:
csv = "01_full_balanced_3class_warped.csv"

# ✅ BIEN:
# CSV para experimento de 2 clases - VERIFICADO 2026-01-07
csv = "02_full_balanced_2class_warped.csv"
```

---

## Contacto y Soporte

- **Asesor:** Revisar `00_OBJETIVOS.md` para requisitos
- **Errores críticos:** Usar `POST_MORTEM_ERROR_CRITICO.md` como template
- **Dudas sobre pipeline:** Consultar `02_PIPELINE.md`
- **Verificación:** Usar `VERIFICATION_CHECKLIST.md`

---

**Última actualización:** 2026-01-07
**Estado del proyecto:** Post error crítico - Corrección pendiente
**Próximos pasos:** Ejecutar `CORRECTION_PLAN.md` en próxima sesión
