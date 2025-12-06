# Resumen Ejecutivo: Plan de Validacion Externa

**Sesiones 36-41 | Diciembre 2024**

---

## OBJETIVO CENTRAL

Demostrar cientificamente si el warping mejora la **generalizacion real** (no solo transferencia entre representaciones del mismo dataset).

---

## DATASETS (Orden de prioridad)

| Prioridad | Dataset | Imagenes | Clases | Razon |
|-----------|---------|----------|--------|-------|
| 1ro | **Dataset3** (FedCOVIDx) | 84,818 | 2 (COVID+/-) | Mas diferente, multiples fuentes |
| 2do | **Dataset2** | 9,208 | 3 | Compatible con nuestros modelos |
| 3ro | Dataset1 (COVID-QU-Ex) | 67,840 | 3 | Similar al original |

---

## CRONOGRAMA DE SESIONES

### Sesion 36: Preparacion y Baseline
- Preprocesar Dataset3 (parsear TXT, resize a 299x299)
- Evaluar modelos existentes en Dataset3 SIN warpear
- Establecer baseline de degradacion

### Sesion 37: Warping y Cross-Evaluation
- Predecir landmarks en Dataset3
- Generar version warpeada de Dataset3
- Cross-evaluation completa: Original vs Warped en datos externos

### Sesion 38: Dataset2 y Ablacion
- Repetir experimentos con Dataset2 (3 clases)
- Experimento de ablacion: Warping vs Crop vs otras condiciones

### Sesion 39: Fondos y Analisis de Errores
- Probar alternativas al fondo negro
- Analisis de errores cruzados con Grad-CAM

### Sesion 40-41: Consolidacion
- Dataset1 si es necesario
- Documentacion final para tesis

---

## EXPERIMENTOS CLAVE

### 1. Validacion Externa (Sesiones 36-37)
```
Gap_Warped < Gap_Original en datos externos?

Gap = Accuracy_test_propio - Accuracy_dataset_externo
```

### 2. Ablacion (Sesion 38)
```
El beneficio viene del warping o de eliminar contexto?

Condiciones:
A) Original -> B) Warped -> C) Crop -> D) Warped+media -> E) Resize
```

### 3. Estrategias de Fondo (Sesion 39)
```
El fondo negro es optimo?

Probar: Negro, Gris, Media, Borde, Ruido, Contexto
```

---

## CRITERIOS DE EXITO

| Resultado | Interpretacion |
|-----------|----------------|
| Gap_Warped << Gap_Original | Warping mejora generalizacion |
| Gap_Warped ~= Gap_Original | No hay ventaja de generalizacion |
| Warped >> Crop | Warping aporta valor mas alla del crop |
| Warped ~= Crop | Beneficio es solo eliminar contexto |

---

## ARCHIVOS DE DOCUMENTACION

```
docs/
├── PLAN_VALIDACION_EXTERNA.md     # Plan detallado
└── RESUMEN_PLAN_SESIONES_36-41.md # Este documento

PROMPT_SESION_36.md  # Preparacion Dataset3
PROMPT_SESION_37.md  # Warping y cross-evaluation
PROMPT_SESION_38.md  # Dataset2 y ablacion
PROMPT_SESION_39.md  # Fondos y errores
```

---

## NOTAS IMPORTANTES

1. **Las mascaras de segmentacion de Dataset1 NO son utiles** porque nuestro warping incluye el mediastino (zona entre pulmones).

2. **Dataset3 tiene solo 2 clases** - mapear Normal+Viral_Pneumonia -> negative.

3. **El fondo negro se evaluara pero no es prioridad** - primero validar generalizacion.

4. **Si Gap_Warped ~= Gap_Original** debemos reformular la hipotesis a "transferencia entre representaciones".

---

## PREGUNTAS FINALES A RESPONDER

1. El warping mejora generalizacion a datos EXTERNOS?
2. O solo mejora transferencia entre representaciones del MISMO dataset?
3. El fondo negro es un problema?
4. El beneficio viene del warping o simplemente de eliminar contexto?
5. Que estrategia de fondo es optima?
6. Cual es la reformulacion correcta de la hipotesis?

---

*Documento creado: Sesion 35, 03-Dic-2024*
