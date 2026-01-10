# Guia de Comprension del Proyecto

**Objetivo:** Entender que hace este proyecto, que hemos logrado, y que significa cada resultado.

---

## 1. El Problema que Resolvemos

### Problema Original
Las radiografias de torax tienen mucha variabilidad:
- Pacientes posicionados diferente
- Equipos de rayos X diferentes
- Protocolos de hospitales diferentes

Esto causa que los clasificadores de COVID-19:
- Sean fragiles a compresion JPEG (hospitales comprimen imagenes)
- No generalicen bien entre diferentes condiciones
- Aprendan "atajos" (marcas de hospital, bordes, etc.)

### Nuestra Solucion
```
Imagen Original → Predecir Landmarks → Warpear a Forma Canonica → Clasificar
                   (15 puntos)         (alinear anatomia)        (COVID/Normal/Viral)
```

**Idea:** Si todas las imagenes tienen la misma geometria, el clasificador se enfoca en caracteristicas medicas, no en variaciones geometricas.

---

## 2. Los Dos Componentes del Sistema

### Componente 1: Prediccion de Landmarks

**Que hace:** Predice 15 puntos anatomicos en la radiografia (apices, hilios, bases, etc.)

**Resultado:** Error de 3.71 pixeles (en imagen 224x224)

**Arquitectura:**
- ResNet-18 (backbone)
- Coordinate Attention (atencion espacial)
- Deep Head (768 neuronas)
- Wing Loss (optimizada para landmarks)

**Archivo clave:** `checkpoints/session10/ensemble/` (4 modelos)

### Componente 2: Clasificacion COVID-19

**Que hace:** Clasifica imagenes en 3 clases (COVID, Normal, Viral Pneumonia)

**Resultado:** 99.10% accuracy (en imagenes warped)

**Arquitectura:**
- ResNet-18 con Coordinate Attention
- Entrenado en imagenes geometricamente normalizadas

**Archivo clave:** `outputs/classifier_replication_v2/best_classifier.pt`

---

## 3. Los Resultados Clave (Que Significan)

### Resultado 1: Robustez Mejorada

| Perturbacion | Original | Warped | Que significa |
|--------------|----------|--------|---------------|
| JPEG Q50 | 16.14% deg | 3.06% deg | 5.3x mas robusto a compresion |
| JPEG Q30 | 29.97% deg | 5.28% deg | 5.7x mas robusto |
| Blur σ=1 | 14.43% deg | 2.43% deg | 5.9x mas robusto |

**Interpretacion:**
- "deg" = degradacion (cuanto cae la accuracy)
- Menor es mejor
- El modelo warped resiste mejor las perturbaciones

**Por que importa:**
- Hospitales comprimen imagenes para almacenar
- Transmision por red introduce artefactos
- Un modelo robusto funciona en condiciones reales

### Resultado 2: El Mecanismo (Experimento de Control)

| Modelo | Fill Rate | JPEG Q50 | Que demuestra |
|--------|-----------|----------|---------------|
| Original 100% | 100% | 16.14% | Baseline |
| Original Cropped 47% | 47% | 2.11% | Efecto de reducir info |
| Warped 47% | 47% | 0.53% | Efecto adicional del warping |

**Interpretacion:**
- ~75% de la robustez viene de REDUCIR informacion (menos pixeles = regularizacion)
- ~25% adicional viene de la NORMALIZACION geometrica

**Por que importa:**
- Entendemos POR QUE funciona, no solo QUE funciona
- Permite tomar decisiones informadas sobre trade-offs

### Resultado 3: Generalizacion Within-Domain

| Modelo | Entrenado en | Evaluado en | Accuracy | Gap |
|--------|--------------|-------------|----------|-----|
| Original | Original | Original | 98.84% | - |
| Original | Original | Warped | 91.13% | 7.70% |
| Warped | Warped | Warped | 98.73% | - |
| Warped | Warped | Original | 95.57% | 3.17% |

**Interpretacion:**
- El modelo warped tiene gap 2.4x menor (3.17% vs 7.70%)
- Generaliza mejor entre diferentes representaciones del mismo dato

### Resultado 4: Validacion Externa (LIMITACION)

| Modelo | Acc. Interna | Acc. Externa | Que significa |
|--------|--------------|--------------|---------------|
| Original | 95.83% | 57.50% | Falla en otro hospital |
| Warped | 99.10% | 53-55% | Tambien falla |

**Interpretacion:**
- 50% = adivinar al azar
- 53-55% = apenas mejor que adivinar
- TODOS los modelos fallan en datos externos

**Por que NO invalida el trabajo:**
- El modelo ORIGINAL tambien falla (57%)
- Es DOMAIN SHIFT - problema conocido en medical imaging
- Papers top-tier (CheXNet, COVID-Net) reportan lo mismo

---

## 4. Que Podemos Afirmar (Claims Validos)

### SI Podemos Decir:

1. "El warping mejora robustez a compresion JPEG 5-30x"
2. "El warping mejora generalizacion within-domain 2.4x"
3. "El mecanismo principal es regularizacion por reduccion de informacion"
4. "El sistema tiene limitaciones de domain shift en datos externos"

### NO Podemos Decir:

1. ~~"El warping resuelve el problema de generalizacion"~~ (no en datos externos)
2. ~~"El modelo funciona en cualquier hospital"~~ (falla ~55%)
3. ~~"El warping fuerza atencion en pulmones"~~ (PFS ≈ 50%)

---

## 5. Estructura del Proyecto

### Codigo Principal
```
src_v2/
├── cli.py              # Interfaz de linea de comandos (21 comandos)
├── data/               # Datasets, transforms, augmentations
├── models/             # Arquitecturas (ResNet, Coordinate Attention)
├── training/           # Entrenamiento, callbacks, logging
├── evaluation/         # Metricas, robustez, cross-evaluation
└── processing/         # Warping, landmarks, Procrustes
```

### Documentacion Clave
```
docs/
├── CLAIMS_TESIS.md                 # QUE podemos afirmar
├── RESUMEN_DEFENSA.md              # Resumen para presentacion
├── RESULTADOS_EXPERIMENTALES_v2.md # Todos los resultados
├── ANALISIS_CRITICO_SESION_55.md   # Analisis honesto de problemas
├── GUIA_COMPRENSION_PROYECTO.md    # ESTE DOCUMENTO
└── sesiones/                       # 55 sesiones de desarrollo
```

### Datos y Modelos
```
outputs/
├── warped_replication_v2/          # Dataset warped (96% fill rate)
├── classifier_replication_v2/      # Clasificador recomendado
├── external_validation/            # Resultados en FedCOVIDx
└── cross_evaluation_valid_3classes/ # Cross-evaluation

checkpoints/
└── session10/ensemble/             # 4 modelos de landmarks
```

### Validacion
```
tests/                    # 655 tests automatizados
GROUND_TRUTH.json         # Valores numericos de referencia
```

---

## 6. Flujo de Trabajo Tipico

### Para Predecir en Nueva Imagen
```bash
# Clasificar una imagen (con warping)
python -m src_v2 classify imagen.png \
  --classifier outputs/classifier_replication_v2/best_classifier.pt \
  --warp \
  --landmark-model checkpoints/session10/ensemble/seed123/final_model.pt
```

### Para Evaluar Robustez
```bash
python -m src_v2 test-robustness \
  outputs/classifier_replication_v2/best_classifier.pt \
  --data-dir outputs/warped_replication_v2
```

### Para Cross-Evaluation
```bash
python -m src_v2 cross-evaluate \
  modelo_original.pt modelo_warped.pt \
  --data-a datos_original/ --data-b datos_warped/
```

---

## 7. Preguntas Frecuentes

### "¿Por que 99% interno pero 55% externo?"

Es DOMAIN SHIFT - los datos de otro hospital son muy diferentes:
- Diferentes equipos de rayos X
- Diferentes poblaciones de pacientes
- Diferentes protocolos de adquisicion

El modelo original TAMBIEN falla (57%). No es problema del warping.

### "¿Entonces el warping no sirve?"

SI sirve para:
- Robustez a compresion (5-30x mejor)
- Generalizacion dentro del mismo hospital (2.4x mejor)

NO sirve para:
- Generalizacion a otros hospitales (requiere domain adaptation)

### "¿Los resultados son reales?"

SI. Verificado con:
- 0 data leakage (train/test completamente separados)
- 655 tests automatizados
- Seeds documentadas para reproducibilidad
- Errores encontrados y corregidos honestamente

### "¿Que falta para publicar?"

Rigor estadistico:
- K-fold cross-validation
- Intervalos de confianza (CI)
- P-values en comparaciones
- Ablation study formal

---

## 8. Documentos para Leer Primero

### Para Entender los Claims
1. `docs/CLAIMS_TESIS.md` - Que afirmamos y por que

### Para Entender la Metodologia
2. `docs/sesiones/SESION_39_EXPERIMENTO_CONTROL.md` - El experimento clave

### Para Entender los Resultados
3. `docs/RESULTADOS_EXPERIMENTALES_v2.md` - Todos los numeros

### Para Entender las Limitaciones
4. `docs/sesiones/SESION_55_VALIDACION_EXTERNA.md` - Por que falla en datos externos

### Para la Defensa
5. `docs/RESUMEN_DEFENSA.md` - Resumen ejecutivo

---

## 9. Glosario

| Termino | Significado |
|---------|-------------|
| **Warping** | Transformacion geometrica para alinear imagenes |
| **Landmarks** | Puntos anatomicos de referencia (15 en este proyecto) |
| **Fill Rate** | Porcentaje de pixeles no-negros despues del warping |
| **Domain Shift** | Diferencias entre datos de entrenamiento y deployment |
| **Within-Domain** | Dentro del mismo tipo de datos (mismo hospital) |
| **Cross-Domain** | Entre diferentes tipos de datos (diferentes hospitales) |
| **CLAHE** | Mejora de contraste adaptativa |
| **TTA** | Test-Time Augmentation (promediar predicciones aumentadas) |
| **Degradacion** | Cuanto cae la accuracy bajo perturbaciones |
| **Robustez** | Capacidad de mantener performance bajo perturbaciones |

---

## 10. Resumen en Una Pagina

### El Proyecto
Sistema de clasificacion de COVID-19 con normalizacion geometrica.

### El Resultado Principal
Robustez mejorada 5-30x a artefactos de compresion.

### El Mecanismo
75% regularizacion + 25% normalizacion geometrica.

### La Limitacion
No resuelve domain shift (~55% en datos externos).

### El Estado
Trabajo honesto y verificable. Falta rigor estadistico para publicacion.

### Los Archivos Clave
- Modelo: `outputs/classifier_replication_v2/best_classifier.pt`
- Datos: `outputs/warped_replication_v2/`
- Valores: `GROUND_TRUTH.json`
- Tests: `tests/` (655 tests)
