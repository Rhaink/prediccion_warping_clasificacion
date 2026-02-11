# Índice Maestro - Documentación Técnica

**Sistema de Detección Automática de Landmarks Pulmonares y Clasificación COVID-19**

**Estancia de Investigación INAOE**
**9 octubre - 9 noviembre 2025**

---

## Bienvenida

Este índice es el **punto de entrada único** a toda la documentación técnica del sistema. Según su rol y objetivos, siga una de las rutas recomendadas abajo.

**¿Es su primera vez con este sistema?** → Comience con [Rutas Recomendadas](#rutas-recomendadas)

---

## Información del Proyecto

**Título:** Normalización y alineación automática de la forma de la región pulmonar integrada con selección de características discriminantes para detección automática de neumonía y COVID-19.

**Estudiante:** Rafael Alejandro Cruz Ovando
**Institución:** Benemérita Universidad Autónoma de Puebla (BUAP)

**Director:** Dr. Leopoldo Altamirano Robles
**Institución:** Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE)
**Email:** robles@inaoep.mx

**Período:** 9 de octubre - 9 de noviembre de 2025
**Fecha de entrega:** 28 de enero de 2026

---

## Resultados Principales

Este sistema alcanzó los siguientes resultados validados:

### Detección de Landmarks Pulmonares
- **Error medio (ensemble + TTA):** 3.61 píxeles (en imágenes 224×224)
- **Desviación estándar:** 2.48 px
- **Error mediano:** 3.07 px
- **Ensemble:** 4 modelos ResNet-18 + Coordinate Attention
- **Mejora vs. mejor individual:** 12.0% (0.49 px)

### Clasificación COVID-19
- **Accuracy (5-fold CV):** 98.60% ± 0.26%
- **F1-macro (5-fold CV):** 98.00% ± 0.36%
- **Ensemble test (con TTA):** 98.26% accuracy, 97.12% F1-macro
- **Clases:** COVID, Normal, Viral Pneumonia
- **Dataset:** 15,153 radiografías de tórax

**Fuente:** Todas las métricas validadas en `GROUND_TRUTH.json` v2.1.0 (13 enero 2026)

---

## Rutas Recomendadas

Seleccione la ruta que mejor describe su objetivo:

### 🎯 Ruta 1: Revisor Académico

**Objetivo:** Validar que los resultados reportados son reproducibles.

**Tiempo estimado:** 2-3 horas

**Secuencia:**
1. **`01_GUIA_INICIO_RAPIDO.md`** (30 min)
   - Instalación básica y ejecución rápida del sistema
2. **`05_REPRODUCIBILIDAD_COMPLETA.md`** (1.5 horas)
   - Reproducción paso a paso de las métricas reportadas
3. **`GROUND_TRUTH.json`** (5 min)
   - Comparar sus resultados con las métricas de referencia
4. **`AUDITORIA_REPORTE_INAOE.md`** (15 min)
   - Auditoría de cumplimiento vs. plan de trabajo firmado

**Meta:** Confirmar que error de landmarks = 3.61 px y accuracy de clasificación ≈ 98.6%.

---

### 🔧 Ruta 2: Usuario Final

**Objetivo:** Usar el sistema pre-entrenado para clasificar nuevas radiografías.

**Tiempo estimado:** 1 hora

**Secuencia:**
1. **`01_GUIA_INICIO_RAPIDO.md`** (30 min)
   - Instalación e inferencia básica
2. **`03_GUIA_USO_CLI.md`** → Sección "Workflows Comunes" (20 min)
   - Comandos para inferencia con modelos pre-entrenados
3. **`08_FORMATOS_DATOS.md`** (10 min)
   - Preparar sus propios datos en el formato correcto

**Meta:** Clasificar radiografías de tórax en COVID/Normal/Viral Pneumonia usando el ensemble.

---

### 🧪 Ruta 3: Investigador (Extender el Trabajo)

**Objetivo:** Modificar el sistema, experimentar con nuevos parámetros, o agregar funcionalidades.

**Tiempo estimado:** 4-6 horas

**Secuencia:**
1. **`02_INSTALACION_REQUISITOS.md`** (45 min)
   - Instalación completa incluyendo entorno de desarrollo
2. **`04_ARQUITECTURA_CODIGO.md`** (1.5 horas)
   - Entender la estructura del código (~25K líneas en 7 módulos)
3. **`06_CONFIGURACIONES_JSON.md`** (45 min)
   - Sistema de configuración para experimentación
4. **`07_MODELOS_ENTRENADOS.md`** (30 min)
   - Detalles del ensemble de 4 modelos
5. **`03_GUIA_USO_CLI.md`** (1 hora)
   - Referencia completa de los 40+ comandos disponibles
6. **`08_FORMATOS_DATOS.md`** (30 min)
   - Especificación de todos los formatos de entrada/salida

**Meta:** Tener conocimiento profundo para modificar arquitecturas, experimentar con hiperparámetros, o agregar nuevas funcionalidades.

---

### 📚 Ruta 4: Estudiante (Aprender la Metodología)

**Objetivo:** Entender el pipeline completo y la metodología científica aplicada.

**Tiempo estimado:** 8-10 horas (estudio profundo)

**Secuencia:**
1. **`00_LEEME.txt`** (15 min)
   - Contexto general del proyecto (en carpeta raíz del USB)
2. **`01_GUIA_INICIO_RAPIDO.md`** (30 min)
   - Familiarizarse con el sistema ejecutándolo
3. **`04_ARQUITECTURA_CODIGO.md`** (2 horas)
   - Entender arquitectura: GPA, warping, ResNet-18, ensemble
4. **`07_MODELOS_ENTRENADOS.md`** (1 hora)
   - Proceso de entrenamiento en 2 fases, Wing Loss, TTA
5. **`06_CONFIGURACIONES_JSON.md`** (1 hora)
   - Entender decisiones de hiperparámetros (margin=1.05, tile_size=4, etc.)
6. **`05_REPRODUCIBILIDAD_COMPLETA.md`** (2 horas)
   - Reproducir experimentos para consolidar aprendizaje
7. **`03_GUIA_USO_CLI.md`** (1 hora)
   - Dominar comandos para experimentación propia
8. **`09_GLOSARIO_TERMINOS.md`** (opcional, 30 min)
   - Clarificar términos técnicos
9. **Reporte ESTANCIA INAOE** (carpeta `01_Reporte/`) (2 horas)
   - Leer el reporte académico completo

**Meta:** Dominar la metodología completa para aplicarla en proyectos propios o continuar esta línea de investigación.

---

## Índice Completo de Documentos

### Documentos de Inicio Rápido

#### `01_GUIA_INICIO_RAPIDO.md` (9.5 KB)
**Ejecutar el Sistema en <30 Minutos**

Instalación básica, ejecución del pipeline completo con scripts automáticos, y verificación de resultados.

**Cuándo leer:** Siempre primero. Permite validar rápidamente que el sistema funciona.

**Contenido clave:**
- Instalación en 3 pasos (venv, pip, organizar archivos)
- Ejecución con `quickstart_warping.sh`
- Verificación de salidas (forma canónica, predicciones, dataset warped)
- Errores comunes y soluciones

---

### Documentos de Instalación

#### `02_INSTALACION_REQUISITOS.md` (21 KB)
**Instalación Completa y Troubleshooting**

Guía exhaustiva de instalación con requisitos de hardware, dependencias explicadas, instalación paso a paso, y troubleshooting detallado.

**Cuándo leer:** Si la instalación rápida falla, o si necesita instalación para desarrollo.

**Contenido clave:**
- Requisitos de hardware (mínimos y recomendados)
- Dependencias Python explicadas (¿por qué PyTorch?, ¿por qué OpenCV?)
- Instalación con GPU (CUDA/ROCm)
- Troubleshooting de 15+ problemas comunes
- Verificación completa de instalación

---

### Documentos de Uso

#### `03_GUIA_USO_CLI.md` (29 KB)
**Referencia Completa de la Interfaz de Línea de Comandos**

Documentación exhaustiva de los 7 comandos principales del pipeline y workflows comunes.

**Cuándo leer:** Para entender cómo ejecutar cada paso del pipeline manualmente o crear workflows personalizados.

**Contenido clave:**
- 7 comandos principales documentados:
  1. `compute-canonical` - Forma canónica via GPA
  2. `evaluate-ensemble` - Evaluar ensemble de landmarks
  3. `generate-dataset` - Generar dataset warped
  4. `train-classifier` - Entrenar clasificador
  5. `evaluate-classifier` - Evaluar clasificador
  6. `cross-validate-classifier` - Validación cruzada
  7. `evaluate-classifier-ensemble` - Ensemble de clasificadores
- Sintaxis completa, ejemplos concretos, entradas y salidas
- 3 workflows completos (desde cero, solo inferencia, experimentación)

---

#### `05_REPRODUCIBILIDAD_COMPLETA.md` (22 KB)
**Reproducción Paso a Paso de Resultados Reportados**

Guía detallada para reproducir exactamente las métricas del reporte (3.61 px landmarks, 98.60% clasificación).

**Cuándo leer:** Para validación académica o cuando necesite estar 100% seguro de la reproducibilidad.

**Contenido clave:**
- Objetivos de reproducción (métricas específicas a alcanzar)
- 3 fases: (1) Validar ensemble landmarks, (2) Generar dataset normalizado, (3) Clasificación CV
- Comandos exactos con salidas esperadas
- Checklist de verificación completo
- Troubleshooting de reproducción (¿qué hacer si métricas no coinciden?)
- Tiempos estimados por fase (GPU vs. CPU)

---

### Documentos Técnicos

#### `04_ARQUITECTURA_CODIGO.md` (36 KB)
**Guía de la Estructura del Sistema para Modificaciones**

Descripción completa de la arquitectura del código (~25K líneas en 7 módulos principales).

**Cuándo leer:** Antes de modificar el código o agregar funcionalidades.

**Contenido clave:**
- Estadísticas del proyecto (43 archivos Python, 7 módulos, 25K líneas)
- Estructura de directorios con tamaño y propósito de cada módulo
- 7 módulos principales documentados:
  - `models/` - ResNet-18, classifier, loss functions
  - `processing/` - GPA, piecewise affine warping
  - `data/` - LandmarkDataset, CLAHE, TTA
  - `training/` - Two-phase trainer, callbacks
  - `evaluation/` - Metrics, ensemble evaluation
  - `visualization/` - GradCAM, ROC curves, científicas
  - `cli.py` - 40+ comandos (~10,895 líneas)
- Flujo de datos completo (diagrama ASCII de 6 fases)
- Patrones de diseño (Factory, Strategy, Observer, Singleton)
- Puntos de extensión (cómo agregar: backbones, pérdidas, métricas, transformaciones, comandos)
- Convenciones del código (naming, type hints, docstrings, imports)

---

#### `06_CONFIGURACIONES_JSON.md` (25 KB)
**Guía Completa del Sistema de Configuración**

Explicación del sistema de configs JSON para reproducibilidad y experimentación.

**Cuándo leer:** Antes de ejecutar comandos complejos o al crear experimentos personalizados.

**Contenido clave:**
- ¿Por qué configs JSON? (Reproducibilidad, versionamiento, legibilidad)
- 4 configuraciones críticas documentadas:
  1. `ensemble_best.json` - Ensemble de 4 modelos (3.61 px)
  2. `landmarks_train_base.json` - Entrenamiento en 2 fases
  3. `warping_best.json` - Parámetros óptimos (margin=1.05, fill=47%)
  4. `classifier_warped_base.json` - Clasificador ResNet-18
- Referencia completa de ~30 parámetros (tabla maestra)
- Cómo crear configs personalizados (3 ejemplos paso a paso)
- Validación de configs (manual y automática)
- Tips y mejores prácticas (versionamiento, documentar cambios, separar por experimento)

---

#### `07_MODELOS_ENTRENADOS.md` (19 KB)
**Documentación del Ensemble de Modelos de Landmarks**

Descripción técnica de los 4 modelos que componen el ensemble (3.61 px).

**Cuándo leer:** Para entender cómo se entrenaron los modelos o cómo usarlos programáticamente.

**Contenido clave:**
- Métricas del ensemble (3.61 px, mejora del 12% vs. best individual)
- 4 modelos individuales documentados (seeds 123, 321, 111, 666)
- Arquitectura completa (ResNet-18 + CoordinateAttention + DeepHead)
- Proceso de entrenamiento:
  - Fase 1: Backbone congelado (15 epochs, LR=1e-3)
  - Fase 2: Fine-tuning (100 epochs, LR diferenciados 2e-5 backbone, 2e-4 head)
  - Wing Loss (¿por qué? ~0.3 px mejor que MSE)
- Cómo cargar y usar modelos (código Python completo)
- Cuándo usar ensemble vs. individual (tabla comparativa)
- Troubleshooting (OOM, predicciones fuera de rango, TTA errors)

---

#### `08_FORMATOS_DATOS.md` (21 KB)
**Especificación Completa de Formatos de Entrada y Salida**

Documentación exhaustiva de todos los formatos de datos del sistema.

**Cuándo leer:** Al preparar datos propios o al entender las salidas del sistema.

**Contenido clave:**
- Dataset de entrada (PNG 299×299, estructura de directorios, 15,153 imágenes)
- Formato de anotaciones (CSV con 32 columnas: nombre, categoría, 30 coordenadas)
- Especificación de 15 landmarks (orden, ubicación anatómica, pares simétricos)
- Formato de predicciones (.npz cache: predictions, image_paths, metadata)
- Dataset normalizado (PNG 224×224, splits 75/12.5/12.5, fill_rate ~47%)
- Salidas de clasificación (CSV/JSON con predicciones, probabilidades, logits)
- Formatos auxiliares (forma canónica, triangulación, historial de entrenamiento)
- Conversiones comunes (normalizar/denormalizar, reshape, clase↔índice)
- Validación de formatos (código Python para validar imágenes y CSVs)

---

### Documentos de Referencia

#### `GROUND_TRUTH.json` (17 KB)
**Métricas Validadas Experimentalmente**

Archivo JSON con todas las métricas de referencia del proyecto (versión 2.1.0, validada 13 enero 2026).

**Cuándo leer:** Para comparar sus resultados con los valores de referencia oficiales.

**Contenido clave:**
- Ensemble landmarks: 3.61 px (TTA + 4 modelos)
- Modelos individuales: seed666 (4.10 px), seed456 (4.04 px histórico)
- Clasificación warped_lung_best: 98.05% accuracy, 97.12% F1-macro
- Cross-validation: 98.60% ± 0.26% accuracy, 98.00% ± 0.36% F1-macro
- Ensemble clasificadores con TTA: 98.26% accuracy, +3 casos vs. sin TTA
- Parámetros óptimos: margin=1.05, clahe_tile=4
- Histórico de métricas obsoletas (con razones de obsolescencia)

---

#### `AUDITORIA_REPORTE_INAOE.md` (11 KB)
**Auditoría del Reporte vs. Plan de Trabajo Firmado**

Documento de auditoría que verifica el cumplimiento del reporte respecto al plan de trabajo firmado.

**Cuándo leer:** Para entender la alineación entre objetivos propuestos y resultados obtenidos.

**Contenido clave:**
- Verificación de cumplimiento de los 5 objetivos específicos
- Cronograma ejecutado vs. planificado (4 semanas)
- Entregables verificados (código, checkpoints, reporte, documentación)
- Análisis de desviaciones y mejoras no planificadas
- Conclusión: 100% de objetivos cumplidos + resultados superiores a metas

---

### Documentos Opcionales

#### `09_GLOSARIO_TERMINOS.md` (Opcional)
**Definiciones de Términos Técnicos**

Glosario de términos de Machine Learning, Medical Imaging, y específicos del proyecto.

**Cuándo leer:** Si encuentra términos técnicos que no entiende en otros documentos.

**Contenido esperado:**
- Machine Learning: Ensemble, TTA, Fine-tuning, Backbone, Overfitting, Early Stopping
- Medical Imaging: Landmarks, CLAHE, ROI, Fill Rate, Radiografía PA
- Proyecto: GPA, Piecewise Affine Warp, Wing Loss, Coordinate Attention
- Acrónimos: TTA, CLAHE, SAHS, CV, GPA, ROI, etc.

---

#### `10_PREGUNTAS_FRECUENTES.md` (Opcional)
**Preguntas Frecuentes**

Respuestas a preguntas comunes sobre el sistema.

**Cuándo leer:** Si tiene preguntas no respondidas en otros documentos.

**Contenido esperado:**
- ¿Cuánto tarda entrenar desde cero? (Landmarks: ~5-7 h/modelo, Clasificador: ~2-3 h)
- ¿Puedo usar mis propias radiografías? (Sí, ver `08_FORMATOS_DATOS.md`)
- ¿Qué hardware necesito? (Mín: CPU + 8GB RAM, Rec: GPU + 16GB RAM)
- ¿Por qué margin=1.05? (Optimizado por grid search, balance fill/contenido)
- ¿Por qué tile_size=4 y no 8? (Validado experimentalmente, mejor contrast)
- ¿Funciona offline? (Sí, una vez instalado)
- ¿Cómo citar este trabajo? (Ver reporte en `01_Reporte/`)

---

### Archivos de Configuración

#### `configs/` (1.6 KB total)

Directorio con las 4 configuraciones JSON críticas:

1. **`ensemble_best.json`** (336 B)
   - 4 modelos del ensemble (seeds 123, 321, 111, 666)
   - TTA: true, CLAHE: true

2. **`landmarks_train_base.json`** (439 B)
   - Two-phase training (15 epochs frozen, 100 fine-tune)
   - Learning rates: 1e-3 (phase1), 2e-5 backbone, 2e-4 head (phase2)
   - Coord Attention, Deep Head, Wing Loss

3. **`warping_best.json`** (544 B)
   - Margin: 1.05, use_full_coverage: false
   - Splits: 75/12.5/12.5, seed: 42
   - CLAHE: clip=2.0, tile=4

4. **`classifier_warped_base.json`** (255 B)
   - ResNet-18, 50 epochs, batch=32, lr=1e-4
   - use_class_weights: true (crítico para dataset desbalanceado)

---

### Otros Archivos

#### `requirements.txt` (47 líneas)
Lista de dependencias Python con versiones específicas.

Ver `02_INSTALACION_REQUISITOS.md` para explicación de cada dependencia.

---

## Mapa de Navegación Rápida

### Por Tarea

| Tarea | Documento(s) Recomendado(s) |
|-------|----------------------------|
| **Instalar el sistema** | `01_GUIA_INICIO_RAPIDO.md` → `02_INSTALACION_REQUISITOS.md` |
| **Reproducir métricas** | `05_REPRODUCIBILIDAD_COMPLETA.md` + `GROUND_TRUTH.json` |
| **Usar modelos pre-entrenados** | `03_GUIA_USO_CLI.md` → "Workflows Comunes" → "Solo Inferencia" |
| **Entrenar desde cero** | `06_CONFIGURACIONES_JSON.md` → `03_GUIA_USO_CLI.md` → `05_REPRODUCIBILIDAD_COMPLETA.md` |
| **Modificar código** | `04_ARQUITECTURA_CODIGO.md` → `06_CONFIGURACIONES_JSON.md` |
| **Preparar mis datos** | `08_FORMATOS_DATOS.md` |
| **Entender ensemble** | `07_MODELOS_ENTRENADOS.md` |
| **Resolver error** | `02_INSTALACION_REQUISITOS.md` → "Troubleshooting" |
| **Crear experimento** | `06_CONFIGURACIONES_JSON.md` → "Crear Configs Personalizados" |

### Por Nivel de Urgencia

**🔴 Crítico (Leer primero):**
- `01_GUIA_INICIO_RAPIDO.md`
- `GROUND_TRUTH.json`

**🟡 Importante (Leer si trabaja con el sistema):**
- `03_GUIA_USO_CLI.md`
- `05_REPRODUCIBILIDAD_COMPLETA.md`
- `06_CONFIGURACIONES_JSON.md`

**🟢 Profundización (Leer para expertise):**
- `04_ARQUITECTURA_CODIGO.md`
- `07_MODELOS_ENTRENADOS.md`
- `08_FORMATOS_DATOS.md`

**⚪ Referencia (Consultar cuando necesario):**
- `02_INSTALACION_REQUISITOS.md`
- `AUDITORIA_REPORTE_INAOE.md`
- `09_GLOSARIO_TERMINOS.md`
- `10_PREGUNTAS_FRECUENTES.md`

---

## Documentos Externos (Otros Directorios del USB)

Este índice cubre la documentación en `05_Documentacion/`. Otros documentos importantes en el USB:

### `00_LEEME.txt` (Raíz del USB)
README general del USB con contexto del proyecto, contenido completo, comandos principales, y verificación de integridad.

### `01_Reporte/`
- **`REPORTE_ESTANCIA_INAOE.pdf`** - Reporte académico completo de la estancia (46 páginas)
- **`REPORTE_ESTANCIA_INAOE.tex`** - Código fuente LaTeX del reporte

### `02_Codigo/src_v2/`
Código fuente completo del sistema (~25K líneas, 43 archivos Python).

Ver `04_ARQUITECTURA_CODIGO.md` para descripción de la estructura.

### `03_Modelos/`
4 checkpoints del ensemble (184 MB total):
- `seed123_final_model.pt` (46 MB)
- `seed321_final_model.pt` (46 MB)
- `seed111_final_model.pt` (46 MB)
- `seed666_final_model.pt` (46 MB)

Ver `07_MODELOS_ENTRENADOS.md` para descripción de cada modelo.

### `04_Configuraciones/`
4 archivos JSON de configuración (copiados también a `05_Documentacion/configs/`).

---

## Convenciones de la Documentación

### Formato

- **Idioma:** Español (términos técnicos en inglés con explicación)
- **Formato:** Markdown (GitHub-flavored)
- **Bloques de código:** Syntax highlighting para bash, python, json
- **Ejemplos:** Comandos ejecutables con salidas esperadas
- **Referencias cruzadas:** Enlaces relativos entre documentos

### Estructura de Documentos

Todos los documentos siguen una estructura estándar:

1. **Título y descripción**
2. **Tabla de contenidos**
3. **Contenido por secciones**
4. **Referencias cruzadas**
5. **Información de contacto**
6. **Última actualización**

### Nomenclatura de Archivos

- **Numeración:** `00-10` (orden de lectura recomendado)
- **`00_`:** Índice maestro (este documento)
- **`01-08`:** Documentación principal
- **`09-10`:** Documentación opcional
- **Sin número:** Archivos de referencia (`GROUND_TRUTH.json`, `AUDITORIA_...`)

---

## Actualizaciones y Versionamiento

**Versión actual de documentación:** 1.0.0 (28 enero 2026)

**Última actualización:** 28 de enero de 2026

**Versionamiento de métricas:** Ver `GROUND_TRUTH.json` (actualmente v2.1.0)

**Changelog:**
- **v1.0.0 (2026-01-28):** Release inicial de documentación técnica completa (10 documentos)

---

## Obtener Ayuda

### Dentro de la Documentación

1. Use este índice para navegar a documentos relevantes
2. Consulte `10_PREGUNTAS_FRECUENTES.md` para preguntas comunes
3. Revise secciones de "Troubleshooting" en cada documento
4. Use `09_GLOSARIO_TERMINOS.md` para clarificar términos

### Contacto Directo

**Estudiante:**
Rafael Alejandro Cruz Ovando
Benemérita Universidad Autónoma de Puebla (BUAP)

**Director del Proyecto:**
Dr. Leopoldo Altamirano Robles
Email: robles@inaoep.mx
Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE)
Laboratorio de Visión por Computadora

---

## Licencia y Uso

Este software y documentación fueron desarrollados como parte de una estancia de investigación en el INAOE.

**Uso permitido:**
- Investigación académica
- Educación
- Evaluación y validación de resultados

**Citar como:**
```
Cruz Ovando, R. A. (2026). Normalización y alineación automática de la forma de
la región pulmonar integrada con selección de características discriminantes para
detección automática de neumonía y COVID-19. Estancia de Investigación, Instituto
Nacional de Astrofísica, Óptica y Electrónica (INAOE). Dirigido por: Dr. Leopoldo
Altamirano Robles.
```

---

## Verificación de Integridad del USB

Para verificar que todos los archivos están presentes:

```bash
# Documentación (10 archivos MD + 2 de referencia + configs + requirements)
ls 05_Documentacion/*.md | wc -l  # Debe mostrar: 10
ls 05_Documentacion/GROUND_TRUTH.json  # Debe existir
ls 05_Documentacion/AUDITORIA_REPORTE_INAOE.md  # Debe existir
ls 05_Documentacion/requirements.txt  # Debe existir
ls 05_Documentacion/configs/*.json | wc -l  # Debe mostrar: 4

# Modelos (4 checkpoints)
ls 03_Modelos/*.pt | wc -l  # Debe mostrar: 4
du -sh 03_Modelos/  # Debe mostrar: ~184M

# Código (43 archivos Python)
find 02_Codigo/src_v2 -name "*.py" | wc -l  # Debe mostrar: 43

# Reporte
ls 01_Reporte/REPORTE_ESTANCIA_INAOE.pdf  # Debe existir
```

**Tamaño total del USB:** ~185 MB

---

**¡Bienvenido al proyecto! Esperamos que esta documentación facilite su trabajo con el sistema.**

---

**Última actualización:** 28 de enero de 2026
**Versión:** 1.0.0
