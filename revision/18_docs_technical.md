# 18. Technical Documentation

Analisis de la documentacion tecnica del proyecto.

**Archivos analizados**: 42

---

## Hallazgo Critico: Documentacion Referenciada en CLAUDE.md No Existe

CLAUDE.md (la guia principal del proyecto) referencia 8 archivos de documentacion en `docs/` que **no existen** en el repositorio:

| Archivo Referenciado | Existe |
|---|---|
| `docs/REPRO_FULL_PIPELINE.md` | NO |
| `docs/REPRO_ENSEMBLE_3_71.md` | NO |
| `docs/QUICKSTART_WARPING.md` | NO |
| `docs/LANDMARK_VISUALIZATION_DATASET.md` | NO |
| `docs/REPRO_CLASSIFIER_RESNET18.md` | NO |
| `docs/CONFIGS.md` | NO |
| `docs/EXPERIMENTS.md` | NO |
| `docs/CHECKPOINTS_CLEANUP_REPORT.md` | NO |

Adicionalmente, CLAUDE.md referencia directorios `docs/sesiones/` y `docs/reportes/` que tampoco existen. El unico archivo tecnico que existe directamente en `docs/` (fuera de Tesis/estancia/carta/manual) es `RELEASE_NOTES_v16.md`.

Esto indica que la documentacion fue eliminada o nunca fue creada, pero CLAUDE.md no fue actualizado para reflejar este estado. La informacion que contenian esos archivos fue aparentemente absorbida en CLAUDE.md y/o migrada al sistema `.planning/`.

**Importancia**: CRITICO -- CLAUDE.md contiene referencias rotas que confundiran a desarrolladores y herramientas de IA.

---

## A. Documentacion en docs/ (excluye Tesis/estancia/carta/manual)

### docs/RELEASE_NOTES_v16.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/docs/RELEASE_NOTES_v16.md
- **Lineas/Tamano**: 285 lineas / 12 KB
- **Proposito**: Documenta la version 16 del paquete portable Windows, incluyendo la nueva funcionalidad de comparticion publica via Gradio tunneling.
- **Contenido clave**:
  - Nueva funcionalidad de RUN_DEMO_SHARE.bat para demostraciones remotas (enlaces publicos de 72 horas via Gradio)
  - Arquitectura de doble modo de lanzamiento (local vs. publico)
  - Consideraciones de seguridad y proteccion de datos (HIPAA/GDPR)
  - Especificaciones del paquete (579 MB, 196 archivos, Python 3.12.8, 86 dependencias)
  - Checksums de modelos (sin cambios respecto a v15)
  - Guia de despliegue para defensa de tesis con plan de contingencia
  - Lista de verificacion pre-release con items marcados como completados o pendientes
  - Instrucciones para desarrolladores (build) y usuarios finales
- **Importancia**: MEDIO
- **Justificacion**: Documenta una release especifica del paquete portable. Util para reproducir builds y entender la configuracion de distribucion, pero no contiene informacion critica sobre la metodologia o resultados de investigacion. La informacion de seguridad es valiosa para el contexto de despliegue.

---

## B. Documentacion de Planificacion (.planning/)

El directorio `.planning/` contiene un sistema de planificacion exhaustivo generado por una herramienta de desarrollo asistida por IA (GSD - Goal-Structured Development). Comprende 42 archivos markdown con un total de 12,420 lineas. Esta organizado en tres categorias: archivos de proyecto raiz, mapeo del codebase, investigacion, y fases de ejecucion.

### B.1 Archivos Raiz del Proyecto

### .planning/PROJECT.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/PROJECT.md
- **Lineas/Tamano**: 93 lineas / 8 KB
- **Proposito**: Define el alcance, valor central y restricciones del esfuerzo de mejora del ensemble clasificador.
- **Contenido clave**:
  - Definicion del problema: mejorar la precision del clasificador usando ensemble de 5 modelos CV con TTA
  - Requisitos validados (existentes) vs. activos (pendientes) vs. fuera de alcance
  - Contexto: baseline individual 97.68%, objetivo 98.2-98.7%
  - Restricciones metodologicas (nunca optimizar en test set), medicas (augmentaciones seguras), y de cronograma
  - Tabla de decisiones clave con razonamiento
- **Importancia**: ALTO
- **Justificacion**: Documento maestro que guia todo el trabajo de ensemble. Define claramente el alcance y las restricciones metodologicas que protegen la integridad de la tesis.

### .planning/REQUIREMENTS.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/REQUIREMENTS.md
- **Lineas/Tamano**: 115 lineas / 8 KB
- **Proposito**: Especifica requisitos formales con identificadores unicos, estado de completitud, y trazabilidad a fases del roadmap.
- **Contenido clave**:
  - 17 requisitos v1 organizados por categoria: Ensemble Core (4), TTA (2), Metrics (5), Output (3), Validation (3)
  - 53% completados (9/17), con ENSEMBLE-04 y METRICS-05 aun pendientes
  - Requisitos v2 diferidos: analisis avanzado, TTA extendido, validacion rigurosa
  - Tabla de trazabilidad requisito-fase con estado actualizado
  - Lista explicita de exclusiones para evitar scope creep
- **Importancia**: MEDIO
- **Justificacion**: Buena practica de ingenieria para rastrear requisitos, pero mas relevante como herramienta de gestion que como referencia tecnica. Los requisitos pendientes indican trabajo incompleto en el proyecto de ensemble.

### .planning/ROADMAP.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/ROADMAP.md
- **Lineas/Tamano**: 111 lineas / 8 KB
- **Proposito**: Define las 5 fases del proyecto de ensemble con criterios de exito observables.
- **Contenido clave**:
  - 5 fases: Pre-Implementation Audit, Ensemble Core, TTA Integration, Analysis & Visualization, Final Test Evaluation
  - Fases 1-3 completadas (2026-01-27), fases 4-5 no iniciadas
  - Criterios de exito detallados para cada fase (verificables como verdadero/falso)
  - Dependencias entre fases (ejecutar en orden 1->2->3->4->5)
  - Tabla de progreso con planes completados por fase
- **Importancia**: MEDIO
- **Justificacion**: Proporciona estructura y visibilidad del progreso del proyecto de ensemble. Las fases 4 y 5 estan marcadas como "TBD" lo que indica trabajo incompleto.

### .planning/STATE.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/STATE.md
- **Lineas/Tamano**: 89 lineas / 8 KB
- **Proposito**: Captura el estado actual del proyecto de ensemble, decisiones acumuladas, y contexto de continuidad entre sesiones.
- **Contenido clave**:
  - Posicion actual: Fase 3 de 5 completada, 60% de progreso
  - Metricas de velocidad: 7 planes completados, promedio 7 min por plan, 0.92 horas totales
  - 13 decisiones acumuladas con estado (IMPLEMENTED/VALIDATED)
  - Resultados clave: Ensemble 98.10%, TTA 98.26% (+0.16pp), 6 muestras ayudadas, 3 perjudicadas
  - Blockers pendientes: limpieza de 9 imagenes duplicadas requerida para Fase 5
  - Continuidad: ultima sesion 2026-01-27, siguiente paso es verificacion de Fase 3
- **Importancia**: ALTO
- **Justificacion**: Esencial para entender el estado actual del trabajo de ensemble y retomar donde se dejo. Contiene las metricas validadas y decisiones que afectan la evaluacion final.

---

### B.2 Mapeo del Codebase (.planning/codebase/)

### .planning/codebase/ARCHITECTURE.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/ARCHITECTURE.md
- **Lineas/Tamano**: 494 lineas / 24 KB
- **Proposito**: Documenta la arquitectura completa del sistema como un pipeline de clasificacion CNN de dos etapas con normalizacion geometrica.
- **Contenido clave**:
  - 7 capas arquitectonicas: Input, Preprocessing, Landmark Detection, Warping, Canonical Shape, Classification, Output
  - Flujo de datos detallado para 4 pipelines: Training (landmarks), Warping (normalizacion), Classification, Inference (end-to-end)
  - Abstracciones clave: GPA, Piecewise Affine Warping, ResNet18Landmarks, CoordinateAttention, Classification Model, Loss Functions, Two-Phase Training, TTA, Cached Predictions
  - Puntos de entrada: CLI (35+ comandos), GUI (Gradio)
  - Patrones de manejo de errores: file existence, device fallback, multiprocessing sandbox, data splits
  - Preocupaciones transversales: logging, validacion, rendimiento, reproducibilidad, escalabilidad
- **Importancia**: CRITICO
- **Justificacion**: Documentacion de referencia fundamental para cualquier desarrollador que trabaje en el proyecto. Explica como todas las piezas encajan y el flujo de datos end-to-end. Equivale a la "guia de arquitectura" que normalmente se esperaria en `docs/`.

### .planning/codebase/STRUCTURE.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/STRUCTURE.md
- **Lineas/Tamano**: 362 lineas / 20 KB
- **Proposito**: Mapa completo de la estructura de directorios y archivos del proyecto con proposito de cada componente.
- **Contenido clave**:
  - Arbol de directorios con anotaciones para cada archivo/directorio
  - Proposito detallado de cada subdirectorio (src_v2/data, models, processing, training, evaluation, visualization, gui, utils)
  - Ubicaciones de archivos clave por funcion (entry points, config, core logic, data handling, evaluation, visualization)
  - Convenciones de nomenclatura (archivos, directorios, clases, funciones)
  - Guia "Where to Add New Code" para 7 tipos de adiciones comunes
  - Directorios especiales con sus estados (archive, glass_box_visualizations, fisher, data, outputs, checkpoints)
  - **NOTA**: El arbol de directorios en la seccion `docs/` lista archivos que no existen (REPRO_FULL_PIPELINE.md, CONFIGS.md, etc.), confirmando que esta informacion esta desactualizada
- **Importancia**: ALTO
- **Justificacion**: Referencia esencial para navegacion del proyecto. La guia de "donde agregar codigo nuevo" es especialmente valiosa. Sin embargo, contiene la misma informacion desactualizada sobre docs/ que CLAUDE.md.

### .planning/codebase/STACK.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/STACK.md
- **Lineas/Tamano**: 188 lineas / 8 KB
- **Proposito**: Documenta el stack tecnologico completo del proyecto incluyendo lenguajes, frameworks, dependencias y requisitos de plataforma.
- **Contenido clave**:
  - Lenguajes: Python 3.9+ (primario), Bash, Batch, JSON, LaTeX (secundarios)
  - Frameworks: PyTorch 2.0+, Typer (CLI), Gradio (UI), pytest (testing), PyInstaller (build)
  - Dependencias criticas: torch, numpy 2.0+, opencv-python, scipy, pandas, scikit-learn
  - Dependencias de visualizacion: matplotlib, seaborn, Pillow
  - Configuracion: JSON configs en configs/, GROUND_TRUTH.json, pyproject.toml
  - Requisitos de hardware: minimo (CPU 4-core, 8GB RAM) y recomendado (GPU 8GB+, 16GB RAM)
  - Instrucciones de instalacion para CPU, CUDA 12.1, y ROCm 6.0
- **Importancia**: ALTO
- **Justificacion**: Referencia esencial para setup y reproducibilidad. Documenta versiones de dependencias, requisitos de hardware, e instrucciones de instalacion para diferentes plataformas.

### .planning/codebase/CONVENTIONS.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/CONVENTIONS.md
- **Lineas/Tamano**: 202 lineas / 8 KB
- **Proposito**: Define las convenciones de codificacion del proyecto (naming, style, imports, error handling, logging, comments, function/module design).
- **Contenido clave**:
  - Naming: snake_case para archivos/funciones/variables, PascalCase para clases, UPPERCASE para constantes
  - Style: PEP 8, 100 chars, 4-space indentation, Black-compatible
  - Imports: stdlib -> third-party -> local, alphabetized
  - Error handling: excepciones especificas, chain con `from e`, log antes de raise
  - Logging: Python logging module, logger per-module
  - Docstrings: Google-style con Args/Returns/Raises
  - Funciones: 20-80 lineas, type hints, keyword args explicitos
  - Configuracion centralizada en constants.py y configs/*.json
- **Importancia**: MEDIO
- **Justificacion**: Util como guia de estilo para contribuyentes. La mayoria de esta informacion ya esta en CLAUDE.md de forma condensada.

### .planning/codebase/TESTING.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/TESTING.md
- **Lineas/Tamano**: 304 lineas / 12 KB
- **Proposito**: Documenta los patrones de testing del proyecto, incluyendo el framework, organizacion de archivos, mocking, fixtures, y coverage.
- **Contenido clave**:
  - Framework: pytest 7.0+ con configuracion en pyproject.toml
  - Estado actual: directorio tests/ referenciado pero la suite de tests formales es minima
  - Tests historicos en scripts/archive/ (test_forward_pass.py, test_dataset.py, etc.)
  - Patron: assertions directas, datos reales (no mocks), torch.rand para dummy tensors
  - Variable FORCE_NUM_WORKERS_ZERO=1 para testing deterministico
  - Coverage: pytest-cov, target src_v2, exclusiones configuradas
  - "To Organize Tests": lista de 6 pasos pendientes para formalizar el testing
- **Importancia**: MEDIO
- **Justificacion**: Documenta el estado real del testing (minimo) y proporciona un plan para mejorarlo. La informacion sobre la variable de entorno FORCE_NUM_WORKERS_ZERO es importante para reproducibilidad de tests.

### .planning/codebase/INTEGRATIONS.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/INTEGRATIONS.md
- **Lineas/Tamano**: 215 lineas / 12 KB
- **Proposito**: Documenta integraciones externas y pipelines de datos. Confirma que el proyecto es autocontenido sin APIs externas.
- **Contenido clave**:
  - APIs externas: ninguna (proyecto autocontenido)
  - Storage: filesystem local unicamente (CSV, NPZ, PT, JSON, PNG/JPEG)
  - Autenticacion: no aplica
  - Monitoreo: solo Python logging (no Sentry, MLflow, W&B)
  - CI/CD: no detectado
  - Variables de entorno opcionales: COVID_DEMO_MODELS_DIR, COVID_DEMO_FROZEN (para PyInstaller)
  - Pipelines de datos: Input (CSV->DataFrame->DataLoader->Model), Cache (NPZ), Warping, Output
  - Riesgo de seguridad: torch.load() con weights_only=False permite ejecucion de codigo arbitrario
  - Formatos de exportacion: PT, CSV, JSON, NPZ, PNG/JPEG, PDF
- **Importancia**: MEDIO
- **Justificacion**: Confirma la naturaleza autocontenida del proyecto. El hallazgo de seguridad sobre torch.load() es valioso. La documentacion de pipelines de datos complementa ARCHITECTURE.md.

### .planning/codebase/CONCERNS.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/CONCERNS.md
- **Lineas/Tamano**: 210 lineas / 16 KB
- **Proposito**: Auditoria exhaustiva de deuda tecnica, bugs conocidos, consideraciones de seguridad, cuellos de botella de rendimiento, areas fragiles, y brechas de testing.
- **Contenido clave**:
  - **Deuda tecnica**: cli.py monolitico (10,520 lineas, 40+ comandos), estado global en GUI config, exception handling silencioso
  - **Bugs conocidos**: Inestabilidad de Delaunay con landmarks colineales, warnings de tamano de imagen no fatales, deteccion de arquitectura de modelo con fallback chain
  - **Seguridad**: Paths sin validacion en CLI, torch.load() sin weights_only=True, input no sanitizado en GUI web
  - **Performance**: copies de numpy en loops de warping, fallback de num_workers, GPA recomputado sin cache, Delaunay recomputado por imagen
  - **Areas fragiles**: Alineacion de coordenadas en pipeline de warping (multiple sistemas de coordenadas), TTA con pares simetricos, balance de pesos por categoria, propagacion de parametros CLI
  - **Brechas de testing**: Sin tests para transformacion de coordenadas E2E, TTA symmetric pair handling, robustez de data loading, propagacion de parametros CLI, compatibilidad de checkpoints
- **Importancia**: CRITICO
- **Justificacion**: El hallazgo mas importante de toda la documentacion de planificacion. Identifica problemas reales y accionables que afectan la fiabilidad del sistema. El bug de cli.py monolitico (10,520 lineas) y las brechas de testing son particularmente criticos. Este documento deberia informar las prioridades de refactorizacion.

---

### B.3 Investigacion (.planning/research/)

### .planning/research/SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/SUMMARY.md
- **Lineas/Tamano**: 330 lineas / 28 KB
- **Proposito**: Resumen ejecutivo de la investigacion de pre-implementacion para el trabajo de ensemble+TTA, incluyendo stack recomendado, features, arquitectura, y pitfalls criticos.
- **Contenido clave**:
  - Stack recomendado: PyTorch nativo (50 lineas), ttach 0.0.3, torchmetrics 1.8.2+
  - 8 features "table stakes", 10 diferenciadores, 7 anti-features
  - Arquitectura de 5 capas para evaluacion inference-only
  - 15 pitfalls identificados con 5 CRITICOS que invalidan resultados de tesis
  - Roadmap propuesto de 6 fases con Phase 0 de auditoria bloqueante
  - Analisis de brechas: mejora modesta esperada (+0.5-1.0pp), validez de flip horizontal, validacion externa esperada fallar
  - Confianza: HIGH en las 4 areas evaluadas (stack, features, arquitectura, pitfalls)
  - 30+ fuentes de investigacion citadas (papers 2020-2026)
- **Importancia**: ALTO
- **Justificacion**: Investigacion exhaustiva que informo todas las decisiones de diseno del esfuerzo de ensemble. Particularmente valiosa por la identificacion de pitfalls y la recomendacion de Phase 0 audit (que revelo el problema de duplicados en los datos).

### .planning/research/FEATURES.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/FEATURES.md
- **Lineas/Tamano**: 276 lineas / 24 KB
- **Proposito**: Analisis detallado de features para el sistema de ensemble clasificador, con priorizacion y definicion de MVP.
- **Contenido clave**:
  - Tabla stakes (9 features obligatorias para thesis baseline): soft voting, hard voting, per-model/ensemble metrics, per-class breakdown, TTA, config, reproducibilidad, confusion matrix
  - Diferenciadores (10 features de ventaja competitiva): calibracion, ECE, disagreement analysis, uncertainty, per-sample confidence
  - Anti-features (7 que crear problemas): training from scratch, test set optimization, aggressive TTA, weighted voting, real-time optimization, MC Dropout, complex voting
  - Grafo de dependencias entre features
  - MVP definido: v1.0 (12 features completadas), v1.1 (6 features pendientes), v2.0+ (6 features diferidas)
  - Matriz de priorizacion (User Value x Implementation Cost x Priority)
  - Analisis comparativo con landmark ensemble existente
  - 20+ fuentes academicas citadas
- **Importancia**: MEDIO
- **Justificacion**: Investigacion solida pero orientada exclusivamente al esfuerzo de ensemble que esta parcialmente completado. La clasificacion de anti-features es particularmente valiosa para evitar errores de diseno.

### .planning/research/PITFALLS.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/PITFALLS.md
- **Lineas/Tamano**: 948 lineas / 56 KB
- **Proposito**: Documentacion exhaustiva de 15 pitfalls potenciales para evaluacion de ensemble en imagenes medicas, con 5 clasificados como CRITICOS.
- **Contenido clave**:
  - Pitfall 1 (CRITICO): Contaminacion del test set via seleccion de ensemble (5-30% inflacion)
  - Pitfall 2 (CRITICO): Data leakage por splitting inapropiado (29-55% inflacion)
  - Pitfall 3 (CRITICO): Augmentaciones medicas inseguras que destruyen features diagnosticos
  - Pitfall 4 (CRITICO): Reporte de metricas infladas (proyecto ya cometio este error una vez)
  - Pitfall 5 (CRITICO): Cherry-picking de modelos de ensemble
  - Pitfalls HIGH: CLAIM 2024 non-compliance, reproducibility failures, overfitting by observer
  - Cada pitfall incluye: que sale mal, por que ocurre, como evitarlo, senales de alerta, fase para abordarlo
  - Basado en 30+ fuentes academicas revisadas por pares (2020-2026)
- **Importancia**: ALTO
- **Justificacion**: El archivo mas largo de toda la documentacion de planificacion (56 KB). Contiene conocimiento critico de dominio especifico a imagenes medicas que protege la validez de la tesis. La documentacion de que el proyecto ya cometio el error de reportar metricas de validacion como test da credibilidad y urgencia.

### .planning/research/ARCHITECTURE.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/ARCHITECTURE.md
- **Lineas/Tamano**: 521 lineas / 28 KB
- **Proposito**: Investigacion de patrones arquitectonicos para evaluacion de ensemble+TTA en clasificacion de imagenes medicas.
- **Contenido clave**:
  - Patron recomendado: Evaluation Orchestrator de 5 capas (config loading, inference, aggregation, analysis, visualization)
  - Componentes: EnsembleEvaluator, EnsembleWrapper, TTA Engine, Metrics Module, CLI Integration
  - Patron de Model Pool con Lazy Loading para gestion de memoria
  - TTA como pipeline de transformaciones composable
  - Evaluacion estratificada por lotes (esencial para datasets desbalanceados)
  - Flujo de datos detallado: Test dataset -> DataLoader -> TTA views -> Models -> Mean(TTA) -> Stack(ensemble) -> Mean(ensemble) -> Argmax -> Metrics
  - Consideraciones de seguridad medica para augmentaciones
  - Reuso de infraestructura existente del landmark ensemble
- **Importancia**: MEDIO
- **Justificacion**: Diseno arquitectonico bien fundamentado que guio la implementacion del ensemble. La mayor parte de esta informacion ya fue implementada en las fases 1-3 completadas.

### .planning/research/STACK.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/STACK.md
- **Lineas/Tamano**: 434 lineas / 20 KB
- **Proposito**: Investigacion de tecnologias y dependencias para la implementacion del ensemble+TTA, con comparaciones y recomendaciones.
- **Contenido clave**:
  - Evaluacion de PyTorch 2.10+ (FlexAttention, numerical debugging, improved profiling)
  - Evaluacion de ttach vs albumentations vs custom implementation para TTA
  - Evaluacion de torchmetrics vs scikit-learn vs custom para metricas
  - Decision: PyTorch nativo + torchmetrics (2 dependencias nuevas vs frameworks pesados)
  - Alertas: Neptune.ai cerrando en marzo 2026, albumentations en modo mantenimiento
  - Restricciones: no ttach ni torchmetrics para PyInstaller frozen builds
  - Riesgos de compatibilidad: NumPy 2.0, scipy.spatial.Delaunay API changes, torch.compile dynamic shapes
- **Importancia**: BAJO
- **Justificacion**: Investigacion util en su momento pero de valor limitado como referencia futura. Las decisiones ya fueron tomadas e implementadas. Los riesgos de compatibilidad son mas relevantes a largo plazo.

---

### B.4 Documentacion de Fases (.planning/phases/)

Las fases documentan la ejecucion plan-por-plan del proyecto de ensemble. Cada fase tiene hasta 7 tipos de documentos: CONTEXT, RESEARCH, PLAN(s), SUMMARY(s), VERIFICATION, y artefactos especiales.

#### Fase 1: Pre-Implementation Audit (9 archivos, completada)

### .planning/phases/01-pre-implementation-audit/01-CONTEXT.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/01-CONTEXT.md
- **Lineas/Tamano**: 77 lineas / 4 KB
- **Proposito**: Contexto de entrada para la Fase 1: estado del proyecto, archivos relevantes, y objetivos de la auditoria.
- **Importancia**: BAJO
- **Justificacion**: Contexto de sesion, valor referencial minimo.

### .planning/phases/01-pre-implementation-audit/01-RESEARCH.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/01-RESEARCH.md
- **Lineas/Tamano**: 835 lineas / 32 KB
- **Proposito**: Investigacion detallada para la auditoria pre-implementacion, incluyendo verificacion de integridad de datos, aislamiento del test set, y metodologia de seleccion de modelos.
- **Importancia**: MEDIO
- **Justificacion**: Documenta la investigacion que llevo al descubrimiento de los 9 duplicados en el dataset.

### .planning/phases/01-pre-implementation-audit/01-01-PLAN.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/01-01-PLAN.md
- **Lineas/Tamano**: 208 lineas / 8 KB
- **Proposito**: Plan de ejecucion para verificacion de integridad de datos y auditoria metodologica.
- **Importancia**: BAJO
- **Justificacion**: Plan de ejecucion ya completado.

### .planning/phases/01-pre-implementation-audit/01-01-SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/01-01-SUMMARY.md
- **Lineas/Tamano**: 123 lineas / 8 KB
- **Proposito**: Resumen de resultados del plan 01-01: conteos de imagenes, distribucion por clase, y hallazgos de integridad.
- **Importancia**: BAJO
- **Justificacion**: Resumen de ejecucion completada.

### .planning/phases/01-pre-implementation-audit/01-02-PLAN.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/01-02-PLAN.md
- **Lineas/Tamano**: 365 lineas / 16 KB
- **Proposito**: Plan para re-evaluacion de baseline y reporte de auditoria final.
- **Importancia**: BAJO
- **Justificacion**: Plan de ejecucion ya completado.

### .planning/phases/01-pre-implementation-audit/01-02-SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/01-02-SUMMARY.md
- **Lineas/Tamano**: 168 lineas / 8 KB
- **Proposito**: Resumen de re-evaluacion: baseline 97.68% confirmado con diferencia=0.000000.
- **Importancia**: BAJO
- **Justificacion**: Resumen de ejecucion completada.

### .planning/phases/01-pre-implementation-audit/01-VERIFICATION.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/01-VERIFICATION.md
- **Lineas/Tamano**: 452 lineas / 20 KB
- **Proposito**: Reporte de verificacion formal de que todos los criterios de exito de la Fase 1 fueron cumplidos.
- **Importancia**: MEDIO
- **Justificacion**: Registro formal de verificacion util para auditoria de proceso.

### .planning/phases/01-pre-implementation-audit/AUDIT_REPORT.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/AUDIT_REPORT.md
- **Lineas/Tamano**: 438 lineas / 16 KB
- **Proposito**: Reporte completo de auditoria pre-implementacion con hallazgos, recomendaciones, y decision de proceder.
- **Contenido clave**:
  - Status: CONDITIONAL PASS -- proceder con limpieza de datos
  - Hallazgo principal: 9 imagenes duplicadas (1 test, 8 val) -- 0.053% leakage en test
  - Metodologia validada por 4 metodos independientes (git, logs, timestamps, configs)
  - Baseline 97.68% confirmado como precision en test set (no validacion)
  - Recomendacion: limpiar duplicados antes de evaluacion final
- **Importancia**: ALTO
- **Justificacion**: Documento clave que valida la integridad metodologica del proyecto y documenta un hallazgo real (duplicados) que debe ser resuelto.

### .planning/phases/01-pre-implementation-audit/SEED_SELECTION_PROTOCOL.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/SEED_SELECTION_PROTOCOL.md
- **Lineas/Tamano**: 204 lineas / 8 KB
- **Proposito**: Documenta el protocolo de seleccion de seeds para los modelos del ensemble y verifica que no hubo cherry-picking.
- **Importancia**: MEDIO
- **Justificacion**: Responde directamente a una de las preocupaciones criticas (Pitfall 5): verificar que los seeds del ensemble no fueron seleccionados optimizando en el test set.

#### Fase 2: Ensemble Core (7 archivos, completada)

### .planning/phases/02-ensemble-core/02-CONTEXT.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/02-ensemble-core/02-CONTEXT.md
- **Lineas/Tamano**: 81 lineas / 8 KB
- **Proposito**: Contexto de entrada para Fase 2 con estado del proyecto post-auditoria.
- **Importancia**: BAJO
- **Justificacion**: Contexto de sesion.

### .planning/phases/02-ensemble-core/02-RESEARCH.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/02-ensemble-core/02-RESEARCH.md
- **Lineas/Tamano**: 567 lineas / 24 KB
- **Proposito**: Investigacion especifica para implementacion del modulo de ensemble: carga de modelos, soft/hard voting, metricas.
- **Importancia**: BAJO
- **Justificacion**: Investigacion pre-implementacion ya ejecutada.

### .planning/phases/02-ensemble-core/02-01-PLAN.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/02-ensemble-core/02-01-PLAN.md
- **Lineas/Tamano**: 269 lineas / 12 KB
- **Proposito**: Plan para implementar modulo de ensemble y comando CLI.
- **Importancia**: BAJO
- **Justificacion**: Plan completado.

### .planning/phases/02-ensemble-core/02-01-SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/02-ensemble-core/02-01-SUMMARY.md
- **Lineas/Tamano**: 192 lineas / 12 KB
- **Proposito**: Resumen de implementacion del modulo ensemble y CLI.
- **Importancia**: BAJO
- **Justificacion**: Resumen de ejecucion completada.

### .planning/phases/02-ensemble-core/02-02-PLAN.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/02-ensemble-core/02-02-PLAN.md
- **Lineas/Tamano**: 280 lineas / 12 KB
- **Proposito**: Plan para crear configuracion y ejecutar evaluacion del ensemble.
- **Importancia**: BAJO
- **Justificacion**: Plan completado.

### .planning/phases/02-ensemble-core/02-02-SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/02-ensemble-core/02-02-SUMMARY.md
- **Lineas/Tamano**: 241 lineas / 12 KB
- **Proposito**: Resumen de evaluacion: ensemble 98.10% accuracy (+0.42pp sobre baseline), soft voting superior a hard voting.
- **Importancia**: MEDIO
- **Justificacion**: Contiene los resultados validados del ensemble base que son referencia para fases posteriores.

### .planning/phases/02-ensemble-core/02-VERIFICATION.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/02-ensemble-core/02-VERIFICATION.md
- **Lineas/Tamano**: 502 lineas / 24 KB
- **Proposito**: Verificacion formal de criterios de exito de la Fase 2.
- **Importancia**: BAJO
- **Justificacion**: Registro de verificacion.

#### Fase 3: TTA Integration (11 archivos, completada)

### .planning/phases/03-tta-integration/03-CONTEXT.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-CONTEXT.md
- **Lineas/Tamano**: 319 lineas / 12 KB
- **Proposito**: Contexto detallado para Fase 3 incluyendo decisiones acumuladas de fases anteriores.
- **Importancia**: BAJO
- **Justificacion**: Contexto de sesion.

### .planning/phases/03-tta-integration/03-RESEARCH.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-RESEARCH.md
- **Lineas/Tamano**: 452 lineas / 24 KB
- **Proposito**: Investigacion para implementacion de TTA en clasificador: horizontal flip, case-level tracking, integracion con ensemble.
- **Importancia**: BAJO
- **Justificacion**: Investigacion pre-implementacion ya ejecutada.

### .planning/phases/03-tta-integration/03-01-PLAN.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-01-PLAN.md
- **Lineas/Tamano**: 419 lineas / 16 KB
- **Proposito**: Plan para implementar funciones de prediccion TTA e integracion CLI.
- **Importancia**: BAJO
- **Justificacion**: Plan completado.

### .planning/phases/03-tta-integration/03-01-SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-01-SUMMARY.md
- **Lineas/Tamano**: 115 lineas / 8 KB
- **Proposito**: Resumen de implementacion de TTA prediction functions.
- **Importancia**: BAJO
- **Justificacion**: Resumen de ejecucion completada.

### .planning/phases/03-tta-integration/03-02-PLAN.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-02-PLAN.md
- **Lineas/Tamano**: 351 lineas / 16 KB
- **Proposito**: Plan para ejecutar evaluacion con tracking de impacto por caso y actualizar GROUND_TRUTH.
- **Importancia**: BAJO
- **Justificacion**: Plan completado.

### .planning/phases/03-tta-integration/03-02-SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-02-SUMMARY.md
- **Lineas/Tamano**: 153 lineas / 8 KB
- **Proposito**: Resultados: TTA 98.26% (+0.16pp), 6 helped, 3 hurt, 1886 neutral. COVID beneficia mas (+0.44% F1).
- **Importancia**: MEDIO
- **Justificacion**: Contiene resultados clave de la evaluacion TTA que son referencia para la tesis.

### .planning/phases/03-tta-integration/03-03-PLAN.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-03-PLAN.md
- **Lineas/Tamano**: 223 lineas / 12 KB
- **Proposito**: Plan de cierre de brecha: conectar funciones huerfanas de case-level impact al CLI.
- **Importancia**: BAJO
- **Justificacion**: Plan completado.

### .planning/phases/03-tta-integration/03-03-SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-03-SUMMARY.md
- **Lineas/Tamano**: 107 lineas / 4 KB
- **Proposito**: Resumen del cierre de brecha: funciones de impacto integradas en CLI.
- **Importancia**: BAJO
- **Justificacion**: Resumen de ejecucion completada.

### .planning/phases/03-tta-integration/03-VERIFICATION.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/03-VERIFICATION.md
- **Lineas/Tamano**: 387 lineas / 16 KB
- **Proposito**: Verificacion formal de Fase 3: 5/5 criterios verificados tras re-verificacion por cierre de brecha.
- **Importancia**: BAJO
- **Justificacion**: Registro de verificacion.

---

## C. Resumen y Recomendaciones

### Estadisticas Generales

| Categoria | Archivos | Lineas | Tamano |
|---|---|---|---|
| docs/ (tecnico, excl. Tesis/estancia/carta/manual) | 1 | 285 | 12 KB |
| .planning/ raiz | 4 | 408 | 32 KB |
| .planning/codebase/ | 7 | 1,975 | 100 KB |
| .planning/research/ | 5 | 2,509 | 156 KB |
| .planning/phases/01 (Pre-Audit) | 9 | 2,870 | 120 KB |
| .planning/phases/02 (Ensemble Core) | 7 | 2,132 | 104 KB |
| .planning/phases/03 (TTA Integration) | 11 | 2,526 | 124 KB |
| **TOTAL** | **44** | **12,705** | **648 KB** |

### Distribucion por Importancia

| Nivel | Archivos | Descripcion |
|---|---|---|
| CRITICO | 2 | ARCHITECTURE.md (codebase), CONCERNS.md |
| ALTO | 7 | PROJECT.md, STATE.md, STRUCTURE.md, STACK.md, SUMMARY.md (research), PITFALLS.md, AUDIT_REPORT.md |
| MEDIO | 11 | REQUIREMENTS.md, ROADMAP.md, CONVENTIONS.md, TESTING.md, INTEGRATIONS.md, FEATURES.md, RELEASE_NOTES_v16.md, 01-RESEARCH.md, 01-VERIFICATION.md, SEED_SELECTION_PROTOCOL.md, 02-02-SUMMARY.md, 03-02-SUMMARY.md |
| BAJO | 24 | Contextos, planes completados, resumenes de ejecucion, investigaciones pre-implementacion |

### Problemas Identificados

1. **CRITICO: Referencias rotas en CLAUDE.md** -- 8 archivos de documentacion referenciados en CLAUDE.md y .planning/codebase/STRUCTURE.md no existen. Los directorios `docs/sesiones/` y `docs/reportes/` referenciados tampoco existen. Esto confunde a desarrolladores y herramientas de IA.

2. **ALTO: Documentacion de planificacion "oculta"** -- La documentacion mas valiosa del proyecto (arquitectura, estructura, concerns, pitfalls) esta en `.planning/` que es un directorio oculto. No hay ninguna referencia desde CLAUDE.md o README a estos documentos. Un desarrollador nuevo no sabria que existen.

3. **ALTO: Proyecto de ensemble incompleto** -- Las fases 4 (Analysis & Visualization) y 5 (Final Test Evaluation) nunca se completaron. El STATE.md muestra 60% de progreso. Los 9 duplicados identificados en la auditoria (Fase 1) no parecen haber sido limpiados, y la evaluacion final en test set nunca se ejecuto.

4. **MEDIO: Volumen excesivo de documentacion de planificacion** -- 12,700 lineas / 648 KB de documentacion para ~1 hora de trabajo de implementacion (7 planes, 0.92 horas). Los planes completados, contextos de sesion, e investigaciones pre-implementacion tienen valor referencial minimo una vez ejecutados. Considerar archivar las fases completadas.

5. **MEDIO: Informacion duplicada** -- La informacion del codebase (ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md) duplica significativamente lo que ya esta en CLAUDE.md. En algunos casos (como la lista de docs/), la version en .planning/ esta desactualizada con respecto a la realidad.

### Recomendaciones

1. **Actualizar CLAUDE.md**: Eliminar las 8 referencias a archivos de documentacion que no existen. Actualizar las referencias a `docs/sesiones/` y `docs/reportes/`.

2. **Referenciar .planning/ desde CLAUDE.md**: Agregar una seccion que apunte a los documentos clave en `.planning/` (al menos ARCHITECTURE.md, CONCERNS.md, y PITFALLS.md).

3. **Resolver trabajo incompleto del ensemble**: Decidir si completar las fases 4-5 del roadmap de ensemble o documentar que se abandono. Los 9 duplicados deben ser limpiados independientemente.

4. **Considerar archivado de fases completadas**: Los 27 archivos de .planning/phases/ para fases 1-3 completadas podrian moverse a un subdirectorio archive/ para reducir ruido, manteniendo solo los artefactos clave (AUDIT_REPORT.md, resultados de verificacion).
