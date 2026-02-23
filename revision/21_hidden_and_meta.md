# 21. Archivos Ocultos y Meta

**Fecha de revision**: 2026-02-11
**Archivos analizados**: ~90 archivos en 5 directorios
**Tamano total**: ~750 MB (613 MB en results/, 46 MB en checkpoints/session13/hierarchical/, resto <1 MB)

---

## 1. Directorio `.claude/` (7 archivos, 92 KB)

Configuracion y prompts para Claude Code (Anthropic CLI). Contiene permisos de ejecucion, comandos slash y frameworks de prompt engineering.

### 1.1 `.claude/settings.local.json`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.claude/settings.local.json`
- **Lineas/Tamano**: 62 lineas
- **Proposito**: Configuracion de permisos de Claude Code. Define lista de comandos Bash permitidos (57 reglas allow), sin reglas deny ni ask. Controla que operaciones puede ejecutar Claude sin solicitar confirmacion.
- **Importancia**: MEDIO
- **Justificacion**: Necesario para el flujo de trabajo con Claude Code. No afecta el codigo ni los resultados del proyecto, pero es esencial para la productividad del desarrollador. Las reglas son permisivas (permite python, git, find, etc.) lo cual facilita la automatizacion pero podria ser un riesgo de seguridad menor. No contiene secretos ni credenciales.

### 1.2 `.claude/commands/improve-prompt.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.claude/commands/improve-prompt.md`
- **Lineas/Tamano**: 55 lineas
- **Proposito**: Slash command (`/improve-prompt`) para analizar y mejorar prompts. Genera analisis del prompt original, version mejorada, tabla de cambios y ejemplo de uso. Texto en espanol.
- **Importancia**: BAJO
- **Justificacion**: Herramienta de productividad del desarrollador. No tiene relacion directa con el pipeline de clasificacion COVID-19. Es utileria generica de prompt engineering que podria vivir fuera del proyecto.

### 1.3 `.claude/commands/refine-prompt.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.claude/commands/refine-prompt.md`
- **Lineas/Tamano**: 113 lineas
- **Proposito**: Slash command (`/refine-prompt`) para refinamiento iterativo con rubrica de 10 criterios (claridad, especificidad, estructura, etc.), puntuacion de 1-5 por criterio. 4 fases: Analisis, Refinamiento, Validacion, Recomendacion.
- **Importancia**: BAJO
- **Justificacion**: Similar a improve-prompt.md. Herramienta generica de prompt engineering sin relacion con el pipeline de clasificacion. Podria extraerse a un repositorio compartido.

### 1.4 `.claude/prompts/meta-prompt-claude-code.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.claude/prompts/meta-prompt-claude-code.md`
- **Lineas/Tamano**: 773 lineas
- **Proposito**: Framework comprehensivo de prompt engineering con 4 niveles de tecnicas: Nivel 1 (Fundamentales: Especificidad, Contexto, Herramientas, Formato de salida), Nivel 2 (Contextuales: CoT, ReAct, Reflexion, Few-Shot, Tree of Thoughts), Nivel 3 (Seguridad: Code Safety, Error Handling), Nivel 4 (Avanzado: Token Optimization, Subagent Orchestration, Constitutional Principles).
- **Importancia**: BAJO
- **Justificacion**: El archivo mas largo en .claude/ (773 lineas). Es un meta-framework academico de prompt engineering, no especifico del proyecto. No se referencia desde ningun script ni configuracion del pipeline. Podria eliminarse sin afectar el proyecto.

### 1.5 `.claude/prompts/prompt-optimizer-v2.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.claude/prompts/prompt-optimizer-v2.md`
- **Lineas/Tamano**: 156 lineas
- **Proposito**: Paso 1 de cadena de mejora de prompts. Incluye instruccion critica de no ejecutar el prompt siendo analizado. Genera analisis estructurado y version mejorada.
- **Importancia**: BAJO
- **Justificacion**: Parte de una cadena de 2 pasos de optimizacion de prompts. Herramienta generica sin relacion con el proyecto de investigacion.

### 1.6 `.claude/prompts/prompt-refinement-chain-claude-code.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.claude/prompts/prompt-refinement-chain-claude-code.md`
- **Lineas/Tamano**: 820 lineas
- **Proposito**: Sistema de refinamiento de 6 fases con criterios de parada, presupuestos de tiempo, advertencias de seguridad. Incluye versionado semantico para versiones de prompts. El segundo archivo mas largo en .claude/.
- **Importancia**: BAJO
- **Justificacion**: Framework elaborado pero sin uso directo en el proyecto de clasificacion. 820 lineas de documentacion de prompt engineering que no contribuyen al pipeline de investigacion.

### 1.7 `.claude/prompts/prompt-refinement-v2.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.claude/prompts/prompt-refinement-v2.md`
- **Lineas/Tamano**: 209 lineas
- **Proposito**: Paso 2+ del loop iterativo de refinamiento. Define criterios para detener vs continuar iteraciones de mejora.
- **Importancia**: BAJO
- **Justificacion**: Complemento de prompt-optimizer-v2.md. Herramienta generica de productividad.

### Resumen `.claude/`

| Archivo | Lineas | Importancia | Relacion con proyecto |
|---------|--------|-------------|----------------------|
| settings.local.json | 62 | MEDIO | Directa (permisos de ejecucion) |
| commands/improve-prompt.md | 55 | BAJO | Ninguna (utileria generica) |
| commands/refine-prompt.md | 113 | BAJO | Ninguna (utileria generica) |
| prompts/meta-prompt-claude-code.md | 773 | BAJO | Ninguna (framework academico) |
| prompts/prompt-optimizer-v2.md | 156 | BAJO | Ninguna (utileria generica) |
| prompts/prompt-refinement-chain-claude-code.md | 820 | BAJO | Ninguna (framework academico) |
| prompts/prompt-refinement-v2.md | 209 | BAJO | Ninguna (utileria generica) |

**Observacion**: 6 de 7 archivos en `.claude/` son herramientas genericas de prompt engineering (2,126 lineas totales) sin relacion con el proyecto de investigacion COVID-19. Solo `settings.local.json` tiene relacion directa. Los prompts podrian moverse a un repositorio compartido o eliminarse para reducir ruido.

---

## 2. Directorio `.planning/` (680 KB, ~40 archivos)

Directorio de planificacion del framework GSD (Get Shit Done). Contiene documentacion del proyecto, requisitos, roadmap, estado actual, investigacion, analisis del codebase y artefactos por fase.

### 2.1 Archivos de nivel superior (5 archivos)

#### `.planning/config.json`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/config.json`
- **Lineas/Tamano**: 13 lineas
- **Proposito**: Configuracion del workflow GSD. Define modo "yolo" (ejecucion rapida), profundidad "standard", paralelizacion habilitada. Controla el comportamiento del framework de planificacion.
- **Importancia**: MEDIO
- **Justificacion**: Necesario para que el framework GSD funcione correctamente. La configuracion "yolo" indica preferencia por ejecucion rapida sobre planificacion exhaustiva. No afecta directamente los resultados cientificos.

#### `.planning/PROJECT.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/PROJECT.md`
- **Lineas/Tamano**: 94 lineas
- **Proposito**: Definicion del proyecto de mejora del clasificador ensemble. Documenta: valor central (maximizar accuracy sin contaminar test set), linea base (97.68%), mejora esperada (98.2-98.7%), conjunto de test (1,895 imagenes), decisiones clave tomadas, y antipatrones prohibidos.
- **Importancia**: ALTO
- **Justificacion**: Documento fundacional que define el alcance, restricciones y objetivos del milestone actual. Referenciado por todos los demas archivos de planificacion. Contiene decisiones criticas como "no usar test set para optimizacion" y "soft voting sobre hard voting".

#### `.planning/REQUIREMENTS.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/REQUIREMENTS.md`
- **Lineas/Tamano**: 116 lineas
- **Proposito**: Lista de 17 requisitos v1 organizados en 5 grupos (ENSEMBLE, TTA, METRICS, OUTPUT, VALID) con trazabilidad a fases del roadmap. Estado actual: 9 completados (53%), 8 pendientes. Incluye requisitos v2 diferidos (analisis de incertidumbre, validacion externa).
- **Importancia**: ALTO
- **Justificacion**: Documento de referencia para verificar completitud del milestone. La matriz de trazabilidad conecta cada requisito con su fase de implementacion. Los requisitos v2 documentan trabajo futuro sin comprometer el alcance actual.

#### `.planning/ROADMAP.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/ROADMAP.md`
- **Lineas/Tamano**: 112 lineas
- **Proposito**: Roadmap de 5 fases con criterios de exito especificos por fase: Phase 1 (Pre-Implementation Audit, COMPLETA), Phase 2 (Ensemble Core, COMPLETA), Phase 3 (TTA Integration, COMPLETA), Phase 4 (Analysis & Visualization, NO INICIADA), Phase 5 (Final Test Evaluation, NO INICIADA). Incluye tabla de progreso con fechas de completitud.
- **Importancia**: ALTO
- **Justificacion**: Documento central de seguimiento del proyecto. Define el orden de ejecucion, dependencias entre fases y criterios de exito verificables. Las fases 4-5 estan pendientes, indicando trabajo restante.

#### `.planning/STATE.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/STATE.md`
- **Lineas/Tamano**: 90 lineas
- **Proposito**: Estado actual del proyecto: Phase 3 de 5, 60% progreso. Registra metricas de velocidad (7 planes completados, promedio 7 min, total 0.92 horas), decisiones acumuladas, bloqueadores pendientes (9 imagenes duplicadas por limpiar), e informacion de continuidad de sesion.
- **Importancia**: ALTO
- **Justificacion**: Documento critico para continuidad entre sesiones. Contiene el contexto acumulado necesario para reanudar trabajo sin perdida de informacion. Los bloqueadores documentados (limpieza de duplicados) son prerequisitos para Phase 5.

### 2.2 Investigacion (5 archivos en `.planning/research/`)

Documentacion de investigacion generada automaticamente por el framework GSD para informar las decisiones de diseno.

#### `.planning/research/ARCHITECTURE.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/ARCHITECTURE.md`
- **Lineas/Tamano**: 522 lineas
- **Proposito**: Arquitectura detallada del sistema de evaluacion ensemble+TTA. Incluye diagramas ASCII del Evaluation Orchestrator, Inference Pipeline y Aggregation Layer. Ejemplos de codigo para EnsembleWrapper, TTAStrategy y evaluate_stratified. Diagramas de flujo de datos para inferencia y configuracion.
- **Importancia**: MEDIO
- **Justificacion**: Referencia arquitectonica valiosa para entender el diseno del sistema ensemble. Algunos patrones fueron implementados directamente en Phase 2-3. Sin embargo, es documentacion de investigacion previa, no documentacion de la implementacion final.

#### `.planning/research/FEATURES.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/FEATURES.md`
- **Lineas/Tamano**: 277 lineas
- **Proposito**: Analisis del panorama de features: 8 table stakes, 10 diferenciadores, 7 anti-features. MVP v1.0 definido (todo completado), v1.1 y v2.0+ diferidos. Grafo de dependencias y matriz de priorizacion.
- **Importancia**: MEDIO
- **Justificacion**: Util para entender las decisiones de alcance. Los anti-features son particularmente valiosos: documentan lo que NO se debe hacer (rotacion agresiva, optimizacion en test set, MC Dropout) y por que.

#### `.planning/research/PITFALLS.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/PITFALLS.md`
- **Lineas/Tamano**: 949 lineas
- **Proposito**: 15 trampas identificadas (5 de severidad CRITICA): contaminacion del test set, fuga de datos, augmentaciones inseguras, metricas infladas, cherry-picking. Incluye 30+ referencias de articulos revisados por pares (2020-2026). Preguntas de preparacion para defensa de tesis y estrategias de recuperacion.
- **Importancia**: ALTO
- **Justificacion**: El archivo mas largo de toda la investigacion (949 lineas). Contiene conocimiento critico sobre errores comunes en imagenes medicas que podrian invalidar la tesis. Las preguntas de defensa son directamente utiles para preparacion academica. Las referencias bibliograficas son verificables y relevantes.

#### `.planning/research/STACK.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/STACK.md`
- **Lineas/Tamano**: 435 lineas
- **Proposito**: Recomendaciones de stack tecnologico. Recomienda: PyTorch nativo + ttach 0.0.3 + torchmetrics 1.8.2+. Rechaza: MONAI (sobredimensionado), Ensemble-PyTorch (sobredimensionado), Neptune.ai (cerrando). Incluye guias de seguridad para augmentaciones en imagenes medicas.
- **Importancia**: MEDIO
- **Justificacion**: Investigacion util que informo las decisiones de implementacion. La justificacion de por que NO usar ciertas bibliotecas es tan valiosa como la recomendacion positiva. El stack recomendado coincide con lo que se implemento.

#### `.planning/research/SUMMARY.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/research/SUMMARY.md`
- **Lineas/Tamano**: 331 lineas
- **Proposito**: Resumen ejecutivo de toda la investigacion con nivel de confianza ALTO. Recomendacion de roadmap de 6 fases con justificacion de ordenamiento estricto. Brechas identificadas: mejora modesta de accuracy, validez del flip horizontal, fallo esperado en validacion externa.
- **Importancia**: MEDIO
- **Justificacion**: Sintesis util de la investigacion. Las brechas identificadas se confirmaron empiricamente (mejora modesta de +0.16pp con TTA, validacion externa no probada aun).

### 2.3 Analisis del Codebase (7 archivos en `.planning/codebase/`)

Analisis automatico del codebase existente generado por el mapper del framework GSD.

#### `.planning/codebase/ARCHITECTURE.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/ARCHITECTURE.md`
- **Lineas/Tamano**: 495 lineas
- **Proposito**: Analisis del sistema existente: pipeline de 3 etapas (landmarks -> warping -> classification). Flujo de datos detallado para training, warping, classification e inference. Abstracciones clave: GPA, Piecewise Affine Warping, ResNet18Landmarks, ImageClassifier. Entry points: CLI (python -m src_v2), GUI (Gradio).
- **Importancia**: MEDIO
- **Justificacion**: Documentacion de arquitectura valiosa como referencia. Facilita la comprension del pipeline completo para nuevos contribuidores o para retomar trabajo despues de una pausa.

#### `.planning/codebase/CONCERNS.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/CONCERNS.md`
- **Lineas/Tamano**: 211 lineas
- **Proposito**: Deuda tecnica, bugs conocidos, seguridad y brechas de rendimiento. Deuda tecnica: CLI monolitico (10,520 lineas), estado global en GUI, manejo amplio de excepciones. Bugs conocidos: inestabilidad de Delaunay, mismatch de tamano de imagen, deteccion de arquitectura. Seguridad: rutas arbitrarias, torch.load sin validacion, sin sanitizacion de input en GUI.
- **Importancia**: MEDIO
- **Justificacion**: Registro valioso de problemas conocidos. El bug de inestabilidad de Delaunay y la falta de validacion en torch.load son riesgos reales. Las brechas de cobertura de tests identifican areas que necesitan trabajo.

#### `.planning/codebase/CONVENTIONS.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/CONVENTIONS.md`
- **Lineas/Tamano**: 203 lineas
- **Proposito**: Convenciones de codigo: PEP 8, limite de 100 caracteres, snake_case, docstrings estilo Google. Orden de imports: stdlib -> third-party -> local. Constantes centralizadas en src_v2/constants.py.
- **Importancia**: BAJO
- **Justificacion**: Referencia util para consistencia de estilo, pero la informacion ya esta documentada en CLAUDE.md. Duplicacion parcial.

#### `.planning/codebase/INTEGRATIONS.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/INTEGRATIONS.md`
- **Lineas/Tamano**: 216 lineas
- **Proposito**: Analisis de integraciones externas. Conclusion: sistema autocontenido sin APIs externas. Solo almacenamiento basado en archivos (CSV, JSON, NPZ, PT). No hay pipeline CI/CD detectado.
- **Importancia**: BAJO
- **Justificacion**: Confirma lo que se infiere del proyecto: es un sistema de investigacion autocontenido. El valor es confirmatorio, no informativo.

#### `.planning/codebase/STACK.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/STACK.md`
- **Lineas/Tamano**: 189 lineas
- **Proposito**: Analisis del stack tecnologico actual: Python 3.9+, PyTorch 2.0+, Typer CLI, Gradio GUI. Dependencias clave: torch, numpy 2.0+, opencv-python, scipy, pandas, scikit-learn.
- **Importancia**: BAJO
- **Justificacion**: Informacion ya presente en requirements.txt y pyproject.toml. Duplicacion con valor agregado minimo.

#### `.planning/codebase/STRUCTURE.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/STRUCTURE.md`
- **Lineas/Tamano**: 363 lineas
- **Proposito**: Layout completo del directorio con propositos. Incluye ubicaciones clave para agregar nuevo codigo.
- **Importancia**: MEDIO
- **Justificacion**: Mapa util del proyecto. Las indicaciones de donde agregar nuevo codigo fueron usadas directamente en Phases 2-3.

#### `.planning/codebase/TESTING.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/codebase/TESTING.md`
- **Lineas/Tamano**: 305 lineas
- **Proposito**: Patrones de testing y brechas. pytest 7.0+ con pyproject.toml. Tests actualmente faltantes del codebase principal (archivados en scripts/archive/). FORCE_NUM_WORKERS_ZERO=1 para testing deterministico.
- **Importancia**: MEDIO
- **Justificacion**: Identifica brechas criticas en cobertura de tests: warping coordinates, ensemble TTA, data loading, CLI params. Estas brechas persisten y representan riesgo.

### 2.4 Fases de Planificacion

Cada fase sigue la estructura estandar del framework GSD con archivos CONTEXT, RESEARCH, VERIFICATION, PLAN(es) y SUMMARY(ies).

#### Phase 01: Pre-Implementation Audit (12 archivos)

- **Ruta base**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/01-pre-implementation-audit/`
- **Archivos**: 01-CONTEXT.md (78 lineas), 01-RESEARCH.md (836 lineas), 01-VERIFICATION.md (453 lineas), 01-01-PLAN.md (209 lineas), 01-01-SUMMARY.md (124 lineas), 01-02-PLAN.md (366 lineas), 01-02-SUMMARY.md (168 lineas), AUDIT_REPORT.md (438 lineas), BASELINE_VERIFICATION.json (46 lineas), DATA_INTEGRITY_CHECK.txt (90 lineas), GIT_HISTORY_AUDIT.txt (135 lineas), SEED_SELECTION_PROTOCOL.md (204 lineas)
- **Lineas totales**: ~3,147 lineas
- **Proposito**: Auditoria pre-implementacion completa. Verifico integridad del test set (1,895 imagenes con distribucion correcta), confirmo baseline de 97.68% con match exacto, valido aislamiento del test set mediante 4 metodos independientes (git, logs, timestamps, configs). Detecto 9 imagenes duplicadas (1 test, 8 validacion) con tasa de fuga de 0.053%.
- **Importancia**: CRITICO
- **Justificacion**: Estos archivos contienen la evidencia de rigor metodologico necesaria para la tesis. El AUDIT_REPORT.md es un documento de referencia para la defensa de tesis. BASELINE_VERIFICATION.json proporciona evidencia reproducible de que los modelos base funcionan como se reporta. DATA_INTEGRITY_CHECK.txt y GIT_HISTORY_AUDIT.txt proporcionan evidencia forense de aislamiento del test set. SEED_SELECTION_PROTOCOL.md documenta la metodologia de validacion cruzada. Sin esta documentacion, la validez cientifica del trabajo seria cuestionable.

**Hallazgos clave de la auditoria**:
- Baseline 97.68% verificado con diferencia = 0.000000 en los 5 folds
- 1 duplicado train-test (Normal-818/817), 8 duplicados train-val
- Test set evaluado 11 dias despues del entrenamiento (aislamiento temporal)
- Ningun metrica de test en logs de entrenamiento
- Protocolo de seleccion: F1-macro en validacion, early stopping con patience=10

#### Phase 02: Ensemble Core (7 archivos)

- **Ruta base**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/02-ensemble-core/`
- **Archivos**: 02-CONTEXT.md (82 lineas), 02-RESEARCH.md (568 lineas), 02-VERIFICATION.md (503 lineas), 02-01-PLAN.md (270 lineas), 02-01-SUMMARY.md (193 lineas), 02-02-PLAN.md (281 lineas), 02-02-SUMMARY.md (242 lineas)
- **Lineas totales**: ~2,139 lineas
- **Proposito**: Implementacion del nucleo ensemble. Carga de 5 modelos CV, implementacion de soft voting (promedio ponderado de probabilidades) y hard voting (voto mayoritario), evaluacion con metricas completas. Resultado: 98.10% accuracy (+0.42pp sobre baseline), reduccion de errores del 47%.
- **Importancia**: ALTO
- **Justificacion**: Documentacion completa del proceso de implementacion del ensemble. El VERIFICATION.md (503 lineas) proporciona evidencia exhaustiva de que los 7 criterios de exito se cumplieron. Los PLANes son ejecutables y reproducibles. Los SUMMARYs documentan decisiones tecnicas (uso de F1-macro de validacion como pesos, einsum para promedio ponderado, device mismatch bug fix).

**Hallazgos clave**:
- Soft voting: 98.10% accuracy (einsum-based probability averaging)
- Hard voting: 98.10% accuracy (identico, indicando consenso fuerte)
- 19 errores de 1,895 muestras (tasa de error 1.0%)
- Bug corregido: torch.ones creado en CPU mientras prob_sum estaba en CUDA

#### Phase 03: TTA Integration (9 archivos)

- **Ruta base**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.planning/phases/03-tta-integration/`
- **Archivos**: 03-CONTEXT.md (320 lineas), 03-RESEARCH.md (453 lineas), 03-VERIFICATION.md (388 lineas), 03-01-PLAN.md (420 lineas), 03-01-SUMMARY.md (115 lineas), 03-02-PLAN.md (352 lineas), 03-02-SUMMARY.md (153 lineas), 03-03-PLAN.md (224 lineas), 03-03-SUMMARY.md (108 lineas)
- **Lineas totales**: ~2,533 lineas
- **Proposito**: Integracion de Test-Time Augmentation con horizontal flip. Implementacion de TTA a dos niveles (model-level + ensemble-level), configuracion via JSON con override CLI (--tta/--no-tta), tracking de impacto por caso (helped/hurt/neutral), metricas delta. Resultado: 98.26% accuracy (+0.16pp sobre ensemble sin TTA).
- **Importancia**: ALTO
- **Justificacion**: Documentacion exhaustiva del proceso TTA. El CONTEXT.md (320 lineas) es particularmente valioso por documentar la decision de NO aplicar correccion de simetria para clasificacion (a diferencia de landmarks donde L/R swap es critico). El VERIFICATION.md documenta un ciclo de re-verificacion donde se cerro una brecha (funciones orphaned) en plan 03-03. Los detalles de impacto por caso (6 helped, 3 hurt, 1886 neutral) son directamente citables en la tesis.

**Hallazgos clave**:
- TTA: 98.26% accuracy, +0.16pp mejora
- COVID se beneficia mas (+0.44% F1), Viral degrada levemente (-0.28% F1)
- No se necesita correccion de simetria (clases anatomicamente simetricas)
- 03-03 fue un plan de cierre de brecha para funciones implementadas pero no conectadas

### Resumen `.planning/`

| Subdirectorio | Archivos | Lineas aprox. | Importancia | Estado |
|---------------|----------|---------------|-------------|--------|
| Nivel superior (config, PROJECT, etc.) | 5 | ~425 | ALTO | Activos |
| research/ | 5 | ~2,514 | MEDIO-ALTO | Referencia |
| codebase/ | 7 | ~2,182 | MEDIO | Referencia |
| phases/01-pre-implementation-audit/ | 12 | ~3,147 | CRITICO | Completo |
| phases/02-ensemble-core/ | 7 | ~2,139 | ALTO | Completo |
| phases/03-tta-integration/ | 9 | ~2,533 | ALTO | Completo |

**Observacion**: El directorio `.planning/` contiene ~12,940 lineas de documentacion estructurada. Es el registro completo del proceso de investigacion y desarrollo. Los archivos de Phase 01 (AUDIT_REPORT, DATA_INTEGRITY_CHECK, etc.) son evidencia critica para la tesis. Los archivos de research/ son referencia valiosa pero no referenciados directamente por el codigo. Las fases 4 y 5 aun no tienen archivos.

---

## 3. Directorio `.pytest_cache/` (92 KB)

### `.pytest_cache/README.md`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/.pytest_cache/README.md`
- **Lineas/Tamano**: 9 lineas
- **Proposito**: Archivo estandar generado automaticamente por pytest. Describe que el directorio contiene datos de cache de pytest y no debe incluirse en control de versiones.
- **Importancia**: ELIMINABLE
- **Justificacion**: Generado automaticamente. Debe estar en .gitignore (y lo esta). El directorio completo es efimero y regenerable. Los 92 KB son probablemente cache de ejecuciones previas de tests.

---

## 4. Directorio `results/` (613 MB, ~17,420 archivos)

Directorio de resultados experimentales con datos historicos y actuales. La gran mayoria del tamano viene de archivos PNG de visualizacion (~17,360 imagenes). Solo 47 archivos son JSON/TXT con metricas y logs.

### 4.1 `results/archive/warping_legacy/validation/` (4 archivos JSON)

**Archivos**:
- `cross_evaluation_valid_3classes/cross_evaluation_results.json` (136 lineas)
- `external_validation/warped_on_warped_results.json`
- `pfs_warped_valid/pfs_warped_details.json`
- `pfs_warped_valid/pfs_warped_summary.json` (29 lineas)

- **Proposito**: Resultados de validacion de versiones anteriores del pipeline de warping. El cross_evaluation_results.json contiene evaluacion cruzada entre modelos entrenados en datos originales vs warped (4 combinaciones: A en A, A en B, B en A, B en B). El pfs_warped_summary.json contiene metricas de Pixel Fill Score para 200 muestras COVID (mean PFS=0.474).
- **Importancia**: BAJO
- **Justificacion**: Resultados historicos de versiones anteriores del pipeline. Estan correctamente archivados. No se referencian en el pipeline actual. Podrian ser utiles para comparaciones historicas en la tesis pero no son criticos. El cross_evaluation muestra que el modelo entrenado en datos warped generaliza mejor (gap_B=3.17 vs gap_A=7.70).

### 4.2 `results/geometry_artifacts/` (3 archivos JSON + visualizaciones PNG)

**Archivos JSON**:
- `canonical_shape_global.json`: Coordenadas de la forma canonica calculada por GPA (15 landmarks x 2 coordenadas para 957 imagenes)
- `delaunay_triangles.json`: 18 triangulos de Delaunay para warping piecewise affine
- `test_predictions.json`: Predicciones de landmarks para el test set

- **Proposito**: Artefactos geometricos del pipeline de warping: forma canonica global, triangulacion y predicciones. Son las entradas fundamentales para el proceso de normalizacion geometrica.
- **Importancia**: MEDIO
- **Justificacion**: Estos archivos son artefactos generados del pipeline, no configuraciones manuales. Se pueden regenerar ejecutando el pipeline completo. Sin embargo, representan el estado validado de la forma canonica (3.61 px de error) y la triangulacion optima. Son referencia util pero no insustituibles.

### 4.3 `results/logs/` (1 log de ejecucion + 14 historicos + 2 latest_run)

**Archivos**:
- `execution_log.txt`: Log completo de la validacion Fisher Linear Analysis con barras de progreso detalladas (miles de lineas de output de tqdm)
- `historical/evaluation_report_2025MMDD_HHMMSS.txt`: 14 reportes de evaluacion historicos del 27-28 de noviembre y 7 de diciembre 2025 (~42 lineas cada uno)
- `latest_run/training_config.json` (38 lineas): Configuracion de la ultima ejecucion de entrenamiento (seed=123, coord_attention=true, wing loss, CLAHE)
- `latest_run/training_history.json`: Historial de entrenamiento de la ultima ejecucion

- **Proposito**: Logs de ejecucion y configuracion para reproducibilidad y debugging.
- **Importancia**: BAJO
- **Justificacion**: Los logs historicos son voluminosos y de valor limitado para el estado actual del proyecto. El execution_log.txt es particularmente ruidoso (barras de progreso completas de tqdm). training_config.json es el unico archivo con valor de referencia, documentando la configuracion exacta de una ejecucion. Los 14 reportes historicos muestran la evolucion experimental pero no son criticos.

### 4.4 `results/metrics/` (8 archivos)

**Archivos**:
- `audit_results.json` (10 lineas): Resultados de auditoria Fisher - accuracy raw=0.847, warped=0.834 (warped ligeramente peor en Fisher analysis)
- `basic_metrics.txt` (4 lineas): Accuracy: 0.7188 con confusion matrix 2x2 (resultados tempranos, binarios)
- `basic_metrics_full_warped_dataset.txt`: Metricas basicas del dataset warped completo
- `eval_subset.json`: Metricas de evaluacion en subconjunto
- `experiment_results.json` (38 lineas): Resultados experimentales con clasificacion raw vs warped (accuracy 0.6875 vs 0.6979, mejora de 1.04%)
- `grid_search_clahe.json`: Resultados de grid search para parametros CLAHE
- `grid_search.json`: Resultados de grid search PCA+clasificadores (k-NN, LogisticRegression, LinearSVM con diferentes n_components)
- `robustness_original_cropped_47.json`: Metricas de robustez en datos recortados

- **Proposito**: Metricas experimentales de diferentes etapas del proyecto. Incluyen desde experimentos tempranos binarios (COVID vs sano, accuracy ~70%) hasta grid searches de hiperparametros y auditorias de calidad.
- **Importancia**: MEDIO
- **Justificacion**: Registro historico valioso que muestra la progresion del proyecto. Los grid searches documentan la seleccion de hiperparametros (PCA components, CLAHE settings). Sin embargo, los resultados finales estan documentados en GROUND_TRUTH.json y los outputs del pipeline actual. Estos archivos son complementarios.

### 4.5 `results/predictions/` (4 archivos JSON)

**Archivos**:
- `classify_efficientnet_e2e.json`: Predicciones de clasificador EfficientNet end-to-end
- `classify_warp_covid.json`: Predicciones para clase COVID con warping
- `classify_warp_normal.json`: Predicciones para clase Normal con warping
- `classify_warp_viral.json`: Predicciones para clase Viral con warping

- **Proposito**: Predicciones por clase de diferentes configuraciones del clasificador. Permiten analisis detallado de errores y comparacion entre modelos.
- **Importancia**: BAJO
- **Justificacion**: Resultados intermedios de experimentos previos. No son referenciados por el pipeline actual ni el ensemble. Utiles solo para analisis retrospectivo.

### 4.6 `results/validation/external_validation/` (9 archivos JSON + datos)

**Archivos JSON**:
- `baseline_results.json` (413 lineas): Evaluacion de 14 modelos (7 arquitecturas x 2 tipos de datos) en Dataset3 FedCOVIDx (8,482 muestras). Mejores resultados: resnet18_original 57.5% accuracy, todos los modelos <58% accuracy.
- `dataset3/preparation_stats.json`: Estadisticas de preparacion del dataset externo
- `dataset3_warped/test_landmarks.json`: Landmarks predichos para el dataset externo
- `dataset3_warped/test_warping_summary.json`: Resumen de warping del dataset externo
- `mapping_analysis_results.json`: Analisis de mapeo entre datasets
- `warped_96_on_d3_original.json`: Evaluacion del modelo warped_96 en D3 original
- `warped_96_on_d3_original_clahe.json`: Lo mismo con CLAHE
- `warped_96_on_d3_warped.json`: Evaluacion del modelo warped_96 en D3 warped
- `warped_96_on_d3_warped_clahe.json`: Lo mismo con CLAHE

- **Proposito**: Validacion externa en el dataset FedCOVIDx. Todos los modelos obtuvieron accuracy cercana al azar (~50-58%), demostrando que los modelos no generalizan a datos de diferente distribucion.
- **Importancia**: ALTO
- **Justificacion**: Resultados criticos para la tesis. La falta de generalizacion a datos externos es un hallazgo importante que debe reportarse honestamente. Documenta los limites del enfoque y es evidencia de integridad cientifica (reportar resultados negativos). El baseline_results.json con 14 modelos evaluados proporciona comparacion comprehensiva.

### 4.7 `results/validation/pfs_warped_valid_full/` (2 archivos JSON)

**Archivos**:
- `pfs_warped_details.json`: Detalles de Pixel Fill Score por muestra
- `pfs_warped_summary.json` (34 lineas): Resumen PFS para 500 muestras (362 COVID, 138 Normal). Mean PFS=0.487, correctas=0.485, incorrectas=0.506.

- **Proposito**: Validacion de calidad de warping mediante Pixel Fill Score (porcentaje de pixeles no negros en imagen warped). Las imagenes clasificadas incorrectamente tienen PFS ligeramente mayor (0.506 vs 0.485), sugiriendo que el warping de baja calidad puede contribuir a errores.
- **Importancia**: MEDIO
- **Justificacion**: Analisis de calidad util para entender la relacion entre calidad de warping y accuracy de clasificacion. El hallazgo de que imagenes con mayor PFS tienen mayor tasa de error es un insight valioso para la tesis.

### 4.8 Subdirectorios de imagenes y visualizaciones

**Directorios con PNGs**:
- `detailed_analysis/` (FN/, FP/, TN/, TP/): Imagenes clasificadas por tipo de error
- `figures/` (archive/, diagrams/, pipeline_viz/, publication/, visual_analysis/, visualizations/): Figuras generadas para la tesis
- `geometry_artifacts/all_visualizations/` (covid/, normal/, viral_pneumonia/): Visualizaciones de landmarks y warping
- `multiclass_experiment/`: Resultados de experimento multiclase
- `BACKUP_ERROR_2026-01-07/`: Respaldo de error

- **Proposito**: ~17,360 imagenes PNG de visualizacion, analisis de errores, figuras de publicacion y respaldos.
- **Importancia**: MEDIO (figuras de publicacion ALTO, resto BAJO)
- **Justificacion**: Las figuras en `figures/publication/` son directamente usables en la tesis. Las imagenes de `detailed_analysis/` son utiles para investigacion de errores. El directorio `BACKUP_ERROR_2026-01-07` deberia investigarse y posiblemente eliminarse. El volumen total (613 MB) es significativo y podria reducirse eliminando visualizaciones intermedias.

### Resumen `results/`

| Subdirectorio | Archivos JSON/TXT | PNGs | Importancia | Estado |
|---------------|-------------------|------|-------------|--------|
| archive/warping_legacy/ | 4 | 0 | BAJO | Historico |
| geometry_artifacts/ | 3 | ~cientos | MEDIO | Generados |
| logs/ | 17 | 0 | BAJO | Historico |
| metrics/ | 8 | 0 | MEDIO | Historico |
| predictions/ | 4 | 0 | BAJO | Historico |
| validation/external_validation/ | 9 | ~miles | ALTO | Actual |
| validation/pfs_warped_valid_full/ | 2 | 0 | MEDIO | Actual |
| detailed_analysis/ | 0 | ~cientos | MEDIO | Generados |
| figures/ | 0 | ~miles | MEDIO-ALTO | Para tesis |

---

## 5. `checkpoints/session13/hierarchical/results.json`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/checkpoints/session13/hierarchical/results.json`
- **Lineas/Tamano**: 22 lineas (directorio completo 46 MB)
- **Proposito**: Resultados del modelo jerarquico (HierarchicalLandmarkModel) con error de 46.62 px (vs 3.61 px del ensemble actual). Configuracion: seed=42, hidden_dim=512, dropout=0.3. El modelo jerarquico fue abandonado debido al rendimiento muy inferior.
- **Importancia**: BAJO
- **Justificacion**: Resultado de un enfoque fallido que fue correctamente abandonado. Los 46 MB del directorio contienen un modelo que no se usa. El archivo JSON documenta por que este enfoque no funciono (error 12x mayor que el ensemble). Podria eliminarse para ahorrar espacio, pero tiene valor como evidencia de exploracion de alternativas.

---

## 6. Resumen General

### Distribucion por importancia

| Importancia | Archivos | Descripcion |
|-------------|----------|-------------|
| CRITICO | 12 | Phase 01 audit artifacts (evidencia metodologica para tesis) |
| ALTO | 24 | Planning top-level, Phase 02-03, validacion externa |
| MEDIO | 25 | Research, codebase analysis, geometry artifacts, metricas |
| BAJO | 22 | Prompts, logs historicos, predicciones legacy, archive |
| ELIMINABLE | 1 | .pytest_cache/README.md |

### Recomendaciones

1. **Preservar intactos**: Todos los archivos `.planning/phases/01-*` (evidencia de rigor metodologico), archivos de nivel superior de `.planning/`, y `results/validation/external_validation/` (resultados negativos importantes).

2. **Considerar mover fuera del proyecto**: Los 6 archivos de prompt engineering en `.claude/prompts/` y `.claude/commands/` (2,126 lineas) no tienen relacion con la investigacion COVID-19.

3. **Considerar limpiar**: `results/logs/historical/` (14 reportes repetitivos), `results/BACKUP_ERROR_2026-01-07/` (respaldo de error sin documentar), y `checkpoints/session13/hierarchical/` (46 MB de modelo abandonado).

4. **Tamano total recuperable**: ~50 MB eliminando modelo jerarquico y logs historicos. ~612 MB si se eliminan las visualizaciones intermedias en results/ (conservando solo figures/publication/).

5. **Fases pendientes**: `.planning/` no tiene archivos para Phase 4 ni Phase 5 (40% del roadmap). Los archivos STATE.md y ROADMAP.md documentan correctamente esta situacion.
