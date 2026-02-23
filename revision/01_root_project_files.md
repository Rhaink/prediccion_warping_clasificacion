# 01. Root Project Files

Analisis de los archivos en la raiz del proyecto: configuracion, documentacion y scripts de entrada.

**Archivos analizados**: 24

---

## Resumen por importancia

| Nivel | Cantidad | Archivos |
|-------|----------|----------|
| CRITICO | 4 | GROUND_TRUTH.json, pyproject.toml, requirements.txt, README.md |
| ALTO | 3 | CLAUDE.md, AGENTS.md, CHANGELOG.md |
| MEDIO | 6 | CONTRIBUTING.md, DEPLOYMENT.md, install.sh, install.bat, run_demo.sh, run_demo.bat |
| BAJO | 7 | CHECKLIST_DEFENSA.txt, README_USUARIO.txt, MIGRATION_PLAN.md, RELEASE_NOTES_v1.0.10.md, TEST_REPORT_20260128.md, VERIFICATION_CHECKLIST_v16.md, HOTFIX_v16_tzdata.md |
| ELIMINABLE | 4 | build_v1.0.5.log, build_v1.0.5_retry.log, README_PORTABLE_WINDOWS.txt, WINDOWS_STANDALONE_SUMMARY.md |

---

## Analisis detallado

### AGENTS.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/AGENTS.md
- **Lineas/Tamano**: 56 lineas / ~2 KB
- **Proposito**: Guia de alto nivel para agentes de IA (Codex/similar) que interactuan con el repositorio. Resume estructura, comandos, estilo de codigo y convenciones de commit.
- **Contenido clave**: Secciones sobre estructura del proyecto, build/test, coding style, testing, commit guidelines, y fuentes de verdad (GROUND_TRUTH.json). Es una version condensada de CLAUDE.md orientada a agentes genericos.
- **Dependencias**: Referencia a `src_v2/`, `configs/`, `tests/`, `GROUND_TRUTH.json`, `requirements.txt`.
- **Importancia**: ALTO
- **Justificacion**: Facilita que cualquier agente de IA trabaje correctamente con el proyecto. Complementa CLAUDE.md para herramientas que no leen ese archivo especifico.

---

### build_v1.0.5.log
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/build_v1.0.5.log
- **Lineas/Tamano**: 50 lineas / ~3 KB
- **Proposito**: Log parcial del primer intento de build del paquete portable Windows v1.0.5. Contiene solo la descarga de torch y torchvision.
- **Contenido clave**: Salida de `pip download` para torch 2.4.1+cpu y torchvision 0.19.1+cpu para win_amd64. El log esta truncado (solo 50 lineas).
- **Dependencias**: Generado por `scripts/build_portable_windows.py`.
- **Importancia**: ELIMINABLE
- **Justificacion**: Log de build historico incompleto sin valor actual. La version v1.0.5 ya fue superada por versiones posteriores (v16). No contiene informacion que no este documentada en DEPLOYMENT.md.

---

### build_v1.0.5_retry.log
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/build_v1.0.5_retry.log
- **Lineas/Tamano**: 404 lineas / ~21 KB
- **Proposito**: Log completo del retry exitoso del build del paquete portable Windows v1.0.5. Documenta la descarga de 85 paquetes wheel y la creacion del ZIP final.
- **Contenido clave**: Descarga completa de dependencias desde `requirements_windows_full.txt` (85 paquetes), copia de modelos (224.4 MB), creacion de batch files, build exitoso en 577.1 MB.
- **Dependencias**: Generado por `scripts/build_portable_windows.py`, usa `scripts/requirements_windows_full.txt`.
- **Importancia**: ELIMINABLE
- **Justificacion**: Log de build historico. El build v1.0.5 fue superado por versiones posteriores. La informacion tecnica relevante (estrategia de dependencias) ya esta documentada en DEPLOYMENT.md.

---

### CHANGELOG.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/CHANGELOG.md
- **Lineas/Tamano**: 166 lineas / ~5 KB
- **Proposito**: Registro historico de cambios del proyecto siguiendo formato Keep a Changelog con versionado semantico.
- **Contenido clave**: Versiones documentadas desde 1.0.0 (2025-11-15) hasta 2.0.0 (2025-12-11), mas seccion Unreleased con mejoras de ensemble (3.61 px), configs, y multiples fixes. Incluye seccion de resultados validados (Session 39-41) al final.
- **Dependencias**: Referencia `GROUND_TRUTH.json` para metricas validadas.
- **Importancia**: ALTO
- **Justificacion**: Registro formal de la evolucion del proyecto, esencial para reproducibilidad y para la tesis. Nota: la seccion "Validated Results" al final usa metricas antiguas (3.71 px ensemble, warped_99) que ya fueron marcadas como obsoletas en GROUND_TRUTH.json -- requiere actualizacion.

---

### CHECKLIST_DEFENSA.txt
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/CHECKLIST_DEFENSA.txt
- **Lineas/Tamano**: 436 lineas / ~12 KB
- **Proposito**: Checklist exhaustivo para preparar la defensa de tesis, cubriendo desde 1 semana antes hasta el dia despues. Incluye guion de demostracion, respuestas a preguntas del jurado, y planes de contingencia.
- **Contenido clave**: Secciones temporales (1 semana, 3 dias, 1 dia, dia de defensa, post-defensa). Listas de verificacion de hardware, build, USB, testing en maquina limpia, cronometraje, narrativa de demo (7 minutos), FAQ para jurado con respuestas modelo, contactos de emergencia.
- **Dependencias**: Referencia `COVID19_Demo.exe` (PyInstaller, ya no es la estrategia actual), `README_USUARIO.txt`.
- **Importancia**: BAJO
- **Justificacion**: Documento de preparacion personal para la defensa. Tiene valor practico pero no es parte del pipeline ni de la documentacion tecnica del proyecto. Nota: referencia la estrategia PyInstaller (.exe de 1.8 GB) que fue reemplazada por la estrategia de paquete portable.

---

### CLAUDE.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/CLAUDE.md
- **Lineas/Tamano**: 251 lineas / ~9 KB
- **Proposito**: Guia principal para Claude Code con instrucciones detalladas sobre el proyecto: estructura, comandos, arquitectura del pipeline, constantes criticas, y convenciones de desarrollo.
- **Contenido clave**: Resumen del proyecto, comandos de setup/CLI/testing, arquitectura del pipeline (GPA -> landmarks -> warping -> clasificacion), descripcion de modulos clave (models/, processing/, data/, training/, evaluation/), estructura de 15 landmarks con pares simetricos, sistema de configuracion JSON, detalles criticos de implementacion (margin 1.05, CLAHE tile 4, TTA, two-phase training), organizacion de datos y checkpoints, codigo legacy a evitar.
- **Dependencias**: Referencia extensiva a `src_v2/`, `configs/`, `scripts/`, `GROUND_TRUTH.json`, `docs/`.
- **Importancia**: ALTO
- **Justificacion**: Es la guia de referencia que permite a Claude Code (y por extension, a cualquier desarrollador) entender y trabajar con el proyecto. Documento vivo que debe mantenerse actualizado.

---

### CONTRIBUTING.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/CONTRIBUTING.md
- **Lineas/Tamano**: 192 lineas / ~4 KB
- **Proposito**: Guia de contribucion para desarrolladores externos: setup del entorno, estilo de codigo, testing, y proceso de pull requests.
- **Contenido clave**: Prerequisites (Python 3.9+, PyTorch 2.0+), instalacion en modo dev, estilo Python (PEP 8, type hints, 100 chars, Google docstrings), ejemplo de docstring, comandos de testing, proceso de PR, convenciones de commit, estructura del proyecto, constantes clave.
- **Dependencias**: Referencia `requirements.txt`, `src_v2/constants.py`, `tests/`.
- **Importancia**: MEDIO
- **Justificacion**: Documento estandar de open source. Util si el proyecto se publica, pero dado que es un proyecto de tesis individual, su utilidad practica es limitada. La informacion se solapa significativamente con CLAUDE.md y AGENTS.md.

---

### DEPLOYMENT.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/DEPLOYMENT.md
- **Lineas/Tamano**: 687 lineas / ~17 KB
- **Proposito**: Guia completa para crear y desplegar el paquete portable Windows del sistema de demostracion. Cubre el proceso de build, estructura del paquete, instrucciones para usuarios finales, testing, troubleshooting, y la solucion al problema de dependencias --no-deps.
- **Contenido clave**: Comparacion Python embeddable vs PyInstaller, pasos de build (11 pasos), estructura del paquete final (~1.2 GB descomprimido, ~800 MB ZIP), instrucciones de distribucion, checklists de testing, troubleshooting detallado, seccion extensa sobre "The --no-deps Problem" (torch vs triton), GitHub Actions CI/CD de ejemplo, historial de versiones.
- **Dependencias**: Referencia `scripts/build_portable_windows.py`, `scripts/requirements_windows_full.txt`, checkpoints, `GROUND_TRUTH.json`.
- **Importancia**: MEDIO
- **Justificacion**: Documentacion esencial para el workflow de deployment a Windows, pero este workflow es especifico para la demo de defensa de tesis y no es parte del pipeline de investigacion principal. La seccion sobre la estrategia de dependencias tiene valor tecnico significativo.

---

### GROUND_TRUTH.json
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/GROUND_TRUTH.json
- **Lineas/Tamano**: 525 lineas / ~9 KB
- **Proposito**: Fuente unica de verdad (single source of truth) para todas las metricas validadas del proyecto. Usado como referencia por scripts de verificacion, documentacion y tests.
- **Contenido clave**: Estructura JSON con secciones:
  - `_metadata`: version 2.1.0, sesiones validadas [39,42,43,52,53,55].
  - `landmarks`: 3 entradas de ensemble (3.71 obsoleta, 3.67 obsoleta, **3.61 actual**), mejor individual (4.04 px), minimo teorico (1.3 px).
  - `classification`: 6 datasets (4 obsoletos, **warped_lung_best** actual con 98.05%), cross-validation (5-fold, val 98.60%), **classifier_ensemble_cv** con baseline y TTA (98.26% con TTA).
  - `robustness`: JPEG Q50/Q30 y blur sigma1 (marcado obsoleto).
  - `cross_evaluation`: Generalizacion cruzada (obsoleto).
  - `pfs`: Pulmonary Focus Score (obsoleto).
  - `fill_rate_tradeoff`: Analisis fill rate (obsoleto).
  - `external_validation`: Dataset3 FedCovidX (obsoleto, domain shift).
  - `preprocessing`: CLAHE (clip=2.0, tile=4), arquitectura (ResNet-18, coord_attention, deep_head), warping (margin=1.05).
  - `tolerances`: Tolerancias para tests de regresion.
  - `historical_baselines`: Sesiones 4, 10, 12.
  - `per_category_landmarks` y `per_landmark_errors`: Detalle por categoria y por landmark.
- **Dependencias**: Referenciado por `CLAUDE.md`, `src_v2/gui/config.py`, scripts de verificacion, tests. Referencia rutas de checkpoints.
- **Importancia**: CRITICO
- **Justificacion**: Es el ancla de validacion de todo el proyecto. Todas las metricas reportadas deben coincidir con este archivo. Nota: muchas secciones estan marcadas como `obsolete` porque los experimentos se hicieron con configuraciones anteriores a warped_lung_best.

---

### HOTFIX_v16_tzdata.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/HOTFIX_v16_tzdata.md
- **Lineas/Tamano**: 177 lineas / ~4 KB
- **Proposito**: Documentacion del hotfix critico para el paquete portable v16 que fallaba en Windows por la dependencia faltante `tzdata` (requerida por pandas 3.0+ en Windows).
- **Contenido clave**: Descripcion del problema (pandas requiere tzdata en Windows, no incluido en requirements), solucion (agregar `tzdata>=2023.3`), verificacion del build (87 paquetes, 578.6 MB), historial de commits (99b040b4 roto -> 1a22fce9 arreglado), lecciones aprendidas sobre testing cross-platform.
- **Dependencias**: Referencia `scripts/requirements_windows_full.txt`, commits especificos.
- **Importancia**: BAJO
- **Justificacion**: Documento historico de un hotfix ya aplicado. La informacion importante (agregar tzdata) ya esta en el requirements file. Util solo como referencia de lecciones aprendidas.

---

### install.bat
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/install.bat
- **Lineas/Tamano**: 40 lineas / ~1 KB
- **Proposito**: Script batch de Windows para instalar el entorno de desarrollo: verifica Python, crea venv, instala dependencias, y verifica modelos.
- **Contenido clave**: Verificacion de Python, creacion de `.venv`, `pip install -r requirements.txt`, verificacion inline de 7 archivos de modelo en `models/` (nota: las rutas de modelo apuntan a `models/landmarks/seed*_final.pt` que es la estructura del paquete portable, no la del repo de desarrollo).
- **Dependencias**: Requiere Python instalado, `requirements.txt`. Referencia rutas de modelos del paquete portable.
- **Importancia**: MEDIO
- **Justificacion**: Util para configuracion rapida en Windows, pero las rutas de verificacion de modelos apuntan a la estructura del paquete portable (`models/`), no a la estructura del repositorio de desarrollo (`checkpoints/`). Podria causar confusion.

---

### install.sh
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/install.sh
- **Lineas/Tamano**: 78 lineas / ~2 KB
- **Proposito**: Script bash de Linux/macOS para instalar el entorno de desarrollo: verifica Python 3, crea venv, instala dependencias, y verifica modelos.
- **Contenido clave**: Verificacion de python3, creacion de `.venv`, upgrade pip, `pip install -r requirements.txt`, verificacion de 7 archivos de modelo. Mismo problema de rutas que `install.bat` (apunta a `models/` del portable, no `checkpoints/` del repo).
- **Dependencias**: Requiere python3 instalado, `requirements.txt`.
- **Importancia**: MEDIO
- **Justificacion**: Util para setup rapido, pero las rutas de verificacion no coinciden con la estructura real del repositorio de desarrollo.

---

### MIGRATION_PLAN.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/MIGRATION_PLAN.md
- **Lineas/Tamano**: 92 lineas / ~3 KB
- **Proposito**: Plan de migracion (Opcion A) para crear una carpeta limpia del proyecto con solo el pipeline Python, excluyendo datos, outputs, y checkpoints. Define que archivos incluir/excluir.
- **Contenido clave**: Inventario del proyecto (src_v2 88 archivos, configs 11 JSON, scripts 174 archivos), lista de contenido obligatorio (src_v2, configs, pyproject.toml, requirements.txt, etc.), scripts recomendados (10 scripts activos), contenido a excluir (data, checkpoints, outputs, docs, scripts historicos), riesgos (rutas relativas hardcodeadas), pasos de migracion (5 pasos).
- **Dependencias**: Referencia la estructura completa del proyecto.
- **Importancia**: BAJO
- **Justificacion**: Plan preparado pero no ejecutado ("Estado: Plan preparado. No se ha realizado ninguna copia ni modificacion."). Tiene valor como referencia para una futura limpieza del repo, pero no es un documento activo.

---

### pyproject.toml
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/pyproject.toml
- **Lineas/Tamano**: 93 lineas / ~2 KB
- **Proposito**: Configuracion principal del paquete Python: metadatos del proyecto, dependencias, scripts de entrada, y configuracion de herramientas (pytest, coverage).
- **Contenido clave**:
  - Build system: setuptools >= 61.0.
  - Proyecto: `prediccion_warping_clasificacion` v2.1.0, MIT, Python >= 3.9.
  - Dependencias: torch, torchvision, numpy, scipy, pandas, opencv-python, Pillow, scikit-learn, matplotlib, seaborn, tqdm, typer.
  - Dev extras: pytest, pytest-cov.
  - Entry point: `covid-landmarks = "src_v2.cli:app"`.
  - Pytest: testpaths `tests/`, env `FORCE_NUM_WORKERS_ZERO=1`, filtro de warnings.
  - Coverage: source `src_v2`, omit tests y scripts.
- **Dependencias**: Referencia `src_v2.cli:app`, `tests/`, `README.md`.
- **Importancia**: CRITICO
- **Justificacion**: Define la instalabilidad del paquete, las dependencias, y la configuracion de testing. Esencial para `pip install -e ".[dev]"` y `pytest`. Nota: no incluye `gradio` en dependencias principales (esta en `requirements.txt` pero no aqui) -- discrepancia con requirements.txt.

---

### README.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/README.md
- **Lineas/Tamano**: 88 lineas / ~3 KB
- **Proposito**: Documentacion principal del repositorio: resumen del pipeline, estructura, requisitos, comandos de uso, y enlaces a documentacion detallada.
- **Contenido clave**: Descripcion del pipeline de 3 pasos (landmarks, warping, clasificacion), metricas resumidas (3.61 px, ~98% accuracy), estructura del repo, requisitos (Python 3.9+), 5 comandos del pipeline actual, seccion de interfaz grafica Gradio, enlaces a documentacion clave.
- **Dependencias**: Referencia `src_v2/gui/README.md`, `docs/REPRO_FULL_PIPELINE.md`, `docs/REPRO_ENSEMBLE_3_71.md`, `docs/QUICKSTART_WARPING.md`, `docs/REPRO_CLASSIFIER_RESNET18.md`.
- **Importancia**: CRITICO
- **Justificacion**: Punto de entrada principal para cualquier persona que visite el repositorio. Conciso y enfocado en lo esencial.

---

### README_PORTABLE_WINDOWS.txt
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/README_PORTABLE_WINDOWS.txt
- **Lineas/Tamano**: 201 lineas / ~5 KB
- **Proposito**: Resumen del resultado del primer build exitoso del paquete portable Windows v1.0.0 (566 MB). Incluye proximos pasos, instrucciones para usuario final, especificaciones tecnicas, comparacion con PyInstaller, y mini-checklist de defensa.
- **Contenido clave**: Confirmacion del build exitoso (566 MB, 22029 archivos), 4 opciones de distribucion (USB, Google Drive, Wine, colega), especificaciones tecnicas (Python 3.12.8, PyTorch 2.4.1+cpu, Gradio 6.3.0, 36 paquetes -- nota: numero de paquetes incorrecto, eran 85+ en builds posteriores), comparacion portable vs PyInstaller, mini-checklist de defensa.
- **Dependencias**: Referencia `build/releases/covid19-demo-v1.0.0-portable-windows.zip`, `DEPLOYMENT.md`, `scripts/build_portable_windows.py`.
- **Importancia**: ELIMINABLE
- **Justificacion**: Resumen historico del primer build v1.0.0 que fue superado multiples veces (hasta v16). La informacion relevante ya esta en DEPLOYMENT.md. Contiene metricas desactualizadas (36 paquetes cuando son 85+).

---

### README_USUARIO.txt
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/README_USUARIO.txt
- **Lineas/Tamano**: 313 lineas / ~9 KB
- **Proposito**: Manual de usuario completo en espanol para la aplicacion de demo standalone. Cubre instrucciones de uso, requisitos del sistema, solucion de problemas, metricas, limitaciones, privacidad, y FAQ.
- **Contenido clave**: Instrucciones paso a paso para ejecutar `COVID19_Demo.exe`, descripcion de las 3 pestanas (Demostracion Completa, Vista Rapida, Acerca del Sistema), troubleshooting detallado (SmartScreen, DLL, navegador, lentitud, antivirus), metricas validadas, formatos soportados, limitaciones (domain shift, solo 3 clases, solo RX frontales), politica de privacidad, FAQ.
- **Dependencias**: Referencia `COVID19_Demo.exe` (estrategia PyInstaller, ya no activa).
- **Importancia**: BAJO
- **Justificacion**: Manual de usuario para la demo standalone. Referencia la estrategia PyInstaller (.exe) que fue reemplazada por el paquete portable. Necesitaria actualizacion si se usa con la version portable actual (RUN_DEMO.bat en lugar de .exe). Sin embargo, el contenido de troubleshooting y metricas sigue siendo valido.

---

### RELEASE_NOTES_v1.0.10.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/RELEASE_NOTES_v1.0.10.md
- **Lineas/Tamano**: 203 lineas / ~5 KB
- **Proposito**: Notas de release de la version v1.0.10 del paquete portable Windows, que agrega la visualizacion de malla de Delaunay a la interfaz de demo.
- **Contenido clave**: Nueva funcionalidad (render_delaunay_mesh en visualizer.py), ubicacion en la interfaz (Row 2), contenido del paquete (4 modelos landmark + 1 clasificador + shape analysis), archivos modificados (visualizer.py, inference_pipeline.py, app.py, __init__.py, config.py), metricas (sin cambios: 3.61 px, 98.60% CV), instrucciones de uso, detalles tecnicos de la triangulacion, checksum SHA256.
- **Dependencias**: Referencia `src_v2/gui/visualizer.py`, `src_v2/gui/inference_pipeline.py`, `src_v2/gui/app.py`.
- **Importancia**: BAJO
- **Justificacion**: Notas de release para una version intermedia del paquete portable (v1.0.10 fue superada por v16). Tiene valor historico pero no es necesaria para el trabajo actual.

---

### requirements.txt
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/requirements.txt
- **Lineas/Tamano**: 46 lineas / ~1 KB
- **Proposito**: Lista de dependencias Python del proyecto para desarrollo. Usado por `pip install -r requirements.txt`.
- **Contenido clave**: Deep learning (torch >= 2.0.0, torchvision >= 0.15.0), scientific computing (numpy >= 2.0.0, scipy, pandas), computer vision (opencv-python, Pillow), ML (scikit-learn), visualization (matplotlib, seaborn), utilities (tqdm), CLI (typer), testing (pytest, pytest-cov), GUI (gradio >= 4.0.0).
- **Dependencias**: Usado por `install.sh`, `install.bat`, `pyproject.toml` (parcialmente duplicado).
- **Importancia**: CRITICO
- **Justificacion**: Esencial para configurar el entorno de desarrollo. Nota: incluye gradio >= 4.0.0, pero `pyproject.toml` no incluye gradio en sus dependencias -- esta es una discrepancia que deberia reconciliarse. Tambien incluye pytest/pytest-cov que en pyproject.toml estan como extras `[dev]`.

---

### run_demo.bat
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/run_demo.bat
- **Lineas/Tamano**: 4 lineas / ~100 bytes
- **Proposito**: Script batch de Windows para lanzar la demo Gradio en modo desarrollo (con venv local).
- **Contenido clave**: Activa `.venv`, establece `COVID_DEMO_MODELS_DIR` al directorio `models/` local, ejecuta `python scripts\run_demo.py`.
- **Dependencias**: Requiere `.venv` configurado, `scripts/run_demo.py`, directorio `models/`.
- **Importancia**: MEDIO
- **Justificacion**: Script de conveniencia para lanzar la demo en desarrollo desde Windows. Corto y funcional.

---

### run_demo.sh
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/run_demo.sh
- **Lineas/Tamano**: 10 lineas / ~200 bytes
- **Proposito**: Script bash de Linux para lanzar la demo Gradio en modo desarrollo (con venv local).
- **Contenido clave**: Activa `.venv`, establece `COVID_DEMO_MODELS_DIR`, ejecuta `python3 scripts/run_demo.py "$@"`.
- **Dependencias**: Requiere `.venv` configurado, `scripts/run_demo.py`, directorio `models/`.
- **Importancia**: MEDIO
- **Justificacion**: Script de conveniencia para lanzar la demo en desarrollo desde Linux. Corto y funcional.

---

### TEST_REPORT_20260128.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/TEST_REPORT_20260128.md
- **Lineas/Tamano**: 567 lineas / ~14 KB
- **Proposito**: Reporte exhaustivo de testing manual del proyecto v2.1.0, documentando la validacion post-migracion. Cubre estructura, checkpoints, CLI, GPA, ensemble, y modulos individuales.
- **Contenido clave**: 13 fases de testing:
  - Fase 1: Estructura y archivos criticos (PASS).
  - Fase 2: Validacion de checkpoints PyTorch (4/4 validos, 11.9M params).
  - Fase 3: CLI (todos los comandos funcionales).
  - Fase 4: GPA (957 shapes, 18 triangulos, PASS).
  - Fase 5: Ensemble evaluado a 3.61 px (MATCHES GROUND TRUTH).
  - Fases 6-7: Warping y classifier SKIPPED (intensivos en tiempo).
  - Fase 8: Modulos (GPA PASS, warping PASS, transforms PARTIAL).
  - Fase 13: Comparacion con GROUND_TRUTH.json (MATCH exacto).
  - Issues encontrados: channel mismatch en unit tests (bajo impacto), GPA no-convergence (esperado).
  - Tiempos: GPA ~3s, ensemble eval ~2.4s.
  - Score: 13/13 tests criticos pasados.
- **Dependencias**: Referencia `configs/ensemble_best.json`, checkpoints, `GROUND_TRUTH.json`, `scripts/evaluate_ensemble_from_config.py`.
- **Importancia**: BAJO
- **Justificacion**: Reporte valioso como evidencia de validacion post-migracion, pero es un snapshot en el tiempo (2026-01-28). No se usa activamente en el pipeline. Podria moverse a `docs/reportes/`.

---

### VERIFICATION_CHECKLIST_v16.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/VERIFICATION_CHECKLIST_v16.md
- **Lineas/Tamano**: 285 lineas / ~7 KB
- **Proposito**: Checklist de verificacion completo para la version v16 del paquete portable Windows, cubriendo build, testing en Windows, documentacion, compatibilidad hacia atras, y seguridad.
- **Contenido clave**: Verificacion de build (COMPLETADA: 196 archivos, 579 MB, 86 wheels), archivos nuevos v16 (RUN_DEMO_SHARE.bat), checksums de 7 modelos, testing en Windows (PENDIENTE: 40+ items de verificacion incluyendo modo local, modo publico con Gradio sharing, error handling), revision de documentacion (COMPLETADA), compatibilidad con v15 (VERIFICADA), revision de seguridad (PASADA), decision de deployment (APROBADA pendiente testing Windows).
- **Dependencias**: Referencia `covid19-demo-v16-portable-windows.zip`, `RUN_DEMO_SHARE.bat`, `docs/RELEASE_NOTES_v16.md`.
- **Importancia**: BAJO
- **Justificacion**: Checklist especifico para v16 del paquete portable. Tiene secciones de Windows testing aun pendientes. Es un documento de proceso, no de produccion. Podria moverse a `docs/`.

---

### HOTFIX_v16_tzdata.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/HOTFIX_v16_tzdata.md
- **Lineas/Tamano**: 177 lineas / ~4 KB
- **Proposito**: (Ya analizado arriba)
- **Importancia**: BAJO

---

### WINDOWS_STANDALONE_SUMMARY.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/WINDOWS_STANDALONE_SUMMARY.md
- **Lineas/Tamano**: 325 lineas / ~8 KB
- **Proposito**: Resumen de la implementacion completa del sistema de build standalone para Windows (estrategia PyInstaller). Documenta las 6 fases de implementacion, proximos pasos, especificaciones tecnicas, y timeline.
- **Contenido clave**: 6 fases completadas:
  - Fase 1: Infraestructura de build (`build_windows_exe.py`, `requirements_windows_cpu.txt`).
  - Fase 2: Configuracion PyInstaller (`covid_demo.spec`).
  - Fase 3: Modificaciones de codigo (`run_demo.py` frozen mode, `config.py` path resolution).
  - Fase 4: Documentacion de usuario (`README_USUARIO.txt`, `CHECKLIST_DEFENSA.txt`).
  - Fase 5: Documentacion tecnica (`BUILD_WINDOWS_STANDALONE.md`).
  - Fase 6: Infraestructura de testing (`test_exe_startup.py`).
  - Specs: ~1.8 GB ejecutable, 20-30s startup, 1-2s inferencia.
  - Pendiente: build real, VM testing, preparacion USB.
- **Dependencias**: Referencia `scripts/build_windows_exe.py`, `scripts/covid_demo.spec`, `scripts/run_demo.py`, `src_v2/gui/config.py`, `CHECKLIST_DEFENSA.txt`, `README_USUARIO.txt`.
- **Importancia**: ELIMINABLE
- **Justificacion**: Documenta la estrategia PyInstaller que fue **abandonada** en favor del paquete portable (Python embeddable). La estrategia portable esta documentada en `DEPLOYMENT.md`. Este archivo no tiene valor actual ya que describe una aproximacion que no se usa.

---

## Observaciones generales

### Problemas identificados

1. **Proliferacion de documentacion de deployment en raiz**: Hay 10 archivos en la raiz relacionados con deployment/Windows que deberian consolidarse o moverse a `docs/`:
   - `DEPLOYMENT.md`, `CHECKLIST_DEFENSA.txt`, `README_PORTABLE_WINDOWS.txt`, `README_USUARIO.txt`, `RELEASE_NOTES_v1.0.10.md`, `VERIFICATION_CHECKLIST_v16.md`, `HOTFIX_v16_tzdata.md`, `WINDOWS_STANDALONE_SUMMARY.md`, `build_v1.0.5.log`, `build_v1.0.5_retry.log`.

2. **Dos estrategias de deployment documentadas**: PyInstaller (WINDOWS_STANDALONE_SUMMARY.md, README_USUARIO.txt, CHECKLIST_DEFENSA.txt) y paquete portable (DEPLOYMENT.md, README_PORTABLE_WINDOWS.txt). La estrategia PyInstaller fue abandonada pero su documentacion persiste.

3. **Discrepancia de dependencias**: `requirements.txt` incluye gradio y pytest/pytest-cov, pero `pyproject.toml` no incluye gradio en dependencias principales y solo tiene pytest en `[dev]` extras. Deberian reconciliarse.

4. **Metricas desactualizadas en CHANGELOG.md**: La seccion "Validated Results" al final del CHANGELOG.md reporta metricas de ensemble 3.71 px y warped_99, ambas marcadas como obsoletas en GROUND_TRUTH.json.

5. **Rutas incorrectas en install scripts**: Tanto `install.bat` como `install.sh` verifican modelos en `models/landmarks/` (estructura del paquete portable), no en `checkpoints/` (estructura del repo de desarrollo).

### Archivos candidatos a limpieza de raiz

Los siguientes archivos podrian eliminarse o moverse a subdirectorios:

**Eliminar** (4 archivos):
- `build_v1.0.5.log` - Log historico sin valor
- `build_v1.0.5_retry.log` - Log historico sin valor
- `README_PORTABLE_WINDOWS.txt` - Superado por DEPLOYMENT.md
- `WINDOWS_STANDALONE_SUMMARY.md` - Estrategia PyInstaller abandonada

**Mover a `docs/`** (5 archivos):
- `CHECKLIST_DEFENSA.txt` -> `docs/CHECKLIST_DEFENSA.txt`
- `HOTFIX_v16_tzdata.md` -> `docs/HOTFIX_v16_tzdata.md`
- `RELEASE_NOTES_v1.0.10.md` -> `docs/RELEASE_NOTES_v1.0.10.md`
- `TEST_REPORT_20260128.md` -> `docs/reportes/TEST_REPORT_20260128.md`
- `VERIFICATION_CHECKLIST_v16.md` -> `docs/VERIFICATION_CHECKLIST_v16.md`
- `MIGRATION_PLAN.md` -> `docs/MIGRATION_PLAN.md`

### Estadisticas

| Metrica | Valor |
|---------|-------|
| Total archivos analizados | 24 |
| Total lineas | 5,289 |
| Archivos criticos | 4 (17%) |
| Archivos eliminables | 4 (17%) |
| Archivos de deployment/Windows | 10 (42%) |
| Archivos de documentacion de proceso | 8 (33%) |
| Archivos de configuracion/codigo | 6 (25%) |
