# 16. Auxiliary Scripts

Analisis de scripts auxiliares: build, deploy, glass box visualizations y miscelaneos.

**Archivos analizados**: 15

---

## Build / Deploy Scripts

### build_portable_windows.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/build_portable_windows.py
- **Lineas/Tamano**: 1083 lineas / 36 KB
- **Proposito**: Crea un paquete portable para Windows que incluye Python embebido (3.12.8), todas las dependencias como wheels, modelos, codigo fuente, scripts batch y documentacion en un ZIP autocontenido (~800 MB).
- **Contenido clave**:
  - Clase `PortableBuilder` con pipeline de 11 pasos: validacion de entorno, descarga de Python embeddable, configuracion de `._pth`, bootstrap de pip, descarga de dependencias (torch con `--no-deps` para evitar triton-rocm + requirements_windows_full.txt), copia de modelos con checksums SHA256, copia de codigo fuente, creacion de `install_deps.py` embebido, generacion de batch files (RUN_DEMO.bat, RUN_DEMO_SHARE.bat, INSTALL.bat), documentacion README.txt/VERSION.txt, empaquetado ZIP con verificacion de integridad.
  - `MODEL_MAPPINGS` define los 4 modelos de landmarks + clasificador + archivos de shape analysis con rutas estandarizadas.
  - Genera script `install_deps.py` inline que instala wheels offline y verifica paquetes criticos (torch, gradio, sympy, networkx, click).
  - Templates para .bat incluyen modo local (puerto 7860) y modo compartido (Gradio tunnel de 72 horas).
  - Verificacion de espacio en disco (3 GB minimo), cache de descargas, limpieza de temporales.
- **Importancia**: ALTO
- **Justificacion**: Script principal para generar el paquete distribuible de la demo. Muy bien estructurado con validaciones exhaustivas, manejo robusto de errores y documentacion generada automaticamente. Critico para la defensa de tesis.

### build_release.sh
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/build_release.sh
- **Lineas/Tamano**: 243 lineas / 7 KB
- **Proposito**: Script bash para crear un paquete de release multiplataforma (Linux/macOS/Windows) copiando codigo fuente, modelos, documentacion y scripts de instalacion en un ZIP con checksums.
- **Contenido clave**:
  - Copia codigo fuente (src_v2, configs, scripts/run_demo.py), ejemplos opcionales y documentacion.
  - Copia los 4 modelos de landmarks, clasificador y archivos de shape analysis con estructura aplanada en `models/`.
  - Cada modelo tiene fallback con warning si no existe (no falla fatalmente).
  - Genera README.md en espanol con instrucciones de instalacion para Linux/macOS/Windows.
  - Genera checksums SHA256 y empaqueta en ZIP.
  - Referencia `install.sh`, `install.bat`, `run_demo.sh`, `run_demo.bat` que copia desde la raiz del proyecto (asume que existen).
- **Importancia**: MEDIO
- **Justificacion**: Complemento del build_portable_windows.py para distribuciones multiplataforma. Menos sofisticado que la version portable (no embebe Python, no descarga dependencias). Los scripts que referencia (`install.sh`, `install.bat`, etc.) no parecen existir en el repositorio, lo que puede causar fallos.

### build_windows_exe.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/build_windows_exe.py
- **Lineas/Tamano**: 380 lineas / 12 KB
- **Proposito**: Automatiza la creacion de un ejecutable standalone de Windows (.exe) usando PyInstaller: crea entorno virtual de build, instala dependencias CPU-only, prepara modelos y ejecuta PyInstaller.
- **Contenido clave**:
  - Tres modos: `--prepare` (crear venv + instalar deps), `--build` (ejecutar PyInstaller), `--clean` (limpiar artefactos), `--all` (todo).
  - `verify_models_exist()` valida los 4 modelos de landmarks, clasificador, shape analysis y GROUND_TRUTH.json con reporte de tamano.
  - `create_build_environment()` crea `.venv_build` limpio con dependencias de `requirements_windows_cpu.txt`.
  - `build_executable()` limpia builds anteriores, ejecuta `prepare_models_for_build.py`, luego PyInstaller con el spec file.
  - Deteccion de plataforma: si no es Windows, advierte que el ejecutable sera nativo Linux/macOS (no .exe).
  - Genera checksum SHA256 del ejecutable final.
  - Mensajes en espanol con colores ANSI.
- **Importancia**: MEDIO
- **Justificacion**: Approach alternativo al portable package. PyInstaller produce un unico .exe de ~1.8 GB pero es mas dificil de depurar. La estrategia portable (build_portable_windows.py) parece mas practica para distribucion.

### prepare_models_for_build.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/prepare_models_for_build.py
- **Lineas/Tamano**: 88 lineas / 3 KB
- **Proposito**: Copia los modelos desde sus ubicaciones de desarrollo a un directorio staging (`build/models_staging/`) con nombres estandarizados para el empaquetado con PyInstaller.
- **Contenido clave**:
  - Mapeo de 7 archivos: 4 modelos de landmarks renombrados a `resnet18_seedXXX_best.pt`, clasificador y 2 archivos JSON de shape analysis.
  - Crea estructura `landmarks/`, `classifier/`, `shape_analysis/` dentro del staging.
  - Reporta tamano de cada archivo copiado y total.
  - Falla si cualquier modelo falta (exit code 1).
- **Importancia**: BAJO
- **Justificacion**: Script auxiliar simple invocado por build_windows_exe.py. Funcion muy especifica y correcta. Solo relevante si se usa la ruta PyInstaller.

### covid_demo.spec
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/covid_demo.spec
- **Lineas/Tamano**: 203 lineas / 5 KB
- **Proposito**: Archivo de especificacion PyInstaller que define como empaquetar la demo en un ejecutable unico (single-file) de Windows con todos los modelos, codigo fuente y dependencias.
- **Contenido clave**:
  - Entry point: `scripts/run_demo.py`.
  - `datas`: modelos desde staging (~227 MB), GROUND_TRUTH.json, src_v2/ completo, configs/, ejemplos opcionales.
  - `hiddenimports`: lista exhaustiva de 40+ modulos (gradio, uvicorn, torch, cv2, scipy, reportlab, etc.) que PyInstaller no detecta automaticamente.
  - `excludes`: pytest, jupyter, tkinter, PyQt, pip, setuptools, tests de numpy/scipy/matplotlib.
  - Single-file EXE con UPX compression habilitado, consola visible, sin icono ni version info (TODO).
  - Target size estimado: ~1.8 GB.
- **Importancia**: MEDIO
- **Justificacion**: Configuracion bien documentada con listas razonables de hiddenimports y excludes. Los TODO de icono y version info son menores. Necesario solo si se usa la ruta PyInstaller.

---

## Requirements Files

### requirements_windows_cpu.txt
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/requirements_windows_cpu.txt
- **Lineas/Tamano**: 32 lineas / 1 KB
- **Proposito**: Dependencias para build del ejecutable PyInstaller con PyTorch CPU-only, incluyendo PyInstaller como herramienta de build.
- **Contenido clave**:
  - torch==2.4.1+cpu y torchvision==0.19.1+cpu via `--extra-index-url` de PyTorch.
  - gradio>=6.0, opencv-python-headless, numpy, scipy, scikit-learn, matplotlib, pandas, reportlab.
  - pyinstaller>=6.3.0 como dependencia de build.
- **Importancia**: BAJO
- **Justificacion**: Archivo de soporte para build_windows_exe.py. Version simplificada de las dependencias.

### requirements_windows_full.txt
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/requirements_windows_full.txt
- **Lineas/Tamano**: 123 lineas / 5 KB
- **Proposito**: Lista completa y exhaustiva de todas las dependencias para el paquete portable de Windows, incluyendo dependencias transitivas perdidas por `--no-deps` de torch y paquetes con markers de plataforma Windows.
- **Contenido clave**:
  - Secciones bien organizadas: PyTorch deps (sympy, mpmath, networkx, filelock, fsspec), Gradio stack completo (typer, uvicorn, starlette, anyio, httpx, websockets), procesamiento de imagenes, computacion cientifica, utilidades, paquetes Windows-specific (colorama, pytz, tzdata).
  - Documentacion inline detallada explicando por que cada paquete es necesario.
  - Nota explicita sobre exclusion de pytorch-triton-rocm.
  - Autodocumentado como "SINGLE SOURCE OF TRUTH" para dependencias del paquete portable.
- **Importancia**: ALTO
- **Justificacion**: Archivo critico para el build portable. La documentacion inline es excelente -- cada dependencia tiene justificacion. Resuelve problemas reales de cross-platform pip download (markers de Windows no evaluados correctamente desde Linux).

### requirements_windows_portable.txt
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/requirements_windows_portable.txt
- **Lineas/Tamano**: 38 lineas / 1 KB
- **Proposito**: Dependencias simplificadas para el paquete portable. Version minimalista sin dependencias transitivas explicitas.
- **Contenido clave**:
  - Las mismas dependencias principales que cpu.txt pero sin pyinstaller.
  - Usa `--find-links` en vez de `--extra-index-url` para torch.
  - No especifica versiones +cpu de torch (potencialmente problematico).
- **Importancia**: BAJO
- **Justificacion**: Parece una version anterior o alternativa de requirements_windows_full.txt. El build_portable_windows.py lo referencia en la variable `REQUIREMENTS_FILE` pero el metodo `install_dependencies()` realmente usa `requirements_windows_full.txt`. Esta inconsistencia podria ser confusa; el archivo full.txt es el que realmente se usa.

---

## Mantenimiento

### cleanup_checkpoints.sh
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/cleanup_checkpoints.sh
- **Lineas/Tamano**: 167 lineas / 6 KB
- **Proposito**: Script de limpieza para liberar ~120 GB de espacio en disco eliminando checkpoints intermedios y experimentos no criticos, preservando los 6 modelos esenciales con backup previo.
- **Contenido clave**:
  - Modo DRY_RUN por defecto (seguro) -- requiere `DRY_RUN=false` para ejecucion real.
  - Backup de 6 modelos criticos (4 de ensemble + seed456 best individual + seed789 historico) en tarball.
  - 6 pasos de limpieza: backup, checkpoints intermedios (`checkpoint_epoch*.pt`), repro_quickstart, 13 experimentos no criticos (repro_split*), 5 ablation experiments, debug runs.
  - Funcion `safe_delete()` que verifica existencia y reporta tamano antes de eliminar.
  - Resumen con tamano antes/despues y instrucciones de verificacion post-limpieza.
  - `set -euo pipefail` para ejecucion estricta.
- **Importancia**: MEDIO
- **Justificacion**: Script de mantenimiento critico que ya fue ejecutado (liberando 133 GB segun docs). Bien disenado con modo seguro por defecto y backup obligatorio. Util como referencia historica de que se elimino.

### test_exe_startup.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/test_exe_startup.py
- **Lineas/Tamano**: 306 lineas / 10 KB
- **Proposito**: Suite de tests automatizados para validar el ejecutable standalone COVID19_Demo.exe: verifica existencia, tamano, checksum SHA256 y capacidad de arranque (smoke test).
- **Contenido clave**:
  - 5 tests: existencia del archivo, tamano razonable (500 MB - 3 GB), verificacion de checksum SHA256, smoke test de arranque (lanza proceso, espera 15 segundos, verifica que no crashea), test de integracion manual completo (requiere interaccion del usuario).
  - Smoke test usa `subprocess.Popen` con timeout y termina el proceso limpiamente.
  - Test de integracion manual guia al usuario por 6 pasos de verificacion.
  - Resumen con conteo de tests pasados/fallidos y colores ANSI.
  - `--full` flag para habilitar el test manual.
- **Importancia**: MEDIO
- **Justificacion**: Complemento necesario del proceso de build. El smoke test es practico; el test manual es la unica forma real de validar la funcionalidad end-to-end de un ejecutable empaquetado.

---

## Glass Box Visualizations

### block_a_pipeline.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/glass_box_visualizations/block_a_pipeline.py
- **Lineas/Tamano**: 353 lineas / 12 KB
- **Proposito**: Genera visualizaciones del pipeline completo (Block A): A1 muestra el recorrido de una imagen a traves de las 4 etapas (original, landmarks, warped, clasificacion); A2 muestra una grilla comparativa de imagenes originales vs normalizadas.
- **Contenido clave**:
  - `generate_A1_complete_flow()`: Figura de 4 paneles (1x4) mostrando imagen original, landmarks detectados con conexiones, imagen warped, y clasificacion con barras de probabilidad. Carga metricas de GROUND_TRUTH.json, usa ClassifierModel para inferencia real, genera warped on-the-fly si no existe.
  - `generate_A2_comparison_grid()`: Grilla de 2xN (original arriba, warped abajo) con N muestras diversas de todas las clases, coloreadas por categoria.
  - `_find_warped_image()`: Busca imagen warped correspondiente en 3 posibles rutas.
  - `main()`: Configura paths hardcoded relativos al proyecto.
  - Las flechas entre paneles en A1 usan `plt.Arrow` (approach basico que puede no renderizar bien con `tight_layout`).
- **Importancia**: MEDIO
- **Justificacion**: Visualizaciones utiles para la tesis y presentacion. El codigo funciona pero tiene algunas fragilidades: paths hardcoded, la carga del clasificador asume estructura especifica de checkpoint, las flechas entre paneles pueden no renderizar correctamente.

### block_b_landmarks.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/glass_box_visualizations/block_b_landmarks.py
- **Lineas/Tamano**: 1195 lineas / 43 KB
- **Proposito**: Genera 6+ figuras de visualizacion del sistema de deteccion de landmarks (Block B): feature hierarchy de ResNet-18 (B1), Coordinate Attention con mapas reales (B2 con 3 variantes), Wing Loss (B4), Ensemble + TTA (B5), error por landmark (B6).
- **Contenido clave**:
  - `load_landmark_model()`: Carga modelo detectando automaticamente arquitectura (coord_attention, deep_head, hidden_dim) desde el checkpoint.
  - `generate_B1_feature_hierarchy()`: Extrae features de las 4 capas de ResNet-18 usando `LandmarkFeatureExtractor` y visualiza con `FeatureVisualizer`.
  - `generate_B2_coordinate_attention()`: Diagrama arquitectonico con mapas reales (promedio de canales) usando `ArchitectureDiagramBuilder`. Extrae tensores intermedios de cada paso de Coordinate Attention.
  - `generate_B2_coordinate_attention_panels()`: Version alternativa con 6 paneles (entrada CLAHE, mapa antes/despues de atencion, atencion H/W separada, mascara combinada). Usa overlays con colormaps.
  - `generate_B2_coordinate_attention_lung_view()`: Tercera variante mostrando layer3 + atencion como overlay anatomico.
  - `generate_B4_wing_loss()`: Comparacion Wing Loss vs MSE vs L1 con 3 subplots (funciones de perdida, gradientes, zoom en errores pequenos).
  - `generate_B5_ensemble_tta()`: Diagrama completo del ensemble con 4 modelos + TTA, barras de error, explicacion de TTA con pares simetricos, tabla resumen.
  - `generate_B6_error_by_landmark()`: Heatmap horizontal de error por landmark con colorbar y opcionalmente overlay sobre imagen.
  - CLI con argparse: seleccion de figuras, checkpoint, imagen, device, directorio de salida.
  - Importa `diagram_utils.ArchitectureDiagramBuilder` con import relativo (`from diagram_utils import ...`), lo que requiere ejecutar desde el directorio correcto o manipulacion de path.
- **Importancia**: ALTO
- **Justificacion**: Archivo mas grande y completo del sistema de visualizaciones. Genera figuras de calidad publicacion (300 DPI) que explican el sistema de landmarks para audiencia no tecnica. Las 3 variantes de B2 con tensores reales son particularmente valiosas para la tesis. La deteccion automatica de arquitectura en `load_landmark_model()` replica logica que existe en otros lugares (potencial DRY violation).

### diagram_utils.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/glass_box_visualizations/diagram_utils.py
- **Lineas/Tamano**: 540 lineas / 17 KB
- **Proposito**: Utilidades para crear diagramas arquitectonicos profesionales de redes neuronales usando matplotlib, con cajas de capas, flechas, grid patterns para feature maps y diagramas de flujo.
- **Contenido clave**:
  - `ArchitectureDiagramBuilder`: Clase builder con metodos para agregar cajas de capas (`add_layer_box` con feature maps opcionales como grid), flechas (`add_arrow`), anotaciones de texto, etiquetas de dimensiones (C x H x W), y guardado.
  - `_add_grid_pattern()`: Dibuja celdas individuales coloreadas segun feature map real (downsampled al tamano del grid).
  - `create_resnet_diagram()`: Genera diagrama completo de ResNet-18 con 8 capas (input, layer1-4, GAP, FC, output) posicionadas horizontalmente con tamanos proporcionales.
  - `create_operation_diagram()`: Genera explicaciones visuales de Conv (kernel 3x3 con Sobel), MaxPooling (2x2 blocks) y ReLU (funcion + antes/despues).
  - `create_flow_diagram()`: Genera diagramas de flujo genericos con cajas y flechas en orientacion horizontal o vertical.
  - Todos los diagramas a 300 DPI con bordes redondeados y annotations limpias.
- **Importancia**: MEDIO
- **Justificacion**: Framework de utilidades bien disenado para generar diagramas consistentes. El `ArchitectureDiagramBuilder` es reutilizable. Los diagramas de operaciones (conv, pooling, relu) son didacticos pero no son especificos del proyecto -- son explicaciones genericas de DL.

### README.md (glass_box_visualizations)
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/glass_box_visualizations/README.md
- **Lineas/Tamano**: 326 lineas / 9 KB
- **Proposito**: Documentacion completa del sistema de visualizaciones glass-box: estructura, figuras planificadas (Blocks A-E), instrucciones de uso, guias de diseno y checklist de implementacion.
- **Contenido clave**:
  - Estructura de 5 bloques: A (pipeline overview), B (landmarks, implementado), C (warping, pendiente), D (clasificador, pendiente), E (comparaciones, pendiente).
  - Checklist detallado: Blocks A y B completados (~60%), Blocks C-E pendientes con firmas de funciones planificadas.
  - Guias de diseno: paleta de colores, tipografia, calidad de figuras (300 DPI, PNG), estilo de annotations en espanol.
  - Ejemplos de uso de utilidades (load_representative_samples, draw_landmarks_on_image, ArchitectureDiagramBuilder).
  - Prerequisitos: 6 archivos de datos necesarios listados.
  - Convenciones: nombrado de funciones `generate_XN_description()`, metricas de GROUND_TRUTH.json, docstrings con Args/Returns.
- **Importancia**: BAJO
- **Justificacion**: Documentacion interna util para desarrollo futuro del sistema de visualizaciones. No afecta funcionalidad. Refleja que el sistema esta ~60% completo (Blocks C-E sin implementar). El generate_all.py orquestador tampoco existe aun.

### utils.py (glass_box_visualizations)
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/glass_box_visualizations/utils.py
- **Lineas/Tamano**: 509 lineas / 15 KB
- **Proposito**: Utilidades comunes para todas las visualizaciones glass-box: carga de muestras representativas, dibujo de landmarks, grillas de comparacion, overlays de heatmaps, gestion de colores y paleta estandarizada.
- **Contenido clave**:
  - `COLORS` y `LANDMARK_COLORS`: paletas consistentes para COVID (verde), Normal (azul), Viral (amarillo/naranja), landmarks centrales (rojo), izquierdos (azul), derechos (verde).
  - `load_ground_truth()`: Carga GROUND_TRUTH.json relativo al archivo.
  - `load_representative_samples()`: Carga muestras desde predictions.npz con 3 criterios de seleccion (diverse=maxima varianza, typical=cercano a mediana, difficult=aleatorio como fallback).
  - `create_comparison_grid()`: Grid de matplotlib generico con titulos y suptitle.
  - `add_panel_labels()`: Agrega etiquetas (a), (b), (c) en 4 posiciones posibles.
  - `draw_landmarks_on_image()`: Dibuja 15 landmarks con colores por grupo (central/left/right) y conexiones opcionales del eje central.
  - `create_heatmap_overlay()`: Overlay de heatmap con colormap JET sobre imagen base.
  - `apply_clahe()`, `load_checkpoint()`, `denormalize_landmarks()`: Utilidades basicas.
  - `save_figure()`: Guarda con 300 DPI, creando directorios automaticamente.
  - `get_landmark_groups()`: Extrae grupos de landmarks desde constants.py.
- **Importancia**: MEDIO
- **Justificacion**: Modulo de utilidades bien organizado que centraliza funcionalidad compartida. `load_representative_samples()` tiene logica interesante de seleccion por diversidad. Algunas funciones duplican logica existente en src_v2 (apply_clahe, denormalize_landmarks) pero con interfaces simplificadas apropiadas para el contexto de visualizacion.

---

## Resumen Ejecutivo

### Estadisticas Generales
| Categoria | Archivos | Lineas Totales |
|-----------|----------|----------------|
| Build/Deploy | 5 | 1,997 |
| Requirements | 3 | 193 |
| Mantenimiento | 2 | 473 |
| Glass Box Visualizations | 5 | 2,923 |
| **Total** | **15** | **5,586** |

### Distribucion de Importancia
| Nivel | Archivos |
|-------|----------|
| CRITICO | 0 |
| ALTO | 3 (build_portable_windows.py, requirements_windows_full.txt, block_b_landmarks.py) |
| MEDIO | 7 (build_release.sh, build_windows_exe.py, covid_demo.spec, cleanup_checkpoints.sh, test_exe_startup.py, block_a_pipeline.py, diagram_utils.py, utils.py) |
| BAJO | 4 (prepare_models_for_build.py, requirements_windows_cpu.txt, requirements_windows_portable.txt, README.md glass_box) |
| ELIMINABLE | 0 |

### Observaciones Principales

1. **Dos estrategias de build**: El proyecto mantiene dos enfoques paralelos para distribucion en Windows: paquete portable con Python embebido (build_portable_windows.py, la estrategia principal y mas robusta) y ejecutable PyInstaller (build_windows_exe.py + covid_demo.spec, mas experimental). Ambas estan bien implementadas pero la duplicacion agrega mantenimiento.

2. **Inconsistencia en requirements**: `build_portable_windows.py` declara `REQUIREMENTS_FILE = requirements_windows_portable.txt` pero el metodo `install_dependencies()` realmente usa `requirements_windows_full.txt`. El archivo portable.txt es mas simple y no incluye dependencias transitivas necesarias, lo que sugiere que full.txt es el correcto y portable.txt puede ser vestigial.

3. **Glass box ~60% completo**: De los 5 bloques planificados (A-E, con ~20 figuras), solo A y B estan implementados. Blocks C (warping/GPA), D (clasificador/GradCAM) y E (comparaciones) y el orquestador generate_all.py quedan pendientes. Block B es el mas completo y valioso.

4. **Duplicacion de logica**: Varios patrones se repiten entre archivos de build: listas de modelos (MODEL_MAPPINGS en 3+ archivos), verificacion de existencia, estructura de destino. `load_landmark_model()` en block_b_landmarks.py replica deteccion de arquitectura que existe en `src_v2/cli.py`.

5. **Import relativo fragil en glass_box**: `block_b_landmarks.py` usa `from diagram_utils import ArchitectureDiagramBuilder` (import relativo sin prefijo de paquete), lo que requiere ejecutar desde el directorio correcto o que el directorio este en sys.path. No hay `__init__.py` en el directorio.

6. **Calidad de documentacion**: Los scripts de build y requirements_windows_full.txt tienen documentacion inline excelente, con justificaciones para cada decision (por que --no-deps, por que cada dependencia, workarounds de plataforma). Este nivel de detalle es particularmente util dado lo complejo del cross-platform packaging.

7. **Metricas hardcodeadas**: `build_release.sh` y `build_portable_windows.py` incluyen metricas (accuracy 98.05%, F1 97.12%) en templates de README generado. Estos valores no provienen dinamicamente de GROUND_TRUTH.json, lo que puede desincronizarse. Los scripts glass_box si cargan metricas de GROUND_TRUTH.json correctamente.
