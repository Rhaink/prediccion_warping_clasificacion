# Plan de Migración (Limpieza Opción A: solo pipeline Python)

## Objetivo
Crear una carpeta limpia del proyecto **solo con el código Python necesario** para replicar el pipeline (CLI + módulos), manteniendo la estructura de rutas esperada por el código, sin incluir datos, outputs ni checkpoints. Este plan **no modifica el código**; únicamente define qué mover y cómo.

## Alcance (Opción A)
Incluye lo mínimo para ejecutar el pipeline desde cero, asumiendo que los datos y checkpoints se colocan en la nueva carpeta en las rutas estándar:
- Código fuente y CLI (`src_v2/`).
- Configuraciones (`configs/`).
- Scripts activos de soporte (ver lista recomendada).
- Metadatos del proyecto (README, licencia, manifests, requirements, pyproject).

Excluye datasets, outputs, checkpoints y todo material histórico o de documentación pesada.

## Supuestos
- Se mantiene la estructura de rutas relativas usadas por el proyecto:
  - `data/` para datasets.
  - `checkpoints/` para modelos.
  - `outputs/` y `results/` para salidas.
- Los archivos grandes (datos/modelos/resultados) **no se versionan** y se añadirán posteriormente fuera de Git.
- No se corrigen referencias en README o docs (solo plan).

## Inventario resumido (base actual)
- Código principal: `src_v2/` (88 archivos).
- Configs: `configs/` (11 JSON).
- Scripts: `scripts/` (174 archivos; muchos son históricos o de visualización).
- Artefactos locales: `data/`, `outputs/`, `results/`, `checkpoints/` (muy grandes).

## Contenido a incluir (core mínimo)
**Obligatorio**
- `src_v2/` (CLI y módulos de data/models/training/processing/evaluation/gui/visualization)
- `configs/`
- `pyproject.toml`
- `requirements.txt`
- `README.md`
- `LICENSE`
- `GROUND_TRUTH.json`
- `MANIFEST.in`
- `.gitignore`

**Scripts recomendados (activos y de soporte directo)**
- `scripts/README.md`
- `scripts/train.py` (equivalente CLI)
- `scripts/predict.py`
- `scripts/predict_landmarks_dataset.py`
- `scripts/train_classifier.py` (wrapper de CLI)
- `scripts/train_hierarchical.py`
- `scripts/evaluate_ensemble_from_config.py`
- `scripts/extract_dataset_splits.py` (usado por CLI `extract-dataset-splits`)
- `scripts/run_demo.py` (GUI)
- `scripts/verify_*` (solo si se desea verificación puntual)

> Nota: el CLI es la fuente principal; los scripts se incluyen solo por compatibilidad y utilidades clave.

## Contenido a excluir
- `data/` (dataset local)
- `checkpoints/` (modelos entrenados)
- `outputs/` y `results/` (salidas y figuras)
- `build/`, `dist/`, `*.egg-info/`
- `.venv/`, `.pytest_cache/`, `__pycache__/`, `.coverage`, `.claude/`
- `docs/` (tesis/manual/archivos LaTeX)
- `scripts/archive/`, `scripts/visualization/`, `scripts/fisher/` (histórico y figuras)
- Logs, imágenes sueltas no esenciales

## Riesgos y mitigaciones
- **Rutas relativas hardcodeadas**: el código asume `data/`, `outputs/`, `checkpoints/`. Mitigación: mantener la misma estructura en la nueva carpeta.
- **Dependencias opcionales**: algunas visualizaciones requieren librerías extra. Mitigación: en Opción A no se incluyen esos scripts; si se usan, instalar dependencias manualmente.
- **Docs faltantes**: README menciona `docs/REPRO_*` que no existen en la raíz. Mitigación: documentar en README futuro (no se corrige en este plan).

## Pasos de migración (solo planificación)
1) Crear nueva carpeta destino (fuera de Git del proyecto actual).
2) Copiar únicamente los directorios y archivos definidos en “Contenido a incluir”.
3) Verificar que `.gitignore` ignore artefactos locales (`data/`, `outputs/`, `checkpoints/`, `results/`).
4) Confirmar que el CLI funciona a nivel de importación:
   - `python -m src_v2 --help`
5) Añadir datos y checkpoints en la nueva carpeta si se busca reproducir resultados.

## Verificación mínima recomendada (no ejecutar aquí)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src_v2 --help
```

## Entregables esperados
- Carpeta limpia lista para crear un repositorio Git.
- Dependencias declaradas y CLI funcional.
- Estructura lista para copiar datasets/checkpoints cuando sea necesario.

---
**Estado**: Plan preparado. No se ha realizado ninguna copia ni modificación.
