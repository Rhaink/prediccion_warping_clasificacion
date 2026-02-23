# 09. src_v2 Utils Module

Analisis de utilidades geometricas.

**Archivos analizados**: 2

---

## Resumen del modulo

El modulo `src_v2/utils/` es un paquete utilitario minimalista que centraliza una unica operacion geometrica: el calculo de vectores perpendiculares. Fue creado para eliminar duplicacion de codigo entre `losses.py` (penalizacion de simetria) y `hierarchical.py` (modelo jerarquico de landmarks). Contiene 49 lineas totales distribuidas en 2 archivos.

---

## Analisis archivo por archivo

### 1. `__init__.py`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/utils/__init__.py`
- **Lineas/Tamano**: 5 lineas / 151 bytes
- **Proposito**: Inicializa el paquete `utils` y re-exporta la funcion principal `compute_perpendicular_vector` desde `geometry.py` para acceso conveniente.
- **Contenido clave**:
  - Re-exporta unicamente `compute_perpendicular_vector` (version PyTorch) en `__all__`
  - No re-exporta `compute_perpendicular_vector_np` (version NumPy), lo cual es consistente dado que esta ultima no se usa en ningun lugar del proyecto
- **Dependencias**:
  - Importa de: `src_v2.utils.geometry`
  - Importado por: Nadie directamente (los consumidores importan desde `src_v2.utils.geometry` directamente)
- **Importancia**: BAJO
- **Justificacion**: Archivo estandar de inicializacion de paquete. Cumple su funcion de forma minima. El re-export en `__all__` es correcto pero en la practica ninguno de los consumidores usa `from src_v2.utils import compute_perpendicular_vector` -- ambos (`losses.py` y `hierarchical.py`) importan directamente desde `src_v2.utils.geometry`.

### 2. `geometry.py`

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/utils/geometry.py`
- **Lineas/Tamano**: 44 lineas / 1.3 KB
- **Proposito**: Centraliza operaciones geometricas comunes para landmarks, especificamente el calculo de vectores perpendiculares al eje L1-L2 (eje central del torax). Provee versiones tanto NumPy como PyTorch.
- **Contenido clave**:
  - `compute_perpendicular_vector_np(axis_vec)` (linea 12): Version NumPy para analisis sin autograd. Acepta vector `(2,)` o `(N, 2)`, retorna vector perpendicular unitario via rotacion 90 grados antihorario `(-y, x)`. **NOTA: esta funcion no se usa en ningun lugar del proyecto.**
  - `compute_perpendicular_vector(axis_vec)` (linea 29): Version PyTorch para uso en entrenamiento (compatible con autograd). Acepta tensores `(B, 2)` batched, retorna vectores perpendiculares unitarios `(B, 2)`.
  - Ambas funciones usan epsilon `1e-8` para estabilidad numerica en la normalizacion.
  - La operacion matematica es identica en ambas: dado `(x, y)` normalizado, retorna `(-y, x)` (rotacion 90 grados antihorario).
- **Dependencias**:
  - Importa: `numpy`, `torch`
  - Importado por:
    - `src_v2/models/losses.py` (linea 24): Usado en `SymmetryLoss.forward()` para calcular distancias perpendiculares de landmarks simetricos al eje central L1-L2
    - `src_v2/models/hierarchical.py` (linea 36): Usado en `HierarchicalLandmarkModel.hierarchical_decode()` para posicionar landmarks bilaterales respecto al eje
    - `src_v2/utils/__init__.py` (linea 3): Re-export
- **Importancia**: MEDIO
- **Justificacion**: Es una pieza correcta de refactorizacion que centraliza logica geometrica compartida entre dos modulos criticos. La funcion PyTorch es utilizada activamente en el pipeline de entrenamiento (symmetry loss y modelo jerarquico). Sin embargo, el modulo es muy pequeno y la operacion es trivial (3 lineas de logica real por funcion). No afecta al pipeline de produccion principal (prediccion + warping + clasificacion), solo al entrenamiento de landmarks.

---

## Observaciones y hallazgos

### Codigo muerto: `compute_perpendicular_vector_np`

La funcion `compute_perpendicular_vector_np` (lineas 12-26) esta definida pero **nunca se importa ni se usa** en ningun lugar del proyecto:
- No aparece en ningun import de `src_v2/`
- No aparece en ningun script de `scripts/`
- No esta incluida en el `__all__` de `__init__.py`
- El docstring del modulo menciona que es "util para funciones de analisis que no requieren autograd", pero no existe tal uso

El docstring de la funcion dice que acepta `(2,)` o `(N, 2)`, pero la implementacion solo maneja correctamente el caso `(2,)` ya que usa `np.linalg.norm(axis_vec)` sin especificar eje -- para un input `(N, 2)` calcularia la norma Frobenius del arreglo completo en lugar de la norma por fila. Esto contrasta con la version PyTorch que maneja correctamente batches via `dim=1`.

**Recomendacion**: Eliminar `compute_perpendicular_vector_np` o, si se planea usar, corregir el manejo de inputs batched con `np.linalg.norm(axis_vec, axis=-1, keepdims=True)`.

### Inconsistencia en el docstring batch

La version NumPy documenta aceptar `(N, 2)` pero no lo implementa correctamente. La version PyTorch documenta y soporta correctamente `(B, 2)`. Esta inconsistencia podria causar bugs si alguien usa la version NumPy con inputs batched.

### Import no aprovechado en `__init__.py`

Los dos consumidores reales (`losses.py` y `hierarchical.py`) importan directamente desde `src_v2.utils.geometry` en lugar de usar el re-export de `src_v2.utils`. El re-export en `__init__.py` no se aprovecha. Esto no es un problema, pero es una inconsistencia menor de estilo.

### Dependencia de numpy no necesaria

Si se elimina `compute_perpendicular_vector_np`, el import de `numpy` en `geometry.py` seria innecesario, reduciendo las dependencias del modulo.

### Sin tests

No existen tests unitarios para este modulo. Dado que la operacion es matematicamente trivial (rotacion 90 grados), el riesgo es bajo, pero un test unitario verificando la perpendicularidad y la norma unitaria seria buena practica, especialmente para documentar el comportamiento esperado con vectores degenerados (norma cercana a cero).

---

## Metricas del modulo

| Metrica | Valor |
|---------|-------|
| Archivos | 2 |
| Lineas totales | 49 |
| Tamano total | ~1.5 KB |
| Funciones publicas | 2 (`compute_perpendicular_vector`, `compute_perpendicular_vector_np`) |
| Funciones usadas | 1 (solo la version PyTorch) |
| Codigo muerto | 1 funcion (`compute_perpendicular_vector_np`, 15 lineas) |
| Consumidores directos | 2 (`losses.py`, `hierarchical.py`) |
| Tests | 0 |
| Importancia general | MEDIO |

---

## Veredicto

Modulo utilitario limpio y bien enfocado que cumple su proposito de evitar duplicacion de una operacion geometrica compartida. La unica accion recomendada es eliminar la funcion NumPy no utilizada (`compute_perpendicular_vector_np`) y su import de `numpy`, o alternativamente, corregir su implementacion batched si se planea usar en el futuro. El modulo no requiere cambios para el funcionamiento actual del pipeline.
