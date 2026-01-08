Protocolo de split y balanceo

Split fijo (sin generar nuevos splits)
- Manual: usar los splits ya existentes en `outputs/warped_dataset`.
- Full: usar los splits ya existentes en `outputs/full_warped_dataset`.
- Para comparar original vs warped, se usa el mismo split por ID de imagen.

Reglas de mapeo
- Full warped usa sufijo `_warped` en el nombre de archivo; se remueve para
  igualar con el ID del original.
- Se excluyen imagenes manuales sin coordenadas (42 casos).

Balanceo del dataset completo (por split)
- 3-clases: ratio 2:1 respecto a Viral_Pneumonia (COVID y Normal se capean).
- 2-clases: ratio 1.5:1 Normal vs Enfermo (Enfermo = COVID + Viral_Pneumonia).
- Muestreo aleatorio con seed 123 para seleccionar subconjuntos balanceados.

Archivos de referencia
- `feature/plan-fisher-warping/results/metrics/01_full_balanced_3class_original.csv`
- `feature/plan-fisher-warping/results/metrics/01_full_balanced_3class_warped.csv`
- `feature/plan-fisher-warping/results/metrics/02_full_balanced_2class_original.csv`
- `feature/plan-fisher-warping/results/metrics/02_full_balanced_2class_warped.csv`

---

## REGLAS DE USO DE ARCHIVOS CSV (PRESCRIPTIVAS)

### REGLA 1: Selección de CSV según Experimento

**SI el experimento es de 2 clases (Enfermo vs Normal):**
- ✅ USAR: `02_full_balanced_2class_*.csv`
- ❌ NUNCA: `01_full_balanced_3class_*.csv`

**SI el experimento es de 3 clases (COVID vs Normal vs Viral Pneumonia):**
- ✅ USAR: `01_full_balanced_3class_*.csv`
- ❌ NUNCA: `02_full_balanced_2class_*.csv`

### REGLA 2: Sufijos Original vs Warped

**Para comparar efecto del warping:**
- Dataset original: `*_original.csv`
- Dataset warped: `*_warped.csv`
- Ambos deben tener el MISMO prefijo (01_* o 02_*)

### REGLA 3: Validación Obligatoria

**Al cargar un CSV, SIEMPRE verificar:**

```python
# Ejemplo para experimento de 2 clases:
import pandas as pd

df = pd.read_csv("02_full_balanced_2class_warped.csv")
test_df = df[df['split'] == 'test']

# VALIDAR tamaño
expected_test_size = 1245  # según especificación
actual_test_size = len(test_df)
assert actual_test_size == expected_test_size, \
    f"ERROR: CSV incorrecto. Esperaba {expected_test_size} test, obtuve {actual_test_size}"

# VALIDAR ratio de clases
class_counts = test_df['label'].value_counts()
print(f"Distribución test: {class_counts}")
# Para 2-class esperamos: Normal ≈ 747, Enfermo ≈ 498 (ratio 1.5:1)
```

### REGLA 4: Hardcodeo Prohibido

**NUNCA hardcodear rutas de CSV en el código sin comentarios explícitos:**

```python
# ❌ MAL (lo que causó el error del 2026-01-07):
datasets = [
    {
        "name": "full_warped",
        "csv": metrics_dir / "01_full_balanced_3class_warped.csv"  # Sin comentario
    },
]

# ✅ BIEN:
datasets = [
    {
        "name": "full_warped",
        # CSV para experimento de 2 clases - VERIFICADO 2026-01-07
        "csv": metrics_dir / "02_full_balanced_2class_warped.csv"
    },
]
```

**MEJOR: Usar configuración centralizada:**

```python
# En config.py
EXPERIMENT_TYPE = "2class"  # o "3class"

CSV_PATHS = {
    "2class": {
        "warped": "02_full_balanced_2class_warped.csv",
        "original": "02_full_balanced_2class_original.csv",
    },
    "3class": {
        "warped": "01_full_balanced_3class_warped.csv",
        "original": "01_full_balanced_3class_original.csv",
    }
}

# En generate_features.py
csv_path = CSV_PATHS[EXPERIMENT_TYPE]["warped"]
```

---

## TAMAÑOS ESPERADOS DE DATASETS

### Dataset Completo de 2 Clases (02_*)

```
Total: 12,402 imágenes
├── Train: 9,873 (79.6%)
├── Val:   1,284 (10.4%)
└── Test:  1,245 (10.0%)

Distribución de clases en test:
- Enfermo (COVID + Viral): 498 (40%)
- Normal: 747 (60%)
- Ratio: 1.5:1 (Normal/Enfermo)
```

### Dataset Completo de 3 Clases (01_*)

```
Total: 6,725 imágenes
├── Train: 5,361 (79.7%)
├── Val:   684 (10.2%)
└── Test:  680 (10.1%)

Distribución de clases en test:
- COVID: 272 (40%)
- Normal: 272 (40%)
- Viral Pneumonia: 136 (20%)
- Ratio: 2:1 respecto a Viral_Pneumonia
```

### Dataset Manual

```
Total manual: ~700 imágenes (solo landmarks manuales)
Con mismas proporciones de split (80/10/10)
```

---

## ERRORES COMUNES A EVITAR

### ❌ ERROR 1: Usar CSV de 3 clases para experimento de 2 clases

**Síntomas:**
- Test size = 680 en lugar de 1,245
- Ratio invertido: 60% Enfermo, 40% Normal
- Resultados no comparables con literatura

**Causa:**
- Hardcodeo incorrecto en `generate_features.py`
- No verificar tamaño de dataset después de cargar

**Solución:**
- Usar `02_full_balanced_2class_*.csv`
- Verificar con checklist obligatorio (ver `docs/VERIFICATION_CHECKLIST.md`)

### ❌ ERROR 2: Mezclar original y warped entre fases

**Síntomas:**
- Resultados inconsistentes entre fases
- Gráficos comparativos incorrectos

**Causa:**
- No mantener coherencia de sufijo (_original vs _warped)

**Solución:**
- Cada pipeline completo debe usar SIEMPRE el mismo sufijo
- Documentar en bitácora qué CSV se usó en cada fase

### ❌ ERROR 3: No verificar tamaño después de cargar

**Síntomas:**
- Experimentos ejecutan sin error pero con datos incorrectos
- Detección tardía del problema

**Causa:**
- Confiar en que "si corre, está bien"
- No validar assumptions

**Solución:**
- SIEMPRE verificar: `assert len(test) == expected_size`
- Loggear explícitamente: `print(f"Test size: {len(test)}")`

---

## CHANGELOG

### 2026-01-07 - Versión 2.0 (Post error crítico)
- **CAMBIO CRÍTICO:** Agregadas reglas prescriptivas de uso de CSVs
- Agregadas validaciones obligatorias
- Agregados tamaños esperados de datasets
- Agregada sección de errores comunes
- Prohibido hardcodeo sin comentarios
- Referencias a VERIFICATION_CHECKLIST.md

### 2024-12-28 - Versión 1.0 (Original)
- Protocolo inicial de splits y balanceo
- Creación de archivos CSV balanceados
