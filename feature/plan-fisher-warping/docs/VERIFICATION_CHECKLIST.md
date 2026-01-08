# CHECKLIST DE VERIFICACIÓN OBLIGATORIO

**Propósito:** Prevenir errores críticos en la configuración y ejecución de experimentos.

**Uso:** COMPLETAR ESTE CHECKLIST después de cada fase crítica antes de continuar.

---

## CHECKLIST FASE 4: Generación de Features

### A. ANTES de ejecutar generate_features.py

- [ ] **Verificar qué experimento vamos a correr:**
  - [ ] ¿Es experimento de 2 clases? → DEBE usar `02_full_balanced_2class_*.csv`
  - [ ] ¿Es experimento de 3 clases? → DEBE usar `01_full_balanced_3class_*.csv`

- [ ] **Verificar código en generate_features.py líneas 191-199:**
  ```python
  # Para experimento 2-class:
  "csv": metrics_dir / "02_full_balanced_2class_warped.csv"  # ✓ CORRECTO
  "csv": metrics_dir / "01_full_balanced_3class_warped.csv"  # ✗ ERROR

  # Para experimento 3-class:
  "csv": metrics_dir / "01_full_balanced_3class_warped.csv"  # ✓ CORRECTO
  "csv": metrics_dir / "02_full_balanced_2class_warped.csv"  # ✗ ERROR
  ```

- [ ] **Verificar que los CSV existen:**
  ```bash
  ls -lh results/metrics/02_full_balanced_2class_*.csv
  ls -lh results/metrics/01_full_balanced_3class_*.csv
  ```

- [ ] **Verificar tamaños esperados según SPLIT_PROTOCOL.md:**
  - [ ] 2-class: 12,402 total (9,300 train, 1,857 val, 1,245 test)
  - [ ] 3-class: 6,725 total (5,040 train, 1,005 val, 680 test)

### B. DESPUÉS de ejecutar generate_features.py

- [ ] **Verificar log de ejecución:**
  - [ ] ¿Qué CSV se cargó? (debe aparecer explícitamente en output)
  - [ ] ¿Cuántas imágenes de test se procesaron?

- [ ] **Verificar archivos generados:**
  ```bash
  # Para 2-class esperamos:
  wc -l results/metrics/phase4_features/*_test.csv
  # Debe mostrar ~1,245 líneas (+ header)
  ```

- [ ] **Verificar distribución de clases:**
  ```python
  import pandas as pd
  df = pd.read_csv("results/metrics/phase4_features/full_warped_test.csv")
  print(df['label'].value_counts())
  # Para 2-class esperamos: Normal (1) ≈ 747, Enfermo (0) ≈ 498
  # Ratio: 1.5:1 (Normal/Enfermo)
  ```

- [ ] **Documentar en bitácora:**
  - CSV usado: `___________________________`
  - Test size: `___________________________`
  - Ratio clases: `___________________________`
  - Fecha: `___________________________`

---

## CHECKLIST FASE 5: Cálculo de Fisher Ratios

### A. ANTES de ejecutar generate_fisher.py

- [ ] **Verificar que Phase 4 completó exitosamente**
- [ ] **Verificar que los CSV de features usan el dataset correcto:**
  ```bash
  # Verificar size de archivos
  wc -l results/metrics/phase4_features/*_test.csv
  ```

### B. DESPUÉS de ejecutar generate_fisher.py

- [ ] **Verificar Fisher ratios calculados:**
  ```python
  import json
  with open("results/metrics/phase5_fisher/full_warped_fisher.json") as f:
      fisher = json.load(f)
  print(f"Top 5 Fisher: {fisher['fisher_ratios'][:5]}")
  # Verificar que suman sentido (típicamente PC1 > PC2 > PC3...)
  ```

- [ ] **Verificar archivos amplificados:**
  ```bash
  wc -l results/metrics/phase5_fisher/*_test_amplified.csv
  # Debe coincidir con test size de Phase 4
  ```

- [ ] **Verificar coherencia con Phase 4:**
  - [ ] Mismo número de muestras test
  - [ ] Misma distribución de clases

---

## CHECKLIST FASE 6: Clasificación KNN

### A. ANTES de ejecutar generate_classification.py

- [ ] **Verificar que Phase 5 completó exitosamente**
- [ ] **Verificar tamaño de datos amplificados:**
  ```bash
  wc -l results/metrics/phase5_fisher/*_test_amplified.csv
  ```

### B. DESPUÉS de ejecutar generate_classification.py

- [ ] **Verificar summary.json:**
  ```python
  import json
  with open("results/metrics/phase6_classification/summary.json") as f:
      summary = json.load(f)

  # Para 2-class:
  print(f"Test samples: {summary['full_warped']['test_metrics']['n_samples']}")
  # DEBE ser 1,245 (o el número correcto según CSV usado)
  ```

- [ ] **Verificar confusion matrix tiene sentido:**
  - [ ] Suma de todas las celdas = test size
  - [ ] Distribución refleja el ratio de clases esperado

- [ ] **Verificar que los resultados son consistentes:**
  - [ ] Accuracy val vs test no tienen gap excesivo (>5%)
  - [ ] K óptimo tiene sentido para el tamaño del dataset

---

## CHECKLIST PRE-NOTEBOOK: Antes de Documentar

### A. Verificación de Coherencia Global

- [ ] **Verificar trazabilidad de datos:**
  ```
  CSV usado → Phase 4 features → Phase 5 Fisher → Phase 6 Classification

  Tamaño test debe ser IDÉNTICO en todas las fases
  ```

- [ ] **Verificar números en summary.json vs CSV original:**
  ```python
  # Cargar CSV original
  df_csv = pd.read_csv("results/metrics/02_full_balanced_2class_warped.csv")
  test_csv = df_csv[df_csv['split'] == 'test']

  # Cargar resultados
  with open("results/metrics/phase6_classification/summary.json") as f:
      summary = json.load(f)

  # VERIFICAR:
  assert len(test_csv) == summary['full_warped']['test_metrics']['n_samples']
  print("✓ Tamaños coinciden")
  ```

- [ ] **Verificar distribución de clases:**
  ```python
  # En CSV original
  print(test_csv['label'].value_counts())

  # En confusion matrix (suma de columnas debe coincidir)
  cm = summary['full_warped']['test_metrics']['confusion_matrix']
  print(f"Enfermo: {cm[0][0] + cm[0][1]}")
  print(f"Normal: {cm[1][0] + cm[1][1]}")
  ```

### B. Verificación de Figuras

- [ ] **Verificar que todas las figuras referenciadas existen:**
  ```bash
  find results/figures -name "*.png" | sort
  # Comparar con rutas en notebooks
  ```

- [ ] **Verificar que las figuras tienen los datos correctos:**
  - [ ] Títulos mencionan el dataset correcto
  - [ ] Números en figuras coinciden con summary.json

---

## CHECKLIST PRE-REUNIÓN: Verificación Final

### A. Números Clave a Memorizar

- [ ] **Tamaño del dataset:**
  - Total: `___________`
  - Train: `___________`
  - Val: `___________`
  - Test: `___________`

- [ ] **Ratio de clases en test:**
  - Normal: `___________` (`____%`)
  - Enfermo: `___________` (`____%`)
  - Ratio: `_____:1`

- [ ] **Resultados de clasificación:**
  - Full Warped: `_____%`
  - Full Original: `_____%`
  - Mejora: `+_____%`

### B. Preguntas a Poder Responder

- [ ] ¿Qué CSV usaste?
- [ ] ¿Por qué ese CSV y no el otro?
- [ ] ¿Cuántas imágenes de test?
- [ ] ¿Cuál es el ratio de clases?
- [ ] ¿Cómo verificaste que usaste el CSV correcto?

### C. Coherencia Narrativa

- [ ] **Revisar notebooks 01-08:**
  - [ ] Todos los números coinciden con summary.json
  - [ ] No hay contradicciones entre notebooks
  - [ ] Todos los TODOs están resueltos
  - [ ] Todas las figuras existen y son correctas

---

## REGLAS DE ORO (NUNCA ROMPER)

### 1. VERIFICACIÓN EXPLÍCITA SIEMPRE
```
✗ "El código funciona → los datos deben ser correctos"
✓ "Verificar EXPLÍCITAMENTE qué archivo se carga"
```

### 2. DOCUMENTACIÓN PRESCRIPTIVA
```
✗ "Existen estos CSVs: 01_*, 02_*"
✓ "SI experimento=2class ENTONCES usar 02_*, NUNCA 01_*"
```

### 3. VALIDACIÓN AUTOMÁTICA
```python
# SIEMPRE agregar validaciones en el código:
expected_test_size = 1245  # para 2-class
actual_test_size = len(test_data)
assert actual_test_size == expected_test_size, \
    f"ERROR: Esperaba {expected_test_size} test, obtuve {actual_test_size}"
```

### 4. LOGGING EXPLÍCITO
```python
# SIEMPRE loggear archivos críticos:
print(f"[INFO] Cargando CSV: {csv_path}")
print(f"[INFO] Test size: {len(test_data)}")
print(f"[INFO] Class distribution: {Counter(test_labels)}")
```

### 5. COHERENCIA INPUT→OUTPUT
```
Antes de cada fase, verificar que el OUTPUT de la fase anterior
tiene el tamaño y distribución ESPERADOS.
```

---

## FORMATO DE BITÁCORA

Completar después de cada fase:

```
FASE: ___________
FECHA: ___________
CSV USADO: ___________________________
COMANDO EJECUTADO: ___________________________

VERIFICACIONES:
[ ] Tamaño test: _____ (esperado: _____)
[ ] Ratio clases: _____ (esperado: _____)
[ ] Archivos generados: _____

PROBLEMAS ENCONTRADOS:
_____________________________________________________

NOTAS:
_____________________________________________________
```

---

**Autor:** Claude (aprendiendo de errores críticos)
**Revisado por:** Usuario
**Fecha:** 2026-01-07
**Versión:** 1.0 - Post error crítico del 2026-01-07
