# Proveniencia de all_landmarks.npz

## Estado actual
- Archivo principal en uso: `outputs/predictions/all_landmarks.npz`
- Respaldo intacto: `results/geometry_artifacts/all_landmarks.npz`

`outputs/predictions/` fue movido a `results/geometry_artifacts/` durante la limpieza
documentada en `audit/OUTPUTS_CLEANUP_20251227.md`, luego restaurado localmente.

## Origen reproducible
El archivo se deriva de `data/coordenadas/coordenadas_maestro.csv`:
- Coordenadas originales en escala 299x299.
- Escalado a 224x224 con factor `224/299`.
- Splits estratificados con semilla 42:
  - train: 717
  - val: 144
  - test: 96

## Script de generación
Script reproducible agregado:
`scripts/generate_all_landmarks_npz.py`

Uso:
```
python scripts/generate_all_landmarks_npz.py
python scripts/generate_all_landmarks_npz.py --compare-to outputs/predictions/all_landmarks.npz
```

## Verificación de equivalencia
Se generó un `.npz` nuevo y se comparó contra el original:
- Resultado: match exacto en todas las claves y valores.

Nota: no se reemplazó el archivo en uso; se preserva el original.
