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
