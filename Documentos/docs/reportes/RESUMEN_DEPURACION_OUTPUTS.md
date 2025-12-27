# Resumen de Depuracion de Outputs (Warping Invalido)

**Fecha:** 2025-12-21
**Autor:** Codex (por solicitud del usuario)

## Motivo
Durante investigacion manual, se determino que los datasets:
- `outputs/warped_replication_v2/`
- `outputs/full_coverage_warped_dataset/`

no cumplen el objetivo de warpear la imagen conservando la ROI del contorno
pulmonar y su informacion interior. Se aplico un procedimiento erroneo, por lo
que estos outputs y cualquier modelo entrenado sobre ellos no deben usarse.

## Acciones realizadas
1. Eliminados los directorios de outputs invalidos:
   - `outputs/warped_replication_v2/`
   - `outputs/full_coverage_warped_dataset/`
2. Archivado el script generador asociado a full coverage:
   - `scripts/archive/invalid_warping/generate_warped_dataset_full_coverage.py`
3. Actualizado `scripts/README.md` para remover el script de la lista activa.

## Impacto y consideraciones
- Cualquier clasificador entrenado con esos datasets es invalido y debe
  revisarse (por ejemplo `outputs/classifier_replication_v2/` y
  `outputs/classifier_warped_full_coverage/`).
- La documentacion que referencia esos datasets debe marcarse como invalida
  o actualizarse con el dataset correcto.

## Pendientes sugeridos
- Auditar documentos que recomiendan `warped_replication_v2` o
  `full_coverage_warped_dataset` para evitar su uso.
- Definir el dataset correcto de warping y regenerar resultados si aplica.
