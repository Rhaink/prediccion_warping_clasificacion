# Checkpoint Cleanup Report

**Fecha**: 2026-01-20
**Ejecutado por**: Script automatizado `scripts/cleanup_checkpoints.sh`

## Resumen Ejecutivo

Se ejecutó una limpieza moderada de la carpeta `/checkpoints` liberando **~133.4 GB** (99.5% del espacio total).

**Resultado**:
- Antes: 134 GB
- Después: 629 MB
- Espacio liberado: 133.4 GB

## Modelos Preservados

Los siguientes 6 modelos fueron preservados y respaldados:

### Modelos Críticos (ensemble_best.json)

1. `checkpoints/session10/ensemble/seed123/final_model.pt` (46 MB)
   - Parte del ensemble best (3.61 px)
   - Fecha: 2025-11-27

2. `checkpoints/session13/seed321/final_model.pt` (46 MB)
   - Parte del ensemble best (3.61 px)
   - Fecha: 2025-11-28

3. `checkpoints/repro_split111/session14/seed111/final_model.pt` (46 MB)
   - Parte del ensemble best (3.61 px)
   - Fecha: 2026-01-10

4. `checkpoints/repro_split666/session16/seed666/final_model.pt` (46 MB)
   - Parte del ensemble best (3.61 px)
   - Fecha: 2026-01-11

### Modelos Históricos

5. `checkpoints/session10/ensemble/seed456/final_model.pt` (46 MB)
   - Mejor modelo individual (4.04 px)
   - Preservado por valor histórico
   - Fecha: 2025-11-27

6. `checkpoints/session13/seed789/final_model.pt` (46 MB)
   - Parte de ensemble obsoleto (3.71 px)
   - Preservado por valor histórico
   - Fecha: 2025-11-27

## Archivos Eliminados

### 1. Checkpoints Intermedios
- **Cantidad**: 1,125 archivos `checkpoint_epoch*.pt`
- **Espacio liberado**: ~43 GB
- **Justificación**: Snapshots de entrenamiento no necesarios. Los `final_model.pt` son suficientes.

### 2. Experimentos de Reproducción
- `repro_quickstart/` - 21 GB
- `repro_split456/` - 13 GB
- `repro_split123/` - 5.9 GB
- `repro_split222/` - 4.9 GB
- `repro_split333/` - 5.0 GB
- `repro_split444/` - 5.2 GB
- `repro_split555/` - 5.8 GB
- `repro_split789/` - 5.7 GB
- `repro_split321/` - 4.0 GB
- `repro_split456_rerun/` - 5.2 GB
- `repro_tuned/` - 5.6 GB
- `repro_tuned2/` - 5.2 GB
- `repro_exact/` - 11 GB
- `repro_exact_longpat/` - 4.5 GB
- `repro/` - 22 GB
- **Espacio total liberado**: ~77 GB
- **Justificación**: Experimentos no críticos, no contienen modelos del ensemble actual

### 3. Experimentos de Ablation
- `session10/exp1_dropout02/` - 46 MB
- `session10/exp2_hidden1024/` - 46 MB
- `session10/exp3_hidden512/` - 45 MB
- `session10/exp4_epochs100/` - 46 MB
- `session10/exp5_lr1e5/` - 4 KB
- **Espacio total liberado**: ~183 MB
- **Justificación**: Resultados documentados en sesiones

### 4. Debug Runs
- `debug_runs/` - 545 MB
- `final_model.pt` (raíz) - 45 MB
- **Espacio total liberado**: ~590 MB
- **Justificación**: Archivos de depuración no necesarios

## Backup

Se creó un backup comprimido de los 6 modelos preservados:

- **Archivo**: `checkpoints_backup_20260120.tar.gz`
- **Tamaño**: 253 MB
- **Ubicación**: `backups/checkpoints_backup_20260120.tar.gz` (raíz del proyecto)
- **Contenido**: 6 archivos `final_model.pt`

### Contenido del Backup

```bash
tar -tzf checkpoints_backup_20260120.tar.gz
seed123_final.pt
seed321_final.pt
seed111_final.pt
seed666_final.pt
seed456_final.pt
seed789_final.pt
```

## Verificación de Integridad

Todos los modelos críticos fueron verificados exitosamente:

```bash
✓ session10/ensemble/seed123/final_model.pt - Cargado correctamente
✓ session13/seed321/final_model.pt - Cargado correctamente
✓ repro_split111/session14/seed111/final_model.pt - Cargado correctamente
✓ repro_split666/session16/seed666/final_model.pt - Cargado correctamente
```

El ensemble puede ser evaluado con:
```bash
python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json
```

Resultado esperado: ~3.61 px (según GROUND_TRUTH.json)

## Archivos Adicionales No Eliminados

Los siguientes archivos no fueron eliminados (no estaban en el plan original):

- `checkpoints/best_model_7.21px.pt` - 46 MB
- `checkpoints/best_model_7.84px.pt` - 45 MB
- `checkpoints/best_model_session10.pt` - 46 MB
- `checkpoints/session13/hierarchical/final_model.pt` - 46 MB

**Total**: ~183 MB

Estos archivos pueden eliminarse manualmente si se desea liberar espacio adicional.

## Estructura Final

Después de la limpieza, `checkpoints/` contiene:

```
checkpoints/
├── best_model_*.pt (3 archivos, ~137 MB) [opcional]
├── session10/
│   └── ensemble/
│       ├── seed123/final_model.pt (46 MB) ✓ CRÍTICO
│       └── seed456/final_model.pt (46 MB) ✓ HISTÓRICO
├── session13/
│   ├── hierarchical/final_model.pt (46 MB) [opcional]
│   ├── seed321/final_model.pt (46 MB) ✓ CRÍTICO
│   └── seed789/final_model.pt (46 MB) ✓ HISTÓRICO
├── repro_split111/
│   └── session14/seed111/final_model.pt (46 MB) ✓ CRÍTICO
└── repro_split666/
    └── session16/seed666/final_model.pt (46 MB) ✓ CRÍTICO
```

**Tamaño total**: 629 MB (de 134 GB originales)

## Próximos Pasos

1. **Verificar ensemble completo** (opcional):
   ```bash
   python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json
   ```

2. **Backup movido a ubicación segura**: ✓ COMPLETADO
   - El backup está en `backups/checkpoints_backup_20260120.tar.gz`
   - Agregado a `.gitignore` para evitar versionado accidental

3. **Eliminar archivos adicionales** (opcional):
   ```bash
   rm checkpoints/best_model_*.pt
   rm checkpoints/session13/hierarchical/final_model.pt
   ```
   Esto liberaría ~183 MB adicionales.

4. **Actualizar scripts que referencian checkpoints eliminados**:
   - `scripts/quickstart_landmarks.sh` - Actualizar paths de repro_quickstart

## Notas Finales

- La limpieza fue **exitosa** y **segura**
- Todos los modelos críticos están **preservados y verificados**
- El backup está disponible en caso de necesidad
- El espacio liberado permite continuar con el trabajo sin restricciones

## Referencias

- Plan completo: `/home/donrobot/.claude/plans/staged-skipping-stream.md`
- Script de limpieza: `scripts/cleanup_checkpoints.sh`
- Configuración de ensemble: `configs/ensemble_best.json`
- Ground truth: `GROUND_TRUTH.json`
