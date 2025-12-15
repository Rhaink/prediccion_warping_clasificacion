# Trabajo Futuro: Pulmonary Focus Score (PFS)

**Fecha:** 2025-12-09
**Sesión:** 24
**Estado:** POSPUESTO - Documentado para trabajo futuro

## Resumen Ejecutivo

El análisis PFS fue implementado pero se identificó una **limitación crítica técnica** que invalida los resultados actuales. Se decidió posponer su corrección ya que la hipótesis principal de la tesis **ya está demostrada** con otras métricas.

## Hallazgos Clave

### 1. Limitación Técnica Crítica

**Problema:** Las máscaras pulmonares NO están warped junto con las imágenes.

```
Imagen Original (299x299) ──[Warping]──> Imagen Warped (224x224)
        ↓                                        ↓
  Máscara Original                      Máscara NO transformada
  (geometría correcta)                  (geometría DESALINEADA)
```

**Impacto:**
- Para imágenes ORIGINALES: PFS es preciso
- Para imágenes WARPED: PFS tiene desalineación geométrica
- Los resultados previos (~35% PFS) son técnicamente inválidos para warped

### 2. Resultados Obtenidos (Inválidos para Comparación)

| Sesión | PFS Warped | PFS Original | Diferencia | P-value |
|--------|------------|--------------|------------|---------|
| 26 | 0.3931 | 0.3721 | +0.0210 | N/A |
| 27 | 0.3865 | 0.3721 | +0.0144 | N/A |
| 29 | 0.3564 | 0.3551 | +0.0013 | **0.856** (ns) |

**Conclusión:** No hay diferencia significativa (p=0.856), pero esto puede deberse a la desalineación de máscaras.

### 3. Landmarks vs Máscaras - Son Complementarios

| Aspecto | Landmarks (Manual) | Máscaras (Dataset) |
|---------|-------------------|-------------------|
| Origen | Anotados manualmente | COVID-19_Radiography_Dataset |
| Formato | 15 puntos (x,y) | PNG binario 299x299 |
| Propósito | Warping geométrico | Segmentación/PFS |
| Cantidad | ~650 imágenes | 21,165 máscaras |

**No son contradictorios:** Los landmarks definen el CONTORNO para warping, las máscaras definen la REGIÓN para PFS.

### 4. El PFS NO Existe en Literatura

El término "Pulmonary Focus Score" no aparece en papers publicados. Si se usa en la tesis, debe:
- Definirse formalmente: `PFS = sum(heatmap * mask) / sum(heatmap)`
- Considerar renombrarlo a "Lung Region Attention Ratio" o similar
- Citar la fórmula como contribución propia

### 5. Métricas Alternativas de Literatura

Métricas aceptadas en papers de clasificación médica:

| Métrica | Descripción | Papers |
|---------|-------------|--------|
| Deletion/Insertion AUC | Faithfulness de explicaciones | Nature, RSNA |
| IoU vs anotaciones | Localización de patologías | MICCAI |
| Cohen's Kappa | Acuerdo con radiólogos | Medical journals |

## Solución Técnica (Para Implementar)

### Opción Recomendada: Warpear Máscaras

```python
def warp_mask(
    mask: np.ndarray,              # Máscara original (H, W)
    source_landmarks: np.ndarray,  # 15 landmarks en imagen original
    target_landmarks: np.ndarray,  # 15 landmarks en forma canónica
    triangles: np.ndarray,         # Delaunay triangles
    output_size: int = 224
) -> np.ndarray:
    """Aplica el MISMO piecewise affine warp a la máscara."""
    # Reutilizar lógica de piecewise_affine_warp()
    # Usar interpolación NEAREST para máscaras binarias
    pass
```

**Viabilidad:** ALTA - El código existente en `warp.py` ya soporta imágenes 2D.

**Archivos a modificar:**
- `src_v2/processing/warp.py` - Agregar `warp_mask()`
- `src_v2/visualization/pfs_analysis.py` - Integrar warping de máscaras
- `src_v2/cli.py` - Parámetro opcional en `pfs-analysis`

### Métrica Adicional: Consistency Score

Más poderosa que PFS absoluto:

```python
Consistency_Score = 1 - (Std(PFS_warped) / Std(PFS_original))
```

**Hipótesis:** Si el warping normaliza la geometría, el modelo debería mirar más CONSISTENTEMENTE las mismas regiones (menor varianza en atención).

## Por Qué Se Pospone

1. **La hipótesis principal YA está demostrada:**
   - Cross-evaluation: 11x mejor generalización
   - Robustez JPEG: 30x mejor
   - Robustez Blur: 3x mejor

2. **El PFS es métrica complementaria, no primaria**

3. **Requiere trabajo técnico adicional** (~2-3 horas)

4. **No afecta conclusiones de la tesis**

## Checklist para Retomar

Cuando se decida implementar:

- [ ] Crear función `warp_mask()` en `src_v2/processing/warp.py`
- [ ] Agregar tests unitarios para `warp_mask()`
- [ ] Modificar `run_pfs_analysis()` para usar máscaras warped
- [ ] Agregar parámetro `--warp-masks` al comando CLI
- [ ] Recalcular PFS con máscaras alineadas
- [ ] Calcular Consistency_Score (varianza)
- [ ] Actualizar documentación de sesión 24
- [ ] Agregar sección en tesis si resultados son significativos

## Bugs Corregidos en Sesión 24

Aunque PFS se pospone, se corrigieron bugs importantes:

1. **BUG CRÍTICO:** `max_per_class = num_samples` → `num_samples // len(class_names)`
2. **BUG:** `find_mask_for_image` no buscaba sufijo `_mask.png`
3. **BUG:** Progress bar con total incorrecto
4. **MEJORA:** Advertencia automática para imágenes warped

## Referencias

- Session 24: `docs/sesiones/SESION_24_PFS_ANALYSIS.md`
- Implementación: `src_v2/visualization/pfs_analysis.py`
- CLI: `src_v2/cli.py` (comandos `pfs-analysis`, `generate-lung-masks`)
- Tests: `tests/test_visualization.py`, `tests/test_cli.py`

---

**Última actualización:** 2025-12-09
**Autor:** Claude Code (Sesión 24)
