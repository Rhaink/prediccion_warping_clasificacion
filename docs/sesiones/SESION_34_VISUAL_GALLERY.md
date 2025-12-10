# Sesión 34: Galería Visual para Defensa de Tesis

**Fecha:** 2025-12-10
**Objetivo:** Crear evidencia visual comparativa GradCAM para demostrar la hipótesis de tesis

## Resumen Ejecutivo

Se generaron **16 figuras de alta calidad (300 DPI)** que demuestran visualmente cómo el warping geométrico mejora la atención de los clasificadores hacia las regiones pulmonares en lugar de artefactos hospitalarios.

### Hipótesis Confirmada Visualmente

> "Las imágenes warped mejoran la generalización 11.3× y la robustez 30× porque eliminan variabilidad geométrica no-anatómica que causa overfitting."

## Figuras Generadas

### 1. Comparaciones Lado-a-Lado (6 figuras)

Ubicación: `outputs/thesis_figures/combined_figures/comparison_*.png`

| Archivo | Clase | Descripción |
|---------|-------|-------------|
| `comparison_COVID_COVID-1000.png` | COVID | Caso típico COVID |
| `comparison_COVID_COVID-1005.png` | COVID | Caso COVID alternativo |
| `comparison_Normal_Normal-10012.png` | Normal | Caso normal típico |
| `comparison_Normal_Normal-10028.png` | Normal | Caso normal alternativo |
| `comparison_Viral_Pneumonia_Viral Pneumonia-100.png` | VP | Neumonía viral 1 |
| `comparison_Viral_Pneumonia_Viral Pneumonia-1011.png` | VP | Neumonía viral 2 |

**Estructura de cada figura:**
```
┌─────────────────────────────────────────┐
│     Imagen Original  │  Imagen Warped   │
├─────────────────────────────────────────┤
│  GradCAM Modelo Orig │ GradCAM Modelo W │
│  (atiende bordes)    │ (atiende pulmón) │
└─────────────────────────────────────────┘
```

### 2. Análisis Cross-Domain (6 figuras)

Ubicación: `outputs/thesis_figures/combined_figures/crossdomain_*.png`

Muestran las 4 combinaciones modelo×imagen:
- Original→Original (in-domain baseline)
- Original→Warped (cross-domain, revela overfitting)
- Warped→Original (cross-domain, muestra robustez)
- Warped→Warped (in-domain)

### 3. Matrices de Atención (3 figuras)

Ubicación: `outputs/thesis_figures/combined_figures/matrix_*.png`

Una figura representativa por clase mostrando la matriz completa 2×2.

### 4. Resumen de Métricas (1 figura)

Ubicación: `outputs/thesis_figures/combined_figures/summary_metrics.png`

Gráfico de barras mostrando:
- Gap de generalización: 25.36% → 2.24% (11.3× mejor)
- Degradación JPEG Q50: 16.14% → 0.53% (30.5× mejor)

## Observaciones Visuales Clave

### Patrón de Atención - Modelo Original

El modelo entrenado en imágenes originales presenta:

1. **Atención dispersa**: El heatmap GradCAM se distribuye en múltiples regiones
2. **Foco en bordes**: Alta activación en esquinas y perímetro de la imagen
3. **Sensibilidad a marcas**: Atiende a etiquetas de texto/números de hospital
4. **Regiones no-anatómicas**: Activación fuera del área pulmonar

### Patrón de Atención - Modelo Warped

El modelo entrenado en imágenes warped presenta:

1. **Atención concentrada**: Heatmap focalizado en región central
2. **Foco pulmonar**: Alta activación sobre los campos pulmonares
3. **Ignora artefactos**: Baja activación en bordes y marcas
4. **Consistencia anatómica**: Patrón similar entre imágenes de la misma clase

## Métricas Verificadas (Datos Reales)

### Generalización
```
Gap Original: 25.36%  (modelo original falla en datos warped)
Gap Warped:    2.24%  (modelo warped generaliza bien)
Mejora:       11.32×
```

### Robustez JPEG Q50
```
Degradación Original: 16.14%
Degradación Warped:    0.53%
Mejora:              30.62×
```

### Cross-Evaluation
```
                    → Original    → Warped
Modelo Original      98.81%       73.45%
Modelo Warped        95.78%       98.02%
```

## Archivos Generados

```
outputs/thesis_figures/
├── source_pairs/
│   ├── original/           # 6 imágenes originales
│   └── warped/             # 6 imágenes warped (correspondientes)
├── gradcam_comparison/
│   ├── original_on_original/   # GradCAM M.orig → I.orig
│   ├── original_on_warped/     # GradCAM M.orig → I.warp
│   ├── warped_on_original/     # GradCAM M.warp → I.orig
│   └── warped_on_warped/       # GradCAM M.warp → I.warp
└── combined_figures/           # 16 figuras finales
    ├── comparison_*.png        # 6 lado-a-lado
    ├── crossdomain_*.png       # 6 cross-domain
    ├── matrix_*.png            # 3 matrices
    └── summary_metrics.png     # 1 resumen
```

## Script Reproducible

```bash
# Regenerar todas las figuras
.venv/bin/python scripts/create_thesis_figures.py
```

El script `scripts/create_thesis_figures.py` incluye:
- `create_side_by_side_comparison()`: Figuras 2×2 comparativas
- `create_cross_domain_figure()`: Análisis 4-way cross-domain
- `create_4x4_matrix_figure()`: Matrices de atención
- `create_summary_figure()`: Gráfico de métricas

## Uso en la Tesis

### Figuras Recomendadas para el Documento

1. **Figura Principal (Capítulo de Resultados)**:
   `comparison_COVID_COVID-1000.png`
   Muestra claramente la diferencia en patrones de atención.

2. **Figura Cross-Domain (Discusión)**:
   `crossdomain_COVID_COVID-1000.png`
   Demuestra la generalización del modelo warped.

3. **Figura de Resumen (Abstract Visual)**:
   `summary_metrics.png`
   Cuantifica las mejoras de manera visual.

### Caption Sugerido

> **Figura X**: Comparación de patrones de atención (Grad-CAM) entre modelos entrenados en imágenes originales y warped. Panel superior: imagen de rayos X original (izquierda) y normalizada geométricamente (derecha). Panel inferior: mapas de calor Grad-CAM mostrando las regiones que influyen en la predicción. El modelo original atiende a bordes y artefactos (marcas hospitalarias), mientras que el modelo warped focaliza su atención en las regiones pulmonares anatómicamente relevantes.

## Conclusiones Visuales

1. **El warping elimina "atajos" de aprendizaje**: Los modelos originales aprenden características espurias (bordes, marcas) que no generalizan.

2. **La atención se redirige a anatomía**: Al normalizar geométricamente, el modelo se ve forzado a aprender características de textura pulmonar.

3. **Evidencia visual de la hipótesis**: Las figuras GradCAM proporcionan interpretabilidad y confianza en las métricas numéricas.

## Próximos Pasos

- [ ] Sesión 35: Validación estadística (p-values, IC 95%)
- [ ] Sesión 36: Tests de integración críticos
- [ ] Sesión 37: Documentación final

## Checkpoints Utilizados

| Modelo | Path | Arquitectura |
|--------|------|--------------|
| Original | `outputs/classifier_comparison/resnet18_original/best_model.pt` | ResNet18 |
| Warped | `outputs/classifier_comparison/resnet18_warped/best_model.pt` | ResNet18 |
| Landmarks | `checkpoints_v2_correct/final_model.pt` | CoordAttention+DeepHead |
