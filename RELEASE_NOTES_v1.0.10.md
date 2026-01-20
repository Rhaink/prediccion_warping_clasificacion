# Release Notes - COVID-19 Demo v1.0.10

**Fecha de lanzamiento**: 20 de enero de 2026
**Paquete**: `covid19-demo-v1.0.10-portable-windows.zip`
**Tamaño**: 578 MB
**Archivos**: 183 archivos

---

## 🎉 Nueva Funcionalidad

### Visualización de Malla de Delaunay

Esta versión agrega una **nueva visualización educativa** que muestra la malla de triangulación de Delaunay superpuesta sobre la imagen original. Esta visualización permite comprender visualmente cómo se divide la imagen en regiones triangulares antes del proceso de warping geométrico.

#### Características principales:

- **Triangulación dinámica**: Calcula la malla de Delaunay sobre los landmarks predichos de cada imagen específica (típicamente ~18 triángulos)
- **Visualización clara**: Bordes de triángulos en color cyan (#00FFFF) con 60% de transparencia
- **Puntos anatómicos**: Muestra los 15 landmarks con colores por grupo (eje, central, lateral, borde, costal)
- **Etiquetas**: Identifica cada landmark como L1-L15
- **Exportación PDF**: La nueva visualización se incluye automáticamente en las exportaciones PDF (página 2)

#### Ubicación en la interfaz:

```
Row 1:  [1️⃣ Original]        [2️⃣ Landmarks]
Row 2:  [🔷 Delaunay Mesh]   [3️⃣ Warped]        ← NUEVA
Row 3:  [4️⃣ Warped + SAHS]
```

#### Propósito educativo:

Esta visualización ayuda a entender:
- Cómo se divide la imagen en regiones triangulares
- Cómo cada triángulo se transforma durante el warping piecewise affine
- La estructura geométrica utilizada para la normalización anatómica

---

## 📦 Contenido del paquete

### Modelos incluidos (224.4 MB):

- 4 modelos de landmarks ResNet-18 (ensemble):
  - `resnet18_seed123_best.pt` (45.4 MB)
  - `resnet18_seed321_best.pt` (45.4 MB)
  - `resnet18_seed111_best.pt` (45.4 MB)
  - `resnet18_seed666_best.pt` (45.4 MB)
- 1 clasificador ResNet-18:
  - `best_classifier.pt` (42.7 MB)
- Archivos de análisis geométrico:
  - `canonical_shape_gpa.json` (forma canónica consenso)
  - `canonical_delaunay_triangles.json` (18 triángulos de referencia)

### Componentes del sistema:

- **Python embeddable**: 3.12.8 (portable, no requiere instalación)
- **PyTorch**: 2.4.1+cpu (CPU-only para compatibilidad universal)
- **Gradio**: 6.0.0 (interfaz web moderna)
- **OpenCV**: 4.10.0 (procesamiento de imágenes)
- **SciPy**: 1.14.1 (triangulación de Delaunay)
- **85 paquetes de dependencias** incluidos

---

## 🔧 Archivos modificados

### Código fuente:

1. **src_v2/gui/visualizer.py** (~130 líneas nuevas)
   - Nueva función `render_delaunay_mesh()` (líneas 243-355)
   - Importaciones: `Delaunay` (scipy.spatial), `Polygon` (matplotlib.patches)
   - Actualización de `export_to_pdf()` para incluir página de Delaunay

2. **src_v2/gui/inference_pipeline.py** (~10 líneas)
   - Integración de `render_delaunay_mesh()` en el pipeline completo
   - Generación de visualización de malla en `process_image_full()`
   - Actualización de diccionario de resultados

3. **src_v2/gui/app.py** (~15 líneas)
   - Nuevo componente `img_delaunay` en Row 2
   - Actualización de función `on_process()` para manejar 5 visualizaciones
   - Ajuste de outputs en click handler

4. **src_v2/gui/__init__.py**
   - Versión actualizada: `1.0.9` → `1.0.10`

5. **src_v2/gui/config.py**
   - Versión en ABOUT_TEXT actualizada

---

## 📊 Métricas validadas (sin cambios)

Los resultados científicos del sistema permanecen idénticos a v1.0.9:

- **Error de landmarks (ensemble)**: 3.61 ± 2.48 px
- **Mediana de error**: 3.07 px
- **Accuracy de clasificación**: 98.60% ± 0.26% (5-fold CV)
- **F1-Score Macro**: 98.00% ± 0.36%
- **F1-Score Weighted**: 98.60% ± 0.25%

---

## 🚀 Instrucciones de uso

### Instalación:

1. Descomprimir `covid19-demo-v1.0.10-portable-windows.zip`
2. Ejecutar `INSTALL.bat` (solo la primera vez, ~2-3 minutos)
3. Ejecutar `RUN_DEMO.bat` para iniciar la aplicación

### Uso de la nueva visualización:

1. Cargar una radiografía de tórax
2. Hacer clic en "🔍 Procesar Imagen"
3. Observar la **malla de Delaunay** en Row 2, columna izquierda
4. (Opcional) Exportar a PDF para incluir todas las visualizaciones

---

## 🔍 Detalles técnicos

### Triangulación de Delaunay:

- **Algoritmo**: `scipy.spatial.Delaunay`
- **Entrada**: 15 landmarks predichos en coordenadas de píxeles (224×224)
- **Salida**: ~18 triángulos (varía según disposición de landmarks)
- **Tiempo de cómputo**: ~1-2 ms (negligible)
- **Propiedades garantizadas**:
  - Maximiza el ángulo mínimo de todos los triángulos
  - Sin superposiciones
  - Cobertura completa del convex hull

### Parámetros de visualización:

```python
mesh_color = '#00FFFF'       # Cyan (color del grupo central)
mesh_alpha = 0.6             # 60% transparencia
mesh_linewidth = 1.5         # Grosor medio
fill_triangles = False       # Solo bordes (sin relleno)
show_labels = True           # Mostrar L1-L15
show_landmark_points = True  # Mostrar puntos de colores
```

---

## 📝 Notas de compatibilidad

- **Windows**: 10/11 (64-bit)
- **Memoria RAM**: Mínimo 4 GB (recomendado 8 GB)
- **Espacio en disco**: ~1.5 GB después de instalación
- **No requiere**: GPU, Python instalado, permisos de administrador

---

## 🐛 Correcciones de errores

Ninguna. Esta es una release de nueva funcionalidad sin correcciones de bugs.

---

## 🔮 Próximas versiones

Mejoras planificadas para futuras versiones:

- Visualización de vectores de desplazamiento (antes/después del warping)
- Comparación lado a lado: Original vs Normalizada
- Métricas de calidad del warping (fill rate, distorsión angular)
- Exportación de landmarks en formato JSON/CSV

---

## 📧 Contacto y soporte

Para reportar problemas o sugerencias:
- GitHub Issues: [Tu repositorio]
- Email: [Tu email de contacto]

---

## 📄 Licencia

[Especificar licencia del proyecto]

---

**Checksum SHA256 del paquete:**

```
cecb7f5466e9a386c3b141357faf4d81cfc4cf3af393dfbdd006b94c504c54c9
```

Para verificar la integridad del archivo descargado:
```bash
sha256sum covid19-demo-v1.0.10-portable-windows.zip
```

---

**Archivo generado automáticamente por el build system**
**Build date**: 2026-01-20 05:22:32 UTC
