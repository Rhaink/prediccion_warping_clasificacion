# Checklist de Verificación Manual de la GUI

## Pre-requisitos
- [ ] Python 3.9+ instalado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Modelos en su lugar (checkpoints, outputs)

## Lanzamiento
- [ ] `python scripts/run_demo.py` ejecuta sin errores
- [ ] Verifica mensaje: "✓ Todas las dependencias instaladas"
- [ ] Verifica mensaje: "✓ Todos los modelos encontrados (7/7)"
- [ ] Interfaz abre en http://localhost:7860
- [ ] No hay warnings de PyTorch en consola

## Tab 1: Demostración Completa

### Carga de Imagen
- [ ] Botón "Cargar Imagen" funciona
- [ ] Drag & drop funciona
- [ ] Ejemplos pre-cargados se seleccionan correctamente
  - [ ] COVID example
  - [ ] Normal example
  - [ ] Viral example

### Procesamiento
- [ ] Click en "Procesar Imagen" inicia procesamiento
- [ ] Mensaje "Procesando..." aparece
- [ ] Primera ejecución toma 2-3 segundos (carga modelos)
- [ ] Ejecuciones posteriores toman ~1-2 segundos

### Outputs (4 Etapas)
- [ ] **Imagen Original**: Se muestra en escala de grises
- [ ] **Landmarks Detectados**:
  - [ ] 15 puntos visibles
  - [ ] Colores correctos por grupo (verde, cyan, amarillo, magenta, rojo)
  - [ ] Etiquetas L1-L15 visibles
- [ ] **Imagen Normalizada (Warped)**:
  - [ ] Imagen warped se muestra
  - [ ] No hay artefactos evidentes
  - [ ] Pulmones centrados
- [ ] **GradCAM**:
  - [ ] Heatmap visible sobre imagen warped
  - [ ] Colormap JET (azul→rojo)
  - [ ] Colorbar en lateral derecho
  - [ ] Título muestra clase predicha

### Clasificación
- [ ] Label muestra clase predicha en español
- [ ] Probabilidades de 3 clases visibles
- [ ] Suman ~100% (±1%)

### Métricas
- [ ] Acordeón "Detalles de Landmarks" desplegable
- [ ] Tabla con 15 filas (L1-L15)
- [ ] Columnas: Landmark, Grupo, X, Y, Error Ref
- [ ] Valores numéricos razonables (0-224 para X/Y)

### Export
- [ ] Botón "Exportar a PDF" funciona
- [ ] PDF se descarga automáticamente
- [ ] PDF contiene todas las visualizaciones
- [ ] PDF tiene múltiples páginas

## Tab 2: Vista Rápida

- [ ] Interfaz simplificada visible
- [ ] Botón "Clasificar" funciona
- [ ] Muestra solo clasificación + tiempo
- [ ] Más rápido que demo completo (~0.5-1s)

## Tab 3: Acerca del Sistema

- [ ] Texto informativo visible
- [ ] Métricas validadas mostradas:
  - [ ] Error landmarks: 3.61 px
  - [ ] Accuracy: 98.05%
  - [ ] F1-Score Macro: 97.12%
- [ ] Secciones desplegables funcionan
- [ ] Referencias legibles

## Edge Cases

- [ ] Imagen < 100x100 muestra error descriptivo
- [ ] Archivo .txt rechazado con mensaje en español
- [ ] Imagen corrupta manejada gracefully
- [ ] Puerto 7860 ocupado → usa siguiente puerto disponible

## Performance

- [ ] Primera carga de modelos < 5 segundos (GPU) o < 30s (CPU)
- [ ] Inferencia posterior < 2 segundos
- [ ] No hay memory leaks visibles después de 10 imágenes
- [ ] CPU usage razonable en idle (<5%)

## Modo Deployment

- [ ] Configurar `export COVID_DEMO_MODELS_DIR=/path/to/models`
- [ ] Lanzar con `bash run_demo.sh`
- [ ] Verifica que carga modelos desde ruta simplificada
- [ ] Funcionalidad idéntica a modo development

## Multiplatform

### Linux
- [ ] Ubuntu 20.04+ funciona
- [ ] No requiere permisos root

### macOS
- [ ] macOS 10.15+ funciona
- [ ] Apple Silicon (M1/M2) funciona en CPU

### Windows
- [ ] Windows 10/11 funciona
- [ ] run_demo.bat ejecuta correctamente

## Veredicto Final

- [ ] **APROBADO**: Todos los tests pasaron
- [ ] **APROBADO CON OBSERVACIONES**: Tests críticos pasaron, issues menores
- [ ] **RECHAZADO**: Tests críticos fallaron

**Observaciones**:
_[Agregar notas sobre comportamiento inesperado]_

**Fecha de verificación**: _______
**Verificado por**: _______
**GPU utilizada**: _______
