# Índice de Documentación - Interfaz Gráfica

## 📚 Guía Rápida de Documentación

Esta guía te ayudará a encontrar la documentación apropiada según tu necesidad.

## 🎯 Por Tipo de Usuario

### Para Usuarios/Tesistas
**Quieres usar la interfaz gráfica:**
1. 📖 Leer: [`src_v2/gui/README.md`](../src_v2/gui/README.md)
   - Instrucciones de instalación
   - Cómo ejecutar la interfaz
   - Guía de uso de los 3 tabs
   - Troubleshooting común

2. 🚀 Ejecutar: `python scripts/run_demo.py`

### Para Desarrolladores/Revisores
**Quieres entender la implementación:**
1. 📋 Leer: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md)
   - Resumen ejecutivo completo
   - Arquitectura técnica
   - Componentes principales
   - Métricas de rendimiento
   - Testing y validación

2. 📝 Revisar: [`src_v2/gui/CHANGELOG.md`](../src_v2/gui/CHANGELOG.md)
   - Historial de cambios
   - Correcciones realizadas
   - Versiones

### Para Investigadores/Académicos
**Quieres reproducir o extender:**
1. 📊 Revisar: Métricas validadas en [`GROUND_TRUTH.json`](../GROUND_TRUTH.json)
2. 🔬 Consultar: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md) sección "Pipeline de Inferencia"
3. 💻 Código: Todo en `src_v2/gui/` con docstrings completos

## 📂 Estructura de Documentación

```
📁 docs/
├── GUI_INDEX.md                    # Este archivo - Índice general
├── GUI_IMPLEMENTATION.md           # Resumen ejecutivo técnico (8,500+ palabras)
└── [otros docs del proyecto]

📁 src_v2/gui/
├── README.md                       # Manual de usuario completo
├── CHANGELOG.md                    # Historial de versiones y cambios
├── app.py                          # Código con docstrings
├── config.py                       # Configuración documentada
├── gradcam_utils.py               # Implementación con docstrings
├── inference_pipeline.py          # Pipeline documentado
├── model_manager.py               # Gestión de modelos documentada
└── visualizer.py                  # Funciones de renderizado

📁 scripts/
├── run_demo.py                    # Launcher con --help
└── verify_gui_setup.py           # Script de verificación

📁 README.md (raíz)                 # Actualizado con sección GUI
```

## 🔍 Por Tema

### Instalación y Setup
- **Requisitos**: [`src_v2/gui/README.md`](../src_v2/gui/README.md#requisitos)
- **Instalación**: [`src_v2/gui/README.md`](../src_v2/gui/README.md#instalación)
- **Verificación**: Ejecutar `python scripts/verify_gui_setup.py`

### Uso de la Interfaz
- **Guía de uso**: [`src_v2/gui/README.md`](../src_v2/gui/README.md#uso-de-la-interfaz)
- **Tab 1 - Demo Completa**: [`src_v2/gui/README.md`](../src_v2/gui/README.md#tab-1-demostración-completa)
- **Tab 2 - Vista Rápida**: [`src_v2/gui/README.md`](../src_v2/gui/README.md#tab-2-vista-rápida)
- **Tab 3 - Información**: Ver directamente en la interfaz

### Arquitectura Técnica
- **Resumen general**: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#arquitectura-técnica)
- **Componentes**: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#componentes-principales)
- **Pipeline**: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#pipeline-de-inferencia-completo)
- **Patrón Singleton**: [`src_v2/gui/model_manager.py`](../src_v2/gui/model_manager.py)

### Visualizaciones
- **Landmarks**: [`src_v2/gui/visualizer.py`](../src_v2/gui/visualizer.py) - `render_landmarks_overlay()`
- **GradCAM**: [`src_v2/gui/gradcam_utils.py`](../src_v2/gui/gradcam_utils.py)
- **Exportación PDF**: [`src_v2/gui/visualizer.py`](../src_v2/gui/visualizer.py) - `export_to_pdf()`
- **Colores**: [`src_v2/gui/config.py`](../src_v2/gui/config.py) - `LANDMARK_COLORS`

### Métricas y Validación
- **Métricas validadas**: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#métricas-validadas)
- **Fuente de verdad**: [`GROUND_TRUTH.json`](../GROUND_TRUTH.json)
- **Rendimiento**: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#rendimiento)
- **Testing**: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#testing-y-validación)

### Troubleshooting
- **Errores comunes**: [`src_v2/gui/README.md`](../src_v2/gui/README.md#troubleshooting)
- **Correcciones aplicadas**: [`src_v2/gui/CHANGELOG.md`](../src_v2/gui/CHANGELOG.md#correcciones)
- **Verificación**: `python scripts/verify_gui_setup.py`

### Desarrollo y Extensión
- **Arquitectura**: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#arquitectura-técnica)
- **Mejoras futuras**: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#mejoras-futuras-sugeridas)
- **Código fuente**: `src_v2/gui/` con docstrings Google-style

## 📊 Documentos por Extensión

### Documentación Principal (4 archivos .md)
1. **README principal** [`README.md`](../README.md)
   - Actualizado con sección de GUI
   - Comando de lanzamiento
   - Link a documentación completa

2. **Manual de Usuario** [`src_v2/gui/README.md`](../src_v2/gui/README.md)
   - ~500 líneas
   - Guía completa de uso
   - Troubleshooting
   - Ejemplos

3. **Resumen Ejecutivo** [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md)
   - ~600 líneas
   - Documentación técnica completa
   - Arquitectura y diseño
   - Métricas y rendimiento

4. **Historial de Cambios** [`src_v2/gui/CHANGELOG.md`](../src_v2/gui/CHANGELOG.md)
   - ~200 líneas
   - Versiones
   - Correcciones
   - Features

### Código Fuente (8 archivos .py)
Todos con docstrings completos:
- `__init__.py` (módulo)
- `app.py` (388 líneas)
- `config.py` (210 líneas)
- `gradcam_utils.py` (261 líneas)
- `inference_pipeline.py` (274 líneas)
- `model_manager.py` (440 líneas)
- `visualizer.py` (482 líneas)

### Scripts de Utilidad (2 archivos .py)
- `scripts/run_demo.py` (218 líneas)
- `scripts/verify_gui_setup.py` (338 líneas)

## 🎓 Para Defensa de Tesis

### Documentos a Revisar Antes
1. [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md) - Sección "Métricas Validadas"
2. [`src_v2/gui/README.md`](../src_v2/gui/README.md) - Sección "Notas para Defensa de Tesis"
3. [`GROUND_TRUTH.json`](../GROUND_TRUTH.json) - Valores exactos

### Preparación
```bash
# 1. Verificar todo está funcionando
python scripts/verify_gui_setup.py

# 2. Probar con ejemplos
python scripts/run_demo.py

# 3. Tener backup de screenshots
# Ver: src_v2/gui/README.md#backup-plan
```

### Durante la Presentación
- Usar **Tab 1: Demostración Completa**
- Ejemplos precargados para velocidad
- Expandir métricas detalladas
- Mostrar GradCAM para explicabilidad
- Referir a valores validados

## 🔧 Para Desarrollo

### Agregar Nueva Funcionalidad
1. Leer arquitectura: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#arquitectura-técnica)
2. Ver patrón existente en código fuente
3. Seguir convenciones:
   - Docstrings Google-style
   - Type hints
   - Manejo de errores en español
4. Actualizar CHANGELOG

### Modificar Configuración
- Editar: [`src_v2/gui/config.py`](../src_v2/gui/config.py)
- Constantes centralizadas
- Validar con: `python scripts/verify_gui_setup.py`

### Testing
```bash
# Verificación completa
python scripts/verify_gui_setup.py

# Test de módulo específico
python -c "from src_v2.gui import model_manager; print('OK')"

# Test de pipeline completo
python -c "from src_v2.gui.inference_pipeline import process_image_full; print('OK')"
```

## 📈 Estadísticas de Documentación

### Archivos Documentados
- ✅ 4 archivos Markdown (README, IMPLEMENTATION, CHANGELOG, INDEX)
- ✅ 8 módulos Python con docstrings
- ✅ 2 scripts de utilidad
- ✅ README principal actualizado

### Líneas de Documentación
- Markdown: ~1,500 líneas
- Docstrings en código: ~800 líneas
- Comentarios inline: ~200 líneas
- **Total**: ~2,500 líneas de documentación

### Cobertura
- ✅ 100% funciones públicas con docstrings
- ✅ 100% módulos con documentation strings
- ✅ 100% parámetros con type hints
- ✅ Todos los componentes explicados

## 🆘 Ayuda Rápida

### No arranca la interfaz
```bash
python scripts/verify_gui_setup.py
# Seguir recomendaciones del script
```

### Error durante inferencia
- Ver: [`src_v2/gui/CHANGELOG.md`](../src_v2/gui/CHANGELOG.md#correcciones)
- Verificar modelos: `ls -lh checkpoints/*/final_model.pt`

### Quiero entender el código
1. Empezar por: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md)
2. Luego revisar: `src_v2/gui/model_manager.py` (bien documentado)
3. Seguir el flujo en: `src_v2/gui/inference_pipeline.py`

### Necesito modificar algo
1. Identificar módulo en: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md#componentes-principales)
2. Leer código fuente con docstrings
3. Hacer cambios
4. Probar con: `python scripts/verify_gui_setup.py`
5. Actualizar: [`src_v2/gui/CHANGELOG.md`](../src_v2/gui/CHANGELOG.md)

## 📞 Contacto y Contribución

Para reportar problemas o sugerir mejoras:
1. Revisar primero: Troubleshooting en README
2. Verificar setup: `python scripts/verify_gui_setup.py`
3. Consultar: [`docs/GUI_IMPLEMENTATION.md`](GUI_IMPLEMENTATION.md)

## 🎯 Checklist de Documentación

- ✅ Manual de usuario completo
- ✅ Guía de instalación
- ✅ Instrucciones de uso
- ✅ Troubleshooting
- ✅ Arquitectura técnica documentada
- ✅ Todos los módulos con docstrings
- ✅ Historial de cambios
- ✅ Scripts de verificación
- ✅ Ejemplos incluidos
- ✅ README principal actualizado
- ✅ Índice de documentación (este archivo)

---

**Última actualización**: 18 de enero de 2026
**Versión**: 1.0.0
**Estado**: Documentación completa
