# Instrucciones para Capturar las Imágenes del Manual

Este documento lista todas las imágenes sugeridas para complementar el Manual de Usuario. Cada imagen debe ser capturada desde la interfaz gráfica v15 en funcionamiento.

## Preparación Inicial

1. **Iniciar el sistema:**
   ```bash
   # Desde la raíz del proyecto
   cd build/releases/covid19-demo-v15-portable-windows
   # Doble clic en RUN_DEMO.bat
   ```

2. **Configurar el navegador:**
   - Usar modo de pantalla completa o ventana maximizada
   - Resolución recomendada: 1920×1080 o superior
   - Zoom del navegador: 100%

3. **Herramienta de captura:**
   - Windows: Usar Recortes (Win+Shift+S)
   - Guardar capturas en: `docs/manual/imagenes/`

## Lista de Imágenes Requeridas

### Capítulo 1: Introducción

#### `portada_sistema.png`
- **Ubicación en LaTeX:** Línea 76, después del título
- **Contenido:** Captura de la interfaz principal con las 3 pestañas visibles
- **Sugerencia de captura:**
  - Abrir la pestaña "Demostración Completa"
  - Capturar toda la ventana del navegador
  - Incluir: título del sistema, las pestañas y el área de carga
- **Tamaño sugerido:** Ancho completo del navegador

#### `interfaz_principal.png` (Figura 1)
- **Ubicación en LaTeX:** Línea 106, final Capítulo 1
- **Contenido:** Vista general de la interfaz con áreas identificables
- **Sugerencia de captura:**
  - Misma captura que portada_sistema.png puede servir
  - O captura enfocada en el área de trabajo
- **Elementos a incluir:**
  - Las 3 pestañas en la parte superior
  - Área de carga de imágenes a la izquierda
  - Mensaje "Esperando imagen..." a la derecha

---

### Capítulo 2: Instalación

#### `extraer_zip.png` (Figura 2)
- **Ubicación en LaTeX:** Línea 142
- **Contenido:** Menú contextual de Windows con opción "Extraer todo"
- **Sugerencia de captura:**
  - En explorador de Windows, hacer clic derecho sobre el archivo .zip
  - Capturar el menú emergente
  - Resaltar con flecha roja la opción "Extraer todo..."
- **Herramienta:** Puede usar Paint o similar para agregar la flecha

#### `instalacion.png` (Figura 3)
- **Ubicación en LaTeX:** Línea 175
- **Contenido:** Ventana de comandos durante la instalación
- **Sugerencia de captura:**
  - Ejecutar INSTALL.bat
  - Capturar la ventana negra mostrando progreso
  - Debe verse texto desplazándose (instalando dependencias)
- **Momento:** Capturar a mitad de la instalación para que se vea actividad

#### `sistema_cargado.png` (Figura 4)
- **Ubicación en LaTeX:** Línea 196
- **Contenido:** Navegador con el sistema completamente cargado
- **Sugerencia de captura:**
  - Interfaz completamente cargada en http://localhost:7860
  - Incluir la barra de direcciones del navegador mostrando la URL
  - Sistema listo para usar con mensaje inicial visible
- **Importante:** Debe verse profesional y limpio

---

### Capítulo 3: Demostración Completa

#### `area_carga.png` (Figura 5)
- **Ubicación en LaTeX:** Línea 234
- **Contenido:** Área de carga de imágenes con flecha indicadora
- **Sugerencia de captura:**
  - Capturar solo el panel izquierdo
  - El área donde dice "Cargar Radiografía de Tórax"
  - Agregar una flecha apuntando al botón/área de carga
- **Edición:** Usar herramienta de edición para agregar flecha

#### `ejemplos.png` (Figura 6)
- **Ubicación en LaTeX:** Línea 248
- **Contenido:** Los 3 ejemplos precargados con miniaturas
- **Sugerencia de captura:**
  - Sección "Ejemplos Precargados" debajo del área de carga
  - Deben verse las 3 miniaturas: COVID, Normal, Viral
- **Nota:** Si no hay miniaturas visibles, capturar los labels

#### `original.png` (Figura 7)
- **Ubicación en LaTeX:** Línea 269
- **Contenido:** Ejemplo de radiografía original sin procesar
- **Sugerencia de captura:**
  - Cargar un ejemplo (COVID preferiblemente)
  - Capturar solo el panel "1️⃣ Imagen Original"
  - Sin anotaciones ni overlays
- **Importante:** Imagen clara y visible

#### `landmarks.png` (Figura 8)
- **Ubicación en LaTeX:** Línea 284
- **Contenido:** Radiografía con 15 puntos de colores y etiquetas L1-L15
- **Sugerencia de captura:**
  - Panel "2️⃣ Puntos de Referencia Detectados"
  - Deben verse claramente:
    * Los 15 puntos de colores
    * Las etiquetas L1-L15
    * Las líneas conectoras (blancas y celestes)
- **Esta es una de las imágenes MÁS IMPORTANTES del manual**

#### `delaunay.png` (Figura 9)
- **Ubicación en LaTeX:** Línea 295
- **Contenido:** Malla triangular celeste sobre la radiografía
- **Sugerencia de captura:**
  - Panel "🔷 Malla de Delaunay"
  - Deben verse los ~18 triángulos formados por los landmarks
  - Líneas celestes sobre la imagen
- **Importancia:** Muestra cómo se divide la imagen para warping

#### `warped.png` (Figura 10)
- **Ubicación en LaTeX:** Línea 309
- **Contenido:** Imagen normalizada con bordes negros
- **Sugerencia de captura:**
  - Panel "3️⃣ Imagen Normalizada (Warped)"
  - Se verán pulmones en forma canónica
  - Bordes negros alrededor (esto es normal)
- **Nota:** Explicar que los bordes negros son esperados

#### `sahs.png` (Figura 11)
- **Ubicación en LaTeX:** Línea 321
- **Contenido:** Imagen normalizada con contraste mejorado (SAHS)
- **Sugerencia de captura:**
  - Panel "4️⃣ Imagen Normalizada con SAHS"
  - Mayor contraste que la imagen warped normal
  - Estructuras pulmonares más visibles
- **Comparar:** Debería verse diferencia clara con figura 10

#### `prediccion.png` (Figura 12)
- **Ubicación en LaTeX:** Línea 341
- **Contenido:** Recuadro de predicción destacado
- **Sugerencia de captura:**
  - Solo el recuadro con fondo oscuro
  - Debe mostrar: "⭐ COVID-19" en rojo (o la clase correspondiente)
  - Texto grande y centrado
- **Importante:** El color debe ser visible (rojo/verde/amarillo)

#### `probabilidades.png` (Figura 13)
- **Ubicación en LaTeX:** Línea 355
- **Contenido:** Las tres barras de probabilidad coloreadas
- **Sugerencia de captura:**
  - Sección "Probabilidades por Clase"
  - Tres barras horizontales con colores (rojo/verde/amarillo)
  - Porcentajes visibles en cada barra
  - La clase ganadora debe tener emoji ⭐
- **Ejemplo ideal:** COVID-19 85%, Normal 10%, Neumonía Viral 5%

#### `metricas.png` (Figura 14)
- **Ubicación en LaTeX:** Línea 391
- **Contenido:** Acordeón de métricas expandido
- **Sugerencia de captura:**
  - Clic en "📈 Métricas Detalladas" para expandir
  - Capturar la tabla completa con coordenadas L1-L15
  - Incluir el tiempo de inferencia debajo
- **Nota:** Tabla debe ser legible

---

### Capítulo 4: Vista Rápida

#### `vista_rapida.png` (Figura 15)
- **Ubicación en LaTeX:** Línea 425
- **Contenido:** Interfaz de la pestaña Vista Rápida
- **Sugerencia de captura:**
  - Cambiar a pestaña "⚡ Vista Rápida"
  - Capturar la interfaz completa
  - Debe verse más simple que Demostración Completa
  - Solo área de carga + botón Clasificar + resultados
- **Comparar:** Mostrar diferencia visual con Tab 1

---

### Anexo D: Interfaz Anotada

#### `interfaz_anotada.png` (Figura 16)
- **Ubicación en LaTeX:** Línea 651, Anexo D
- **Contenido:** Captura completa con anotaciones y flechas
- **Sugerencia de captura:**
  - Captura de pantalla completa de la interfaz procesada
  - Usar herramienta de edición (Paint, GIMP, Photoshop, etc.)
  - Agregar FLECHAS ROJAS apuntando a:
    1. Área de carga de imágenes (panel izquierdo)
    2. Botón "Procesar Imagen"
    3. Ejemplos precargados
    4. Las 4 visualizaciones (numerarlas 1️⃣ 2️⃣ 🔷 3️⃣ 4️⃣)
    5. Recuadro de predicción destacado
    6. Barras de probabilidad
    7. Botón "Exportar a PDF"
    8. Acordeón "Métricas Detalladas"
  - Agregar ETIQUETAS DE TEXTO junto a cada flecha
- **Esta es la imagen más compleja y requiere edición**

---

## Proceso de Captura Recomendado

### Paso 1: Capturar las Bases (sin procesar)
1. Iniciar sistema
2. Capturar interfaz inicial (portada, interfaz_principal)
3. Capturar área de carga y ejemplos

### Paso 2: Procesar un Ejemplo COVID
1. Cargar ejemplo de COVID-19
2. Clic en "Procesar Imagen"
3. Esperar a que termine (1-2 segundos)
4. Capturar cada panel individualmente:
   - Original
   - Landmarks
   - Delaunay
   - Warped
   - SAHS
5. Capturar predicción y probabilidades
6. Expandir métricas y capturar

### Paso 3: Vista Rápida
1. Cambiar a pestaña "Vista Rápida"
2. Cargar mismo ejemplo
3. Clic en "Clasificar"
4. Capturar interfaz

### Paso 4: Instalación (requiere reinicio)
1. Cerrar sistema
2. Capturar proceso de instalación (INSTALL.bat)
3. Capturar inicio (RUN_DEMO.bat)
4. Capturar navegador cargando

### Paso 5: Ediciones
1. Agregar flechas donde sea necesario
2. Crear la imagen anotada del Anexo D
3. Redimensionar si es necesario (mantener proporciones)

---

## Especificaciones Técnicas

### Formato de Guardado
- **Formato:** PNG (preferido) o JPG de alta calidad
- **Resolución:** Mínimo 1200px de ancho
- **Nombres:** Usar exactamente los nombres indicados arriba
- **Ubicación:** `docs/manual/imagenes/`

### Calidad
- Capturas nítidas, no borrosas
- Colores fieles (importante para rojo/verde/amarillo)
- Texto legible al 100% de zoom
- Sin elementos personales (cerrar otras pestañas del navegador)

### Post-Procesamiento
- Si las capturas son muy grandes, redimensionar manteniendo aspect ratio
- Asegurar que el texto sea legible después de redimensionar
- Para flechas y anotaciones, usar color rojo (#FF0000) con borde blanco

---

## Estructura de Carpetas

```
docs/manual/
├── manual_usuario.tex          # Archivo LaTeX principal
├── INSTRUCCIONES_IMAGENES.md   # Este archivo
├── imagenes/                   # Crear esta carpeta
│   ├── portada_sistema.png
│   ├── interfaz_principal.png
│   ├── extraer_zip.png
│   ├── instalacion.png
│   ├── sistema_cargado.png
│   ├── area_carga.png
│   ├── ejemplos.png
│   ├── original.png
│   ├── landmarks.png
│   ├── delaunay.png
│   ├── warped.png
│   ├── sahs.png
│   ├── prediccion.png
│   ├── probabilidades.png
│   ├── metricas.png
│   ├── vista_rapida.png
│   └── interfaz_anotada.png
└── manual_usuario.pdf          # Generado después de compilar
```

---

## Compilación del PDF

Una vez capturadas todas las imágenes:

```bash
cd docs/manual
pdflatex manual_usuario.tex
pdflatex manual_usuario.tex  # Segunda pasada para índice y referencias
```

O si hay errores de imágenes faltantes, el PDF se generará mostrando espacios vacíos donde irían las imágenes.

---

## Notas Finales

1. **Prioridad:** Las imágenes más críticas son:
   - `landmarks.png` (muestra el corazón del sistema)
   - `probabilidades.png` (resultados principales)
   - `interfaz_anotada.png` (guía visual completa)

2. **Alternativa temporal:** Si no se pueden capturar todas las imágenes de inmediato, el manual puede compilarse con las imágenes que haya. LaTeX mostrará un recuadro vacío donde falten imágenes.

3. **Consistencia:** Usar el mismo ejemplo (preferiblemente COVID) en todas las capturas de procesamiento para mantener coherencia visual.

4. **Revisión:** Después de generar el PDF, revisar que todas las imágenes se vean bien y sean legibles.
