# Manual de Usuario - Sistema de Detección COVID-19 v15

Este directorio contiene el manual de usuario completo para la Interfaz Gráfica versión 15 del sistema de detección de COVID-19.

## 📄 Archivos Generados

- **`manual_usuario.tex`** - Código fuente LaTeX del manual (archivo editable)
- **`manual_usuario.pdf`** - Manual compilado en PDF (23 páginas, 274 KB)
- **`INSTRUCCIONES_IMAGENES.md`** - Guía detallada para capturar las imágenes sugeridas
- **`README.md`** - Este archivo

## 📋 Contenido del Manual

El manual está organizado en 9 capítulos principales:

### 1. Introducción (págs. 3-4)
- ¿Qué es el sistema?
- ¿Para quién es este manual?
- Requisitos previos
- Avisos importantes

### 2. Instalación y Configuración (págs. 5-7)
- Descarga del sistema
- Proceso de instalación paso a paso
- Inicio del sistema
- Verificación de funcionamiento

### 3. Uso del Sistema: Demostración Completa (págs. 8-14)
- Descripción de la interfaz
- Cómo cargar una radiografía
- Interpretación de las 4 visualizaciones principales:
  - Imagen original
  - Puntos de referencia detectados (15 landmarks)
  - Malla de Delaunay
  - Imagen normalizada y con SAHS
- Resultados de clasificación
- Métricas detalladas
- Exportación a PDF

### 4. Uso del Sistema: Vista Rápida (págs. 15)
- Modo de clasificación rápida
- Cuándo usar este modo

### 5. Solución de Problemas (págs. 16-17)
- Sistema no inicia
- Página no carga
- Procesamiento lento
- Error al cargar imágenes
- Resultados inesperados
- Mensajes de advertencia

### 6. Preguntas Frecuentes (págs. 18-19)
- Uso diagnóstico
- Precisión del sistema
- Procesamiento por lotes
- Tipos de radiografías soportadas
- Privacidad de datos
- Funcionamiento offline

### 7. Glosario de Términos (pág. 20)
- Definiciones de términos médicos y técnicos
- Conceptos clave del sistema

### 8. Información de Contacto (pág. 21)
- Soporte técnico
- Uso académico
- Reporte de errores

### 9. Anexos (págs. 22-23)
- Lista de verificación pre-análisis
- Interpretación de colores
- Especificaciones técnicas
- Captura de pantalla anotada

## 🎯 Público Objetivo

Este manual está diseñado específicamente para **usuarios sin conocimientos técnicos**:
- Personal médico sin experiencia en informática
- Estudiantes de ciencias de la salud
- Investigadores de áreas no técnicas
- Cualquier persona interesada en usar el sistema

## ✨ Características del Manual

- **Lenguaje sencillo**: Sin jerga técnica innecesaria
- **Instrucciones paso a paso**: Cada proceso explicado en detalle
- **Cajas destacadas**: Notas, advertencias y consejos visuales
- **Glosario completo**: Todos los términos técnicos definidos
- **Solución de problemas**: Sección dedicada a errores comunes
- **Diseño profesional**: Formato limpio y fácil de navegar

## 📸 Imágenes Sugeridas

El manual incluye **16 espacios para imágenes** que complementan las instrucciones. Actualmente, estos espacios muestran recuadros con descripciones de lo que debe capturarse.

### Para Capturar las Imágenes:

1. **Lee el archivo**: `INSTRUCCIONES_IMAGENES.md`
2. **Inicia el sistema**: Ejecuta la interfaz gráfica v15
3. **Sigue las instrucciones**: Captura cada imagen según lo especificado
4. **Guarda en**: `docs/manual/imagenes/` con los nombres exactos
5. **Recompila el PDF**: Ejecuta `pdflatex manual_usuario.tex` (3 veces)

### Imágenes Más Importantes:

1. **`landmarks.png`** - Muestra los 15 puntos de referencia detectados (CRÍTICA)
2. **`probabilidades.png`** - Barras de clasificación coloreadas (CRÍTICA)
3. **`interfaz_anotada.png`** - Vista completa con anotaciones (CRÍTICA)
4. **`portada_sistema.png`** - Imagen de la portada del manual

## 🔨 Compilación del PDF

### Requisitos:
- LaTeX completo (texlive-full en Ubuntu/Debian)
- Paquetes necesarios: babel, inputenc, fancyhdr, hyperref, tcolorbox, etc.

### Comandos:

```bash
cd docs/manual

# Primera compilación (genera estructura)
pdflatex -interaction=nonstopmode manual_usuario.tex

# Segunda compilación (actualiza referencias)
pdflatex -interaction=nonstopmode manual_usuario.tex

# Tercera compilación (finaliza índice y referencias cruzadas)
pdflatex -interaction=nonstopmode manual_usuario.tex
```

### Archivos Generados Durante Compilación:

- `manual_usuario.aux` - Referencias auxiliares
- `manual_usuario.log` - Log de compilación
- `manual_usuario.out` - Metadatos PDF
- `manual_usuario.toc` - Tabla de contenidos
- `manual_usuario.pdf` - **Documento final**

## 📝 Edición del Manual

### Para Modificar el Contenido:

1. Abre `manual_usuario.tex` en tu editor preferido
2. Busca la sección que deseas modificar
3. Edita el texto (respeta la sintaxis LaTeX)
4. Recompila con pdflatex (3 veces)

### Comandos LaTeX Personalizados:

El manual incluye comandos personalizados para facilitar la edición:

```latex
\paso{Paso 1:}                    % Texto en negrita azul
\boton{Procesar Imagen}           % Nombre de botón con fondo gris
\nota{...}                        % Caja azul con nota informativa
\advertencia{...}                 % Caja roja con advertencia importante
\consejo{...}                     % Caja verde con consejo útil
```

### Estructura de Secciones:

```latex
\section{Título}           % Capítulo principal
\subsection{Subtítulo}     % Subsección
\subsubsection{Detalle}    % Sub-subsección
```

## 🎨 Personalización

### Colores Definidos:

```latex
\definecolor{covidred}{RGB}{255,107,107}      % Rojo para COVID-19
\definecolor{normalgreen}{RGB}{81,207,102}     % Verde para Normal
\definecolor{virusyellow}{RGB}{255,212,59}     % Amarillo para Neumonía Viral
\definecolor{primaryblue}{RGB}{41,128,185}     % Azul principal
```

### Modificar Encabezado:

Editar las líneas 38-43 en `manual_usuario.tex`

### Modificar Geometría de Página:

Editar las líneas 22-28 (márgenes, tamaño de página)

## 📊 Estadísticas del Manual

- **Páginas totales**: 23
- **Tamaño del PDF**: 274 KB
- **Idioma**: Español
- **Formato papel**: A4
- **Tamaño de fuente**: 12pt
- **Imágenes planificadas**: 16
- **Capítulos**: 9
- **Anexos**: 4

## ⚠️ Advertencias Importantes

### Disclaimers Incluidos:

1. **NO es un dispositivo médico aprobado para diagnóstico clínico**
2. **Herramienta de apoyo para investigación académica**
3. **Los resultados deben ser interpretados por profesionales médicos**
4. **No reemplaza el diagnóstico médico profesional**

Estas advertencias aparecen en:
- Introducción (caja roja destacada)
- Interpretación de resultados
- Preguntas frecuentes
- Página final

## 🔄 Próximos Pasos

### Para Completar el Manual:

1. [ ] Capturar las 16 imágenes sugeridas
2. [ ] Guardar imágenes en `imagenes/` con nombres correctos
3. [ ] Recompilar el PDF
4. [ ] Revisar que todas las imágenes se vean correctamente
5. [ ] Verificar que todos los enlaces internos funcionen
6. [ ] Validar con usuarios no técnicos (feedback)
7. [ ] Ajustar contenido según feedback

### Mejoras Futuras (Opcional):

- [ ] Agregar ejemplos de casos de uso reales
- [ ] Incluir troubleshooting visual (screenshots de errores)
- [ ] Crear versión en inglés
- [ ] Generar video tutorial complementario
- [ ] Agregar sección de interpretación médica detallada

## 📞 Contacto

Para preguntas sobre el manual o sugerencias de mejora, contactar al investigador principal o equipo de desarrollo del sistema.

---

**Versión del Manual**: 1.0
**Fecha de Creación**: Enero 2026
**Sistema Compatible**: Interfaz Gráfica v15 (v1.0.10)
**Última Actualización**: 27 de enero de 2026
