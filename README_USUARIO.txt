===============================================================
   COVID-19 Detection System - Demo para Defensa de Tesis
   Sistema de Detección de COVID-19 - Demostración
===============================================================

INSTRUCCIONES DE USO:

1. EJECUTAR LA APLICACIÓN:
   ✓ Hacer doble click en "COVID19_Demo.exe"
   ✓ Aparecerá una ventana negra con texto (¡NO CERRAR!)
   ✓ Esperar 10-30 segundos mientras carga los modelos
   ✓ Su navegador web se abrirá automáticamente con la interfaz

   Nota: La primera ejecución puede tardar más tiempo.

2. USAR LA INTERFAZ:

   Pestaña "Demostración Completa":
   ----------------------------------
   • Arrastre una imagen de rayos X de tórax al área de carga
   • O use uno de los ejemplos precargados (COVID, Normal, Viral)
   • Click en el botón "🔍 Procesar Imagen"
   • Espere 1-2 segundos mientras procesa

   Resultados mostrados:
   ✓ Imagen Original con Landmarks (15 puntos anatómicos de colores)
   ✓ Imagen Normalizada (warping geométrico aplicado)
   ✓ Mapa de Calor GradCAM (explica qué ve el modelo)
   ✓ Resultado de Clasificación con probabilidades:
     - COVID-19 (rojo)
     - Normal (verde)
     - Neumonía Viral (amarillo)

   Funcionalidades adicionales:
   • Expandir "Métricas del Sistema" para ver detalles técnicos
   • Click en "💾 Exportar a PDF" para guardar resultados

   Pestaña "Vista Rápida":
   -----------------------
   • Procesamiento más rápido (solo muestra resultado de clasificación)
   • Útil cuando solo necesita el diagnóstico sin visualizaciones

   Pestaña "Acerca del Sistema":
   ------------------------------
   • Información sobre la metodología
   • Métricas validadas del sistema
   • Referencias científicas

3. CERRAR LA APLICACIÓN:
   ✓ Cerrar la pestaña del navegador
   ✓ En la ventana negra, presionar Ctrl+C
   ✓ O simplemente cerrar la ventana negra con la X
   ✓ La aplicación le pedirá confirmar para salir

===============================================================
REQUISITOS DEL SISTEMA:
===============================================================

Hardware Mínimo:
  • Procesador: Intel/AMD dual-core o superior (2 GHz+)
  • Memoria RAM: 4 GB (8 GB recomendado)
  • Espacio en disco: 2 GB libres
  • Pantalla: 1280×720 o superior

Software:
  • Sistema Operativo: Windows 10 o 11 (64-bit)
  • Navegador web: Chrome, Firefox, Edge (cualquier versión reciente)

NO REQUIERE:
  ✗ Python instalado
  ✗ NVIDIA GPU o drivers CUDA
  ✗ Conexión a Internet (funciona completamente offline)
  ✗ Instalación de dependencias

===============================================================
SOLUCIÓN DE PROBLEMAS:
===============================================================

Problema: "Windows protegió su PC" (SmartScreen)
-------------------------------------------------
Solución:
  1. Click en "Más información"
  2. Click en "Ejecutar de todas formas"

Causa: Windows no reconoce el ejecutable porque no está
firmado digitalmente. Es normal para aplicaciones académicas.

---

Problema: La aplicación no abre o muestra error de DLL
-------------------------------------------------------
Solución:
  1. Instalar "Microsoft Visual C++ Redistributable 2015-2022"
  2. Descargar desde: https://aka.ms/vs/17/release/vc_redist.x64.exe
  3. Ejecutar el instalador
  4. Reintentar abrir COVID19_Demo.exe

---

Problema: El navegador no abre automáticamente
-----------------------------------------------
Solución:
  1. Abrir manualmente su navegador web
  2. Ir a la dirección: http://localhost:7860
  3. Verificar que la ventana negra muestre "Running on local URL"

---

Problema: La aplicación es muy lenta
-------------------------------------
Causa: Procesamiento en CPU (no GPU)
Solución:
  • Cerrar otras aplicaciones para liberar memoria
  • Esperar pacientemente (1-2 segundos por imagen es normal)
  • Primer procesamiento puede tardar más (cold start)

---

Problema: Error "Failed to load model" en la interfaz
------------------------------------------------------
Solución:
  1. Verificar que tiene al menos 4 GB RAM disponible
  2. Cerrar otras aplicaciones
  3. Reiniciar la aplicación

---

Problema: El archivo .exe es muy grande (1.8 GB)
-------------------------------------------------
Respuesta: Es normal. El ejecutable incluye:
  • PyTorch completo (framework de deep learning)
  • 5 modelos de redes neuronales preentrenados
  • Todas las librerías científicas (OpenCV, NumPy, etc.)
  • Interfaz web Gradio

---

Problema: Antivirus bloquea o elimina el archivo
-------------------------------------------------
Solución:
  1. Agregar excepción en el antivirus para COVID19_Demo.exe
  2. Verificar checksum SHA256 (ver archivo .sha256) para
     confirmar que no está corrupto
  3. Si persiste, contactar al desarrollador

===============================================================
MÉTRICAS VALIDADAS DEL SISTEMA:
===============================================================

El sistema ha sido validado con los siguientes resultados:

Detección de Landmarks:
  • Error medio: 3.61 píxeles (en imágenes 224×224)
  • Desviación estándar: ±2.48 píxeles
  • Mediana: 3.07 píxeles

Clasificación de COVID-19:
  • Accuracy: 98.05%
  • F1-Score (macro): 97.12%
  • F1-Score (weighted): 98.04%

Preprocesamiento:
  • CLAHE clip limit: 2.0
  • CLAHE tile size: 4×4
  • Margen de warping: 1.05 (5% expansión)

Dataset:
  • Fuente: COVID-19 Radiography Database (Kaggle)
  • Clases: COVID-19, Normal, Neumonía Viral
  • Tamaño de entrada: 224×224 píxeles

===============================================================
FORMATOS DE IMAGEN SOPORTADOS:
===============================================================

✓ PNG (.png)
✓ JPEG (.jpg, .jpeg)
✓ BMP (.bmp)

Tamaño recomendado: Al menos 224×224 píxeles
Imágenes muy pequeñas (<100×100) pueden dar error.
Imágenes muy grandes (>10 MB) pueden tardar más en procesar.

===============================================================
LIMITACIONES Y ADVERTENCIAS:
===============================================================

⚠️ IMPORTANTE: Este sistema es una herramienta de investigación
académica y NO debe usarse para diagnóstico clínico real sin
supervisión médica profesional.

Limitaciones conocidas:
  1. Domain Shift: El modelo está entrenado en un dataset
     específico. Resultados en nuevas fuentes de rayos X
     pueden variar.

  2. No detecta otras patologías: El sistema solo clasifica
     entre COVID-19, Normal, y Neumonía Viral. No detecta
     otras enfermedades pulmonares.

  3. Requiere rayos X de tórax frontales: Vistas laterales
     u otros ángulos no son soportados.

  4. Sensibilidad a calidad de imagen: Imágenes muy oscuras,
     borrosas o con artefactos pueden dar resultados
     incorrectos.

===============================================================
PRIVACIDAD Y DATOS:
===============================================================

✓ Todas las imágenes se procesan LOCALMENTE en su computadora
✓ NO se envían datos a Internet
✓ NO se guardan imágenes automáticamente (solo si exporta PDF)
✓ Al cerrar la aplicación, todo se borra de la memoria

===============================================================
SOPORTE TÉCNICO:
===============================================================

Para reportar problemas o solicitar ayuda:
  • Email: [Agregar email del investigador]
  • GitHub Issues: [Agregar URL del repositorio]

Al reportar un problema, incluya:
  1. Versión de Windows (Ej: Windows 10 21H2)
  2. Captura de pantalla del error
  3. Mensaje completo de error de la ventana negra
  4. Características del hardware (RAM, procesador)

===============================================================
INFORMACIÓN DEL PROYECTO:
===============================================================

Sistema de Detección de COVID-19 mediante Landmarks Anatómicos
y Normalización Geométrica

Autor: [Nombre del tesista]
Institución: [Universidad]
Año: 2026

Tecnologías:
  • PyTorch 2.x (Deep Learning)
  • ResNet-18 con Coordinate Attention
  • Generalized Procrustes Analysis (GPA)
  • Piecewise Affine Warping
  • GradCAM (Explainability)
  • Gradio (Interfaz web)

Versión: 1.0.0
Última actualización: Enero 2026

===============================================================
LICENCIA:
===============================================================

[Especificar licencia si aplica - Ej: MIT, GPL, Académica]

Este software se proporciona "tal cual", sin garantías de
ningún tipo. El uso es bajo su propio riesgo.

===============================================================
AGRADECIMIENTOS:
===============================================================

Dataset: COVID-19 Radiography Database
  - Chowdhury et al. (2020)
  - Kaggle: covid19-radiography-database

Frameworks y Librerías:
  • PyTorch (Facebook AI Research)
  • Gradio (Hugging Face)
  • OpenCV (Open Source Computer Vision)
  • ResNet (Microsoft Research)

===============================================================

¿PREGUNTAS FRECUENTES?

P: ¿Puedo usar este sistema en un hospital?
R: NO sin validación clínica adicional y aprobación regulatoria.
   Es una herramienta de investigación académica.

P: ¿Funciona sin Internet?
R: SÍ, completamente offline.

P: ¿Necesito GPU/NVIDIA?
R: NO, funciona solo con CPU (es más lento pero funcional).

P: ¿Puedo procesar múltiples imágenes a la vez?
R: Actualmente solo una a la vez. Para procesamiento batch,
   use la versión Python del sistema.

P: ¿Los resultados se guardan?
R: NO automáticamente. Use "Exportar a PDF" para guardar.

P: ¿Puedo modificar el código?
R: Esta es una versión standalone (ejecutable). Para modificar
   el código, descargue el repositorio fuente de GitHub.

P: ¿Por qué tarda tanto en abrir?
R: Debe descomprimir internamente 1.8 GB de datos y cargar
   modelos de redes neuronales. Es normal en la primera
   ejecución.

===============================================================

¡Gracias por usar el sistema!

Para más información, consulte la documentación completa
en el repositorio del proyecto.

===============================================================
