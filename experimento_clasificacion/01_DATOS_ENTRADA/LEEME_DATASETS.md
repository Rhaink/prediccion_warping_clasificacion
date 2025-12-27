# DOCUMENTACIÓN DE DATASETS
Este directorio contiene los tres conjuntos de datos utilizados durante el método de Warping.

## 1. COVID-19_Radiography_Dataset
- **Descripción:** Dataset original descargado de Kaggle/Fuente oficial.
- **Uso:** Sirve como línea base y fuente para comparar las imágenes "Antes" del procesamiento.
- **Contenido:** Imágenes en escala de grises original, sin alineación.

## 2. warped_dataset
- **Descripción:** Primer conjunto de datos procesado con el algoritmo de Warping.
- **Características:** 
    - Dataset más pequeño (957 imágenes).
    - Usado para pruebas iniciales.
- **Uso:** Primeras pruebas.

## 3. full_warped_dataset
- **Descripción:** Dataset masivo generado automáticamente aplicando el Warping a todos las imágenes disponibles (>15,000 imágenes).
- **Características:**
    - Dividido en Train/Test/Validation.
    - **Es la fuente de los resultados finales (~85% Accuracy).**
- **Uso:** Clasificación completa.
