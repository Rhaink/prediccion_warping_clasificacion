# Plan de trabajo – Estancia de investigación

**Nombre del proyecto:** Normalización y alineación automática de la forma de la región pulmonar integrada con selección de características discriminantes para detección automática de neumonía y COVID-19.

**Director del proyecto:** Dr. Leopoldo Altamirano Robles

**Estudiante:** Rafael Alejandro Cruz Ovando

**Fechas de la estancia de investigación:** Inicio 9 de octubre del 2025 - finalización 9 de noviembre del 2025.

**Lugar donde se desarrollará la estancia:** Laboratorio de visión por computadora del Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE).

**Objetivo general:** Desarrollar un prototipo de sistema para la detección automática de landmarks pulmonares en radiografías de tórax utilizando técnicas de aprendizaje profundo y modelado geométrico.

## Objetivos Específicos:

1. Recopilar y normalizar un conjunto de radiografías de tórax con anotaciones de landmarks anatómicos, garantizando calidad y consistencia.

2. Implementar una arquitectura basada en ResNet-18 adaptada a la regresión de coordenadas anatómicas con funciones de pérdida que incorporen restricciones geométricas.

3. Diseñar y ejecutar un plan de entrenamiento que permita evaluar la contribución de cada componente del modelo.

4. Establecer un protocolo de evaluación cuantitativa y cualitativa con métricas clínicas relevantes.

5. Documentar el flujo de trabajo y los resultados.

## Metodología:

### 1. Preparación de Datos

- Realizar un diagnóstico inicial del dataset disponible (cantidad de imágenes, calidad de anotaciones, balance por categoría).

- Definir procedimientos de limpieza, normalización y aumentación de datos acordes a principios de imagenología médica.

- Dividir el dataset en subconjuntos de entrenamiento, validación y prueba asegurando representatividad.

### 2. Diseño del Modelo

- Analizar alternativas de arquitecturas con base en ResNet y seleccionar una configuración inicial.

- Incorporar una cabeza de regresión con capas totalmente conectadas y funciones de activación adecuadas.

- Investigar funciones de pérdida candidatas (Wing Loss, Symmetry Loss, Distance Preservation) y establecer un plan de experimentación.

### 3. Plan de Entrenamiento

- Fase 1: Entrenar únicamente la cabeza de regresión con el backbone congelado para estabilizar la predicción inicial.

- Fase 2: Descongelar gradualmente el backbone aplicando tasas de aprendizaje diferenciadas.

- Fase 3: Evaluar la incorporación de pérdidas geométricas incrementales y comparar su efecto.

- Fase 4: Ajustar hiperparámetros y consolidar la configuración que alcance los objetivos de desempeño.

### 4. Evaluación y Análisis

- Definir métricas objetivo (p. ej. error medio por landmark en píxeles, distribución por categorías clínicas).

- Diseñar un protocolo de validación cruzada.

- Elaborar herramientas para visualizar resultados (superposición de landmarks, mapas de error) que permitan la interpretación clínica.

### 5. Documentación y Difusión

- Mantener bitácoras de experimentos con parámetros, configuraciones y resultados.

- Preparar guías técnicas y manuales de uso preliminares.

## Cronograma:

| Semana | Periodo | Actividades |
|--------|---------|-------------|
| 1 | 9 - 15 Oct | Revisión del estado del arte, análisis del dataset, definición de criterios de calidad y preparación del entorno de trabajo. |
| 2 | 16 – 22 Oct | Implementación de la arquitectura base, configuración del flujo de trabajo del entrenamiento y ejecución de la Fase 1. |
| 3 | 23 – 29 Oct | Ajustes de *fine-tuning*, incorporación de pérdidas geométricas y ejecución de las Fases 2 y 3. |
| 4 | 30 Oct – 9 Nov | Experimentos finales, análisis de métricas, generación de visualizaciones y documentación del prototipo. |

## Entregables:

- Prototipo funcional del modelo con código documentado.

- Checkpoints y configuraciones de entrenamiento de cada fase experimental.

- Reporte de resultados con análisis cuantitativo y visualizaciones representativas.

- Documentación técnica.

---

**Dr. Leopoldo Altamirano Robles**

Investigador titular en el Instituto Nacional de Astrofísica, Óptica y Electrónica, en la Coordinación de Ciencias Computacionales.

Responsable del Laboratorio de Visión por computadora de la coordinación.

robles@inaoep.mx
