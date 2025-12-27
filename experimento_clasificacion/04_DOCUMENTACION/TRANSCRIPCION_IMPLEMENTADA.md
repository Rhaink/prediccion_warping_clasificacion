# INFORME DE AUDITORÍA: DEL AUDIO AL ALGORITMO

Este documento certifica la correspondencia exacta entre las instrucciones verbales del asesor y la implementación técnica realizada en los scripts de esta carpeta.

| Instrucción del Asesor (Audio) | Implementación Técnica (Código) | Referencia en Script (Numpy) |
| :--- | :--- | :--- |
| **"Simplificarlo... pensando que solo hay dos clases: neumonía o sano."** | Mapeo binario: Normal=0, (Neumonía/COVID)=1. | `lib/utils.py` (Clasificación Binaria) |
| **"A esos dos grupos, construyes un solo Eigen-space."** | SVD calculado sobre la matriz completa X que mezcla ambas clases. | `1_demostracion...py` (Bloque 1) |
| **"Las características no serían las Eigenfaces. Serían los ponderantes [pesos]."** | Los vectores de imagen se proyectan al espacio latente; solo se usan las coordenadas resultantes. | `1_demostracion...py` (Función `calcular_espacio_propio`) |
| **"Estandarizar solo la característica 1... (X - media) / std."** | Aplicación de Z-score columna por columna sobre la matriz de ponderantes. | `1_demostracion...py` (Bloque 2) |
| **"Tomas los que nada más son para la neumonía... sacas media 1... sanos... media 2."** | Segmentación de la matriz estandarizada en dos nubes de puntos por clase. | `1_demostracion...py` (Bloque 3) |
| **"Si esa Razón de Fisher es grande, significa que esa característica separa bien."** | Cálculo del score J como métrica de discriminación lineal. | `1_demostracion...py` (Bloque 3) |
| **"Le pegas un ponderante [Fisher]... la estás amplificando."** | Multiplicación de cada columna por su importancia de Fisher calculada. | `1_demostracion...py` (Bloque 4) |
| **"Y ahora sí ya las metes al clasificador... puede ser un KNN."** | Clasificación final mediante distancia Euclidiana pura (Manual). | `1_demostracion...py` (Bloque 4) |
| **"Lo que hace la diferencia es que hayas alineado esas canijas imágenes."** | Comparación de resultados: RAW vs WARPED (Demostrando la superioridad de la alineación). | `1_demostracion...py` (Flujo Principal) |

---
**Nota Metodológica:** Para la implementación de "Máxima Potencia" (GPU), se utiliza la raíz cuadrada del score de Fisher ($\sqrt{J}$) para mantener la estabilidad dimensional de la amplitud, cumpliendo con la intención física descrita por el asesor.
