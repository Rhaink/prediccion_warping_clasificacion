# INFORME METODOLÓGICO: LA GEOMETRÍA COMO AMPLIFICADOR

## Resumen Ejecutivo
Este experimento demuestra que la alineación geométrica (Warping) no solo estandariza la forma, sino que actúa como un **multiplicador de fuerza** para el análisis de textura.

Al comparar el rendimiento en el mismo dataset masivo:
- **RAW + CLAHE:** ~81.4% (El realce de contraste amplifica el ruido geométrico).
- **WARPED + CLAHE:** **86.03%** (El realce de contraste amplifica la señal patológica).

## Conclusión Científica
El algoritmo de Warping reduce la **Varianza Intra-clase Geométrica** (las diferencias de forma irrelevantes entre pacientes), permitiendo que el clasificador lineal (Fisher/KNN) se enfoque puramente en la **Varianza de Textura** (la patología).

Esto confirma la hipótesis del asesor: al desenredar el "manifold" de formas, la separación lineal se vuelve trivialmente más efectiva.
