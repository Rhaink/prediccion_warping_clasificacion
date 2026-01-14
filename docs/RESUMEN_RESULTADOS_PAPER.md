# Resumen para paper (LaTeX)

Este archivo contiene un resumen corto en LaTeX con protocolo y resultados
para incluir en el manuscrito. Texto sin acentos para mantener compatibilidad
ASCII.

## LaTeX

```latex
\paragraph{Protocolo de validacion.}
Se utilizo el dataset COVID-19 Radiography con tres clases (COVID, Normal,
Viral\_Pneumonia). Para el clasificador se trabajo con el dataset warpeado
\texttt{warped\_lung\_best} (fill rate \(\sim\)47\%), con splits
train/val/test = 11{,}364/1{,}894/1{,}895. El modelo fue ResNet-18
preentrenado en ImageNet, con preprocesamiento CLAHE (clip=2.0, tile=4),
entrenado hasta 50 epocas con early stopping segun F1 macro en validacion.
Se reporta el mejor modelo en test y la media\(\pm\)desviacion estandar con
3 semillas. Para landmarks se evaluo un ensemble de 4 modelos con TTA+CLAHE,
midiendo error en pixeles a escala 224x224.

\paragraph{Resultados.}
\textbf{Predicciones (landmarks):} error medio 3.61 px con TTA+CLAHE.
\textbf{Warping/robustez:} degradacion por JPEG Q50 de 0.53\% (vs. 16.14\%
en original) y por blur \(\sigma\)=1 de 6.06\% (vs. 14.43\%),
mejoras aproximadas de 30x y 2.38x, respectivamente; el gap de generalizacion
within-domain mejora 2.43x.
\textbf{Clasificacion en \texttt{warped\_lung\_best}:} mejor run con accuracy
98.05\%, F1 macro 97.12\% y F1 weighted 98.04\%.
La estabilidad con 3 semillas (lr=1.5e-4, class\_weights=on) es accuracy
97.89\% +/- 0.13, F1 macro 96.82\% +/- 0.12.
```

## Notas

- Fuente de metricas: resultados documentados en `docs/EXPERIMENTS.md` y
  `GROUND_TRUTH.json`.
- Ajusta el texto si cambian los best actual o los protocolos de validacion.
