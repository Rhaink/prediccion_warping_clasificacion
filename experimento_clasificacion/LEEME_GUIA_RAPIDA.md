# EXPERIMENTO DE CLASIFICACIÓN: VALIDACIÓN GEOMÉTRICA (FISHER)

Este proyecto valida la hipótesis de que el alineamiento pulmonar vía **Piecewise Affine Warping** mejora significativamente la separabilidad de las clases "Sano" y "Enfermo" mediante métodos estadísticos clásicos.

## Estructura de la Carpeta

- **01_DATOS_ENTRADA/**: Contiene los datasets originales (Raw), el prototipo (Warped Small) y el masivo (Warped Full).
- **02_SCRIPTS_ANALISIS/**:
    - `1_demostracion_matematica_numpy.py`: **USAR PARA DEFENSA.** Implementación artesanal paso a paso según el guion del asesor.
    - `2_fisher_estricto_gpu.py`: **RESULTADO FINAL.** Ejecución a gran escala (86% accuracy) usando GPU.
    - `3_generar_evidencia_visual.py`: Generador de láminas comparativas Raw vs Warped.
- **03_EVIDENCIA_RESULTADOS/**: Capturas de matrices de confusión, gráficas de importancia de Fisher y ejemplos visuales de pacientes.
- **04_DOCUMENTACION/**: Informe de auditoría y mapeo entre el audio y el código.

## Cómo ejecutar la demostración

Para ver la validación matemática en vivo (NumPy):
```bash
python3 02_SCRIPTS_ANALISIS/1_demostracion_matematica_numpy.py
```

Para generar la evidencia visual:
```bash
python3 02_SCRIPTS_ANALISIS/3_generar_evidencia_visual.py
```

---
*Este paquete de experimentos ha sido diseñado para demostrar que el Warping es un "multiplicador de fuerza" para técnicas de análisis de textura y ML clásico.*
