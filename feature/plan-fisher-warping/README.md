# Plan Fisher-Warping: Validacion de Alineacion Geometrica

## Objetivo

Demostrar que la alineacion geometrica (warping) mejora la discriminacion entre clases usando un pipeline clasico: **PCA + Fisher + KNN**. NO se usa CNN para esta demostracion.

## Requisitos del Asesor

Estos requisitos provienen de `conversacion.txt` y deben seguirse exactamente:

1. **2 clases**: Enfermo (COVID + Neumonia Viral) vs Normal
2. **Imagenes warped**: Usar imagenes ya alineadas (224x224)
3. **Eigen-space**: Construir UN solo Eigen-space con las imagenes warped
4. **Eigenfaces**: Seleccionar N principales (ej: 10)
5. **Caracteristicas = Ponderantes**: Los pesos de la proyeccion PCA son las caracteristicas
6. **Z-score**: Estandarizar cada caracteristica
7. **Fisher**: Calcular ratio por caracteristica para medir separabilidad
8. **Amplificacion**: Multiplicar caracteristicas por su Fisher ratio
9. **Clasificador simple**: KNN (no CNN)
10. **Mostrar imagenes warped**: Entregar muestras al asesor

## Estado Actual

Ver `checklist/PROGRESS.md` para el progreso detallado.

## Estructura de Archivos

```
feature/plan-fisher-warping/
├── README.md                 # Este archivo
├── conversacion.txt          # Transcripcion original con el asesor
├── docs/
│   ├── 00_OBJETIVOS.md       # Objetivos y criterios de evidencia
│   ├── 01_MATEMATICAS.md     # Explicaciones matematicas (PCA, Fisher, etc.)
│   ├── 02_PIPELINE.md        # Pipeline paso a paso
│   └── 03_ASESOR_CHECKLIST.md # Verificacion de requisitos del asesor
├── config/
│   ├── LABEL_MAPPING.md      # Mapeo de etiquetas 2/3 clases
│   └── SPLIT_PROTOCOL.md     # Protocolo de splits y balanceo
├── checklist/
│   └── PROGRESS.md           # Progreso incremental
├── src/                      # Codigo a implementar
│   ├── data_loader.py        # Carga de imagenes
│   ├── pca.py                # PCA/Eigenfaces
│   ├── fisher.py             # Fisher ratio
│   ├── classifier.py         # KNN
│   └── pipeline.py           # Pipeline completo
├── notebooks/                # Tutoriales interactivos
└── results/
    ├── logs/                 # Reportes
    ├── metrics/              # CSVs con metricas
    └── figures/              # Visualizaciones
```

## Como Trabajar con Claude

1. **Inicio de sesion**: Leer `checklist/PROGRESS.md` para ver donde quedamos
2. **Dudas matematicas**: Consultar `docs/01_MATEMATICAS.md`
3. **Ejecutar**: Siguiente item pendiente en la checklist
4. **Explicacion**: Claude explica codigo/matematicas mientras implementa
5. **Evidencia**: Generar archivo/figura como entregable
6. **Verificar**: Actualizar `docs/03_ASESOR_CHECKLIST.md` si aplica
7. **Cierre**: Marcar item completado en PROGRESS.md

## Datos

- **Dataset warped**: `outputs/full_warped_dataset/` (15,153 imagenes)
- **Dataset manual**: `outputs/warped_dataset/` (957 imagenes)
- **Splits balanceados**: Ver `results/metrics/`

## Referencias

- Conversacion con asesor: `conversacion.txt`
- Pipeline: `docs/02_PIPELINE.md`
- Matematicas: `docs/01_MATEMATICAS.md`
