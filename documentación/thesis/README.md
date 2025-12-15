# Tesis de Maestría - Predicción de Landmarks Anatómicos

## Estructura del Proyecto

```
thesis/
├── main.tex                    # Documento principal
├── bibliography.bib            # Referencias bibliográficas (50+ refs)
├── config/
│   └── preamble.tex           # Configuración de paquetes LaTeX
├── chapters/
│   ├── 01_introduccion.tex    # Capítulo 1: Introducción
│   ├── 02_marco_teorico.tex   # Capítulo 2: Marco Teórico
│   ├── 03_estado_del_arte.tex # Capítulo 3: Estado del Arte
│   ├── 04_metodologia.tex     # Capítulo 4: Metodología
│   ├── 05_experimentacion.tex # Capítulo 5: Experimentación y Resultados
│   ├── 06_discusion.tex       # Capítulo 6: Discusión
│   └── 07_conclusiones.tex    # Capítulo 7: Conclusiones
├── appendices/
│   ├── A_codigo.tex           # Apéndice A: Fragmentos de Código
│   ├── B_hiperparametros.tex  # Apéndice B: Configuración Completa
│   └── C_visualizaciones.tex  # Apéndice C: Visualizaciones
└── figures/                    # Figuras (15+ imágenes)
```

## Compilación

### Requisitos

- TeX Live 2022 o posterior (recomendado)
- Paquetes requeridos: babel, biblatex, biber, tikz, listings, booktabs, etc.

### Comandos de Compilación

```bash
# Opción 1: Compilación completa con latexmk (recomendado)
cd thesis/
latexmk -pdf -biber main.tex

# Opción 2: Compilación manual
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex

# Opción 3: Usando make (si se crea un Makefile)
make
```

### Limpieza de archivos auxiliares

```bash
latexmk -c main.tex
# o
rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.toc *.lof *.lot *.run.xml
```

## Contenido

### Resumen del Proyecto

- **Tema**: Predicción de 15 Landmarks Anatómicos en Radiografías de Tórax
- **Técnica**: Deep Learning con ResNet-18 + Coordinate Attention
- **Resultado Final**: Error de **3.71 px** (mejora del 59% sobre baseline)
- **Dataset**: 957 radiografías (COVID-19, Normal, Neumonia Viral)

### Estructura de Capítulos

| Capítulo | Título | Páginas Aprox. |
|----------|--------|----------------|
| 1 | Introducción | 10 |
| 2 | Marco Teórico | 20 |
| 3 | Estado del Arte | 15 |
| 4 | Metodología | 25 |
| 5 | Experimentación y Resultados | 20 |
| 6 | Discusión | 10 |
| 7 | Conclusiones | 5 |
| A-C | Apéndices | 15 |
| | **Total aproximado** | **~120** |

## Figuras Incluidas

### Diagramas de Arquitectura
- `model_architecture.png` - Arquitectura completa del modelo
- `coordinate_attention.png` - Módulo Coordinate Attention
- `ensemble_tta_pipeline.png` - Pipeline de inferencia
- `training_pipeline.png` - Entrenamiento en 2 fases
- `data_flow.png` - Flujo de datos

### Figuras de Resultados
- `progress_by_session.png` - Evolución del error
- `error_by_landmark.png` - Error por landmark
- `error_by_category.png` - Error por categoría
- `heatmap_landmark_category.png` - Heatmap de errores
- `ensemble_comparison.png` - Comparación de ensemble
- `ablation_study.png` - Estudio de ablación
- `clahe_comparison.png` - Efecto de CLAHE
- `prediction_examples.png` - Ejemplos de predicciones
- `best_worst_cases.png` - Mejores y peores casos

## Personalización

### Información del Autor

Editar en `main.tex`:
- Nombre del autor
- Nombre del director de tesis
- Nombre de la universidad
- Ciudad y país

### Formato

El documento usa:
- Tamaño A4
- Márgenes: izq=3cm, der=2.5cm, arriba/abajo=2.5cm
- Espaciado 1.5
- Fuente Latin Modern 12pt
- Estilo de citas IEEE

## Notas

- El documento está en español
- Usa UTF-8 para caracteres especiales
- Las figuras se buscan en `figures/`, `../outputs/diagrams/`, `../outputs/thesis_figures/`
- La bibliografía requiere `biber` (no `bibtex`)
