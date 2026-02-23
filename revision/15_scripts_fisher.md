# 15. Fisher Analysis Scripts

Analisis de los scripts de validacion estadistica Fisher para la tesis.

**Archivos analizados**: 11

---

## Resumen Ejecutivo

Este directorio contiene 11 scripts que implementan variaciones de un pipeline de validacion geometrica basado en el Criterio Discriminante de Fisher (Fisher Linear Discriminant). El proposito general es demostrar que las imagenes warped (geometricamente normalizadas) son mas separables que las raw en un espacio PCA ponderado por Fisher, usando clasificacion k-NN como proxy. Los scripts representan una evolucion iterativa del mismo concepto, desde versiones didacticas CPU hasta versiones GPU optimizadas, pasando por busqueda de semillas y optimizacion de hiperparametros.

**Pipeline comun**: Carga imagenes warped -> PCA (reduccion dimensional) -> Estandarizacion de ponderantes -> Fisher Score por componente -> Amplificacion (multiplicar por J o sqrt(J)) -> Clasificacion k-NN -> Accuracy.

**Problema principal**: Existe una duplicacion masiva de codigo entre los 9 scripts principales (el DatasetLoader, criterio_fisher_manual, knn_predict se repiten casi identicos). Las variantes difieren solo en detalles menores (CPU vs GPU, J vs sqrt(J), binario vs multiclase, con/sin CLAHE, orden de estandarizacion).

---

## Analisis Individual

### 1. thesis_fisher_tutorial.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_fisher_tutorial.py
- **Lineas/Tamano**: 292 lineas / 11.4 KB
- **Proposito**: Tutorial didactico completo que implementa el pipeline Fisher desde cero usando solo NumPy, diseñado para explicar cada paso matematico durante la defensa de tesis (PCA via SVD, Z-score, Fisher Score, k-NN manual).
- **Contenido clave**:
  - `pca_artesanal_svd()`: PCA manual via np.linalg.svd sin sklearn
  - `estandarizador_manual()`: Z-score manual
  - `criterio_fisher_score()`: Fisher ratio J = (mu0-mu1)^2 / (var0+var1) por componente
  - `clasificador_knn_manual()`: k-NN con distancia euclidiana bruta
  - Usa datos sinteticos como fallback si no existe el dataset
  - Imagenes a 64x64 para velocidad, 20 componentes PCA, k=5
  - Amplificacion con J directo (no sqrt)
- **Importancia**: MEDIO
- **Justificacion**: Tiene valor pedagogico para la defensa de tesis al explicar cada operacion matricial sin cajas negras. Sin embargo, no produce resultados reproducibles del proyecto (usa dataset reducido, resolucion baja). Es material de soporte para la presentacion, no para el pipeline de investigacion.

---

### 2. thesis_optimization_fisher.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_optimization_fisher.py
- **Lineas/Tamano**: 261 lineas / 8.6 KB
- **Proposito**: Grid search para optimizar parametros de preprocesamiento CLAHE (clip_limit x tile_size) maximizando el accuracy de clasificacion binaria Fisher-Linear en GPU.
- **Contenido clave**:
  - `DatasetLoader.load_dataset_with_params()`: Carga con CLAHE parametrizable on-the-fly
  - `evaluate_configuration()`: Ejecuta pipeline completo (PCA centrado -> estandarizar weights -> Fisher sqrt(J) -> k-NN) por configuracion
  - Grid: clip_limits=[1.0, 2.0, 4.0] x tile_sizes=[2, 4, 8] = 9 combinaciones
  - Gestion de memoria GPU con gc.collect() y torch.cuda.empty_cache()
  - Guarda resultados en `results/metrics/grid_search_clahe.json`
  - Bug: `run_optimization` esta definida dos veces (lineas 117 y 219); la primera definicion (vacia) se sobreescribe
  - Usa logica "Strict" (PCA sobre raw centrado, estandarizacion de weights, sqrt(J))
- **Importancia**: MEDIO
- **Justificacion**: Es una herramienta de optimizacion experimental util para encontrar los mejores parametros CLAHE. Sin embargo, los parametros optimos de CLAHE del proyecto ya estan validados (clip=2.0, tile=4) en el pipeline principal. Los resultados de este grid search son complementarios. El bug de funcion duplicada indica desarrollo apresurado.

---

### 3. thesis_seed_search.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_seed_search.py
- **Lineas/Tamano**: 173 lineas / 5.8 KB
- **Proposito**: Busqueda exhaustiva de la semilla aleatoria (0-500) que maximiza el accuracy del pipeline Fisher-PCA-kNN en GPU, intentando superar el record de 86.03% (seed 8).
- **Contenido clave**:
  - `test_seed()`: Ejecuta pipeline completo con una semilla dada (afecta torch.pca_lowrank)
  - Fija torch.manual_seed, np.random.seed, torch.cuda.manual_seed_all
  - Pipeline: PCA centrado (q=50) -> estandarizar weights -> sqrt(J) Fisher -> k-NN(k=5)
  - Umbral de detencion temprana: acc > 87%
  - Carga datos una sola vez y reitera sobre semillas (eficiente)
  - CLAHE hardcoded a clip=2.0, tile=4x4
- **Importancia**: BAJO
- **Justificacion**: La busqueda de semillas para maximizar accuracy en un clasificador no-parametrico es una practica metodologicamente cuestionable (cherry-picking). El hecho de que el accuracy dependa fuertemente de la semilla de PCA sugiere inestabilidad del metodo. Este script es evidencia del proceso experimental pero no deberia citarse como resultado validado.

---

### 4. thesis_validation_fisher_basic.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_validation_fisher_basic.py
- **Lineas/Tamano**: 239 lineas / 8.5 KB
- **Proposito**: Version didactica CPU de la validacion Fisher usando scikit-learn (PCA, StandardScaler, KNeighborsClassifier). Implementa el flujo basico: PCA -> Estandarizar ponderantes -> Fisher manual -> Amplificar con J -> k-NN.
- **Contenido clave**:
  - `cargar_datos_simples()`: Carga desde CSV con busqueda multi-ruta, sin CLAHE
  - `criterio_fisher_manual()`: Fisher ratio por componente (identico en 5+ scripts)
  - Pipeline: sklearn PCA(n=50) -> StandardScaler sobre weights -> Fisher J -> amplificacion J directo -> KNN(k=5)
  - Genera grafico de barras de Fisher scores (`results/figures/fisher_scores_basic.png`)
  - Guarda metricas en `results/metrics/basic_metrics.txt`
  - Imagen a 224x224, sin CLAHE
  - Diferencia clave vs "strict": Amplificacion con J directo, no sqrt(J)
- **Importancia**: MEDIO
- **Justificacion**: Primera implementacion funcional del concepto Fisher para la tesis. Sirve como baseline de referencia y como version didactica comprensible. El codigo es limpio y bien comentado. No usa CLAHE, lo que lo hace comparable pero suboptimo frente a las versiones con preprocesamiento.

---

### 5. thesis_validation_fisher_basic_standard.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_validation_fisher_basic_standard.py
- **Lineas/Tamano**: 132 lineas / 4.7 KB
- **Proposito**: Variante que invierte el orden de preprocesamiento: estandariza pixeles ANTES de PCA (practica comun en ML), a diferencia de la version del asesor que estandariza los ponderantes DESPUES de PCA.
- **Contenido clave**:
  - Flujo diferente: Estandarizacion de pixeles -> PCA -> Fisher -> Amplificacion con J -> k-NN
  - `cargar_datos_simples()`: Identica a la version basic (copiar-pegar)
  - `criterio_fisher_manual()`: Identica (copiar-pegar)
  - Sin CLAHE, imagen 224x224, 50 componentes
  - Objetivo explicito: comparar contra la version "estricta" del asesor
- **Importancia**: BAJO
- **Justificacion**: Variante experimental para comparar el orden de estandarizacion (pixeles vs ponderantes). El resultado de esta comparacion es interesante metodologicamente pero el script es practicamente un copiar-pegar del basic con una linea movida. Podria haber sido un parametro en un script unificado.

---

### 6. thesis_validation_fisher_massive_cpu.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_validation_fisher_massive_cpu.py
- **Lineas/Tamano**: 235 lineas / 9.2 KB
- **Proposito**: Version CPU con argparse para ejecutar la validacion Fisher basica sobre datasets masivos (full_warped_dataset), identica a thesis_validation_fisher_basic.py pero con ruta parametrizable por linea de comandos.
- **Contenido clave**:
  - `argparse` con `--dataset-dir` (default: outputs/warped_dataset)
  - `cargar_datos_simples()`: Identica a basic (copiar-pegar)
  - `criterio_fisher_manual()`: Identica (copiar-pegar)
  - Mismo pipeline: PCA -> Estandarizar weights -> Fisher J -> Amplificar J -> k-NN
  - Guarda figuras y metricas con sufijo del nombre del dataset
- **Importancia**: BAJO
- **Justificacion**: Es una copia literal de thesis_validation_fisher_basic.py con la unica adicion de argparse para parametrizar la ruta del dataset. La duplicacion de codigo es innecesaria; bastaba anadir argparse al script original. El nombre "massive" es engañoso ya que no tiene ninguna optimizacion de memoria o procesamiento por lotes para datasets grandes.

---

### 7. thesis_validation_fisher_multiclass.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_validation_fisher_multiclass.py
- **Lineas/Tamano**: 294 lineas / 10.6 KB
- **Proposito**: Extension multiclase (Normal vs Neumonia vs COVID) de la validacion Fisher, usando F-Statistic (ANOVA) como generalizacion del Fisher Score binario. Ejecuta en GPU.
- **Contenido clave**:
  - `TorchAnalysisMulticlass.fisher_score_multiclass()`: Calcula F-statistic ANOVA por componente (S_B entre clases / S_W intra clases, normalizado por grados de libertad)
  - `DatasetLoader.load_full_dataset()`: Mapeo 3 clases: Normal=0, Viral Pneumonia=1, COVID=2
  - Pipeline: Estandarizacion pixeles -> PCA(k=50) -> Fisher multiclase sqrt(J) -> k-NN(k=5)
  - Genera confusion matrix heatmap (`results/multiclass_experiment/confusion_matrix.png`)
  - Genera proyeccion 2D Fisher-PCA para visualizar separabilidad
  - k-NN con batching para datasets grandes (>2000 muestras)
  - CLAHE habilitado por defecto
  - Nota: Usa estandarizacion de pixeles (no ponderantes), difiere de la version "strict"
- **Importancia**: ALTO
- **Justificacion**: Es la unica version que implementa clasificacion triclase, lo cual es mas representativo del problema real (el pipeline CNN principal clasifica 3 clases). La generalizacion del Fisher Score a F-statistic es matematicamente correcta. Sin embargo, la estandarizacion de pixeles (no ponderantes) contradice la version "strict" del asesor usada en otros scripts. Los resultados son valiosos para la tesis si se reportan como analisis complementario.

---

### 8. thesis_validation_fisher.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_validation_fisher.py
- **Lineas/Tamano**: 404 lineas / 13.1 KB
- **Proposito**: Version mas completa de la validacion Fisher con generacion de evidencia forense detallada: comparativas individuales RAW vs WARPED por imagen, calculo de SSIM/PSNR para cuantificar deformacion, y clasificacion de errores (TP/TN/FP/FN).
- **Contenido clave**:
  - `DatasetLoader.get_raw_path()`: Busca imagen original raw para comparacion visual
  - `TorchAnalysis.save_detailed_results()`: Genera reportes visuales individuales (RAW vs WARPED) categorizados en TP/TN/FP/FN con SSIM y PSNR
  - Pipeline "Strict": PCA centrado (no estandarizado) -> estandarizar weights -> sqrt(J) Fisher -> k-NN
  - Semilla fija: SEED=8 (la ganadora del seed search, 86.03%)
  - k-NN con batching para datasets grandes
  - CLAHE habilitado por defecto
  - Genera 50 reportes por categoria de error en `results/detailed_analysis/`
  - Es el script mas largo (404 lineas) del directorio
- **Importancia**: ALTO
- **Justificacion**: Es la version mas completa y util para la tesis. Genera evidencia visual (RAW vs WARPED) que puede incluirse directamente en el documento. Las metricas SSIM/PSNR cuantifican la transformacion geometrica. La categorizacion TP/TN/FP/FN permite analisis cualitativo de errores. Es el unico script que produce material publicable mas alla de un numero de accuracy.

---

### 9. thesis_validation_fisher_strict.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/thesis_validation_fisher_strict.py
- **Lineas/Tamano**: 236 lineas / 8.4 KB
- **Proposito**: Implementacion estricta de las instrucciones del asesor de tesis: PCA sobre pixeles crudos (centrados), estandarizacion de ponderantes (no pixeles), Fisher sobre ponderantes estandarizados, amplificacion con sqrt(J).
- **Contenido clave**:
  - `TorchAnalysis.fit_pca_raw()`: PCA explicito sobre datos centrados (no escalados)
  - `TorchAnalysis.standardize_weights()`: Estandarizacion feature-wise de los ponderantes PCA
  - Pipeline explicito y limpio: fit_pca_raw -> standardize_weights -> fisher_score -> sqrt(J) amplificacion -> k-NN
  - CLAHE habilitado por defecto
  - No genera visualizaciones ni reportes detallados
  - Es la implementacion GPU mas limpia del metodo del asesor
- **Importancia**: ALTO
- **Justificacion**: Es la implementacion canonica del "Metodo del Asesor" con la logica mas clara y modular. El codigo separa explicitamente cada paso del pipeline en metodos nombrados (`fit_pca_raw`, `standardize_weights`). Sirve como referencia de implementacion para documentar en la tesis. La ausencia de visualizaciones es una limitacion menor comparada con thesis_validation_fisher.py.

---

### 10. study_pca_real_data.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/studies/study_pca_real_data.py
- **Lineas/Tamano**: 74 lineas / 2.4 KB
- **Proposito**: Estudio visual comparativo que demuestra la importancia de centrar los datos antes de PCA: genera eigenfaces con y sin centrado sobre imagenes reales del dataset warped.
- **Contenido clave**:
  - PCA "malo" (sin centrar): SVD directo sobre pixeles raw -> primer componente captura el promedio
  - PCA "bueno" (centrado): SVD sobre (X - mean) -> primer componente captura la variacion real (forma)
  - Genera imagen comparativa triple: imagen promedio, eigenface sin centrar, eigenface centrada
  - Usa 100 imagenes a 128x128 del subset train
  - Output: `results/figures/pca_real_comparison.png`
- **Importancia**: MEDIO
- **Justificacion**: Genera una figura didactica muy util para explicar PCA en la tesis. La comparacion visual centrado vs sin centrar es intuitiva y pedagogicamente valiosa. Es un script auxiliar corto y enfocado.

---

### 11. study_variance_visual.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/fisher/studies/study_variance_visual.py
- **Lineas/Tamano**: 80 lineas / 2.6 KB
- **Proposito**: Estudio visual que compara la imagen promedio de radiografias RAW vs WARPED para demostrar que el warping reduce la varianza geometrica (ghosting), lo cual mejora la eficiencia de PCA.
- **Contenido clave**:
  - Acumula y promedia 100 imagenes RAW y 100 WARPED del mismo subset
  - Hipotesis visual: promedio RAW es borroso (alta varianza geometrica) vs promedio WARPED es nitido (geometria alineada)
  - Justifica teoricamente por que PCA funciona mejor sobre datos warped
  - Output: `results/figures/variance_ghosting_proof.png`
  - Requiere acceso tanto al dataset raw como al warped
- **Importancia**: ALTO
- **Justificacion**: Esta es la evidencia visual mas directa y convincente de que el warping geometrico cumple su objetivo: reducir varianza posicional. La figura generada ("ghosting proof") es candidata principal para inclusion en la tesis. Demuestra la hipotesis central del proyecto de forma intuitiva. A pesar de ser solo 80 lineas, produce evidencia de alto valor.

---

## Resumen de Importancia

| Archivo | Lineas | Importancia | Rol |
|---------|--------|-------------|-----|
| thesis_validation_fisher.py | 404 | ALTO | Validacion completa con evidencia forense visual |
| thesis_validation_fisher_strict.py | 236 | ALTO | Implementacion canonica del metodo del asesor |
| thesis_validation_fisher_multiclass.py | 294 | ALTO | Extension a 3 clases con F-statistic ANOVA |
| study_variance_visual.py | 80 | ALTO | Prueba visual ghosting RAW vs WARPED |
| thesis_fisher_tutorial.py | 292 | MEDIO | Tutorial didactico NumPy para defensa |
| thesis_validation_fisher_basic.py | 239 | MEDIO | Baseline CPU con scikit-learn |
| thesis_optimization_fisher.py | 261 | MEDIO | Grid search CLAHE |
| study_pca_real_data.py | 74 | MEDIO | Estudio visual PCA centrado vs no centrado |
| thesis_seed_search.py | 173 | BAJO | Busqueda de semilla optima (metodologicamente dudoso) |
| thesis_validation_fisher_basic_standard.py | 132 | BAJO | Variante menor del orden de estandarizacion |
| thesis_validation_fisher_massive_cpu.py | 235 | BAJO | Copia de basic con argparse |

## Observaciones Criticas

### 1. Duplicacion de Codigo Masiva
Los 9 scripts principales comparten ~70% de su codigo. Las funciones `cargar_datos_simples()`, `criterio_fisher_manual()`, `DatasetLoader`, `TorchAnalysis.fisher_score()`, y `TorchAnalysis.knn_predict()` se repiten con variaciones minimas. Un refactor a un unico script parametrizable (o un modulo compartido) eliminaria ~1,500 lineas redundantes.

### 2. Inconsistencia en la Amplificacion (J vs sqrt(J))
- Scripts "basic" (CPU): Multiplican por J directo (mas agresivo)
- Scripts "strict" (GPU): Multiplican por sqrt(J) (mas suave)
- Esta diferencia no esta documentada formalmente y dificulta la comparacion de resultados entre versiones

### 3. Inconsistencia en la Estandarizacion
- Version "basic" y "standard": Estandarizan pixeles ANTES de PCA o ponderantes DESPUES
- Version "strict": PCA sobre datos centrados (no escalados), luego estandariza ponderantes
- Version "multiclass": Estandariza pixeles antes de PCA (contradice la version "strict")

### 4. Seed Search como Practica Cuestionable
El script `thesis_seed_search.py` busca la semilla que maximiza el accuracy. Esto es efectivamente data snooping ya que el resultado depende de la semilla de `torch.pca_lowrank` (un algoritmo aproximado). Los resultados con seed=8 no deberian citarse como evidencia de rendimiento generalizable.

### 5. Ninguno de los Scripts Tiene Tests
No existen tests unitarios para las funciones Fisher, PCA manual, o k-NN manual. Dado que estos scripts implementan matematicas desde cero, la ausencia de tests es un riesgo de errores silenciosos.

### 6. Relacion con el Pipeline Principal
Estos scripts son **complementarios y ortogonales** al pipeline principal de la tesis (landmark detection -> warping -> CNN classification). El pipeline principal alcanza 99.10% accuracy con ResNet-18 CNN. Los scripts Fisher demuestran que incluso un metodo lineal simple (PCA + Fisher + k-NN) puede discriminar clases en el espacio de imagenes warped (~86% accuracy), lo cual valida la hipotesis de que el warping geometrico mejora la separabilidad.

### 7. Recomendacion de Consolidacion
Para mantener a largo plazo, los 9 scripts principales podrian consolidarse en 2-3:
- `fisher_validation.py`: Script unificado con flags para CPU/GPU, binario/multiclase, version basic/strict, con/sin evidencia visual
- `fisher_optimization.py`: Grid search unificado
- `studies/`: Mantener los 2 scripts de estudio visual tal cual (son cortos y enfocados)
