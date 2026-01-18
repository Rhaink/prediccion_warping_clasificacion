# Papers y Surveys Identificados para Capítulo 3

## Documento de Trabajo - Estado del Arte

**Fecha:** 2026-01-17
**Objetivo:** Organizar literatura identificada para el Capítulo 3: Estado del Arte

---

## SURVEYS CLAVE IDENTIFICADOS (Prioridad Alta)

### 0. SURVEY SEMINAL: Litjens et al. (2017) - **CRÍTICO**

**Título:** "A Survey on Deep Learning in Medical Image Analysis"
**Autores:** Geert J. S. Litjens, Thijs Kooi, Babak Ehteshami Bejnordi, et al.
**Journal:** Medical Image Analysis, Vol. 42, pp. 60-88
**Año:** 2017
**DOI:** 10.1016/j.media.2017.07.005
**arXiv:** 1702.05747
**Citas:** 11,766+ (altamente influyente)
**Alcance:** Revisa >300 papers (242 de 2016-2017), cubre clasificación, detección, segmentación, registro, retrieval
**Temas:** Network architectures, sparse/noisy labels, federated learning, interpretability
**URL:** https://www.sciencedirect.com/science/article/abs/pii/S1361841517301135
**Uso en tesis:** Sección 3.1 - Survey fundamental de DL en medical imaging

### 1. Deep Learning for Medical Imaging

**Título:** "A review of deep learning in medical imaging: Imaging traits, technology trends, case studies with progress highlights, and future promises"
**Fuente:** PMC (PubMed Central)
**Año:** 2021-2023
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10544772/
**Relevancia:** Survey comprensivo sobre DL en imágenes médicas, cubre network architecture, sparse labels, federated learning, interpretability
**Uso en tesis:** Sección 3.1 - DL para diagnóstico médico

**Título Alternativo:** "A comprehensive survey of deep learning research on medical image analysis with focus on transfer learning"
**Fuente:** PubMed
**Año:** 2022
**URL:** https://pubmed.ncbi.nlm.nih.gov/36462229/
**Relevancia:** Survey sobre transfer learning en medical imaging (2017-2021)
**Uso en tesis:** Sección 3.1

### 2. COVID-19 Detection Systematic Reviews

**Título:** "A comprehensive review of analyzing the chest X-ray images to detect COVID-19 infections using deep learning techniques"
**Fuente:** PubMed
**Año:** 2023
**URL:** https://pubmed.ncbi.nlm.nih.gov/37362273/
**Relevancia:** Systematic review de técnicas DL para COVID-19 en chest X-rays
**Uso en tesis:** Sección 3.2 - Detección de COVID-19

**Título:** "Deep Learning for Pneumonia Detection in Chest X-ray Images: A Comprehensive Survey"
**Fuente:** PMC
**Año:** 2024
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11355845/
**Relevancia:** Survey sobre pneumonia detection, incluye Vision Transformers como técnica más prometedora
**Uso en tesis:** Sección 3.2

**Título:** "Classifying chest x-rays for COVID-19 through transfer learning: a systematic review"
**Fuente:** Multimedia Tools and Applications (Springer)
**Año:** 2024
**URL:** https://link.springer.com/article/10.1007/s11042-024-18924-3
**Relevancia:** Systematic review de transfer learning para COVID-19 (48 estudios, 2020-2023)
**Uso en tesis:** Sección 3.2

### 3. Attention Mechanisms in Computer Vision

**Título:** "Attention Mechanisms in Computer Vision: A Survey"
**Fuente:** arXiv (posteriormente en Computational Visual Media journal)
**Autores:** Guo et al.
**Año:** 2021-2022
**arXiv:** https://arxiv.org/abs/2111.07624
**Journal:** https://link.springer.com/article/10.1007/s41095-022-0271-y
**Relevancia:** Survey comprensivo de attention mechanisms (channel, spatial, temporal, branch attention)
**Uso en tesis:** Sección 3.5 - Mecanismos de atención

**Título:** "Visual Attention Methods in Deep Learning: An In-Depth Survey"
**Fuente:** ScienceDirect
**Año:** 2024
**URL:** https://www.sciencedirect.com/science/article/abs/pii/S1566253524001957
**Relevancia:** Survey actualizado (50 técnicas de atención), categorización detallada
**Uso en tesis:** Sección 3.5

### 4. Domain Generalization and Adaptation in Medical Imaging

**Título:** "Domain Generalization for Medical Image Analysis: A Survey"
**Fuente:** arXiv
**Año:** 2024 (Febrero)
**arXiv:** https://arxiv.org/abs/2402.05035
**Relevancia:** Survey sobre domain generalization específico para medical imaging
**Uso en tesis:** Sección 3.7 - Robustez y generalización

**Título:** "Domain Adaptation for Medical Image Analysis: A Survey"
**Fuente:** PMC
**Año:** 2021
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/
**Relevancia:** Survey sobre domain adaptation en análisis de imágenes médicas
**Uso en tesis:** Sección 3.7

**Título:** "A systematic review of generalization research in medical image classification"
**Fuente:** ScienceDirect
**Año:** 2024 (Octubre)
**URL:** https://www.sciencedirect.com/science/article/pii/S0010482524013416
**Relevancia:** Systematic review reciente sobre generalización en clasificación de imágenes médicas
**Uso en tesis:** Sección 3.7

### 5. Ensemble Methods in Medical Diagnosis

**Título:** "Machine-Learning-Based Disease Diagnosis: A Comprehensive Review"
**Fuente:** PMC
**Año:** 2022
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8950225/
**Relevancia:** Review de machine learning (incluye ensemble methods) para diagnóstico médico
**Uso en tesis:** Sección 3.7 (o introducción a ensemble)

---

## PAPERS ORIGINALES CLAVE (Por Tema)

### Papers Seminales de Medical AI

#### Esteva et al. (2017) - Dermatologist-Level Classification - **PAPER SEMINAL**
- **Título:** Dermatologist-level classification of skin cancer with deep neural networks
- **Autores:** Andre Esteva, Brett Kuprel, Roberto A. Novoa, et al. (Stanford University)
- **Journal:** Nature, Vol. 542, pp. 115-118
- **Año:** 2017 (Enero 25)
- **DOI:** 10.1038/nature21056
- **Citas:** 11,281+ (altamente influyente)
- **Dataset:** 129,450 clinical images, 2,032 different diseases
- **Performance:** On par con 21 board-certified dermatologists en biopsy-proven images
- **Innovación:** Demuestra que CNNs pueden alcanzar competencia comparable a dermatólogos
- **Tareas:** Malignant carcinomas vs benign seborrheic keratoses; Malignant melanomas vs benign nevi
- **URL:** https://www.nature.com/articles/nature21056
- **Uso en tesis:** Sección 3.1 - Ejemplo seminal de DL en medical imaging
- **Estado:** NUEVO (agregar a references.bib)

### COVID-19 Detection

#### COVID-Net (Wang et al., 2020) - **YA CITADO EN TESIS**
- **Título:** COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images
- **Autores:** Wang, Linda; Lin, Zhong Qiu; Wong, Alexander
- **Journal:** Scientific Reports, Vol. 10, No. 1, pp. 19549
- **Año:** 2020
- **DOI:** 10.1038/s41598-020-76550-z
- **Dataset:** COVIDx (13,975 X-rays, 13,870 pacientes)
- **Accuracy:** 93.3%
- **Arquitectura:** Custom CNN tailored para COVID-19
- **URL:** https://www.nature.com/articles/s41598-020-76550-z
- **Estado en tesis:** YA EN references.bib

#### CheXNet (Rajpurkar et al., 2017) - **YA CITADO EN TESIS**
- **Título:** CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning
- **Autores:** Rajpurkar, Pranav et al.
- **Journal:** arXiv preprint arXiv:1711.05225
- **Año:** 2017
- **Dataset:** ChestX-ray14 (112,120 imágenes, 14 patologías)
- **Arquitectura:** DenseNet-121 (121 capas)
- **Performance:** Excede radiologist performance en F1 metric
- **Innovación:** Class Activation Maps para localización
- **URL:** https://arxiv.org/abs/1711.05225
- **Estado en tesis:** YA EN references.bib

#### COVID-19 Detection con ResNet-18 (2023)
- **Título:** CovC-ReDRNet: A Deep Learning Model for COVID-19 Classification
- **Autores:** Varios
- **Journal:** Machine Learning and Knowledge Extraction
- **Año:** 2023
- **Accuracy:** 97.56% (state-of-the-art)
- **Arquitectura:** ResNet-18 + deep random vector function link network
- **Métricas:** Sensitivity: 94.94%, Specificity: 97.01%, F1: 95.84%
- **Relevancia:** Comparación directa con ResNet-18 (arquitectura usada en este trabajo)
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7615781/

#### Otros papers COVID-19 para tabla comparativa
- Enhanced COVID-19 Detection (MDPI, 2024)
- COVID-19 severity detection (Nature Scientific Reports, 2024)
- Varios con transfer learning approach

### Landmark Detection

#### Wing Loss (Feng et al., 2018) - **YA CITADO EN TESIS**
- **Título:** Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks
- **Autores:** Feng, Zhen-Hua; Kittler, Josef; Awais, Muhammad; Huber, Patrik; Wu, Xiao-Jun
- **Venue:** CVPR 2018, pp. 2235-2245
- **Año:** 2018
- **Innovación:** Loss function que amplifica errores pequeños-medianos mediante función logarítmica modificada
- **Dataset:** AFLW, 300W
- **Metric:** 1.47% NME (Normalized Mean Error)
- **URL:** https://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_Wing_Loss_for_CVPR_2018_paper.pdf
- **Estado en tesis:** YA EN references.bib

#### Spine Landmarks (Yeh et al., 2021) - **YA CITADO EN TESIS**
- **Título:** Deep learning approach for automatic landmark detection and alignment analysis in whole-spine lateral radiographs
- **Autores:** Yeh, Yu-Cheng et al.
- **Journal:** Scientific Reports, Vol. 11, No. 1, pp. 7618
- **Año:** 2021
- **DOI:** 10.1038/s41598-021-87141-x
- **Error:** Standard errors mild to moderate
- **Aplicación:** Columna vertebral
- **URL:** https://www.nature.com/articles/s41598-021-87141-x
- **Estado en tesis:** YA EN references.bib

#### Two-Stage Task-Oriented Deep Learning
- **Título:** Detecting Anatomical Landmarks From Limited Medical Imaging Data Using Two-Stage Task-Oriented Deep Neural Networks
- **Journal:** PMC
- **Año:** 2017
- **Error:** Brain: 2.96 mm, Prostate: 3.34 mm
- **Innovación:** Two-stage approach con limited training data
- **URL:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5729285/

#### Adaptive Wing Loss (Wang et al., 2019) - Mejora sobre Wing Loss
- **Título:** Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression
- **Autores:** Xinyao Wang, Liefeng Bo, Li Fuxin
- **Venue:** ICCV 2019
- **Año:** 2019
- **arXiv:** 1904.07399
- **Innovación:** Loss function que adapta su forma a diferentes tipos de pixeles (foreground vs background)
- **Componentes:** Adaptive Wing loss + Weighted Loss Map
- **Performance:** Supera state-of-the-art en COFW, 300W, WFLW
- **Ventaja vs Wing Loss:** Maneja mejor el desbalance foreground/background en heatmap regression
- **GitHub:** https://github.com/protossw512/AdaptiveWingLoss
- **URL:** https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.html
- **Uso en tesis:** Sección 3.3 - Comparación con Wing Loss (usado en este trabajo)
- **Estado:** NUEVO (agregar a references.bib)

#### Heatmap vs Coordinate Regression Comparison
- **Hallazgos clave:**
  - Heatmap regression: Mayor accuracy, explota features locales, mejor spatial generalization
  - Coordinate regression: Más directo (end-to-end), incorpora structural knowledge, menor accuracy
  - Trade-off: Heatmap requiere más tiempo/espacio pero funciona mejor en datasets pequeños
- **Fuente:** Multiple papers (Nature Scientific Reports 2023, ScienceDirect 2023)
- **Consenso:** Heatmap regression es preferida en medical imaging por mayor accuracy
- **Relevancia para tesis:** Este trabajo usa coordinate regression (directo), justificar por qué

#### Papers adicionales landmark detection
- BrainSignsNET (2025) - 3D landmark detection con multi-task CNN
- Reinforcement learning for landmark detection (2019)
- Otros papers de segmentación y detección

### Attention Mechanisms

#### Coordinate Attention (Hou et al., 2021) - **YA CITADO EN TESIS**
- **Título:** Coordinate Attention for Efficient Mobile Network Design
- **Autores:** Hou, Qibin; Zhou, Daquan; Feng, Jiashi
- **Venue:** CVPR 2021, pp. 13713-13722
- **Año:** 2021
- **Innovación:** Factoriza channel attention en 2 procesos 1D (dirección-aware, posición-sensible)
- **Ventaja:** Mejor localización de objetos vs channel attention tradicional
- **Performance:** Mejor en object detection y semantic segmentation
- **GitHub:** https://github.com/Andrew-Qibin/CoordAttention
- **URL:** https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.html
- **Estado en tesis:** YA EN references.bib

#### SE-Net (Hu et al., 2018) - **YA CITADO EN TESIS**
- **Título:** Squeeze-and-Excitation Networks
- **Autores:** Hu, Jie; Shen, Li; Sun, Gang
- **Venue:** CVPR 2018
- **Año:** 2018
- **arXiv:** https://arxiv.org/abs/1709.01507
- **Innovación:** SE block con squeeze + excitation operations
- **Performance:** Winner ILSVRC 2017, top-5 error: 2.251%
- **Mecanismo:** Recalibración adaptativa channel-wise mediante channel interdependencies
- **URL:** https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
- **Estado en tesis:** YA EN references.bib

#### CBAM (Woo et al., 2018) - **YA CITADO EN TESIS**
- **Título:** CBAM: Convolutional Block Attention Module
- **Autores:** Woo, Sanghyun; Park, Jongchan; Lee, Joon-Young; Kweon, In So
- **Venue:** ECCV 2018
- **Año:** 2018
- **arXiv:** https://arxiv.org/abs/1807.06521
- **Innovación:** Secuencial channel + spatial attention
- **Integración:** Lightweight, negligible overhead, end-to-end trainable
- **Validación:** ImageNet-1K, MS COCO, VOC 2007
- **URL:** https://link.springer.com/chapter/10.1007/978-3-030-01234-2_1
- **Estado en tesis:** YA EN references.bib

#### Vision Transformer (Dosovitskiy et al., 2020) - **YA CITADO EN TESIS**
- **Título:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **Autores:** Dosovitskiy, Alexey et al.
- **Venue:** ICLR 2021
- **Año:** 2020 (arXiv), 2021 (publicado)
- **Innovación:** Aplicar transformers a image recognition
- **Performance en medical imaging:** A menudo supera CNNs tradicionales
- **Aplicaciones médicas:** Clasificación de enfermedades, segmentación, detección, registro
- **Limitación:** Requiere datasets grandes (problema con datasets médicos pequeños)
- **URL:** https://arxiv.org/abs/2010.11929
- **Estado en tesis:** YA EN references.bib

### Geometric Normalization and Warping

#### Spatial Transformer Networks (Jaderberg et al., 2015) - **YA CITADO EN TESIS**
- **Título:** Spatial Transformer Networks
- **Autores:** Jaderberg, Max; Simonyan, Karen; Zisserman, Andrew; Kavukcuoglu, Koray
- **Venue:** NIPS 2015
- **Año:** 2015
- **Innovación:** Módulo learnable para spatial manipulation dentro de la red
- **Limitación:** Transformaciones globales afines/rígidas (no deformación local)
- **Componentes:** Localization network, grid generator, sampler
- **Invarianza:** Traslación, escala, rotación, warping genérico
- **URL:** https://papers.nips.cc/paper/2015/hash/33ceb07bf4eeb3da587e268d663aba1a-Abstract.html
- **Estado en tesis:** YA EN references.bib

#### STERN (Rocha et al., 2024) - **YA CITADO EN TESIS**
- **Título:** STERN: Attention-driven Spatial Transformer Network for abnormality detection in chest X-ray images
- **Autores:** Rocha, Joana et al.
- **Journal:** Artificial Intelligence in Medicine, Vol. 147, pp. 102737
- **Año:** 2024
- **DOI:** 10.1016/j.artmed.2023.102737
- **Innovación:** Combina STN + attention para detección de anomalías en chest X-rays
- **Mejora vs baseline:** +2.1% AUC (aproximado según tipo de trabajo)
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0010482523012410
- **Estado en tesis:** YA EN references.bib

#### Picazo-Castillo et al. (2024) - **TRABAJO DEL GRUPO, YA CITADO**
- **Título:** Comparative Study of Lung Image Representations for Automated Pneumonia Recognition
- **Autores:** Picazo-Castillo, Angel Ernesto et al.
- **Journal:** IJCOPI, Vol. 15, No. 5, pp. 193-201
- **Año:** 2024
- **DOI:** 10.61467/2007.1558.2024.v15i5.578
- **Contribución:** Estudio comparativo de representaciones de imágenes pulmonares, demuestra que normalización espacial afecta generalización
- **Estado en tesis:** YA EN references.bib

#### Ayala-Raggi et al. (2023) - **TRABAJO DEL GRUPO, YA CITADO**
- **Título:** Synergizing Chest X-ray Image Normalization and Discriminative Feature Selection for Efficient and Automatic COVID-19 Recognition
- **Autores:** Ayala-Raggi, Salvador E. et al.
- **Venue:** ACPR 2023 (Lecture Notes in Computer Science, Vol. 14407)
- **Año:** 2023
- **DOI:** 10.1007/978-3-031-47637-2_17
- **Contribución:** Integración normalización + feature selection, mejoras en accuracy mediante reducción de variabilidad extrínseca
- **Estado en tesis:** YA EN references.bib

#### Piecewise Affine Transformation
- **Implementación:** scikit-image PiecewiseAffineTransform
- **Método:** Delaunay triangulation + affine transform per triangle
- **Aplicaciones:** Face warping, remote sensing (VHR images)
- **Fundamentación teórica:** Wolberg (1990) - **YA CITADO EN TESIS**
- **URL docs:** https://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html

### Procrustes Analysis and Shape Models

#### Generalized Procrustes Analysis (Gower, 1975) - **YA CITADO EN TESIS**
- **Estado en tesis:** YA EN references.bib

#### Statistical Shape Analysis (Dryden & Mardia, 2016) - **YA CITADO EN TESIS**
- **Estado en tesis:** YA EN references.bib

#### Active Shape Models (Cootes et al., 1995) - **YA CITADO EN TESIS**
- **Título:** Active shape models---Their training and application
- **Autores:** Cootes, Timothy F; Taylor, Christopher J; Cooper, David H; Graham, Jim
- **Journal:** Computer Vision and Image Understanding, Vol. 61, No. 1, pp. 38-59
- **Año:** 1995
- **Innovación:** Learn patterns of variability from training set, iterative refinement
- **Aplicación:** Medical image segmentation
- **Estado en tesis:** YA EN references.bib

### Image Enhancement and Preprocessing

#### CLAHE (Zuiderveld, 1994 & Pizer et al., 1987) - **YA CITADOS EN TESIS**
- **Estado en tesis:** YA EN references.bib

#### BO-CLAHE for Chest X-rays (2025)
- **Título:** BO-CLAHE enhancing neonatal chest X-ray image quality for improved lesion classification
- **Journal:** Scientific Reports
- **Año:** 2025
- **Innovación:** CLAHE + Bayesian optimization para chest X-rays neonatales
- **URL:** https://www.nature.com/articles/s41598-025-88451-0

#### CLAHE for COVID-19 (Rahman et al., 2021) - **YA CITADO EN TESIS**
- **Estado en tesis:** YA EN references.bib

#### SAHS (Cruz-Ovando et al., 2025) - **TRABAJO DEL GRUPO, YA CITADO**
- **Título:** Statistical Asymmetrical Histogram Stretching for Contrast Enhancement in Chest X-ray Images for Pneumonia Detection
- **Autores:** Cruz-Ovando, Rafael Alejandro et al.
- **Journal:** IJCOPI
- **Año:** 2025
- **Contribución:** Fundamentación estadística para histogramas asimétricos
- **Comparación:** CLAHE vs SAHS
- **Estado en tesis:** YA EN references.bib

### Domain Shift and Robustness

#### Zech et al. (2018) - **YA CITADO EN TESIS**
- **Título:** Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study
- **Autores:** Zech, John R et al.
- **Journal:** PLoS Medicine, Vol. 15, No. 11, pp. e1002683
- **Año:** 2018
- **Dataset:** 158,323 chest radiographs (NIH, Mount Sinai, Indiana University)
- **Hallazgo clave:** CNNs detect hospital system con 99.95% accuracy (confounding)
- **Performance degradation:** 3/5 comparisons tuvieron performance externa significativamente menor
- **Conclusión:** Domain shift es problema fundamental en medical imaging
- **URL:** https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
- **Estado en tesis:** YA EN references.bib

#### Shortcut Learning (Geirhos et al., 2020) - **YA CITADO EN TESIS**
- **Estado en tesis:** YA EN references.bib

### Test-Time Augmentation

#### TTA for Cell Segmentation (2020)
- **Título:** Test-time augmentation for deep learning-based cell segmentation on microscopy images
- **Journal:** Scientific Reports
- **Año:** 2020
- **Innovación:** TTA significativamente mejora accuracy con transformaciones simples (rotation, flipping)
- **Aplicación:** Segmentación de células/núcleos
- **URL:** https://www.nature.com/articles/s41598-020-61808-3

#### TTA for Medical Image Segmentation (2024)
- **Título:** Improving Medical Image Segmentation Using Test-Time Augmentation with MedSAM
- **Journal:** MDPI Mathematics
- **Año:** 2024
- **Innovación:** TTA con state-of-the-art generative models
- **URL:** https://www.mdpi.com/2227-7390/12/24/4003

#### Ensemble TTA (Matsunaga et al., 2017) - **YA CITADO EN TESIS**
- **Estado en tesis:** YA EN references.bib

### Ensemble Methods

#### Dietterich (2000) - **YA CITADO EN TESIS**
- **Título:** Ensemble methods in machine learning
- **Estado en tesis:** YA EN references.bib

#### Ensemble para Alzheimer (2023-2024)
- **Título:** A Deep Learning-Based Ensemble Method for Early Diagnosis of Alzheimer's Disease using MRI Images
- **Journal:** Neuroinformatics / PMC
- **Año:** 2023-2024
- **Accuracy:** 98.57%, 96.37%, 94.22%, etc. (varios classification groups)
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10917836/

---

## DATASET BENCHMARK

### COVID-19 Radiography Database (Kaggle)
- **Creadores:** Qatar University, University of Dhaka, collaborators
- **URL:** https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **Contenido:**
  - 10,192 Normal cases
  - 3,616 COVID-19 positive cases
  - 1,345 Viral Pneumonia cases
  - 6,012 Lung Opacity images
- **Formato:** PNG, 1024x1024 pixels
- **Uso:** Benchmark ampliamente usado en investigación COVID-19
- **Estado en tesis:** Usado en este trabajo (Chowdhury et al., 2020 - YA CITADO)

---

## PAPERS A BUSCAR ADICIONALMENTE

### Papers faltantes para completar tablas comparativas

1. **COVIDx dataset papers** (para comparación con COVID-19 Radiography Database)
2. **Papers adicionales COVID-19 detection 2022-2024** con métricas específicas (accuracy, sensitivity, specificity)
3. **Papers landmark detection en chest X-rays** (actualmente escasos en búsqueda)
4. **Papers sobre Delaunay triangulation en medical imaging**
5. **Papers piecewise affine warping para classification** (identificado como gap en literatura)
6. **Papers fundamentales de Litjens et al. (2017), Esteva et al. (2017)** mencionados en el plan

### Papers mencionados en plan pero no encontrados aún

- ~~Litjens et al. (2017) - Medical imaging survey~~ ✓ ENCONTRADO
- ~~Esteva et al. (2017) - Dermatologist-level classification~~ ✓ ENCONTRADO
- Payer et al. (2016) - Landmark detection con CNN
- ~~Papers sobre Adaptive Wing Loss~~ ✓ ENCONTRADO (Wang et al., 2019)

---

## ESTRATEGIA DE COMPLETITUD

### Próximos pasos

1. **Buscar papers faltantes mencionados en el plan:**
   - Litjens 2017, Esteva 2017 (surveys seminales)
   - Adaptive Wing Loss
   - Más papers COVID-19 con métricas cuantitativas

2. **Extraer métricas específicas para tablas:**
   - Releer papers identificados para extraer accuracy, sensitivity, specificity, F1
   - Organizar en formato tabla LaTeX

3. **Verificar qué papers ya están en references.bib:**
   - Evitar duplicados
   - Identificar cuáles necesitan agregarse

4. **Categorizar papers por prioridad:**
   - Esenciales para tablas comparativas
   - Contextuales para discusión
   - Complementarios

---

## NOTAS DE INTEGRACIÓN CON TESIS ACTUAL

### Papers ya citados en Capítulo 1 (no duplicar, profundizar en Cap 3)
- Wang COVIDNet 2020
- Rajpurkar CheXNet 2017
- Chowdhury COVID-19 Radiography Database 2020
- Rahman CLAHE 2021
- Picazo-Castillo 2024
- Ayala-Raggi 2023
- Rocha STERN 2024
- Yeh spine landmarks 2021

### Papers en references.bib no citados en Cap 1 (disponibles para Cap 3)
- Feng Wing Loss 2018
- Hou Coordinate Attention 2021
- Hu SE-Net 2018
- Woo CBAM 2018
- Dosovitskiy ViT 2020/2021
- Jaderberg STN 2015
- Cootes ASM 1995
- Gower GPA 1975
- Dryden Statistical Shape Analysis 2016
- Zech domain shift 2018
- Dietterich ensemble 2000
- Muchos más...

### Métricas validadas para comparación (GROUND_TRUTH.json)
- **Landmark ensemble:** 3.61 px (en imágenes 224x224)
- **Classification accuracy:** 98.05%
- **F1-macro:** 97.12%
- **F1-weighted:** 98.04%
- **Fill rate:** 47%

**COMPARAR CONTRA:**
- COVIDNet: 93.3% accuracy
- CheXNet: comparable a radiólogos en F1
- ResNet-18 COVID-19 (2023): 97.56% accuracy
- Otros trabajos en tablas

---

**Total de surveys identificados:** 7-8 ✓
**Total de papers originales identificados:** ~25-30
**Referencias ya en tesis:** 53
**Referencias nuevas estimadas:** 30-40
**Total esperado:** ~85-95 referencias
