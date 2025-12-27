# thesis_validation_fisher_basic.py
# -----------------------------------------------------------------------------
# ESTA ES LA VERSIÓN DIDÁCTICA (Paso 1 del desarrollo)
# Objetivo: Comprender y validar la hipótesis geométrica usando herramientas estándar.
# Librerías: CPU solamente (NumPy, Scikit-Learn, OpenCV).
# -----------------------------------------------------------------------------

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Configuración básica
IMAGE_SIZE = 224
COMPONENTS = 50  # Número de componentes principales a analizar


def cargar_datos_simples(image_root_dir, split_name):
    """
    Carga imágenes de forma sencilla leyendo un CSV.
    Sin optimizaciones complejas, solo un bucle for estándar.
    """
    # Construir ruta al CSV específico del split
    csv_path = Path(image_root_dir) / split_name / "images.csv"
    print(f"\n[1] Cargando datos del conjunto '{split_name}' desde {csv_path}...")

    if not csv_path.exists():
        print(f"Error: No se encontró el archivo CSV en {csv_path}")
        return np.array([]), np.array([])

    # 1. Leer el archivo CSV que tiene la lista de imágenes
    df = pd.read_csv(csv_path)

    # Filtrar solo las filas que corresponden al split deseado (train o test)
    if "split" in df.columns:
        df = df[df["split"] == split_name]

    imagenes = []
    etiquetas = []

    # 2. Recorrer el dataframe fila por fila (Iteración estándar)
    # Usamos un contador simple para informar progreso
    total = len(df)
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"    Procesando imagen {i}/{total}...", end="\r")

        # Obtener nombre y categoría
        nombre = row["image_name"]
        categoria = row["category"]

        # Determinar el nombre del archivo en disco
        # El CSV de warped dataset suele tener la columna 'warped_filename'
        if "warped_filename" in row:
            nombre_archivo = row["warped_filename"]
        else:
            nombre_archivo = f"{nombre}.png"

        # Construir ruta (asumimos estructura estándar: root/categoria/nombre.png)
        # Intentamos varias rutas comunes por si acaso
        posibles_rutas = [
            Path(image_root_dir)
            / split_name
            / categoria
            / nombre_archivo,  # Estructura completa
            Path(image_root_dir) / categoria / nombre_archivo,  # Estructura simple
            Path(image_root_dir) / nombre_archivo,  # Estructura plana
        ]

        ruta_valida = None
        for p in posibles_rutas:
            if p.exists():
                ruta_valida = p
                break

        if ruta_valida:
            # 3. Lectura y Preprocesamiento Básico
            img = cv2.imread(str(ruta_valida), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # Aplanar la imagen: De matriz 2D a vector 1D
            # Ejemplo: 224x224 -> Vector de 50,176 números
            vector_imagen = img.flatten()

            imagenes.append(vector_imagen)
            # Etiqueta: 0 = Normal, 1 = Neumonía/COVID
            etiquetas.append(0 if categoria == "Normal" else 1)

    print(f"    Carga completada: {len(imagenes)} imágenes encontradas.")
    return np.array(imagenes), np.array(etiquetas)


def criterio_fisher_manual(X_pca, y):
    """
    Implementación MANUAL del Criterio de Fisher.
    Objetivo: Calcular qué componentes separan mejor las clases.

    Fórmula: J = (media1 - media2)^2 / (varianza1 + varianza2)
    """
    print("\n[3] Calculando Criterio de Fisher (Manual)...")

    clases = np.unique(y)  # [0, 1]
    fisher_ratios = []

    # X_pca es una matriz donde cada columna es un Componente Principal
    num_componentes = X_pca.shape[1]

    for i in range(num_componentes):
        # Extraer la columna i-ésima (el componente i)
        componente = X_pca[:, i]

        # Separar los datos por clase
        datos_clase_0 = componente[y == 0]  # Sano
        datos_clase_1 = componente[y == 1]  # Enfermo

        # Calcular medias (promedios)
        media_0 = np.mean(datos_clase_0)
        media_1 = np.mean(datos_clase_1)

        # Calcular varianzas (dispersión)
        var_0 = np.var(datos_clase_0)
        var_1 = np.var(datos_clase_1)

        # APLICAR LA FÓRMULA DE FISHER
        numerador = (media_0 - media_1) ** 2
        denominador = (
            var_0 + var_1 + 1e-9
        )  # Pequeño valor para evitar división por cero

        J = numerador / denominador
        fisher_ratios.append(J)

    return np.array(fisher_ratios)


def main():
    print("=" * 60)
    print("VALIDACIÓN GEOMÉTRICA - VERSIÓN DIDÁCTICA (CPU/NumPy)")
    print("=" * 60)

    # Rutas (Ajustables según tu entorno)
    dataset_dir = "outputs/warped_dataset"  # Usamos el dataset procesado

    # 1. Cargar Datos REALES (Train para aprender, Test para evaluar)
    print("\n--- FASE 1: CARGA DE DATOS ---")
    X_train, y_train = cargar_datos_simples(dataset_dir, "train")
    X_test, y_test = cargar_datos_simples(dataset_dir, "test")

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Faltan datos. Verifica las rutas.")
        return

    # 2. PCA (Obtención de Ponderantes/Características)
    # EL ASESOR DICE: "Construyes un solo Eigen-space... las características serían los ponderantes."
    print("\n--- FASE 2: EXTRACCIÓN DE CARACTERÍSTICAS (PCA) ---")
    print(f"[2] Calculando Eigenfaces y Ponderantes (PCA n={COMPONENTS})...")
    # Nota: Sklearn PCA centra los datos automáticamente (resta la media de píxeles), lo cual es correcto.
    pca = PCA(n_components=COMPONENTS)

    # Obtenemos los "Ponderantes Crudos" (Weights)
    X_train_weights = pca.fit_transform(X_train)
    X_test_weights = pca.transform(X_test)

    print(
        f"    Varianza explicada acumulada: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
    )

    # 3. Estandarización de Ponderantes
    # EL ASESOR DICE: "Tienes 1,000 valores para la característica 1... vamos a estandarizar solo la característica 1."
    print("\n--- FASE 3: ESTANDARIZACIÓN DE PONDERANTES ---")
    print("[3] Estandarizando cada característica (Media=0, Var=1)...")
    scaler = StandardScaler()

    # Estandarizamos los PESOS, no los píxeles
    X_train_std = scaler.fit_transform(X_train_weights)
    X_test_std = scaler.transform(X_test_weights)

    # 4. Análisis de Fisher (Sobre Ponderantes Estandarizados)
    # EL ASESOR DICE: "Bueno, ya que la estandarizaste... aplicas el Criterio de Fisher."
    print("\n--- FASE 4: CRITERIO DE FISHER ---")
    fisher_scores = criterio_fisher_manual(X_train_std, y_train)

    # Visualizar los Scores de Fisher
    plt.figure(figsize=(10, 5))
    plt.bar(range(COMPONENTS), fisher_scores)
    plt.title(f"Importancia Discriminante (Fisher) - {Path(dataset_dir).name}")
    plt.xlabel("Componente Principal (Ponderante)")
    plt.ylabel("Fisher Ratio")
    plt.savefig("fisher_scores_basic.png")
    print("    Gráfico guardado en 'fisher_scores_basic.png'")

    # --- AMPLIFICACIÓN (MÉTODO DEL ASESOR) ---
    print("\n--- FASE 5: AMPLIFICACIÓN (MÉTODO DEL ASESOR) ---")
    print("[5] Amplificando Ponderantes Estandarizados con Fisher Scores...")
    # Multiplicamos: Característica_Std * Fisher_Score
    X_train_final = X_train_std * fisher_scores
    X_test_final = X_test_std * fisher_scores

    # 5. Clasificación
    print("\n--- FASE 6: CLASIFICACIÓN ---")
    print("[6] Entrenando clasificador k-NN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train_final, y_train)
    print("    Evaluando en conjunto de TEST...")
    preds = knn.predict(X_test_final)

    acc = accuracy_score(y_test, preds)
    print("\n" + "#" * 40)
    print(f"RESULTADO FINAL REAL: Accuracy = {acc * 100:.2f}%")
    print("#" * 40)

    # Matriz de confusión
    cm = confusion_matrix(y_test, preds)
    print("\nMatriz de Confusión:")
    print(cm)

    # Guardar métricas básicas
    with open("basic_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")


if __name__ == "__main__":
    main()
