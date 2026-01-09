"""
Modulo de Clasificacion KNN (K-Nearest Neighbors).

FASE 6 DEL PIPELINE FISHER-WARPING
===================================

Este modulo implementa:
1. Clasificador KNN desde cero (sin sklearn)
2. Seleccion de K optimo usando validacion
3. Evaluacion con metricas: accuracy, precision, recall, F1
4. Visualizacion de matriz de confusion

MATEMATICAS DE KNN
==================

1. EL PROBLEMA
--------------
Tenemos N muestras de entrenamiento, cada una con:
- Vector de caracteristicas: x_i = (x_i1, x_i2, ..., x_iK) en R^K
- Etiqueta de clase: y_i en {0, 1} (Enfermo o Normal)

Dado un nuevo punto x_query, queremos predecir su clase.

2. LA INTUICION
---------------
"Dime con quien andas y te dire quien eres"

Si los K vecinos mas cercanos de x_query son mayormente de clase 0,
entonces x_query probablemente es de clase 0.

Es como preguntar: "De mis K vecinos mas cercanos, cuantos son Enfermos
y cuantos son Normales?" y elegir la clase mayoritaria.

3. DISTANCIA EUCLIDIANA
-----------------------
Para encontrar "cercania", usamos la distancia euclidiana:

    d(x, y) = sqrt( sum_i (x_i - y_i)^2 )

En 2D, esto es el teorema de Pitagoras.
En K dimensiones, es la generalizacion natural.

EJEMPLO:
    x = (1, 2, 3)
    y = (4, 5, 6)
    d(x, y) = sqrt((1-4)^2 + (2-5)^2 + (3-6)^2)
            = sqrt(9 + 9 + 9)
            = sqrt(27)
            = 5.196

4. EL ALGORITMO
---------------
Para clasificar un punto x_query:

    PASO 1: Calcular distancia a TODOS los puntos de entrenamiento
            d_i = d(x_query, x_i) para i = 1, 2, ..., N

    PASO 2: Ordenar las distancias de menor a mayor
            Obtener los indices de los K mas cercanos

    PASO 3: Contar votos de cada clase entre los K vecinos
            votos_clase_0 = numero de vecinos con y_i = 0
            votos_clase_1 = numero de vecinos con y_i = 1

    PASO 4: Asignar la clase mayoritaria
            prediccion = clase con mas votos

5. SELECCION DE K
-----------------
- K muy pequeno (ej. K=1): Muy sensible a ruido
- K muy grande (ej. K=N): Siempre predice la clase mayoritaria
- K optimo: Balance entre bias y varianza

Estrategia: Probar varios valores de K y elegir el que maximiza
accuracy en el conjunto de VALIDACION (NO test).

6. EMPATES
----------
Si hay empate (ej. K=4, 2 votos por clase):
- Opcion 1: Elegir la clase del vecino mas cercano
- Opcion 2: Elegir la clase con menor indice (deterministico)
- Opcion 3: Elegir aleatoriamente

En esta implementacion usamos Opcion 1 (vecino mas cercano gana empate).

NOTA DEL ASESOR
---------------
> "Tengo una imagen nueva... pues calculale la diferencia
>  (distancia euclidiana) con respecto a todos. Y luego ordena
>  las distancias de la menor a la mayor. Y los primeros K
>  vecinos pues vota por ellos."

METRICAS DE EVALUACION
======================

Para clasificacion binaria (Enfermo=0, Normal=1):

                        Prediccion
                    Enfermo    Normal
    Real  Enfermo    TP         FN
          Normal     FP         TN

- Accuracy  = (TP + TN) / (TP + TN + FP + FN)
            = Proporcion de predicciones correctas

- Precision = TP / (TP + FP)
            = De todos los que predije como Enfermo, cuantos SI son Enfermo

- Recall    = TP / (TP + FN)
            = De todos los Enfermos reales, cuantos detecte

- F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
            = Media armonica entre Precision y Recall

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import Counter
import json


@dataclass
class ClassificationResult:
    """
    Resultado de la clasificacion.

    Atributos:
        y_true: Etiquetas reales (N,)
        y_pred: Etiquetas predichas (N,)
        accuracy: Proporcion de predicciones correctas
        precision: TP / (TP + FP) por clase
        recall: TP / (TP + FN) por clase
        f1: Media armonica de precision y recall por clase
        confusion_matrix: Matriz de confusion (n_classes, n_classes)
        class_names: Nombres de las clases
        k: Valor de K usado
    """
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    precision: Dict[int, float]
    recall: Dict[int, float]
    f1: Dict[int, float]
    confusion_matrix: np.ndarray
    class_names: List[str]
    k: int
    n_samples: int = field(init=False)

    def __post_init__(self):
        self.n_samples = len(self.y_true)

    def get_macro_f1(self) -> float:
        """Calcula F1 macro (promedio de F1 por clase)."""
        return np.mean(list(self.f1.values()))

    def get_weighted_f1(self) -> float:
        """Calcula F1 weighted (ponderado por soporte de cada clase)."""
        unique, counts = np.unique(self.y_true, return_counts=True)
        total = len(self.y_true)
        weighted = sum(self.f1[cls] * count / total for cls, count in zip(unique, counts))
        return weighted


@dataclass
class KOptimizationResult:
    """
    Resultado de la optimizacion de K.

    Atributos:
        k_values: Lista de valores de K probados
        val_accuracies: Accuracy en validacion para cada K
        best_k: Valor optimo de K
        best_val_accuracy: Accuracy en validacion con K optimo
    """
    k_values: List[int]
    val_accuracies: List[float]
    best_k: int
    best_val_accuracy: float


class KNNClassifier:
    """
    Clasificador K-Nearest Neighbors implementado desde cero (sin sklearn).

    Este clasificador usa distancia euclidiana y votacion mayoritaria
    para clasificar nuevas muestras basandose en sus K vecinos mas cercanos.

    Ejemplo de uso:
        >>> knn = KNNClassifier(k=5)
        >>> knn.fit(X_train, y_train)
        >>> predictions = knn.predict(X_test)
        >>> accuracy = knn.score(X_test, y_test)
    """

    def __init__(self, k: int = 5):
        """
        Inicializa el clasificador KNN.

        Args:
            k: Numero de vecinos a considerar (default: 5)
        """
        if k < 1:
            raise ValueError(f"K debe ser >= 1, pero se recibio k={k}")

        self.k = k
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.classes_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> 'KNNClassifier':
        """
        Entrena el clasificador KNN.

        En realidad KNN no "entrena" en el sentido tradicional.
        Solo almacena los datos de entrenamiento para usarlos
        durante la prediccion.

        Args:
            X: Matriz de caracteristicas (N, K)
            y: Vector de etiquetas (N,)
            verbose: Si True, imprime informacion

        Returns:
            self (para encadenar metodos)
        """
        N, K = X.shape

        if len(y) != N:
            raise ValueError(f"X tiene {N} muestras pero y tiene {len(y)}")

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.n_samples_ = N
        self.n_features_ = K
        self.classes_ = np.unique(y)
        self._fitted = True

        if self.k > N:
            if verbose:
                print(f"AVISO: k={self.k} > N={N}. Ajustando k={N}")
            self.k = N

        if verbose:
            print(f"KNN entrenado:")
            print(f"  Muestras: {N}")
            print(f"  Caracteristicas: {K}")
            print(f"  Clases: {self.classes_}")
            print(f"  K vecinos: {self.k}")

        return self

    def _compute_distances(self, X_query: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia euclidiana entre X_query y todos los puntos de entrenamiento.

        FORMULA:
            d(x, y) = sqrt( sum_i (x_i - y_i)^2 )

        Optimizacion: Usamos broadcasting de NumPy para calcular
        todas las distancias de forma vectorizada.

        Args:
            X_query: Matriz de consultas (M, K)

        Returns:
            Matriz de distancias (M, N) donde M=queries, N=training
        """
        # X_query: (M, K)
        # X_train: (N, K)
        # Resultado: (M, N)

        # Expansion: X_query[:, np.newaxis, :] tiene shape (M, 1, K)
        # Diferencia: (M, 1, K) - (N, K) = (M, N, K) por broadcasting
        # Suma de cuadrados: (M, N)
        # Raiz: (M, N)

        diff = X_query[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))

        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice las clases para nuevas muestras.

        ALGORITMO:
        1. Calcular distancias a todos los puntos de entrenamiento
        2. Para cada query, encontrar los K vecinos mas cercanos
        3. Contar votos de cada clase
        4. Asignar la clase mayoritaria (en empate, vecino mas cercano gana)

        Args:
            X: Matriz de consultas (M, K)

        Returns:
            Vector de predicciones (M,)
        """
        if not self._fitted:
            raise RuntimeError("KNN no ha sido entrenado. Llama a fit() primero.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X tiene {X.shape[1]} caracteristicas, pero KNN fue "
                f"entrenado con {self.n_features_} caracteristicas."
            )

        M = X.shape[0]
        predictions = np.zeros(M, dtype=self.y_train_.dtype)

        # Calcular todas las distancias de una vez
        distances = self._compute_distances(X)

        # Para cada muestra de consulta
        for i in range(M):
            # Obtener indices de los K vecinos mas cercanos
            # argsort ordena de menor a mayor, tomamos los primeros K
            neighbor_indices = np.argsort(distances[i])[:self.k]

            # Obtener etiquetas de los K vecinos
            neighbor_labels = self.y_train_[neighbor_indices]

            # Contar votos
            vote_counts = Counter(neighbor_labels)

            # Encontrar la clase con mas votos
            max_votes = max(vote_counts.values())
            winners = [cls for cls, count in vote_counts.items() if count == max_votes]

            if len(winners) == 1:
                # Caso normal: una clase gana
                predictions[i] = winners[0]
            else:
                # Empate: elegir la clase del vecino mas cercano
                for idx in neighbor_indices:
                    if self.y_train_[idx] in winners:
                        predictions[i] = self.y_train_[idx]
                        break

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predice probabilidades de clase para nuevas muestras.

        La "probabilidad" es simplemente la proporcion de votos de cada clase
        entre los K vecinos.

        Args:
            X: Matriz de consultas (M, K)

        Returns:
            Matriz de probabilidades (M, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("KNN no ha sido entrenado.")

        M = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((M, n_classes))

        distances = self._compute_distances(X)

        for i in range(M):
            neighbor_indices = np.argsort(distances[i])[:self.k]
            neighbor_labels = self.y_train_[neighbor_indices]

            for j, cls in enumerate(self.classes_):
                probas[i, j] = np.sum(neighbor_labels == cls) / self.k

        return probas

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula el accuracy en un conjunto de datos.

        Args:
            X: Matriz de caracteristicas (N, K)
            y: Vector de etiquetas reales (N,)

        Returns:
            Accuracy (proporcion de predicciones correctas)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None
) -> np.ndarray:
    """
    Calcula la matriz de confusion.

    FORMATO:
                    Prediccion
                    Clase 0    Clase 1
    Real  Clase 0    C[0,0]     C[0,1]
          Clase 1    C[1,0]     C[1,1]

    C[i, j] = numero de muestras de clase real i predichas como clase j

    Args:
        y_true: Etiquetas reales (N,)
        y_pred: Etiquetas predichas (N,)
        n_classes: Numero de clases (si None, se infiere de los datos)

    Returns:
        Matriz de confusion (n_classes, n_classes)
    """
    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    return cm


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Tuple[float, Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Calcula metricas de clasificacion.

    Args:
        y_true: Etiquetas reales (N,)
        y_pred: Etiquetas predichas (N,)
        class_names: Nombres de las clases

    Returns:
        Tuple de (accuracy, precision_dict, recall_dict, f1_dict)
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Accuracy global
    accuracy = np.mean(y_true == y_pred)

    # Metricas por clase
    precision = {}
    recall = {}
    f1 = {}

    for cls in classes:
        # True Positives: predichos como cls Y realmente son cls
        tp = np.sum((y_pred == cls) & (y_true == cls))

        # False Positives: predichos como cls PERO NO son cls
        fp = np.sum((y_pred == cls) & (y_true != cls))

        # False Negatives: NO predichos como cls PERO SI son cls
        fn = np.sum((y_pred != cls) & (y_true == cls))

        # Precision: TP / (TP + FP)
        if tp + fp > 0:
            precision[cls] = tp / (tp + fp)
        else:
            precision[cls] = 0.0

        # Recall: TP / (TP + FN)
        if tp + fn > 0:
            recall[cls] = tp / (tp + fn)
        else:
            recall[cls] = 0.0

        # F1: 2 * (P * R) / (P + R)
        if precision[cls] + recall[cls] > 0:
            f1[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls])
        else:
            f1[cls] = 0.0

    return accuracy, precision, recall, f1


def evaluate_classifier(
    knn: KNNClassifier,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    verbose: bool = True
) -> ClassificationResult:
    """
    Evalua el clasificador en un conjunto de datos.

    Args:
        knn: Clasificador KNN entrenado
        X: Caracteristicas (N, K)
        y: Etiquetas reales (N,)
        class_names: Nombres de las clases
        verbose: Si True, imprime resultados

    Returns:
        ClassificationResult con todas las metricas
    """
    y_pred = knn.predict(X)

    accuracy, precision, recall, f1 = compute_metrics(y, y_pred, class_names)
    cm = compute_confusion_matrix(y, y_pred, n_classes=len(class_names))

    result = ClassificationResult(
        y_true=y,
        y_pred=y_pred,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
        class_names=class_names,
        k=knn.k
    )

    if verbose:
        print(f"\nResultados de clasificacion (K={knn.k}):")
        print(f"  Muestras: {len(y)}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print()
        for cls in np.unique(y):
            print(f"  {class_names[cls]}:")
            print(f"    Precision: {precision[cls]:.4f}")
            print(f"    Recall:    {recall[cls]:.4f}")
            print(f"    F1-Score:  {f1[cls]:.4f}")
        print()
        print(f"  Macro F1:    {result.get_macro_f1():.4f}")
        print(f"  Weighted F1: {result.get_weighted_f1():.4f}")

    return result


def find_best_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_values: Optional[List[int]] = None,
    verbose: bool = True
) -> KOptimizationResult:
    """
    Encuentra el mejor valor de K usando validacion.

    Prueba varios valores de K y selecciona el que maximiza
    accuracy en el conjunto de validacion.

    Args:
        X_train: Caracteristicas de entrenamiento (N, K)
        y_train: Etiquetas de entrenamiento (N,)
        X_val: Caracteristicas de validacion (M, K)
        y_val: Etiquetas de validacion (M,)
        k_values: Lista de valores de K a probar (default: [1,3,5,7,9,11,15,21])
        verbose: Si True, imprime progreso

    Returns:
        KOptimizationResult con el mejor K y resultados
    """
    if k_values is None:
        # Valores tipicos de K (impares para evitar empates en binario)
        max_k = min(len(X_train), 51)
        k_values = [k for k in [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51] if k <= max_k]

    if verbose:
        print("Buscando K optimo...")
        print(f"  Valores a probar: {k_values}")
        print()

    val_accuracies = []

    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train, verbose=False)
        acc = knn.score(X_val, y_val)
        val_accuracies.append(acc)

        if verbose:
            print(f"  K={k:2d}: Val Accuracy = {acc:.4f} ({acc*100:.2f}%)")

    # Encontrar el mejor K
    best_idx = np.argmax(val_accuracies)
    best_k = k_values[best_idx]
    best_acc = val_accuracies[best_idx]

    if verbose:
        print()
        print(f"  Mejor K: {best_k} (Val Accuracy = {best_acc:.4f})")

    return KOptimizationResult(
        k_values=k_values,
        val_accuracies=val_accuracies,
        best_k=best_k,
        best_val_accuracy=best_acc
    )


def plot_confusion_matrix(
    result: ClassificationResult,
    output_path: Optional[Path] = None,
    normalize: bool = False,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la matriz de confusion.

    Args:
        result: Resultado de clasificacion
        output_path: Ruta para guardar la figura
        normalize: Si True, normaliza por fila (porcentajes)
        title: Titulo de la figura

    Returns:
        Figura de matplotlib
    """
    cm = result.confusion_matrix.copy()

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    n_classes = len(result.class_names)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Colormap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Etiquetas
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=result.class_names,
        yticklabels=result.class_names,
        ylabel='Clase Real',
        xlabel='Clase Predicha'
    )

    # Rotar etiquetas del eje X
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Agregar texto en cada celda
    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            text = f'{val:{fmt}}' if normalize else f'{int(val)}'
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        title_str = f'Matriz de Confusion (K={result.k})\nAccuracy: {result.accuracy:.2%}'
        ax.set_title(title_str, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Matriz de confusion guardada en: {output_path}")

    return fig


def plot_k_optimization(
    opt_result: KOptimizationResult,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza los resultados de la optimizacion de K.

    Args:
        opt_result: Resultado de la optimizacion
        output_path: Ruta para guardar la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Grafico de linea
    ax.plot(opt_result.k_values, opt_result.val_accuracies,
            'b-o', linewidth=2, markersize=8, label='Val Accuracy')

    # Marcar el mejor K
    best_idx = opt_result.k_values.index(opt_result.best_k)
    ax.plot(opt_result.best_k, opt_result.best_val_accuracy,
            'ro', markersize=8, label=f'Mejor K={opt_result.best_k}')

    # Linea horizontal en el mejor accuracy
    ax.axhline(y=opt_result.best_val_accuracy, color='red',
               linestyle='--', alpha=0.5)

    ax.set_xlabel('K (n\u00famero de vecinos)', fontsize=12)
    ax.set_ylabel('Accuracy en Validaci\u00f3n', fontsize=12)
    ax.set_title('Selecci\u00f3n de K \u00f3ptimo', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Etiquetas en el eje X
    ax.set_xticks(opt_result.k_values)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Optimizaci\u00f3n de K guardada en: {output_path}")

    return fig


def plot_metrics_comparison(
    results: Dict[str, ClassificationResult],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Compara metricas entre varios modelos/datasets.

    Args:
        results: Dict {nombre: ClassificationResult}
        output_path: Ruta para guardar la figura

    Returns:
        Figura de matplotlib
    """
    names = list(results.keys())
    n_models = len(names)

    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    values = []

    for name in names:
        res = results[name]
        values.append([
            res.accuracy,
            res.get_macro_f1(),
            res.get_weighted_f1()
        ])

    values = np.array(values)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.8 / n_models

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, (name, vals) in enumerate(zip(names, values)):
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i], alpha=0.8)

        # Agregar valores encima de las barras
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metrica', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparacion de Metricas', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Comparacion de metricas guardada en: {output_path}")

    return fig


def save_classification_results(
    result: ClassificationResult,
    dataset_name: str,
    output_path: Path
) -> None:
    """
    Guarda los resultados de clasificacion en CSV.

    Args:
        result: Resultado de clasificacion
        dataset_name: Nombre del dataset
        output_path: Ruta del archivo CSV
    """
    data = {
        'dataset': [dataset_name],
        'k': [result.k],
        'n_samples': [result.n_samples],
        'accuracy': [result.accuracy],
        'macro_f1': [result.get_macro_f1()],
        'weighted_f1': [result.get_weighted_f1()],
    }

    for cls in sorted(result.precision.keys()):
        class_name = result.class_names[cls]
        data[f'precision_{class_name}'] = [result.precision[cls]]
        data[f'recall_{class_name}'] = [result.recall[cls]]
        data[f'f1_{class_name}'] = [result.f1[cls]]

    df = pd.DataFrame(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Resultados guardados en: {output_path}")


def save_predictions(
    result: ClassificationResult,
    image_ids: List[str],
    output_path: Path
) -> None:
    """
    Guarda las predicciones individuales en CSV.

    Args:
        result: Resultado de clasificacion
        image_ids: Lista de IDs de imagen
        output_path: Ruta del archivo CSV
    """
    data = {
        'image_id': image_ids,
        'y_true': result.y_true,
        'y_pred': result.y_pred,
        'true_class': [result.class_names[y] for y in result.y_true],
        'pred_class': [result.class_names[y] for y in result.y_pred],
        'correct': result.y_true == result.y_pred
    }

    df = pd.DataFrame(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")


def load_amplified_features(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga caracteristicas amplificadas desde CSV.

    Args:
        csv_path: Ruta al archivo CSV (formato de fase 5)

    Returns:
        Tuple de (features, labels, image_ids)
    """
    df = pd.read_csv(csv_path)

    # Extraer columnas de features (PC1_amp, PC2_amp, ...)
    feature_cols = [col for col in df.columns if col.endswith('_amp')]
    features = df[feature_cols].values

    labels = df['label'].values
    image_ids = df['image_id'].tolist()

    return features, labels, image_ids


if __name__ == "__main__":
    """
    Script de prueba para el modulo de clasificacion.

    Demuestra:
    1. Creacion de datos sinteticos
    2. Entrenamiento de KNN
    3. Optimizacion de K
    4. Evaluacion
    5. Visualizacion
    """
    print("="*60)
    print("PRUEBA DEL MODULO DE CLASIFICACION KNN")
    print("="*60)
    print()

    # Crear datos sinteticos con dos clases separadas
    np.random.seed(42)

    # Clase 0: centrado en (-2, -2)
    # Clase 1: centrado en (+2, +2)
    n_train = 200
    n_val = 50
    n_test = 50

    # Training
    X_train_0 = np.random.randn(n_train // 2, 10) + np.array([-2] * 10)
    X_train_1 = np.random.randn(n_train // 2, 10) + np.array([2] * 10)
    X_train = np.vstack([X_train_0, X_train_1])
    y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))

    # Validation
    X_val_0 = np.random.randn(n_val // 2, 10) + np.array([-2] * 10)
    X_val_1 = np.random.randn(n_val // 2, 10) + np.array([2] * 10)
    X_val = np.vstack([X_val_0, X_val_1])
    y_val = np.array([0] * (n_val // 2) + [1] * (n_val // 2))

    # Test
    X_test_0 = np.random.randn(n_test // 2, 10) + np.array([-2] * 10)
    X_test_1 = np.random.randn(n_test // 2, 10) + np.array([2] * 10)
    X_test = np.vstack([X_test_0, X_test_1])
    y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))

    print("Datos sinteticos creados:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print()

    # Buscar K optimo
    opt_result = find_best_k(X_train, y_train, X_val, y_val)

    # Entrenar con K optimo
    print()
    print("-" * 60)
    knn = KNNClassifier(k=opt_result.best_k)
    knn.fit(X_train, y_train)

    # Evaluar en test
    class_names = ['Enfermo', 'Normal']
    result = evaluate_classifier(knn, X_test, y_test, class_names)

    print()
    print("-" * 60)
    print("Matriz de confusion:")
    print(result.confusion_matrix)

    print()
    print("Prueba completada.")
