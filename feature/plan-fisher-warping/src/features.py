"""
Módulo de extracción de características y estandarización.

FASE 4 DEL PIPELINE FISHER-WARPING
==================================

Este módulo implementa:
1. Extracción de ponderantes (proyección PCA)
2. Estandarización Z-score
3. Verificación estadística

MATEMÁTICAS DE LA ESTANDARIZACIÓN Z-SCORE
==========================================

1. EL PROBLEMA
--------------
Después de PCA, tenemos K ponderantes por imagen. Pero cada ponderante
tiene escala diferente:
- Ponderante 1 (PC1): valores entre -1000 y +1000
- Ponderante 2 (PC2): valores entre -500 y +500
- ...
- Ponderante K (PCK): valores entre -0.01 y +0.01

Si usamos estos valores directamente para clasificar (ej. KNN con distancia
euclidiana), los ponderantes de escala grande dominarán el cálculo,
independientemente de su importancia real.

2. LA SOLUCIÓN: Z-SCORE
-----------------------
Para cada característica i, calculamos con los datos de TRAINING:

    μᵢ = (1/N) Σⱼ xᵢⱼ          (media de la característica i)
    σᵢ = √[(1/N) Σⱼ (xᵢⱼ - μᵢ)²]  (desviación estándar)

Luego transformamos TODOS los datos (train, val, test):

    zᵢⱼ = (xᵢⱼ - μᵢ) / σᵢ

Después de la estandarización, cada característica tiene:
- Media = 0
- Desviación estándar = 1

3. IMPORTANCIA DE USAR SOLO TRAINING
------------------------------------
NUNCA calculamos μ y σ con datos de validación o test.
Esto causaría "data leakage" (fuga de información):
- Estaríamos usando información del futuro para transformar datos
- El modelo tendría acceso indirecto a datos de test
- Las métricas de evaluación serían optimistas (falsas)

Lo correcto:
- Calcular μ y σ SOLO con training
- Aplicar la misma transformación a val y test
- Los valores estandarizados de val/test NO tendrán media=0 exacta (¡y está bien!)

NOTA DEL ASESOR
---------------
> "De esos 1,000 valores sacas la media. Y le sacas la desviación estándar...
>  a cada valor le restas la media y luego esa diferencia la divides entre
>  la desviación estándar."
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ScalerResult:
    """
    Resultado del ajuste del StandardScaler.

    Atributos:
        mean: Media por característica (K,)
        std: Desviación estándar por característica (K,)
        n_features: Número de características K
        n_samples_fit: Número de muestras usadas para ajustar
    """
    mean: np.ndarray
    std: np.ndarray
    n_features: int
    n_samples_fit: int


@dataclass
class FeaturesResult:
    """
    Resultado de la extracción de características.

    Contiene ponderantes estandarizados para train, val, test.
    """
    train_features: np.ndarray  # (N_train, K)
    val_features: np.ndarray    # (N_val, K)
    test_features: np.ndarray   # (N_test, K)
    train_labels: np.ndarray    # (N_train,)
    val_labels: np.ndarray      # (N_val,)
    test_labels: np.ndarray     # (N_test,)
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    scaler_params: ScalerResult
    n_components: int


class StandardScaler:
    """
    Estandarización Z-score implementada desde cero (sin sklearn).

    Esta implementación calcula media y desviación estándar SOLO con
    los datos de entrenamiento, luego aplica la misma transformación
    a cualquier conjunto de datos.

    Ejemplo de uso:
        >>> scaler = StandardScaler()
        >>> scaler.fit(X_train)
        >>> X_train_scaled = scaler.transform(X_train)
        >>> X_test_scaled = scaler.transform(X_test)  # Usa media/std de train
    """

    def __init__(self):
        """Inicializa el StandardScaler."""
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: int = 0
        self.n_samples_fit_: int = 0
        self._fitted = False

    def fit(self, X: np.ndarray, verbose: bool = True) -> 'StandardScaler':
        """
        Calcula media y desviación estándar de los datos de entrenamiento.

        FÓRMULAS:
        - Media: μᵢ = (1/N) Σⱼ xᵢⱼ
        - Std:   σᵢ = √[(1/N) Σⱼ (xᵢⱼ - μᵢ)²]

        Usamos N (no N-1) para consistencia con el estándar de ML.

        Args:
            X: Matriz de datos (N, K) donde N=muestras, K=características
            verbose: Si True, imprime información

        Returns:
            self (para encadenar métodos)
        """
        N, K = X.shape

        if verbose:
            print(f"Ajustando StandardScaler...")
            print(f"  Datos: {N} muestras × {K} características")

        # Calcular media por característica (columna)
        # μ = (1/N) Σ xᵢ
        self.mean_ = np.mean(X, axis=0)

        # Calcular desviación estándar por característica
        # σ = √[(1/N) Σ (xᵢ - μ)²]
        self.std_ = np.std(X, axis=0)

        # Evitar división por cero: si std=0, usar 1
        # Esto ocurre cuando una característica es constante
        zero_std_mask = self.std_ == 0
        if np.any(zero_std_mask):
            n_zero = np.sum(zero_std_mask)
            if verbose:
                print(f"  AVISO: {n_zero} características con std=0 (constantes)")
            self.std_[zero_std_mask] = 1.0

        self.n_features_ = K
        self.n_samples_fit_ = N
        self._fitted = True

        if verbose:
            print(f"  Media: min={self.mean_.min():.4f}, max={self.mean_.max():.4f}")
            print(f"  Std: min={self.std_.min():.4f}, max={self.std_.max():.4f}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica la transformación Z-score a los datos.

        FÓRMULA:
            z = (x - μ) / σ

        donde μ y σ son los calculados con fit() (solo training).

        Args:
            X: Matriz de datos (N, K)

        Returns:
            Matriz estandarizada (N, K)
        """
        if not self._fitted:
            raise RuntimeError("StandardScaler no ha sido ajustado. Llama a fit() primero.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X tiene {X.shape[1]} características, pero el scaler fue "
                f"ajustado con {self.n_features_} características."
            )

        # z = (x - μ) / σ
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Ajusta el scaler y transforma en un solo paso.

        Args:
            X: Matriz de datos (N, K)
            verbose: Si True, imprime información

        Returns:
            Matriz estandarizada (N, K)
        """
        self.fit(X, verbose=verbose)
        return self.transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Revierte la transformación Z-score.

        FÓRMULA:
            x = z * σ + μ

        Args:
            X_scaled: Matriz estandarizada (N, K)

        Returns:
            Matriz en escala original (N, K)
        """
        if not self._fitted:
            raise RuntimeError("StandardScaler no ha sido ajustado.")

        return X_scaled * self.std_ + self.mean_

    def get_params(self) -> ScalerResult:
        """
        Obtiene los parámetros del scaler como dataclass.

        Returns:
            ScalerResult con media, std y metadatos
        """
        if not self._fitted:
            raise RuntimeError("StandardScaler no ha sido ajustado.")

        return ScalerResult(
            mean=self.mean_.copy(),
            std=self.std_.copy(),
            n_features=self.n_features_,
            n_samples_fit=self.n_samples_fit_
        )


def verify_standardization(
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Verifica que la estandarización sea correcta.

    VERIFICACIONES:
    1. Training debe tener media ≈ 0 (dentro de tolerancia numérica)
    2. Training debe tener std ≈ 1 (dentro de tolerancia numérica)
    3. Val/Test pueden diferir de 0/1 (es normal y esperado)

    Args:
        X_train_scaled: Ponderantes estandarizados de training (N, K)
        X_val_scaled: Ponderantes estandarizados de validación (N, K)
        X_test_scaled: Ponderantes estandarizados de test (N, K)
        tolerance: Tolerancia para verificación de media=0
        verbose: Si True, imprime resultados

    Returns:
        Dict con estadísticas por split
    """
    results = {}

    for name, X in [("train", X_train_scaled),
                     ("val", X_val_scaled),
                     ("test", X_test_scaled)]:
        mean_per_feature = np.mean(X, axis=0)
        std_per_feature = np.std(X, axis=0)

        results[name] = {
            "mean_of_means": float(np.mean(mean_per_feature)),
            "std_of_means": float(np.std(mean_per_feature)),
            "mean_of_stds": float(np.mean(std_per_feature)),
            "std_of_stds": float(np.std(std_per_feature)),
            "max_abs_mean": float(np.max(np.abs(mean_per_feature))),
            "min_std": float(np.min(std_per_feature)),
            "max_std": float(np.max(std_per_feature)),
        }

    if verbose:
        print("\n" + "="*60)
        print("VERIFICACIÓN DE ESTANDARIZACIÓN")
        print("="*60)

        print("\nTraining (debe tener media≈0 y std≈1):")
        train_stats = results["train"]
        mean_ok = abs(train_stats["mean_of_means"]) < tolerance
        std_ok = abs(train_stats["mean_of_stds"] - 1.0) < 0.01
        print(f"  Media de medias: {train_stats['mean_of_means']:.2e} {'✓' if mean_ok else '✗'}")
        print(f"  Media de stds:   {train_stats['mean_of_stds']:.6f} {'✓' if std_ok else '✗'}")
        print(f"  Max |media|:     {train_stats['max_abs_mean']:.2e}")

        print("\nValidation (puede diferir de 0/1):")
        val_stats = results["val"]
        print(f"  Media de medias: {val_stats['mean_of_means']:.4f}")
        print(f"  Media de stds:   {val_stats['mean_of_stds']:.4f}")

        print("\nTest (puede diferir de 0/1):")
        test_stats = results["test"]
        print(f"  Media de medias: {test_stats['mean_of_means']:.4f}")
        print(f"  Media de stds:   {test_stats['mean_of_stds']:.4f}")

        # Verificar que training esté correcto
        if mean_ok and std_ok:
            print("\n✓ Estandarización verificada correctamente.")
        else:
            print("\n✗ ERROR: La estandarización no es correcta.")

    return results


def plot_standardization_distribution(
    X_train_scaled: np.ndarray,
    labels_train: np.ndarray,
    class_names: List[str],
    n_features_to_show: int = 6,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza la distribución de características estandarizadas.

    Genera histogramas para las primeras K características,
    coloreados por clase, para verificar visualmente:
    1. Que la distribución está centrada en 0
    2. Que la escala es comparable entre características
    3. Cómo se separan las clases en cada característica

    Args:
        X_train_scaled: Características estandarizadas (N, K)
        labels_train: Etiquetas de clase (N,)
        class_names: Nombres de las clases
        n_features_to_show: Número de características a visualizar
        output_path: Ruta para guardar la figura

    Returns:
        Figura de matplotlib
    """
    n_features = min(n_features_to_show, X_train_scaled.shape[1])
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Distribución de Características Estandarizadas (Training)',
                 fontsize=14, fontweight='bold', y=1.02)

    colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Rojo, Azul, Verde

    for i in range(n_features):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        feature_data = X_train_scaled[:, i]

        # Histograma por clase
        for cls_idx in np.unique(labels_train):
            mask = labels_train == cls_idx
            ax.hist(feature_data[mask], bins=30, alpha=0.5,
                    label=class_names[cls_idx], color=colors[cls_idx % len(colors)],
                    density=True)

        # Línea vertical en 0
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Estadísticas
        mean_val = np.mean(feature_data)
        std_val = np.std(feature_data)
        ax.set_title(f'PC{i+1}\nμ={mean_val:.2e}, σ={std_val:.2f}', fontsize=10)
        ax.set_xlabel('Valor estandarizado')
        ax.set_ylabel('Densidad')

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Ocultar ejes vacíos
    for i in range(n_features, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Figura guardada en: {output_path}")

    return fig


def save_features_to_csv(
    features: np.ndarray,
    labels: np.ndarray,
    ids: List[str],
    class_names: List[str],
    output_path: Path,
    prefix: str = "PC"
) -> None:
    """
    Guarda características en formato CSV.

    Formato del CSV:
    - image_id: Identificador de la imagen
    - label: Índice numérico de la clase
    - class: Nombre de la clase
    - PC1, PC2, ..., PCK: Valores de cada componente

    Args:
        features: Matriz de características (N, K)
        labels: Etiquetas numéricas (N,)
        ids: Lista de IDs de imagen
        class_names: Nombres de las clases
        output_path: Ruta del archivo CSV
        prefix: Prefijo para columnas de características (default: "PC")
    """
    n_samples, n_features = features.shape

    # Crear DataFrame
    data = {
        'image_id': ids,
        'label': labels,
        'class': [class_names[l] for l in labels]
    }

    # Añadir columnas de características
    for i in range(n_features):
        data[f'{prefix}{i+1}'] = features[:, i]

    df = pd.DataFrame(data)

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Características guardadas en: {output_path}")
    print(f"  Shape: {n_samples} muestras × {n_features} características")


def plot_feature_statistics(
    scaler_params: ScalerResult,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza estadísticas del scaler (media y std por característica).

    Esta figura muestra:
    1. Media por componente principal (debe variar, refleja la escala original)
    2. Std por componente principal (debe variar, refleja la dispersión)

    Args:
        scaler_params: Parámetros del scaler (ScalerResult)
        output_path: Ruta para guardar la figura

    Returns:
        Figura de matplotlib
    """
    n_features = scaler_params.n_features
    x = np.arange(1, n_features + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    fig.suptitle('Parámetros de Estandarización (calculados solo con Training)',
                 fontsize=12, fontweight='bold')

    # Panel 1: Media por característica
    ax1.bar(x, scaler_params.mean, color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Media (μ)')
    ax1.set_title('Media por Componente')
    ax1.set_xticks(x[::max(1, n_features//10)])

    # Panel 2: Std por característica
    ax2.bar(x, scaler_params.std, color='darkorange', alpha=0.7)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1, label='std=1')
    ax2.set_xlabel('Componente Principal')
    ax2.set_ylabel('Desviación Estándar (σ)')
    ax2.set_title('Desviación Estándar por Componente')
    ax2.set_xticks(x[::max(1, n_features//10)])

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Estadísticas del scaler guardadas en: {output_path}")

    return fig


if __name__ == "__main__":
    """
    Script de prueba para el módulo de features.

    Demuestra:
    1. Creación de datos sintéticos
    2. Estandarización
    3. Verificación
    4. Visualización
    """
    print("="*60)
    print("PRUEBA DEL MÓDULO DE FEATURES")
    print("="*60)
    print()

    # Crear datos sintéticos con diferentes escalas
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Simular ponderantes con diferentes escalas (como después de PCA)
    scales = np.array([1000, 500, 200, 100, 50, 20, 10, 5, 2, 1])
    X_train = np.random.randn(n_samples, n_features) * scales
    X_val = np.random.randn(200, n_features) * scales + 0.1 * scales  # Ligero shift
    X_test = np.random.randn(100, n_features) * scales - 0.1 * scales

    print("Datos sintéticos creados:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Escalas por característica: {scales}")
    print()

    # Estandarizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print()

    # Verificar
    verify_standardization(X_train_scaled, X_val_scaled, X_test_scaled)

    # Mostrar ejemplo de valores antes/después
    print("\n" + "-"*60)
    print("Ejemplo de transformación (primera muestra, primeras 5 características):")
    print("-"*60)
    print(f"Antes:   {X_train[0, :5]}")
    print(f"Después: {X_train_scaled[0, :5]}")

    print("\n✓ Prueba completada.")
