"""
Módulo de PCA (Principal Component Analysis) / Eigenfaces.

MATEMÁTICAS DE PCA - EXPLICACIÓN COMPLETA
==========================================

1. EL PROBLEMA
--------------
Tenemos N imágenes de D dimensiones (D = 224 × 224 = 50,176 pixeles).
Es imposible visualizar o procesar eficientemente en 50,176 dimensiones.

PCA encuentra las K direcciones (K << D) donde los datos varían más.
Proyectar a estas direcciones reduce dimensiones preservando información.


2. IMAGEN PROMEDIO (Mean Face)
------------------------------
Sumamos todas las imágenes pixel a pixel y dividimos entre N:

    μ = (1/N) Σᵢ xᵢ

donde xᵢ es la imagen i como vector.

Resultado: El "rostro promedio" o "radiografía promedio".


3. CENTRADO DE DATOS
--------------------
A cada imagen le restamos la media:

    x̃ᵢ = xᵢ - μ

Esto mueve el centroide de los datos al origen.
Es CRUCIAL porque PCA busca direcciones desde el origen.


4. MATRIZ DE COVARIANZA
-----------------------
La covarianza mide cómo varían los pixeles juntos:

    C = (1/N) X̃ᵀ X̃

donde X̃ es la matriz de imágenes centradas (N × D).

C es una matriz (D × D) = (50176 × 50176) = 2.5 mil millones de elementos.
¡Esto es computacionalmente intratable!


5. EL TRUCO DE LA COVARIANZA PEQUEÑA
------------------------------------
En lugar de calcular C (D × D), calculamos:

    C' = (1/N) X̃ X̃ᵀ    (N × N)

Si N < D (típicamente 1000-5000 imágenes << 50176 dimensiones),
C' es MUCHO más pequeña y manejable.

Los eigenvectores de C se obtienen a partir de los de C':

    Si C' v' = λ v'
    Entonces v = X̃ᵀ v'  es eigenvector de C con el mismo eigenvalor λ

(Luego normalizamos v para que tenga norma 1)


6. EIGENVECTORES Y EIGENVALORES
-------------------------------
Resolvemos: C v = λ v

- eigenvector v: Dirección principal de varianza (una "eigenface")
- eigenvalor λ: Cantidad de varianza en esa dirección

Los eigenvalores se ordenan: λ₁ ≥ λ₂ ≥ ... ≥ λₖ
Los primeros eigenvectores capturan la mayor parte de la varianza.


7. VARIANZA EXPLICADA
---------------------
La proporción de varianza explicada por el componente i es:

    varianza_explicada_i = λᵢ / Σⱼ λⱼ

La varianza acumulada hasta K componentes es:

    varianza_acumulada_K = Σᵢ₌₁ᴷ λᵢ / Σⱼ λⱼ

Típicamente queremos ~95% de varianza explicada.


8. PROYECCIÓN (Obtener Ponderantes)
-----------------------------------
Para proyectar una imagen al espacio reducido:

    ponderantes = (x - μ)ᵀ · Vₖ

donde Vₖ son los K primeros eigenvectores (cada uno de dimensión D).

Resultado: Vector de K números = los "ponderantes" o "pesos".
ESTOS SON LAS CARACTERÍSTICAS que usaremos para clasificar.


9. RECONSTRUCCIÓN (Opcional)
----------------------------
Podemos reconstruir una imagen aproximada:

    x_reconstruida = μ + Vₖ · ponderantes

El error de reconstrucción depende de cuántos componentes K usamos.


NOTA IMPORTANTE DEL ASESOR
--------------------------
> "Las características no serían las Eigenfaces. Las características serían
>  los ponderantes... Los pesos. Esos serían las características."

Es decir: NO usamos los eigenvectores directamente como características.
Los eigenvectores son las BASES del nuevo espacio.
Los PONDERANTES (coeficientes de la proyección) son las características.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class PCAResult:
    """
    Resultado del entrenamiento de PCA.

    Atributos:
        mean: Imagen promedio (D,) - el "rostro promedio"
        components: Eigenvectores principales (K, D) - las "eigenfaces"
        eigenvalues: Eigenvalores (K,) - varianza por componente
        explained_variance_ratio: Proporción de varianza explicada (K,)
        n_components: Número de componentes K
        n_features: Dimensionalidad original D
    """
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    n_components: int
    n_features: int


class PCA:
    """
    Implementación de PCA desde cero (sin sklearn).

    Esta implementación usa el truco de la covarianza pequeña
    para manejar eficientemente datos de alta dimensión.

    Ejemplo de uso:
        >>> pca = PCA(n_components=10)
        >>> pca.fit(X_train)
        >>> X_train_pca = pca.transform(X_train)
        >>> X_test_pca = pca.transform(X_test)
    """

    def __init__(self, n_components: int = 10):
        """
        Inicializa PCA.

        Args:
            n_components: Número de componentes principales a retener
        """
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray, verbose: bool = True) -> 'PCA':
        """
        Ajusta PCA a los datos de entrenamiento.

        PASOS:
        1. Calcular imagen promedio
        2. Centrar datos
        3. Calcular matriz de covarianza pequeña (N×N)
        4. Obtener eigenvectores/eigenvalores
        5. Convertir a eigenvectores del espacio original
        6. Normalizar y ordenar

        Args:
            X: Matriz de datos (N, D) donde N=muestras, D=dimensiones
            verbose: Si True, imprime información del proceso

        Returns:
            self (para encadenar métodos)
        """
        N, D = X.shape

        if verbose:
            print(f"Ajustando PCA...")
            print(f"  Datos: {N} muestras × {D} dimensiones")
            print(f"  Componentes a extraer: {self.n_components}")
            print()

        # ===== PASO 1: Calcular imagen promedio =====
        if verbose:
            print("  [1/5] Calculando imagen promedio...")

        # μ = (1/N) Σᵢ xᵢ
        self.mean_ = np.mean(X, axis=0)

        if verbose:
            print(f"        Shape de mean: {self.mean_.shape}")

        # ===== PASO 2: Centrar datos =====
        if verbose:
            print("  [2/5] Centrando datos (restando media)...")

        # X̃ = X - μ
        X_centered = X - self.mean_

        if verbose:
            print(f"        Verificación: media de X_centered ≈ 0: {np.abs(X_centered.mean()):.2e}")

        # ===== PASO 3: Calcular matriz de covarianza pequeña =====
        if verbose:
            print("  [3/5] Calculando matriz de covarianza (usando truco N×N)...")

        # C' = (1/N) X̃ X̃ᵀ   (tamaño N × N en lugar de D × D)
        # Esto es clave cuando N << D (típicamente 5000 << 50176)
        cov_small = (1.0 / N) * np.dot(X_centered, X_centered.T)

        if verbose:
            print(f"        Shape de covarianza pequeña: {cov_small.shape}")
            print(f"        Memoria ahorrada: {D}×{D} → {N}×{N}")

        # ===== PASO 4: Calcular eigenvectores de C' =====
        if verbose:
            print("  [4/5] Calculando eigenvectores y eigenvalores...")

        # Resolver C' v' = λ v'
        eigenvalues_small, eigenvectors_small = np.linalg.eigh(cov_small)

        # eigh retorna eigenvalores en orden ascendente, los invertimos
        eigenvalues_small = eigenvalues_small[::-1]
        eigenvectors_small = eigenvectors_small[:, ::-1]

        if verbose:
            print(f"        Eigenvalores (top 5): {eigenvalues_small[:5]}")

        # ===== PASO 5: Convertir a eigenvectores del espacio original =====
        if verbose:
            print("  [5/5] Convirtiendo eigenvectores al espacio original...")

        # v = X̃ᵀ v' / ||X̃ᵀ v'||
        # Tomamos solo los primeros n_components
        n_comp = min(self.n_components, N - 1)  # No podemos tener más que N-1 componentes

        components = np.zeros((n_comp, D), dtype=np.float32)

        for i in range(n_comp):
            # Proyectar eigenvector pequeño al espacio grande
            v = np.dot(X_centered.T, eigenvectors_small[:, i])
            # Normalizar para que tenga norma 1
            v = v / np.linalg.norm(v)
            components[i] = v

        self.components_ = components
        self.eigenvalues_ = eigenvalues_small[:n_comp]

        # Calcular varianza explicada
        total_variance = np.sum(eigenvalues_small)
        self.explained_variance_ratio_ = self.eigenvalues_ / total_variance

        if verbose:
            cumulative_var = np.cumsum(self.explained_variance_ratio_)
            print()
            print(f"  Varianza explicada:")
            print(f"    Top 1 componente:  {cumulative_var[0]*100:.1f}%")
            if n_comp >= 5:
                print(f"    Top 5 componentes: {cumulative_var[4]*100:.1f}%")
            if n_comp >= 10:
                print(f"    Top 10 componentes: {cumulative_var[9]*100:.1f}%")
            print(f"    Total ({n_comp} componentes): {cumulative_var[-1]*100:.1f}%")

        self._fitted = True
        self.n_components = n_comp

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Proyecta datos al espacio de componentes principales.

        OPERACIÓN:
        ponderantes = (X - μ) · Vᵀ

        Los ponderantes son las CARACTERÍSTICAS que usaremos
        para clasificación.

        Args:
            X: Matriz de datos (N, D)

        Returns:
            Matriz de ponderantes (N, K) donde K = n_components
        """
        if not self._fitted:
            raise RuntimeError("PCA no ha sido ajustado. Llama a fit() primero.")

        # Centrar datos usando la media del training
        X_centered = X - self.mean_

        # Proyectar: cada fila de X_centered se multiplica por cada componente
        # X_centered (N, D) · components.T (D, K) = (N, K)
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Ajusta PCA y transforma los datos en un solo paso.

        Args:
            X: Matriz de datos (N, D)
            verbose: Si True, imprime información

        Returns:
            Matriz de ponderantes (N, K)
        """
        self.fit(X, verbose=verbose)
        return self.transform(X)

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Reconstruye datos aproximados desde los ponderantes.

        OPERACIÓN:
        X_reconstruida = μ + ponderantes · V

        Útil para visualizar qué información se preserva/pierde.

        Args:
            X_pca: Matriz de ponderantes (N, K)

        Returns:
            Matriz reconstruida (N, D)
        """
        if not self._fitted:
            raise RuntimeError("PCA no ha sido ajustado. Llama a fit() primero.")

        # Reconstruir: X_pca (N, K) · components (K, D) + mean (D,)
        return np.dot(X_pca, self.components_) + self.mean_

    def get_result(self) -> PCAResult:
        """
        Obtiene el resultado del PCA como dataclass.

        Returns:
            PCAResult con todos los parámetros del modelo
        """
        if not self._fitted:
            raise RuntimeError("PCA no ha sido ajustado. Llama a fit() primero.")

        return PCAResult(
            mean=self.mean_,
            components=self.components_,
            eigenvalues=self.eigenvalues_,
            explained_variance_ratio=self.explained_variance_ratio_,
            n_components=self.n_components,
            n_features=len(self.mean_)
        )


def plot_eigenfaces(
    pca_result: PCAResult,
    image_shape: Tuple[int, int],
    n_show: int = 10,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza las eigenfaces (componentes principales).

    Cada eigenface muestra un "modo de variación" en las imágenes.
    Las primeras eigenfaces capturan las variaciones más importantes.

    Args:
        pca_result: Resultado del PCA
        image_shape: (alto, ancho) de las imágenes originales
        n_show: Número de eigenfaces a mostrar
        output_path: Ruta para guardar la figura (opcional)

    Returns:
        Figura de matplotlib
    """
    n_show = min(n_show, pca_result.n_components)

    # Calcular layout: intentar hacer un grid casi cuadrado
    n_cols = min(5, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Eigenfaces (Componentes Principales)', fontsize=14, fontweight='bold')

    for i in range(n_show):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        # Reshape eigenvector a imagen
        eigenface = pca_result.components[i].reshape(image_shape)

        # Normalizar para visualización
        eigenface_norm = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())

        ax.imshow(eigenface_norm, cmap='gray')
        var_explained = pca_result.explained_variance_ratio[i] * 100
        ax.set_title(f'PC{i+1}\n({var_explained:.1f}%)', fontsize=10)
        ax.axis('off')

    # Ocultar axes vacíos
    for i in range(n_show, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Eigenfaces guardadas en: {output_path}")

    return fig


def plot_variance_explained(
    pca_result: PCAResult,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza la varianza explicada por componente.

    Esta gráfica ayuda a decidir cuántos componentes retener.
    Típicamente buscamos el "codo" donde la curva se aplana.

    Args:
        pca_result: Resultado del PCA
        output_path: Ruta para guardar la figura (opcional)

    Returns:
        Figura de matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_components = pca_result.n_components
    x = np.arange(1, n_components + 1)
    cumulative = np.cumsum(pca_result.explained_variance_ratio) * 100

    # Panel izquierdo: varianza individual
    ax1.bar(x, pca_result.explained_variance_ratio * 100, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Componente Principal', fontsize=11)
    ax1.set_ylabel('Varianza Explicada (%)', fontsize=11)
    ax1.set_title('Varianza por Componente', fontsize=12, fontweight='bold')
    ax1.set_xticks(x[::max(1, n_components//10)])

    # Panel derecho: varianza acumulada
    ax2.plot(x, cumulative, 'o-', color='steelblue', linewidth=2, markersize=4)
    ax2.axhline(y=95, color='red', linestyle='--', label='95% umbral')
    ax2.axhline(y=99, color='orange', linestyle='--', label='99% umbral')

    # Encontrar componentes para 95% y 99%
    k_95 = np.argmax(cumulative >= 95) + 1 if np.any(cumulative >= 95) else n_components
    k_99 = np.argmax(cumulative >= 99) + 1 if np.any(cumulative >= 99) else n_components

    ax2.axvline(x=k_95, color='red', linestyle=':', alpha=0.7)
    ax2.axvline(x=k_99, color='orange', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Número de Componentes', fontsize=11)
    ax2.set_ylabel('Varianza Acumulada (%)', fontsize=11)
    ax2.set_title('Varianza Acumulada', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim([0, 105])
    ax2.set_xlim([0, n_components + 1])

    # Añadir texto informativo
    info_text = f'K={k_95} para 95%\nK={k_99} para 99%'
    ax2.text(0.95, 0.05, info_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Varianza explicada guardada en: {output_path}")

    return fig


def plot_mean_face(
    pca_result: PCAResult,
    image_shape: Tuple[int, int],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza la imagen promedio (mean face).

    La imagen promedio representa el "caso típico" de todas
    las radiografías. PCA centra los datos restando esta imagen.

    Args:
        pca_result: Resultado del PCA
        image_shape: (alto, ancho) de las imágenes originales
        output_path: Ruta para guardar la figura (opcional)

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    mean_image = pca_result.mean.reshape(image_shape)

    ax.imshow(mean_image, cmap='gray')
    ax.set_title('Radiografía Promedio\n(Mean Face)', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Imagen promedio guardada en: {output_path}")

    return fig


if __name__ == "__main__":
    """
    Script de prueba para PCA.

    Este script:
    1. Carga el dataset de imágenes warped
    2. Aplica PCA solo al set de training
    3. Genera visualizaciones de eigenfaces y varianza
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_loader import load_dataset

    # Rutas
    base_path = Path(__file__).parent.parent.parent.parent
    csv_path = Path(__file__).parent.parent / "results" / "metrics" / "01_full_balanced_3class_warped.csv"
    figures_dir = Path(__file__).parent.parent / "results" / "figures"

    print("=" * 60)
    print("PRUEBA DE PCA / EIGENFACES")
    print("=" * 60)
    print()

    # Cargar datos
    dataset = load_dataset(csv_path, base_path, scenario="2class", verbose=True)

    print()
    print("=" * 60)
    print("APLICANDO PCA")
    print("=" * 60)
    print()

    # Crear y ajustar PCA
    # Usamos más componentes inicialmente para ver la curva de varianza
    n_components_initial = min(100, dataset.train.X.shape[0] - 1)
    pca = PCA(n_components=n_components_initial)
    X_train_pca = pca.fit_transform(dataset.train.X, verbose=True)

    print()
    print(f"Shape de ponderantes (train): {X_train_pca.shape}")

    # Transformar val y test
    X_val_pca = pca.transform(dataset.val.X)
    X_test_pca = pca.transform(dataset.test.X)
    print(f"Shape de ponderantes (val): {X_val_pca.shape}")
    print(f"Shape de ponderantes (test): {X_test_pca.shape}")

    # Obtener resultado
    pca_result = pca.get_result()

    print()
    print("=" * 60)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 60)
    print()

    # Generar visualizaciones
    plot_mean_face(
        pca_result,
        dataset.image_shape,
        output_path=figures_dir / "mean_face.png"
    )

    plot_eigenfaces(
        pca_result,
        dataset.image_shape,
        n_show=10,
        output_path=figures_dir / "eigenfaces_top10.png"
    )

    plot_variance_explained(
        pca_result,
        output_path=figures_dir / "varianza_explicada.png"
    )

    plt.close('all')

    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"\nComponentes calculados: {pca_result.n_components}")
    print(f"Varianza explicada (total): {sum(pca_result.explained_variance_ratio)*100:.2f}%")

    # Encontrar K óptimo
    cumulative = np.cumsum(pca_result.explained_variance_ratio)
    k_95 = np.argmax(cumulative >= 0.95) + 1 if np.any(cumulative >= 0.95) else pca_result.n_components
    k_99 = np.argmax(cumulative >= 0.99) + 1 if np.any(cumulative >= 0.99) else pca_result.n_components

    print(f"\nComponentes sugeridos:")
    print(f"  Para 95% de varianza: K = {k_95}")
    print(f"  Para 99% de varianza: K = {k_99}")
    print(f"\nFiguras generadas en: {figures_dir}")
