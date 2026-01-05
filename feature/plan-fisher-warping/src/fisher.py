"""
Modulo de Criterio de Fisher (Fisher Ratio / Fisher's Linear Discriminant).

FASE 5 DEL PIPELINE FISHER-WARPING
==================================

Este modulo implementa:
1. Calculo del Fisher Ratio por caracteristica
2. Amplificacion de caracteristicas usando Fisher como ponderador
3. Visualizacion y verificacion de resultados

MATEMATICAS DEL CRITERIO DE FISHER
==================================

1. EL PROBLEMA
--------------
Tenemos K caracteristicas (ponderantes estandarizados de PCA) y queremos saber
cual caracteristica separa MEJOR las dos clases (Enfermo vs Normal).

Imaginemos cada caracteristica como un histograma:
- Los valores de pacientes "Enfermos" forman una distribucion (campana)
- Los valores de pacientes "Normales" forman otra distribucion (campana)

El problema es: ?Que tan bien separa esta caracteristica las clases?

2. LA INTUICION
---------------
Una caracteristica es BUENA si:
- Las medias de las dos distribuciones estan MUY SEPARADAS (lejos una de otra)
- Las distribuciones son MUY ESTRECHAS (poca varianza/dispersion)

Una caracteristica es MALA si:
- Las medias estan cerca (mucho traslape)
- Las distribuciones son muy anchas (alta varianza)

3. LA FORMULA
-------------
Para la caracteristica i, el Criterio de Fisher es:

                 (mu_enfermo - mu_normal)^2
    J_i = ------------------------------------
              sigma_enfermo^2 + sigma_normal^2

Donde:
- mu_enfermo: Media de la caracteristica i para pacientes enfermos
- mu_normal: Media de la caracteristica i para pacientes normales
- sigma_enfermo: Desviacion estandar para enfermos
- sigma_normal: Desviacion estandar para normales

NUMERADOR: Separacion entre clases (between-class variance)
           - Mide que tan lejos estan las medias
           - Se eleva al cuadrado para que sea siempre positivo
           - Valor grande = clases bien separadas

DENOMINADOR: Dispersion dentro de las clases (within-class variance)
             - Suma de varianzas de ambas clases
             - Valor pequeno = distribuciones estrechas, poco traslape

4. INTERPRETACION
-----------------
- J grande (>> 1): Caracteristica separa MUY BIEN las clases
- J pequeno (~0): Caracteristica NO separa las clases (se traslapan)
- J = 0: Las medias son identicas (completamente inutiles)

5. EJEMPLO NUMERICO
-------------------
Caracteristica 1 con 1000 imagenes (500 enfermos, 500 normales):

Enfermos:
- Valores: [2.1, 2.3, 1.9, 2.0, ...]
- Media (mu_e) = 2.0
- Sigma (sigma_e) = 0.5

Normales:
- Valores: [-1.8, -2.1, -1.9, -2.0, ...]
- Media (mu_n) = -2.0
- Sigma (sigma_n) = 0.4

Calculo:
    J_1 = (2.0 - (-2.0))^2 / (0.5^2 + 0.4^2)
        = (4.0)^2 / (0.25 + 0.16)
        = 16 / 0.41
        = 39.02

Un J=39 es muy alto, indica excelente separacion.

6. AMPLIFICACION
----------------
Una vez calculado J_i para cada caracteristica, podemos usarlo como
AMPLIFICADOR o PONDERADOR:

    caracteristica_amplificada_i = caracteristica_estandarizada_i * J_i

Esto:
- AMPLIFICA las caracteristicas que separan bien (J grande)
- ATENUA las caracteristicas que no separan (J pequeno)

Es como subir el volumen de los instrumentos que suenan bien y bajar
los que hacen ruido.

NOTA DEL ASESOR
---------------
> "Si esa Razon de Fisher es grande, significa que esa caracteristica separa
>  bien las clases. Si esa Razon de Fisher es cero o chiquita, significa que
>  esa caracteristica no separa bien las clases."

> "Con esa Razon de Fisher, puedes usarla como un ponderante... la estas
>  amplificando. Si separaba muy bien, como quien dice la ganancia o la
>  amplificacion pues va a ser grande."
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class FisherResult:
    """
    Resultado del calculo del Criterio de Fisher.

    Atributos:
        fisher_ratios: Vector de Fisher ratios (K,) - uno por caracteristica
        class_means: Dict con medias por clase {0: array(K,), 1: array(K,)}
        class_stds: Dict con stds por clase {0: array(K,), 1: array(K,)}
        n_features: Numero de caracteristicas K
        n_samples_per_class: Dict con conteo por clase {0: N0, 1: N1}
        class_names: Nombres de las clases
    """
    fisher_ratios: np.ndarray
    class_means: Dict[int, np.ndarray]
    class_stds: Dict[int, np.ndarray]
    n_features: int
    n_samples_per_class: Dict[int, int]
    class_names: List[str]

    def get_top_k(self, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene las top-K caracteristicas con mayor Fisher ratio.

        Args:
            k: Numero de caracteristicas a retornar

        Returns:
            Tuple de (indices, valores) ordenados de mayor a menor
        """
        k = min(k, self.n_features)
        indices = np.argsort(self.fisher_ratios)[::-1][:k]
        values = self.fisher_ratios[indices]
        return indices, values

    def get_ranking(self) -> np.ndarray:
        """
        Obtiene el ranking de todas las caracteristicas (de mejor a peor).

        Returns:
            Array de indices ordenados por Fisher ratio descendente
        """
        return np.argsort(self.fisher_ratios)[::-1]


class FisherRatio:
    """
    Calculo del Criterio de Fisher implementado desde cero (sin sklearn).

    Esta implementacion calcula el Fisher ratio para cada caracteristica,
    midiendo que tan bien separa las dos clases.

    Ejemplo de uso:
        >>> fisher = FisherRatio()
        >>> fisher.fit(X_train, y_train, class_names=['Enfermo', 'Normal'])
        >>> ratios = fisher.fisher_ratios_
        >>> X_amplified = fisher.amplify(X_train)
    """

    def __init__(self, epsilon: float = 1e-9):
        """
        Inicializa FisherRatio.

        Args:
            epsilon: Valor pequeno para evitar division por cero
        """
        self.epsilon = epsilon
        self.fisher_ratios_: Optional[np.ndarray] = None
        self.class_means_: Optional[Dict[int, np.ndarray]] = None
        self.class_stds_: Optional[Dict[int, np.ndarray]] = None
        self.n_features_: int = 0
        self.n_samples_per_class_: Optional[Dict[int, int]] = None
        self.class_names_: List[str] = []
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'FisherRatio':
        """
        Calcula el Fisher ratio para cada caracteristica.

        ALGORITMO:
        Para cada caracteristica i (columna de X):
        1. Separar valores por clase usando y
        2. Calcular mu_0, mu_1 (medias por clase)
        3. Calcular sigma_0, sigma_1 (stds por clase)
        4. Aplicar formula: J_i = (mu_0 - mu_1)^2 / (sigma_0^2 + sigma_1^2 + eps)

        Args:
            X: Matriz de caracteristicas (N, K) - valores estandarizados
            y: Vector de etiquetas (N,) - debe contener exactamente 2 clases
            class_names: Nombres de las clases (opcional)
            verbose: Si True, imprime informacion del proceso

        Returns:
            self (para encadenar metodos)
        """
        N, K = X.shape

        # Verificar que hay exactamente 2 clases
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"Fisher ratio requiere exactamente 2 clases, "
                f"pero se encontraron {len(unique_classes)}: {unique_classes}"
            )

        class_0, class_1 = unique_classes[0], unique_classes[1]

        if class_names is None:
            class_names = [f"Clase {class_0}", f"Clase {class_1}"]

        if verbose:
            print("Calculando Fisher Ratios...")
            print(f"  Datos: {N} muestras x {K} caracteristicas")
            print(f"  Clases: {class_names[0]} (label={class_0}), {class_names[1]} (label={class_1})")

        # Separar datos por clase
        mask_0 = (y == class_0)
        mask_1 = (y == class_1)

        X_0 = X[mask_0]  # Datos de clase 0
        X_1 = X[mask_1]  # Datos de clase 1

        n_0, n_1 = len(X_0), len(X_1)

        if verbose:
            print(f"  {class_names[0]}: {n_0} muestras")
            print(f"  {class_names[1]}: {n_1} muestras")
            print()

        # ===== Calcular estadisticas por clase =====
        # Media por caracteristica para cada clase
        # mu = (1/N) * sum(x_i)
        mu_0 = np.mean(X_0, axis=0)  # (K,)
        mu_1 = np.mean(X_1, axis=0)  # (K,)

        # Desviacion estandar por caracteristica para cada clase
        # sigma = sqrt( (1/N) * sum( (x_i - mu)^2 ) )
        sigma_0 = np.std(X_0, axis=0)  # (K,)
        sigma_1 = np.std(X_1, axis=0)  # (K,)

        # ===== Calcular Fisher Ratio =====
        # J_i = (mu_0 - mu_1)^2 / (sigma_0^2 + sigma_1^2 + epsilon)

        # Numerador: separacion entre clases (between-class)
        numerator = (mu_0 - mu_1) ** 2

        # Denominador: dispersion dentro de clases (within-class)
        denominator = sigma_0 ** 2 + sigma_1 ** 2 + self.epsilon

        # Fisher ratio
        fisher_ratios = numerator / denominator

        # Guardar resultados
        self.fisher_ratios_ = fisher_ratios
        self.class_means_ = {class_0: mu_0, class_1: mu_1}
        self.class_stds_ = {class_0: sigma_0, class_1: sigma_1}
        self.n_features_ = K
        self.n_samples_per_class_ = {class_0: n_0, class_1: n_1}
        self.class_names_ = class_names
        self._fitted = True

        if verbose:
            print("  Resultados Fisher Ratio:")
            print(f"    Min:  {fisher_ratios.min():.4f}")
            print(f"    Max:  {fisher_ratios.max():.4f}")
            print(f"    Mean: {fisher_ratios.mean():.4f}")
            print(f"    Std:  {fisher_ratios.std():.4f}")
            print()

            # Mostrar top 5
            top_indices, top_values = self.get_result().get_top_k(5)
            print("  Top 5 caracteristicas (mejor separacion):")
            for i, (idx, val) in enumerate(zip(top_indices, top_values)):
                print(f"    {i+1}. PC{idx+1}: J = {val:.4f}")

        return self

    def amplify(self, X: np.ndarray) -> np.ndarray:
        """
        Amplifica las caracteristicas multiplicando por Fisher ratios.

        OPERACION:
            X_amplificado = X * J

        Donde J es el vector de Fisher ratios (K,).
        Esto amplifica las caracteristicas que separan bien las clases.

        Args:
            X: Matriz de caracteristicas (N, K) - estandarizadas

        Returns:
            Matriz amplificada (N, K)
        """
        if not self._fitted:
            raise RuntimeError("FisherRatio no ha sido ajustado. Llama a fit() primero.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X tiene {X.shape[1]} caracteristicas, pero Fisher fue "
                f"ajustado con {self.n_features_} caracteristicas."
            )

        # X_amplificado = X * J (broadcasting: cada columna se multiplica por su J)
        return X * self.fisher_ratios_

    def get_result(self) -> FisherResult:
        """
        Obtiene el resultado como dataclass.

        Returns:
            FisherResult con todos los parametros
        """
        if not self._fitted:
            raise RuntimeError("FisherRatio no ha sido ajustado.")

        return FisherResult(
            fisher_ratios=self.fisher_ratios_.copy(),
            class_means=self.class_means_.copy(),
            class_stds=self.class_stds_.copy(),
            n_features=self.n_features_,
            n_samples_per_class=self.n_samples_per_class_.copy(),
            class_names=self.class_names_.copy()
        )


class FisherRatioMulticlass:
    """
    Calculo del Criterio de Fisher para 2 o mas clases (extension pairwise).

    EXTENSION A MULTICLASE
    ======================

    Para 2 clases: Comportamiento identico a FisherRatio.

    Para 3+ clases: Calcula Fisher para cada PAR de clases y promedia.

    Ejemplo con 3 clases (COVID, Normal, Viral):
        J_1 = Fisher(COVID vs Normal)
        J_2 = Fisher(COVID vs Viral)
        J_3 = Fisher(Normal vs Viral)

        J_final = (J_1 + J_2 + J_3) / 3

    Esto mide que tan bien separa cada caracteristica TODOS los pares de clases.

    MATEMATICAS
    -----------
    Para cada par de clases (a, b):

        J_ab = (mu_a - mu_b)^2 / (sigma_a^2 + sigma_b^2)

    Fisher final (promedio de pares):

        J = (1 / n_pares) * sum(J_ab)

    Donde n_pares = C(n_clases, 2) = n_clases * (n_clases - 1) / 2

    Ejemplo de uso:
        >>> fisher = FisherRatioMulticlass()
        >>> fisher.fit(X_train, y_train, class_names=['COVID', 'Normal', 'Viral'])
        >>> ratios = fisher.fisher_ratios_
        >>> X_amplified = fisher.amplify(X_train)
    """

    def __init__(self, epsilon: float = 1e-9):
        """
        Inicializa FisherRatioMulticlass.

        Args:
            epsilon: Valor pequeno para evitar division por cero
        """
        self.epsilon = epsilon
        self.fisher_ratios_: Optional[np.ndarray] = None
        self.pairwise_ratios_: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        self.class_means_: Optional[Dict[int, np.ndarray]] = None
        self.class_stds_: Optional[Dict[int, np.ndarray]] = None
        self.n_features_: int = 0
        self.n_classes_: int = 0
        self.n_samples_per_class_: Optional[Dict[int, int]] = None
        self.class_names_: List[str] = []
        self._fitted = False

    def _compute_pairwise_fisher(
        self,
        X_a: np.ndarray,
        X_b: np.ndarray
    ) -> np.ndarray:
        """
        Calcula Fisher ratio para un par de clases.

        Args:
            X_a: Datos de clase a (n_a, K)
            X_b: Datos de clase b (n_b, K)

        Returns:
            Vector de Fisher ratios (K,)
        """
        mu_a = np.mean(X_a, axis=0)
        mu_b = np.mean(X_b, axis=0)
        sigma_a = np.std(X_a, axis=0)
        sigma_b = np.std(X_b, axis=0)

        numerator = (mu_a - mu_b) ** 2
        denominator = sigma_a ** 2 + sigma_b ** 2 + self.epsilon

        return numerator / denominator

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'FisherRatioMulticlass':
        """
        Calcula el Fisher ratio para cada caracteristica (2+ clases).

        ALGORITMO:
        1. Identificar todas las clases unicas
        2. Para cada par de clases (i, j) con i < j:
           - Calcular J_ij usando la formula clasica
        3. Promediar todos los J_ij para obtener J final

        Args:
            X: Matriz de caracteristicas (N, K) - valores estandarizados
            y: Vector de etiquetas (N,) - puede contener 2 o mas clases
            class_names: Nombres de las clases (opcional)
            verbose: Si True, imprime informacion del proceso

        Returns:
            self (para encadenar metodos)
        """
        N, K = X.shape

        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        if n_classes < 2:
            raise ValueError(
                f"Fisher ratio requiere al menos 2 clases, "
                f"pero se encontro {n_classes}: {unique_classes}"
            )

        if class_names is None:
            class_names = [f"Clase {c}" for c in unique_classes]

        if verbose:
            print("Calculando Fisher Ratios (Multiclase - Pairwise)...")
            print(f"  Datos: {N} muestras x {K} caracteristicas")
            print(f"  Clases: {n_classes} -> {class_names}")

        # Separar datos por clase y calcular estadisticas
        class_data = {}
        class_means = {}
        class_stds = {}
        n_samples_per_class = {}

        for cls in unique_classes:
            mask = (y == cls)
            class_data[cls] = X[mask]
            class_means[cls] = np.mean(X[mask], axis=0)
            class_stds[cls] = np.std(X[mask], axis=0)
            n_samples_per_class[cls] = int(np.sum(mask))

        if verbose:
            for cls, name in zip(unique_classes, class_names):
                print(f"    {name}: {n_samples_per_class[cls]} muestras")

        # Calcular Fisher para cada par de clases
        pairwise_ratios = {}
        fisher_sum = np.zeros(K)
        n_pairs = 0

        if verbose:
            print(f"\n  Calculando Fisher para cada par de clases:")

        for i, cls_i in enumerate(unique_classes):
            for j, cls_j in enumerate(unique_classes):
                if i < j:  # Solo pares unicos (evitar duplicados)
                    J_pair = self._compute_pairwise_fisher(
                        class_data[cls_i],
                        class_data[cls_j]
                    )
                    pairwise_ratios[(cls_i, cls_j)] = J_pair
                    fisher_sum += J_pair
                    n_pairs += 1

                    if verbose:
                        name_i = class_names[i]
                        name_j = class_names[j]
                        print(f"    {name_i} vs {name_j}: "
                              f"J_mean={J_pair.mean():.4f}, J_max={J_pair.max():.4f}")

        # Promediar
        fisher_ratios = fisher_sum / n_pairs

        # Guardar resultados
        self.fisher_ratios_ = fisher_ratios
        self.pairwise_ratios_ = pairwise_ratios
        self.class_means_ = class_means
        self.class_stds_ = class_stds
        self.n_features_ = K
        self.n_classes_ = n_classes
        self.n_samples_per_class_ = n_samples_per_class
        self.class_names_ = class_names
        self._fitted = True

        if verbose:
            print(f"\n  Resultados Fisher Ratio (promedio de {n_pairs} pares):")
            print(f"    Min:  {fisher_ratios.min():.4f}")
            print(f"    Max:  {fisher_ratios.max():.4f}")
            print(f"    Mean: {fisher_ratios.mean():.4f}")
            print(f"    Std:  {fisher_ratios.std():.4f}")
            print()

            # Mostrar top 5
            top_indices = np.argsort(fisher_ratios)[::-1][:5]
            print("  Top 5 caracteristicas (mejor separacion):")
            for rank, idx in enumerate(top_indices):
                print(f"    {rank+1}. PC{idx+1}: J = {fisher_ratios[idx]:.4f}")

        return self

    def amplify(self, X: np.ndarray) -> np.ndarray:
        """
        Amplifica las caracteristicas multiplicando por Fisher ratios.

        Args:
            X: Matriz de caracteristicas (N, K) - estandarizadas

        Returns:
            Matriz amplificada (N, K)
        """
        if not self._fitted:
            raise RuntimeError("FisherRatioMulticlass no ha sido ajustado. Llama a fit() primero.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X tiene {X.shape[1]} caracteristicas, pero Fisher fue "
                f"ajustado con {self.n_features_} caracteristicas."
            )

        return X * self.fisher_ratios_

    def get_result(self) -> FisherResult:
        """
        Obtiene el resultado como dataclass (compatible con FisherRatio).

        Returns:
            FisherResult con todos los parametros
        """
        if not self._fitted:
            raise RuntimeError("FisherRatioMulticlass no ha sido ajustado.")

        return FisherResult(
            fisher_ratios=self.fisher_ratios_.copy(),
            class_means=self.class_means_.copy(),
            class_stds=self.class_stds_.copy(),
            n_features=self.n_features_,
            n_samples_per_class=self.n_samples_per_class_.copy(),
            class_names=self.class_names_.copy()
        )


def plot_fisher_ratios(
    fisher_result: FisherResult,
    output_path: Optional[Path] = None,
    top_k: Optional[int] = None
) -> plt.Figure:
    """
    Visualiza los Fisher ratios de todas las caracteristicas.

    Genera un grafico de barras mostrando el Fisher ratio de cada PC,
    ordenado por componente. Resalta las top-K mejores.

    Args:
        fisher_result: Resultado del calculo Fisher
        output_path: Ruta para guardar la figura
        top_k: Numero de top caracteristicas a resaltar

    Returns:
        Figura de matplotlib
    """
    n_features = fisher_result.n_features
    x = np.arange(1, n_features + 1)
    ratios = fisher_result.fisher_ratios

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fig.suptitle('Criterio de Fisher por Caracteristica',
                 fontsize=14, fontweight='bold')

    # === Panel 1: Barras ordenadas por PC ===
    colors = ['steelblue'] * n_features

    # Resaltar top-K si se especifica
    if top_k:
        top_indices, _ = fisher_result.get_top_k(top_k)
        for idx in top_indices:
            colors[idx] = 'darkorange'

    ax1.bar(x, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Componente Principal (PC)', fontsize=11)
    ax1.set_ylabel('Fisher Ratio (J)', fontsize=11)
    ax1.set_title('Fisher Ratio por PC', fontsize=12)
    ax1.set_xticks(x[::max(1, n_features//10)])

    # Linea de media
    mean_j = ratios.mean()
    ax1.axhline(y=mean_j, color='red', linestyle='--',
                label=f'Media = {mean_j:.3f}', linewidth=1.5)
    ax1.legend(loc='upper right')

    # === Panel 2: Barras ordenadas por valor ===
    ranking = fisher_result.get_ranking()
    sorted_ratios = ratios[ranking]
    sorted_labels = [f'PC{i+1}' for i in ranking]

    # Colores: gradiente de mejor a peor
    n_show = min(20, n_features)  # Mostrar top 20
    colors_sorted = plt.cm.RdYlGn_r(np.linspace(0, 1, n_show))

    ax2.barh(range(n_show), sorted_ratios[:n_show][::-1],
             color=colors_sorted[::-1], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(n_show))
    ax2.set_yticklabels(sorted_labels[:n_show][::-1], fontsize=9)
    ax2.set_xlabel('Fisher Ratio (J)', fontsize=11)
    ax2.set_ylabel('Componente Principal', fontsize=11)
    ax2.set_title(f'Top {n_show} Caracteristicas (ordenadas)', fontsize=12)

    # Anotar valores
    for i, v in enumerate(sorted_ratios[:n_show][::-1]):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Fisher ratios guardados en: {output_path}")

    return fig


def plot_class_separation(
    X: np.ndarray,
    y: np.ndarray,
    fisher_result: FisherResult,
    top_k: int = 6,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza la separacion de clases para las top-K caracteristicas.

    Genera histogramas mostrando como cada caracteristica separa
    las dos clases. Util para entender visualmente el Fisher ratio.

    Args:
        X: Matriz de caracteristicas (N, K)
        y: Vector de etiquetas (N,)
        fisher_result: Resultado del calculo Fisher
        top_k: Numero de caracteristicas a mostrar
        output_path: Ruta para guardar la figura

    Returns:
        Figura de matplotlib
    """
    top_indices, top_values = fisher_result.get_top_k(top_k)

    n_cols = min(3, top_k)
    n_rows = (top_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Separacion de Clases (Top Caracteristicas por Fisher)',
                 fontsize=14, fontweight='bold', y=1.02)

    classes = np.unique(y)
    colors = ['#E74C3C', '#3498DB']  # Rojo para clase 0, Azul para clase 1

    for i, (feat_idx, feat_j) in enumerate(zip(top_indices, top_values)):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        # Histograma por clase
        for cls_idx, cls_label in enumerate(classes):
            mask = (y == cls_label)
            data = X[mask, feat_idx]

            ax.hist(data, bins=30, alpha=0.6,
                    label=f'{fisher_result.class_names[cls_idx]} (n={np.sum(mask)})',
                    color=colors[cls_idx], density=True)

            # Linea vertical en la media
            mean_val = np.mean(data)
            ax.axvline(mean_val, color=colors[cls_idx], linestyle='--',
                       linewidth=2, alpha=0.8)

        # Titulo con Fisher ratio
        ax.set_title(f'PC{feat_idx+1}\nJ = {feat_j:.3f}', fontsize=11)
        ax.set_xlabel('Valor estandarizado')
        ax.set_ylabel('Densidad')
        ax.legend(loc='upper right', fontsize=8)

    # Ocultar ejes vacios
    for i in range(top_k, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Separacion de clases guardada en: {output_path}")

    return fig


def plot_amplification_effect(
    X_original: np.ndarray,
    X_amplified: np.ndarray,
    fisher_result: FisherResult,
    top_k: int = 6,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza el efecto de la amplificacion Fisher.

    Compara la escala de las caracteristicas antes y despues
    de multiplicar por el Fisher ratio.

    Args:
        X_original: Caracteristicas estandarizadas (N, K)
        X_amplified: Caracteristicas amplificadas (N, K)
        fisher_result: Resultado del calculo Fisher
        top_k: Numero de caracteristicas a mostrar
        output_path: Ruta para guardar la figura

    Returns:
        Figura de matplotlib
    """
    top_indices, top_values = fisher_result.get_top_k(top_k)

    fig, axes = plt.subplots(2, top_k, figsize=(3 * top_k, 6))

    fig.suptitle('Efecto de la Amplificacion Fisher',
                 fontsize=14, fontweight='bold', y=1.02)

    for i, (feat_idx, feat_j) in enumerate(zip(top_indices, top_values)):
        # Fila superior: Original
        ax_orig = axes[0, i]
        ax_orig.hist(X_original[:, feat_idx], bins=30, color='steelblue',
                     alpha=0.7, density=True)
        std_orig = np.std(X_original[:, feat_idx])
        ax_orig.set_title(f'PC{feat_idx+1} Original\nstd={std_orig:.2f}', fontsize=10)
        if i == 0:
            ax_orig.set_ylabel('Densidad', fontsize=10)

        # Fila inferior: Amplificado
        ax_amp = axes[1, i]
        ax_amp.hist(X_amplified[:, feat_idx], bins=30, color='darkorange',
                    alpha=0.7, density=True)
        std_amp = np.std(X_amplified[:, feat_idx])
        ax_amp.set_title(f'PC{feat_idx+1} Amplificado\nJ={feat_j:.2f}, std={std_amp:.2f}',
                         fontsize=10)
        ax_amp.set_xlabel('Valor')
        if i == 0:
            ax_amp.set_ylabel('Densidad', fontsize=10)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Efecto de amplificacion guardado en: {output_path}")

    return fig


def save_fisher_results(
    fisher_result: FisherResult,
    output_path: Path
) -> None:
    """
    Guarda los resultados de Fisher en formato CSV.

    Formato del CSV:
    - PC: Numero de componente principal (1-indexed)
    - fisher_ratio: Valor del Fisher ratio
    - rank: Posicion en el ranking (1 = mejor)
    - mean_class0: Media para clase 0
    - mean_class1: Media para clase 1
    - std_class0: Std para clase 0
    - std_class1: Std para clase 1

    Args:
        fisher_result: Resultado del calculo Fisher
        output_path: Ruta del archivo CSV
    """
    n_features = fisher_result.n_features
    ranking = fisher_result.get_ranking()

    # Obtener claves de clase
    class_keys = list(fisher_result.class_means.keys())

    data = {
        'PC': [f'PC{i+1}' for i in range(n_features)],
        'fisher_ratio': fisher_result.fisher_ratios,
        'rank': [np.where(ranking == i)[0][0] + 1 for i in range(n_features)],
        f'mean_{fisher_result.class_names[0]}': fisher_result.class_means[class_keys[0]],
        f'mean_{fisher_result.class_names[1]}': fisher_result.class_means[class_keys[1]],
        f'std_{fisher_result.class_names[0]}': fisher_result.class_stds[class_keys[0]],
        f'std_{fisher_result.class_names[1]}': fisher_result.class_stds[class_keys[1]],
    }

    df = pd.DataFrame(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Resultados Fisher guardados en: {output_path}")
    print(f"  {n_features} caracteristicas")


def save_amplified_features(
    X_amplified: np.ndarray,
    labels: np.ndarray,
    ids: List[str],
    class_names: List[str],
    output_path: Path
) -> None:
    """
    Guarda caracteristicas amplificadas en formato CSV.

    Args:
        X_amplified: Matriz de caracteristicas amplificadas (N, K)
        labels: Etiquetas numericas (N,)
        ids: Lista de IDs de imagen
        class_names: Nombres de las clases
        output_path: Ruta del archivo CSV
    """
    n_samples, n_features = X_amplified.shape

    data = {
        'image_id': ids,
        'label': labels,
        'class': [class_names[l] for l in labels]
    }

    for i in range(n_features):
        data[f'PC{i+1}_amp'] = X_amplified[:, i]

    df = pd.DataFrame(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Caracteristicas amplificadas guardadas en: {output_path}")
    print(f"  Shape: {n_samples} muestras x {n_features} caracteristicas")


def verify_fisher_calculation(
    X: np.ndarray,
    y: np.ndarray,
    fisher_result: FisherResult,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verifica que el calculo de Fisher sea correcto.

    Comprueba:
    1. Que los Fisher ratios sean no negativos
    2. Recalcula manualmente algunos valores para verificar
    3. Verifica la consistencia de las estadisticas

    Args:
        X: Matriz de caracteristicas (N, K)
        y: Vector de etiquetas (N,)
        fisher_result: Resultado a verificar
        verbose: Si True, imprime resultados

    Returns:
        Dict con resultados de verificacion
    """
    results = {
        'all_non_negative': bool(np.all(fisher_result.fisher_ratios >= 0)),
        'no_nan': bool(not np.any(np.isnan(fisher_result.fisher_ratios))),
        'no_inf': bool(not np.any(np.isinf(fisher_result.fisher_ratios))),
    }

    # Verificar recalculando el primer y ultimo Fisher ratio
    classes = np.unique(y)
    for check_idx in [0, -1]:
        feat_idx = check_idx if check_idx >= 0 else fisher_result.n_features + check_idx

        X_0 = X[y == classes[0], feat_idx]
        X_1 = X[y == classes[1], feat_idx]

        mu_0, mu_1 = np.mean(X_0), np.mean(X_1)
        sigma_0, sigma_1 = np.std(X_0), np.std(X_1)

        expected_j = (mu_0 - mu_1)**2 / (sigma_0**2 + sigma_1**2 + 1e-9)
        actual_j = fisher_result.fisher_ratios[feat_idx]

        results[f'PC{feat_idx+1}_verified'] = np.isclose(expected_j, actual_j, rtol=1e-5)

    if verbose:
        print("\n" + "="*60)
        print("VERIFICACION DE CALCULO FISHER")
        print("="*60)

        print(f"\n  Todos no-negativos: {'OK' if results['all_non_negative'] else 'FALLO'}")
        print(f"  Sin NaN: {'OK' if results['no_nan'] else 'FALLO'}")
        print(f"  Sin Inf: {'OK' if results['no_inf'] else 'FALLO'}")

        for key, value in results.items():
            if 'verified' in key:
                print(f"  {key}: {'OK' if value else 'FALLO'}")

        all_ok = all(results.values())
        print(f"\n  {'VERIFICACION EXITOSA' if all_ok else 'HAY ERRORES'}")

    return results


if __name__ == "__main__":
    """
    Script de prueba para el modulo Fisher.

    Demuestra:
    1. Creacion de datos sinteticos
    2. Calculo de Fisher ratios
    3. Amplificacion
    4. Visualizacion
    """
    print("="*60)
    print("PRUEBA DEL MODULO FISHER")
    print("="*60)
    print()

    # Crear datos sinteticos con separacion conocida
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    # Simular K caracteristicas con diferente capacidad de separacion
    # Caracteristica 0: Muy buena separacion (medias muy distintas)
    # Caracteristica 1: Buena separacion
    # ...
    # Caracteristica 9: Sin separacion (misma distribucion)

    X_class0 = np.zeros((n_samples // 2, n_features))
    X_class1 = np.zeros((n_samples // 2, n_features))

    separations = [4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.1, 0.05, 0.0]

    for i, sep in enumerate(separations):
        X_class0[:, i] = np.random.randn(n_samples // 2) * 0.5 - sep/2
        X_class1[:, i] = np.random.randn(n_samples // 2) * 0.5 + sep/2

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    print("Datos sinteticos creados:")
    print(f"  Total: {n_samples} muestras x {n_features} caracteristicas")
    print(f"  Separaciones programadas: {separations}")
    print()

    # Calcular Fisher
    fisher = FisherRatio()
    fisher.fit(X, y, class_names=['Enfermo', 'Normal'])

    print()

    # Verificar
    verify_fisher_calculation(X, y, fisher.get_result())

    # Amplificar
    X_amp = fisher.amplify(X)

    print()
    print("-"*60)
    print("Comparacion de escalas (primeras 3 caracteristicas):")
    print("-"*60)
    print(f"Original std: {np.std(X, axis=0)[:3]}")
    print(f"Amplificado std: {np.std(X_amp, axis=0)[:3]}")
    print(f"Factor (Fisher): {fisher.fisher_ratios_[:3]}")

    print()
    print("Prueba completada.")
