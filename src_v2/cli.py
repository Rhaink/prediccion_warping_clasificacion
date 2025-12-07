"""
CLI principal para COVID-19 Landmark Detection.

Uso:
    python -m src_v2 train --help
    python -m src_v2 evaluate --checkpoint model.pt
    python -m src_v2 predict --image xray.png --checkpoint model.pt
    python -m src_v2 warp --input-dir data/ --output-dir warped/
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from src_v2.constants import DEFAULT_IMAGE_SIZE, NUM_LANDMARKS


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = typer.Typer(
    name="src_v2",
    help="COVID-19 Detection via Anatomical Landmarks - CLI",
    add_completion=False,
)


def get_device(device: str) -> "torch.device":
    """Obtener dispositivo de PyTorch."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def setup_hydra_config(config_path: str, config_name: str, overrides: list):
    """Cargar configuracion con Hydra."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(config_path).resolve()

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


@app.command()
def train(
    config_path: str = typer.Option(
        "src_v2/conf",
        "--config-path",
        "-c",
        help="Path al directorio de configuracion Hydra"
    ),
    config_name: str = typer.Option(
        "config",
        "--config-name",
        "-n",
        help="Nombre del archivo de configuracion (sin .yaml)"
    ),
    data_root: Optional[str] = typer.Option(
        None,
        "--data-root",
        "-d",
        help="Directorio raiz de datos"
    ),
    csv_path: Optional[str] = typer.Option(
        None,
        "--csv-path",
        help="Path al CSV de coordenadas"
    ),
    checkpoint_dir: str = typer.Option(
        "checkpoints",
        "--checkpoint-dir",
        help="Directorio para guardar checkpoints"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Seed para reproducibilidad"
    ),
    phase1_epochs: int = typer.Option(
        15,
        "--phase1-epochs",
        help="Epocas para fase 1 (backbone congelado)"
    ),
    phase2_epochs: int = typer.Option(
        100,
        "--phase2-epochs",
        help="Epocas para fase 2 (fine-tuning)"
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="Tamano de batch"
    ),
):
    """
    Entrenar modelo de prediccion de landmarks.

    Entrenamiento en dos fases:
    - Fase 1: Backbone congelado, entrenar solo cabeza
    - Fase 2: Fine-tuning completo con LR diferenciado
    """
    import torch
    from torch.utils.data import DataLoader, random_split

    from src_v2.data import LandmarkDataset, get_train_transforms, get_val_transforms
    from src_v2.models import create_model, CombinedLandmarkLoss
    from src_v2.training.trainer import LandmarkTrainer

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Training")
    logger.info("=" * 60)

    # Configurar seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Cargar configuracion si existe
    overrides = []
    if data_root:
        overrides.append(f"paths.data_root={data_root}")
    if csv_path:
        overrides.append(f"paths.csv_path={csv_path}")

    config_dir = Path(config_path)
    if config_dir.exists():
        try:
            cfg = setup_hydra_config(config_path, config_name, overrides)
            logger.info("Configuracion cargada desde %s/%s.yaml", config_path, config_name)

            # Usar valores de config si no se especificaron en CLI
            if data_root is None:
                data_root = cfg.paths.data_root
            if csv_path is None:
                csv_path = cfg.paths.csv_path
        except Exception as e:
            logger.warning("No se pudo cargar config Hydra: %s", e)

    # Valores por defecto si no se especificaron
    if data_root is None:
        data_root = "data/"
    if csv_path is None:
        csv_path = "data/coordenadas/coordenadas_maestro.csv"

    # Verificar paths
    if not Path(data_root).exists():
        logger.error("Directorio de datos no existe: %s", data_root)
        raise typer.Exit(code=1)

    if not Path(csv_path).exists():
        logger.error("CSV de coordenadas no existe: %s", csv_path)
        raise typer.Exit(code=1)

    # Crear dataset
    logger.info("Cargando dataset desde %s", data_root)

    full_dataset = LandmarkDataset(
        csv_path=csv_path,
        root_dir=data_root,
        transform=None  # Se asigna despues del split
    )

    # Split train/val/test (75/15/10)
    n_total = len(full_dataset)
    n_train = int(0.75 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    logger.info("Dataset splits: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    # Aplicar transforms
    train_transform = get_train_transforms(image_size=DEFAULT_IMAGE_SIZE)
    val_transform = get_val_transforms(image_size=DEFAULT_IMAGE_SIZE)

    # Wrapper para aplicar transforms
    class TransformWrapper:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, landmarks, meta = self.dataset[idx]
            if self.transform:
                image, landmarks = self.transform(image, landmarks)
            return image, landmarks, meta

    train_set = TransformWrapper(train_dataset, train_transform)
    val_set = TransformWrapper(val_dataset, val_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Crear modelo
    logger.info("Creando modelo ResNet18Landmarks")
    model = create_model(pretrained=True)
    model = model.to(torch_device)

    # Loss function
    criterion = CombinedLandmarkLoss(
        image_size=DEFAULT_IMAGE_SIZE,
        central_weight=1.0,
        symmetry_weight=0.5
    )

    # Trainer
    trainer = LandmarkTrainer(
        model=model,
        device=torch_device,
        save_dir=checkpoint_dir,
        image_size=DEFAULT_IMAGE_SIZE
    )

    # Entrenar
    logger.info("Iniciando entrenamiento...")
    history = trainer.train_full(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
    )

    # Guardar modelo final
    final_path = Path(checkpoint_dir) / "final_model.pt"
    trainer.save_model(str(final_path))
    logger.info("Modelo guardado en: %s", final_path)

    logger.info("=" * 60)
    logger.info("Entrenamiento completado!")
    logger.info("=" * 60)


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(
        ...,
        help="Path al checkpoint del modelo"
    ),
    data_root: str = typer.Option(
        "data/",
        "--data-root",
        "-d",
        help="Directorio raiz de datos"
    ),
    csv_path: str = typer.Option(
        "data/coordenadas/coordenadas_maestro.csv",
        "--csv-path",
        help="Path al CSV de coordenadas"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Tamano de batch"
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar resultados en JSON"
    ),
    use_tta: bool = typer.Option(
        False,
        "--tta",
        help="Usar Test-Time Augmentation"
    ),
):
    """
    Evaluar modelo en dataset de test.

    Calcula metricas de error por landmark y por categoria.
    """
    import json

    import torch
    from torch.utils.data import DataLoader

    from src_v2.data import LandmarkDataset, get_val_transforms
    from src_v2.models import create_model
    from src_v2.evaluation.metrics import evaluate_model, evaluate_model_with_tta

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Evaluation")
    logger.info("=" * 60)

    # Verificar checkpoint
    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo desde: %s", checkpoint)
    model = create_model(pretrained=False)

    checkpoint_data = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model = model.to(torch_device)
    model.eval()

    # Crear dataset de test
    logger.info("Cargando dataset desde: %s", data_root)

    test_transform = get_val_transforms(image_size=DEFAULT_IMAGE_SIZE)
    test_dataset = LandmarkDataset(
        csv_path=csv_path,
        root_dir=data_root,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluar
    logger.info("Evaluando modelo...")
    if use_tta:
        logger.info("Usando Test-Time Augmentation")
        results = evaluate_model_with_tta(model, test_loader, torch_device)
    else:
        results = evaluate_model(model, test_loader, torch_device)

    # Mostrar resultados
    logger.info("-" * 40)
    logger.info("RESULTADOS")
    logger.info("-" * 40)
    logger.info("Error promedio: %.2f px", results["mean_error"])
    logger.info("Error mediana: %.2f px", results["median_error"])
    logger.info("Error std: %.2f px", results["std_error"])

    if "per_landmark" in results:
        logger.info("\nError por landmark:")
        for name, error in results["per_landmark"].items():
            logger.info("  %s: %.2f px", name, error)

    if "per_category" in results:
        logger.info("\nError por categoria:")
        for cat, error in results["per_category"].items():
            logger.info("  %s: %.2f px", cat, error)

    # Guardar JSON si se especifica
    if output_json:
        # Convertir numpy types a Python types para JSON
        results_json = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_json[key] = {k: float(v) for k, v in value.items()}
            elif hasattr(value, "item"):
                results_json[key] = value.item()
            else:
                results_json[key] = float(value)

        with open(output_json, "w") as f:
            json.dump(results_json, f, indent=2)
        logger.info("Resultados guardados en: %s", output_json)

    logger.info("=" * 60)
    logger.info("Evaluacion completada!")
    logger.info("=" * 60)


@app.command()
def predict(
    image: str = typer.Argument(
        ...,
        help="Path a la imagen de rayos X"
    ),
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path al checkpoint del modelo"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar imagen con landmarks visualizados"
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--json",
        "-j",
        help="Guardar coordenadas en JSON"
    ),
):
    """
    Predecir landmarks en una imagen de rayos X.

    Muestra coordenadas predichas y opcionalmente guarda visualizacion.
    """
    import json

    import cv2
    import numpy as np
    import torch
    from PIL import Image as PILImage

    from src_v2.constants import LANDMARK_NAMES, IMAGENET_MEAN, IMAGENET_STD
    from src_v2.models import create_model

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Predict")
    logger.info("=" * 60)

    # Verificar archivos
    if not Path(image).exists():
        logger.error("Imagen no existe: %s", image)
        raise typer.Exit(code=1)

    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo desde: %s", checkpoint)
    model = create_model(pretrained=False)

    checkpoint_data = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model = model.to(torch_device)
    model.eval()

    # Cargar y preprocesar imagen
    logger.info("Procesando imagen: %s", image)

    img = PILImage.open(image).convert("RGB")
    original_size = img.size

    # Resize a 224x224
    img_resized = img.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), PILImage.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    # Normalizar con ImageNet stats
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img_normalized = (img_array - mean) / std

    # Convertir a tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor.to(torch_device)

    # Predecir
    with torch.no_grad():
        predictions = model(img_tensor)

    # Convertir a coordenadas en pixeles
    landmarks = predictions.squeeze().cpu().numpy()
    landmarks = landmarks.reshape(NUM_LANDMARKS, 2) * DEFAULT_IMAGE_SIZE

    # Mostrar resultados
    logger.info("-" * 40)
    logger.info("LANDMARKS PREDICHOS (pixeles en imagen %dx%d):", DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    logger.info("-" * 40)

    landmark_dict = {}
    for i, name in enumerate(LANDMARK_NAMES):
        x, y = landmarks[i]
        logger.info("  %s: (%.1f, %.1f)", name, x, y)
        landmark_dict[name] = {"x": float(x), "y": float(y)}

    # Guardar JSON si se especifica
    if output_json:
        result = {
            "image": str(image),
            "image_size": DEFAULT_IMAGE_SIZE,
            "original_size": list(original_size),
            "landmarks": landmark_dict
        }
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Coordenadas guardadas en: %s", output_json)

    # Guardar visualizacion si se especifica
    if output:
        img_vis = np.array(img_resized)

        # Dibujar landmarks
        for i, (x, y) in enumerate(landmarks):
            x, y = int(x), int(y)
            cv2.circle(img_vis, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(
                img_vis,
                f"L{i+1}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 0),
                1
            )

        cv2.imwrite(output, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        logger.info("Visualizacion guardada en: %s", output)

    logger.info("=" * 60)
    logger.info("Prediccion completada!")
    logger.info("=" * 60)


@app.command()
def warp(
    input_dir: str = typer.Argument(
        ...,
        help="Directorio con imagenes de entrada"
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directorio de salida para imagenes warpeadas"
    ),
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path al checkpoint del modelo para prediccion de landmarks"
    ),
    canonical_shape: str = typer.Option(
        "data/canonical/canonical_shape.npy",
        "--canonical",
        help="Path a la forma canonica (.npy)"
    ),
    triangles: str = typer.Option(
        "data/canonical/delaunay_triangles.npy",
        "--triangles",
        help="Path a los triangulos de Delaunay (.npy)"
    ),
    margin_scale: float = typer.Option(
        1.05,
        "--margin-scale",
        "-m",
        help="Factor de escala para margenes"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    pattern: str = typer.Option(
        "**/*.png",
        "--pattern",
        "-p",
        help="Patron glob para buscar imagenes"
    ),
):
    """
    Aplicar warping geometrico a un dataset de imagenes.

    Usa el modelo de landmarks para predecir puntos anatomicos
    y aplica transformacion piecewise affine a forma canonica.
    """
    import sys

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from src_v2.constants import IMAGENET_MEAN, IMAGENET_STD
    from src_v2.models import create_model

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Warp Dataset")
    logger.info("=" * 60)

    # Verificar paths
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Directorio de entrada no existe: %s", input_dir)
        raise typer.Exit(code=1)

    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    # Cargar forma canonica y triangulos
    if not Path(canonical_shape).exists():
        logger.error("Forma canonica no existe: %s", canonical_shape)
        logger.info("Genera la forma canonica primero con scripts de GPA")
        raise typer.Exit(code=1)

    if not Path(triangles).exists():
        logger.error("Triangulos de Delaunay no existen: %s", triangles)
        raise typer.Exit(code=1)

    canonical = np.load(canonical_shape)
    tri = np.load(triangles)
    logger.info("Forma canonica cargada: %s", canonical.shape)

    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo desde: %s", checkpoint)
    model = create_model(pretrained=False)

    checkpoint_data = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model = model.to(torch_device)
    model.eval()

    # Importar funcion de warp (del proyecto existente)
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from scripts.piecewise_affine_warp import piecewise_affine_warp
    except ImportError:
        logger.error("No se pudo importar piecewise_affine_warp")
        logger.info("Verifica que scripts/piecewise_affine_warp.py existe")
        raise typer.Exit(code=1)

    # Buscar imagenes
    images = list(input_path.glob(pattern))
    logger.info("Encontradas %d imagenes con patron '%s'", len(images), pattern)

    if len(images) == 0:
        logger.warning("No se encontraron imagenes")
        raise typer.Exit(code=0)

    # Procesar imagenes
    success = 0
    failed = 0

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    for img_path in tqdm(images, desc="Warping"):
        try:
            # Cargar imagen
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed += 1
                continue

            # Resize a 224x224
            if img.shape[0] != DEFAULT_IMAGE_SIZE or img.shape[1] != DEFAULT_IMAGE_SIZE:
                img = cv2.resize(img, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

            # Preprocesar para modelo (convertir a RGB y normalizar)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            img_normalized = (img_float - mean) / std

            # Convertir a tensor
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor.to(torch_device)

            # Predecir landmarks
            with torch.no_grad():
                predictions = model(img_tensor)

            landmarks = predictions.squeeze().cpu().numpy()
            landmarks = landmarks.reshape(NUM_LANDMARKS, 2) * DEFAULT_IMAGE_SIZE

            # Escalar landmarks desde centroide
            centroid = landmarks.mean(axis=0)
            landmarks_scaled = centroid + (landmarks - centroid) * margin_scale

            # Clip a limites de imagen
            landmarks_clipped = np.clip(landmarks_scaled, 2, DEFAULT_IMAGE_SIZE - 3)

            # Escalar canonical shape
            canonical_scaled = centroid + (canonical - canonical.mean(axis=0)) * margin_scale
            canonical_clipped = np.clip(canonical_scaled, 2, DEFAULT_IMAGE_SIZE - 3)

            # Aplicar warp
            warped = piecewise_affine_warp(
                img,
                landmarks_clipped.astype(np.float32),
                canonical_clipped.astype(np.float32),
                tri
            )

            # Guardar
            relative_path = img_path.relative_to(input_path)
            out_file = output_path / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_file), warped)
            success += 1

        except Exception as e:
            logger.debug("Error procesando %s: %s", img_path, e)
            failed += 1

    logger.info("-" * 40)
    logger.info("RESUMEN")
    logger.info("-" * 40)
    logger.info("Imagenes procesadas: %d", success)
    logger.info("Imagenes fallidas: %d", failed)
    logger.info("Directorio de salida: %s", output_dir)

    logger.info("=" * 60)
    logger.info("Warping completado!")
    logger.info("=" * 60)


@app.command()
def version():
    """Mostrar version del paquete."""
    from src_v2 import __version__
    typer.echo(f"COVID-19 Landmark Detection v{__version__}")


def main():
    """Entry point principal."""
    app()


if __name__ == "__main__":
    main()
