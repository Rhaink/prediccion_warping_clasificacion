#!/usr/bin/env python3
"""
Generate per-layer feature maps for landmark prediction and classification.

This script saves ordered visual outputs under outputs/mapas_caracteristicas/
for a single input image:
  - Original + resized inputs
  - Landmark model feature maps (ResNet layers, coord attention, avgpool, head)
  - Warped image
  - Classifier feature maps + logits/probabilities
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src_v2.constants import DEFAULT_IMAGE_SIZE
from src_v2.gui import config as gui_config
from src_v2.gui.visualizer import render_landmarks_overlay
from src_v2.models.classifier import load_classifier_checkpoint
from src_v2.models.resnet_landmark import ResNet18Landmarks
from src_v2.processing.warp import piecewise_affine_warp, scale_landmarks_from_centroid
from src_v2.visualization.feature_extractor import FeatureExtractor, LandmarkFeatureExtractor
from src_v2.visualization.feature_visualizer import FeatureVisualizer
from src_v2.visualization.utils import normalize_feature_map, select_top_channels_by_variance


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ordered feature-map visualizations for the pipeline."
    )
    parser.add_argument("--image", required=True, help="Path to input X-ray image.")
    parser.add_argument(
        "--output-dir",
        default="outputs/mapas_caracteristicas",
        help="Root output directory for generated images.",
    )
    parser.add_argument(
        "--landmark-checkpoint",
        default=None,
        help="Optional path to landmark checkpoint. Defaults to GUI config.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default=None,
        help="Optional path to classifier checkpoint. Defaults to GUI config.",
    )
    parser.add_argument(
        "--landmark-model-index",
        type=int,
        default=0,
        help="Index within GUI LANDMARK_MODELS to select when no checkpoint is given.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device override (cpu or cuda). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--channels-per-layer",
        type=int,
        default=16,
        help="Number of channels to show per layer grid.",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE preprocessing for landmark input.",
    )
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    """Return a filesystem-safe stem for output directory names."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "image"


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 for saving."""
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        return (image * 255).clip(0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def save_gray_image(image: np.ndarray, path: Path) -> None:
    """Save a grayscale image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), to_uint8(image))


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """Load grayscale image from disk."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize image to square size."""
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """Apply CLAHE to a grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def prepare_landmark_input(
    image_gray: np.ndarray,
    device: torch.device,
    use_clahe: bool,
    clip_limit: float,
    tile_size: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Prepare landmark model input tensor and the processed grayscale image."""
    resized = resize_image(image_gray, DEFAULT_IMAGE_SIZE)
    processed = apply_clahe(resized, clip_limit, tile_size) if use_clahe else resized
    image_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    image_tensor = (image_tensor - mean) / std
    return image_tensor, processed


def prepare_classifier_input(
    image_gray: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Prepare classifier input tensor from grayscale image."""
    resized = resize_image(image_gray, DEFAULT_IMAGE_SIZE)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (image_tensor - mean) / std


def load_landmark_model(checkpoint_path: Path, device: torch.device) -> ResNet18Landmarks:
    """Load landmark model and auto-detect coord-attention/head depth."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    has_coord_attention = any("coord_attention" in key for key in checkpoint["model_state_dict"])
    head_keys = [k for k in checkpoint["model_state_dict"] if "head" in k and "weight" in k]
    is_deep_head = "head.9.weight" in head_keys or "head.1.weight" in head_keys

    hidden_dim = 768
    if is_deep_head and "head.5.weight" in checkpoint["model_state_dict"]:
        hidden_dim = checkpoint["model_state_dict"]["head.5.weight"].shape[0]
    elif not is_deep_head and "head.2.weight" in checkpoint["model_state_dict"]:
        hidden_dim = checkpoint["model_state_dict"]["head.2.weight"].shape[0]

    model = ResNet18Landmarks(
        pretrained=False,
        freeze_backbone=True,
        hidden_dim=hidden_dim,
        use_coord_attention=has_coord_attention,
        deep_head=is_deep_head,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def unwrap_feature(feature_value: object) -> Optional[np.ndarray]:
    """Extract a numpy feature map from tensors or tuple/list outputs."""
    if feature_value is None:
        return None
    if isinstance(feature_value, (list, tuple)):
        for item in feature_value:
            if isinstance(item, (torch.Tensor, np.ndarray)):
                feature_value = item
                break
    if isinstance(feature_value, torch.Tensor):
        feature_value = feature_value.detach().cpu().numpy()
    if isinstance(feature_value, np.ndarray):
        return feature_value
    return None


def ensure_spatial_feature(feature_map: np.ndarray) -> np.ndarray:
    """Return spatial feature map with shape (C, H, W)."""
    if feature_map.ndim == 4:
        feature_map = feature_map[0]
    if feature_map.ndim != 3:
        raise ValueError(f"Expected 3D feature map, got shape {feature_map.shape}")
    return feature_map


def save_feature_grid(
    feature_map: np.ndarray,
    output_path: Path,
    channels_per_layer: int,
    title: str,
) -> None:
    """Save grid visualization for a feature map."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_map = ensure_spatial_feature(feature_map)
    channel_indices = select_top_channels_by_variance(feature_map, channels_per_layer)
    n_cols = int(math.ceil(math.sqrt(len(channel_indices))))
    visualizer = FeatureVisualizer(dpi=200, font_size=9)
    visualizer.plot_feature_grid(
        feature_map=feature_map,
        n_cols=n_cols,
        channel_indices=channel_indices,
        title=title,
    )
    visualizer.save(str(output_path))
    visualizer.close()


def save_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str = "magma",
    alpha: float = 0.4,
) -> None:
    """Save an overlay of heatmap on top of a grayscale image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = resize_image(image, DEFAULT_IMAGE_SIZE)
    heatmap = resize_image(heatmap, DEFAULT_IMAGE_SIZE)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(image, cmap="gray")
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_vector_heatmap(values: np.ndarray, output_path: Path, title: str) -> None:
    """Save vector values as a square heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = values.flatten().astype(np.float32)
    grid_size = int(math.ceil(math.sqrt(values.size)))
    padded = np.zeros(grid_size * grid_size, dtype=np.float32)
    padded[: values.size] = values
    grid = padded.reshape(grid_size, grid_size)
    grid = normalize_feature_map(grid, method="percentile")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(grid, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_vector_bar(
    values: np.ndarray,
    output_path: Path,
    title: str,
    labels: Optional[List[str]] = None,
) -> None:
    """Save vector values as a bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = values.flatten().astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.bar(range(values.size), values, color="#3b6ea8")
    ax.set_title(title, fontsize=10)
    if labels:
        ax.set_xticks(range(values.size))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_canonical_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load canonical shape and Delaunay triangles from config paths."""
    with open(gui_config.CANONICAL_SHAPE, "r") as handle:
        canonical_data = json.load(handle)
    with open(gui_config.DELAUNAY_TRIANGLES, "r") as handle:
        triangulation_data = json.load(handle)
    canonical_shape = np.array(canonical_data["canonical_shape_pixels"])
    triangles = np.array(triangulation_data["triangles"])
    return canonical_shape, triangles


def get_classifier_layers(classifier: torch.nn.Module) -> List[Tuple[str, str]]:
    """Return ordered layer names and labels for ResNet backbones."""
    backbone = getattr(classifier, "backbone", None)
    if backbone is None or not hasattr(backbone, "layer1"):
        return []
    return [
        ("backbone.layer1", "layer1"),
        ("backbone.layer2", "layer2"),
        ("backbone.layer3", "layer3"),
        ("backbone.layer4", "layer4"),
        ("backbone.avgpool", "avgpool"),
    ]


def main() -> None:
    """Run the feature-map pipeline generator."""
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    image_path = Path(args.image)
    image_gray = load_grayscale_image(image_path)
    image_resized = resize_image(image_gray, DEFAULT_IMAGE_SIZE)

    output_root = Path(args.output_dir) / safe_stem(image_path)
    output_root.mkdir(parents=True, exist_ok=True)

    original_dir = output_root / "00_original"
    save_gray_image(image_gray, original_dir / "original.png")
    save_gray_image(image_resized, original_dir / "original_224.png")

    # Landmark model setup
    if args.landmark_checkpoint:
        landmark_checkpoint = Path(args.landmark_checkpoint)
    else:
        if args.landmark_model_index >= len(gui_config.LANDMARK_MODELS):
            raise ValueError("Invalid landmark model index for GUI config.")
        landmark_checkpoint = gui_config.LANDMARK_MODELS[args.landmark_model_index]

    landmark_model = load_landmark_model(landmark_checkpoint, device)
    landmark_tensor, landmark_input = prepare_landmark_input(
        image_resized,
        device,
        use_clahe=not args.no_clahe,
        clip_limit=gui_config.CLAHE_CLIP_LIMIT,
        tile_size=gui_config.CLAHE_TILE_SIZE
        if isinstance(gui_config.CLAHE_TILE_SIZE, int)
        else gui_config.CLAHE_TILE_SIZE[0],
    )

    landmark_dir = output_root / "01_landmarks"
    save_gray_image(landmark_input, landmark_dir / "input" / "landmark_input.png")

    landmark_extractor = LandmarkFeatureExtractor(landmark_model)
    with torch.no_grad():
        landmark_output = landmark_model(landmark_tensor)
    landmark_features = landmark_extractor.get_features()
    landmark_extractor.remove_hooks()

    landmarks_norm = landmark_output.view(15, 2).detach().cpu().numpy()
    landmarks_px = landmarks_norm * DEFAULT_IMAGE_SIZE

    layer_map = [
        ("backbone_conv.4", "layer1"),
        ("backbone_conv.5", "layer2"),
        ("backbone_conv.6", "layer3"),
        ("backbone_conv.7", "layer4"),
    ]

    backbone_dir = landmark_dir / "backbone"
    for layer_key, layer_label in layer_map:
        feature_value = unwrap_feature(landmark_features.get(layer_key))
        if feature_value is None:
            continue
        feature_map = ensure_spatial_feature(feature_value)
        grid_path = backbone_dir / f"{layer_label}_grid.png"
        title = f"Landmarks {layer_label} ({feature_map.shape[0]}x{feature_map.shape[1:]})"
        save_feature_grid(feature_map, grid_path, args.channels_per_layer, title)

        mean_map = normalize_feature_map(feature_map.mean(axis=0), method="percentile")
        overlay_path = backbone_dir / f"{layer_label}_mean_overlay.png"
        save_heatmap_overlay(
            landmark_input,
            mean_map,
            overlay_path,
            title=f"Landmarks {layer_label} mean activation",
        )

    coord_output = unwrap_feature(landmark_features.get("coord_attention"))
    layer4_output = unwrap_feature(landmark_features.get("backbone_conv.7"))
    if coord_output is not None:
        coord_dir = landmark_dir / "coord_attention"
        coord_map = ensure_spatial_feature(coord_output)
        coord_grid_path = coord_dir / "coord_attention_grid.png"
        save_feature_grid(
            coord_map,
            coord_grid_path,
            args.channels_per_layer,
            title="Coordinate attention output",
        )

        coord_mean = normalize_feature_map(coord_map.mean(axis=0), method="percentile")
        coord_overlay_path = coord_dir / "coord_attention_mean_overlay.png"
        save_heatmap_overlay(
            landmark_input,
            coord_mean,
            coord_overlay_path,
            title="Coordinate attention mean activation",
        )

        if layer4_output is not None:
            base_map = ensure_spatial_feature(layer4_output)
            ratio = coord_map / (base_map + 1e-6)
            ratio_mean = normalize_feature_map(ratio.mean(axis=0), method="percentile")
            ratio_path = coord_dir / "coord_attention_ratio_overlay.png"
            save_heatmap_overlay(
                landmark_input,
                ratio_mean,
                ratio_path,
                title="Coordinate attention ratio (output/input)",
            )

    avgpool_output = unwrap_feature(landmark_features.get("avgpool"))
    if avgpool_output is not None:
        avgpool_dir = landmark_dir / "avgpool"
        avgpool_map = ensure_spatial_feature(avgpool_output)
        avgpool_vector = avgpool_map.reshape(avgpool_map.shape[0])
        save_vector_heatmap(
            avgpool_vector,
            avgpool_dir / "avgpool_feature_vector.png",
            title="AvgPool feature vector (landmarks)",
        )

    head_output = unwrap_feature(landmark_features.get("head"))
    if head_output is not None:
        head_dir = landmark_dir / "head"
        head_vector = head_output.reshape(-1)
        coord_labels = [f"L{i+1}_{axis}" for i in range(15) for axis in ("x", "y")]
        save_vector_bar(
            head_vector,
            head_dir / "head_output_vector.png",
            title="Landmark head output (normalized coords)",
            labels=coord_labels,
        )

    head_dir = landmark_dir / "head"
    head_dir.mkdir(parents=True, exist_ok=True)
    overlay = render_landmarks_overlay(image_resized, landmarks_px, show_labels=True)
    overlay.save(head_dir / "landmarks_overlay.png")
    with open(head_dir / "landmarks_px.json", "w") as handle:
        json.dump(landmarks_px.tolist(), handle, indent=2)

    # Warping
    warping_dir = output_root / "02_warping"
    warping_dir.mkdir(parents=True, exist_ok=True)
    canonical_shape, triangles = load_canonical_data()
    warping_failed = False
    try:
        scaled_landmarks = scale_landmarks_from_centroid(
            landmarks_px, scale=gui_config.MARGIN_SCALE
        )
        warped = piecewise_affine_warp(
            image=image_resized,
            source_landmarks=scaled_landmarks,
            target_landmarks=canonical_shape,
            triangles=triangles,
            use_full_coverage=gui_config.USE_FULL_COVERAGE,
        )
    except Exception:
        warped = image_resized.copy()
        warping_failed = True

    save_gray_image(warped, warping_dir / "warped.png")

    # Classification model setup
    classifier_checkpoint = (
        Path(args.classifier_checkpoint)
        if args.classifier_checkpoint
        else gui_config.CLASSIFIER_CHECKPOINT
    )
    classifier, metadata = load_classifier_checkpoint(str(classifier_checkpoint), device=device)
    classifier.eval()

    classifier_dir = output_root / "03_classifier"
    save_gray_image(warped, classifier_dir / "input" / "classifier_input.png")

    classifier_layers = get_classifier_layers(classifier)
    classifier_tensor = prepare_classifier_input(warped, device)

    classifier_features: Dict[str, np.ndarray] = {}
    if classifier_layers:
        layer_names = [layer[0] for layer in classifier_layers]
        classifier_extractor = FeatureExtractor(classifier, target_layers=layer_names)
        with torch.no_grad():
            logits = classifier(classifier_tensor)
        classifier_features = classifier_extractor.get_features()
        classifier_extractor.remove_hooks()
    else:
        with torch.no_grad():
            logits = classifier(classifier_tensor)

    backbone_dir = classifier_dir / "backbone"
    for layer_key, layer_label in classifier_layers:
        feature_value = unwrap_feature(classifier_features.get(layer_key))
        if feature_value is None:
            continue
        feature_map = ensure_spatial_feature(feature_value)
        grid_path = backbone_dir / f"{layer_label}_grid.png"
        title = f"Classifier {layer_label} ({feature_map.shape[0]}x{feature_map.shape[1:]})"
        save_feature_grid(feature_map, grid_path, args.channels_per_layer, title)

        mean_map = normalize_feature_map(feature_map.mean(axis=0), method="percentile")
        overlay_path = backbone_dir / f"{layer_label}_mean_overlay.png"
        save_heatmap_overlay(
            warped,
            mean_map,
            overlay_path,
            title=f"Classifier {layer_label} mean activation",
        )

        if layer_label == "avgpool":
            avgpool_dir = classifier_dir / "avgpool"
            avgpool_vector = feature_map.reshape(feature_map.shape[0])
            save_vector_heatmap(
                avgpool_vector,
                avgpool_dir / "avgpool_feature_vector.png",
                title="AvgPool feature vector (classifier)",
            )

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    class_names = metadata.get("class_names", ["COVID", "Normal", "Viral_Pneumonia"])
    probabilities = {name: float(prob) for name, prob in zip(class_names, probs)}
    predicted_class = class_names[int(np.argmax(probs))]

    logits_dir = classifier_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    save_vector_bar(
        probs,
        logits_dir / "classification_probabilities.png",
        title="Classifier probabilities",
        labels=class_names,
    )
    with open(logits_dir / "classification_probabilities.json", "w") as handle:
        json.dump(probabilities, handle, indent=2)

    manifest = {
        "image_path": str(image_path),
        "output_root": str(output_root),
        "landmark_checkpoint": str(landmark_checkpoint),
        "classifier_checkpoint": str(classifier_checkpoint),
        "landmark_model_index": args.landmark_model_index,
        "use_clahe": not args.no_clahe,
        "clahe_clip_limit": gui_config.CLAHE_CLIP_LIMIT,
        "clahe_tile_size": gui_config.CLAHE_TILE_SIZE,
        "margin_scale": gui_config.MARGIN_SCALE,
        "use_full_coverage": gui_config.USE_FULL_COVERAGE,
        "warping_failed": warping_failed,
        "predicted_landmarks_px": landmarks_px.tolist(),
        "predicted_landmarks_norm": landmarks_norm.tolist(),
        "predicted_class": predicted_class,
        "probabilities": probabilities,
    }
    with open(output_root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Visualizaciones listas en: {output_root}")


if __name__ == "__main__":
    main()
