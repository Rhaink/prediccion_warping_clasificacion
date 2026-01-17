#!/usr/bin/env python3
"""
Generate Coordinate Attention figures for the thesis (paper-style).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from src_v2.constants import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src_v2.data.transforms import apply_clahe
from src_v2.models.resnet_landmark import ResNet18Landmarks


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def list_dataset_images(max_samples: int = 200) -> list[Path]:
    """Collect a sample of dataset images for representative selection."""
    roots = [
        Path("data/dataset/Normal"),
        Path("data/dataset/COVID"),
        Path("data/dataset/Viral_Pneumonia"),
        Path("data/dataset/COVID-19_Radiography_Dataset/Normal"),
        Path("data/dataset/COVID-19_Radiography_Dataset/COVID"),
        Path("data/dataset/COVID-19_Radiography_Dataset/Viral Pneumonia"),
        Path("data/dataset/COVID-19_Radiography_Dataset/Viral_Pneumonia"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob("*.png")))
    if not candidates:
        candidates = sorted(Path("data/dataset").rglob("*.png"))
    if not candidates:
        return []
    step = max(1, len(candidates) // max_samples)
    return candidates[::step][:max_samples]


def find_sample_image() -> Path:
    """Find a fallback X-ray image from the dataset."""
    candidates = list_dataset_images(max_samples=1)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No PNG images found under data/dataset")


def should_apply_clahe(img_path: Path) -> bool:
    """Skip CLAHE if the file looks already preprocessed."""
    return "clahe" not in str(img_path).lower()


def preprocess_image(img_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
    """Apply optional CLAHE, resize, and ImageNet normalization."""
    img = Image.open(img_path).convert("RGB")
    if should_apply_clahe(img_path):
        img = apply_clahe(img)
    img = img.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), Image.BILINEAR)
    img_np = np.array(img)

    img_norm = img_np.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    img_norm = (img_norm - mean) / std

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return img_np, tensor


def load_model(checkpoint_path: Path, device: torch.device) -> ResNet18Landmarks:
    """Load the landmark model with Coordinate Attention enabled."""
    model = ResNet18Landmarks(
        pretrained=False,
        freeze_backbone=False,
        dropout_rate=DEFAULT_DROPOUT_RATE,
        hidden_dim=256,
        use_coord_attention=True,
        deep_head=True,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def compute_attention_maps(
    model: ResNet18Landmarks,
    x: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a_h, a_w and their outer product A."""
    with torch.no_grad():
        feats = model.backbone_conv(x)
        ca = model.coord_attention
        _, _, h, w = feats.shape

        x_h = ca.pool_h(feats)
        x_w = ca.pool_w(feats).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = ca.conv1(y)
        y = ca.bn1(y)
        y = ca.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = ca.conv_h(x_h).sigmoid()
        a_w = ca.conv_w(x_w).sigmoid()

        a_h_mean = a_h.mean(dim=1).squeeze().cpu().numpy()
        a_w_mean = a_w.mean(dim=1).squeeze().cpu().numpy()
        if a_h_mean.ndim == 2:
            a_h_mean = a_h_mean[:, 0]
        if a_w_mean.ndim == 2:
            a_w_mean = a_w_mean[0, :]
        att2d = np.outer(a_h_mean, a_w_mean)
        return a_h_mean, a_w_mean, att2d


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    gamma: float = 0.65,
    cmap: str = "magma",
) -> np.ndarray:
    """Overlay a heatmap on an RGB image with contrast shaping."""
    hmin, hmax = heatmap.min(), heatmap.max()
    norm = (heatmap - hmin) / (hmax - hmin + 1e-8)
    norm = np.clip(norm ** gamma, 0.0, 1.0)
    colored = (plt.get_cmap(cmap)(norm)[:, :, :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * image_rgb + alpha * colored
    return blended.astype(np.uint8)


def crop_warped_black_border(image_rgb: np.ndarray, pad: int = 6) -> np.ndarray:
    """Crop black margins from warped images to focus on anatomy."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image_rgb
    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, image_rgb.shape[1])
    y1 = min(y + h + pad, image_rgb.shape[0])
    return image_rgb[y0:y1, x0:x1]


def score_attention(a_h: np.ndarray, a_w: np.ndarray, att2d: np.ndarray) -> float:
    """Score attention for visual clarity (higher variation and peaks are better)."""
    peak_h = float(np.max(a_h) - np.min(a_h))
    peak_w = float(np.max(a_w) - np.min(a_w))
    std_h = float(np.std(a_h))
    std_w = float(np.std(a_w))
    return peak_h + peak_w + 0.3 * (std_h + std_w) + 0.2 * float(np.std(att2d))


def overlay_peak_lines(
    image_rgb: np.ndarray,
    a_h: np.ndarray,
    a_w: np.ndarray,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> np.ndarray:
    """Overlay peak coordinate lines from a_h/a_w on the image."""
    if image_rgb.ndim != 3:
        return image_rgb
    h, w = image_rgb.shape[:2]
    row = int(round(np.argmax(a_h) * (h - 1) / max(len(a_h) - 1, 1)))
    col = int(round(np.argmax(a_w) * (w - 1) / max(len(a_w) - 1, 1)))
    out = image_rgb.copy()
    cv2.line(out, (0, row), (w - 1, row), color, thickness)
    cv2.line(out, (col, 0), (col, h - 1), color, thickness)
    return out


def select_representative_image(
    model: ResNet18Landmarks,
    candidates: list[Path],
    device: torch.device,
) -> Path:
    """Pick the image with the most distinct coordinate attention response."""
    best_path = candidates[0]
    best_score = -1.0
    for path in candidates:
        try:
            _, tensor = preprocess_image(path)
            tensor = tensor.to(device)
            a_h, a_w, att2d = compute_attention_maps(model, tensor)
            score = score_attention(a_h, a_w, att2d)
            if score > best_score:
                best_score = score
                best_path = path
        except Exception:
            continue
    return best_path


def draw_cube(ax, x, y, w, h, d, fc="#e6e6e6", ec="#555555", lw=1.0, zorder=2):
    front = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(front)

    top = Polygon(
        [(x, y + h), (x + d, y + h + d * 0.5), (x + w + d, y + h + d * 0.5), (x + w, y + h)],
        closed=True,
        facecolor="#f4f4f4",
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(top)

    side = Polygon(
        [(x + w, y), (x + w + d, y + d * 0.5), (x + w + d, y + h + d * 0.5), (x + w, y + h)],
        closed=True,
        facecolor="#d8d8d8",
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(side)


def draw_slice(ax, x, y, w, h, d, axis="h", fc="#e6e6e6", zorder=2):
    draw_cube(ax, x, y, w, h, d, fc=fc, zorder=zorder)
    if axis == "h":
        ax.plot(
            [x + w * 0.1, x + w * 0.9],
            [y + h * 0.5, y + h * 0.5],
            color="#444444",
            lw=1.2,
            zorder=zorder + 1,
        )
    else:
        ax.plot(
            [x + w * 0.5, x + w * 0.5],
            [y + h * 0.1, y + h * 0.9],
            color="#444444",
            lw=1.2,
            zorder=zorder + 1,
        )


def draw_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle=None,
    fc="#ffffff",
    ec="#000000",
    lw=1.0,
    fs=8,
    zorder=2,
):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        zorder=zorder + 1,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=fs - 1,
            color="#333333",
            zorder=zorder + 1,
        )


def draw_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    lw: float = 1.0,
    shrink_a: float = 4,
    shrink_b: float = 4,
    connectionstyle: str = "arc3,rad=0.0",
    zorder: int = 3,
):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=lw,
            color="#111111",
            shrinkA=shrink_a,
            shrinkB=shrink_b,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def figure_color3d(output_path: Path) -> None:
    """Paper-style Coordinate Attention diagram (3D blocks with subtle color accents)."""
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 7)
    ax.axis("off")

    input_x, input_y, input_w, input_h, input_d = 1.0, 2.0, 2.7, 2.4, 0.9
    pool_x, pool_w, pool_h, pool_d = 6.0, 2.0, 0.75, 0.6
    pool_h_y, pool_w_y = 4.1, 1.2
    concat_x, concat_y, concat_w, concat_h = 9.4, 2.3, 3.1, 1.6
    conv_x, conv_w, conv_h = 13.3, 2.2, 1.0
    conv_h_y, conv_w_y = 4.0, 1.2
    weight_x, weight_w, weight_h, weight_d = 16.3, 1.3, 0.95, 0.45
    weight_h_y, weight_w_y = 4.0, 1.2
    mult_x, mult_y = 18.4, 2.6
    out_x, out_y, out_w, out_h, out_d = 19.6, 2.0, 2.6, 2.4, 0.9

    # Input
    draw_cube(ax, input_x, input_y, input_w, input_h, input_d)
    ax.text(input_x, 5.2, "Input X", fontsize=9)
    ax.text(input_x, 1.4, r"$C\times H\times W$", fontsize=8, color="#333333")

    # Pool H / W with accents
    draw_slice(ax, pool_x, pool_h_y, pool_w, pool_h, pool_d, axis="h", fc="#dff4f2")
    ax.text(pool_x, 5.2, "Pool H", fontsize=8)
    ax.text(pool_x, 3.6, r"$C\times H\times 1$", fontsize=7, color="#333333")

    draw_slice(ax, pool_x, pool_w_y, pool_w, pool_h, pool_d, axis="w", fc="#fde9d4")
    ax.text(pool_x, 0.45, "Pool W", fontsize=8)
    ax.text(pool_x, 0.1, r"$C\times 1\times W$", fontsize=7, color="#333333")

    # Concat + Conv
    draw_box(
        ax,
        concat_x,
        concat_y,
        concat_w,
        concat_h,
        "Concat + Conv 1x1",
        r"$C/r\times(H+W)\times1$",
        fc="#eef2ff",
    )
    ax.text(concat_x + concat_w / 2, 1.75, "BN + ReLU", ha="center", fontsize=7, color="#333333")

    # Branch convs
    draw_box(ax, conv_x, conv_h_y, conv_w, conv_h, "Conv 1x1", r"$C\times H\times1$")
    draw_box(ax, conv_x, conv_w_y, conv_w, conv_h, "Conv 1x1", r"$C\times 1\times W$")
    ax.text(conv_x + conv_w / 2, conv_h_y + conv_h + 0.28, "sigmoid", fontsize=8, ha="center")
    ax.text(conv_x + conv_w / 2, conv_w_y - 0.48, "sigmoid", fontsize=8, ha="center")

    # Weights
    draw_cube(ax, weight_x, weight_h_y, weight_w, weight_h, weight_d, fc="#dff4f2")
    ax.text(weight_x, weight_h_y + 1.2, "A_h", fontsize=8)
    draw_cube(ax, weight_x, weight_w_y, weight_w, weight_h, weight_d, fc="#fde9d4")
    ax.text(weight_x, weight_w_y - 0.75, "A_w", fontsize=8)

    # Multiply + output
    circ = Circle((mult_x, mult_y), 0.35, edgecolor="#111111", facecolor="white", linewidth=1.0)
    ax.add_patch(circ)
    ax.text(mult_x, mult_y, r"$\times$", ha="center", va="center", fontsize=10)

    draw_cube(ax, out_x, out_y, out_w, out_h, out_d)
    ax.text(out_x, 5.2, "Output", fontsize=9)

    # Arrows (offset to avoid text)
    draw_arrow(
        ax,
        input_x + input_w + 0.3,
        input_y + input_h * 0.70,
        pool_x - 0.1,
        pool_h_y + pool_h * 0.5,
    )
    draw_arrow(
        ax,
        input_x + input_w + 0.3,
        input_y + input_h * 0.35,
        pool_x - 0.1,
        pool_w_y + pool_h * 0.5,
    )
    draw_arrow(ax, pool_x + pool_w + 0.2, pool_h_y + pool_h * 0.5, concat_x - 0.1, concat_y + 1.05)
    draw_arrow(ax, pool_x + pool_w + 0.2, pool_w_y + pool_h * 0.5, concat_x - 0.1, concat_y + 0.55)
    draw_arrow(
        ax,
        concat_x + concat_w + 0.2,
        concat_y + 1.05,
        conv_x - 0.1,
        conv_h_y + conv_h * 0.5,
    )
    draw_arrow(
        ax,
        concat_x + concat_w + 0.2,
        concat_y + 0.55,
        conv_x - 0.1,
        conv_w_y + conv_h * 0.5,
    )
    draw_arrow(
        ax,
        conv_x + conv_w + 0.2,
        conv_h_y + conv_h * 0.5,
        weight_x - 0.1,
        weight_h_y + weight_h * 0.5,
        zorder=1,
    )
    draw_arrow(
        ax,
        conv_x + conv_w + 0.2,
        conv_w_y + conv_h * 0.5,
        weight_x - 0.1,
        weight_w_y + weight_h * 0.5,
        zorder=1,
    )
    draw_arrow(
        ax,
        weight_x + weight_w + 0.2,
        weight_h_y + weight_h * 0.5,
        mult_x - 0.3,
        mult_y + 0.2,
    )
    draw_arrow(
        ax,
        weight_x + weight_w + 0.2,
        weight_w_y + weight_h * 0.5,
        mult_x - 0.3,
        mult_y - 0.2,
    )
    draw_arrow(ax, mult_x + 0.35, mult_y, out_x - 0.1, out_y + out_h * 0.5)

    ax.text(
        1.0,
        6.3,
        "Coordinate Attention (C=512, H=W=7, r=32)",
        fontsize=10,
        fontweight="bold",
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def figure_mechanism_real(output_path: Path, overlay: np.ndarray) -> None:
    """Mechanism diagram + real attention map panel (clean spacing)."""
    fig = plt.figure(figsize=(13.8, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.08)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_xlim(0, 21)
    ax_left.set_ylim(0, 6.3)
    ax_left.axis("off")

    # Left: block diagram
    input_x, input_y, input_w, input_h = 0.9, 2.3, 2.7, 1.6
    pool_x, pool_w, pool_h = 4.6, 2.7, 1.1
    pool_h_y, pool_w_y = 3.8, 1.0
    concat_x, concat_y, concat_w, concat_h = 8.3, 2.3, 3.2, 1.6
    conv_x, conv_w, conv_h = 12.6, 2.4, 1.1
    conv_h_y, conv_w_y = 3.8, 1.0
    rew_x, rew_y, rew_w, rew_h = 16.2, 2.3, 3.3, 1.6

    draw_box(ax_left, input_x, input_y, input_w, input_h, "Input X", r"$C\times H\times W$")
    draw_box(ax_left, pool_x, pool_h_y, pool_w, pool_h, "Pool H", r"$C\times H\times 1$", fc="#e8f6f4")
    draw_box(ax_left, pool_x, pool_w_y, pool_w, pool_h, "Pool W", r"$C\times 1\times W$", fc="#fdebd3")

    draw_box(
        ax_left,
        concat_x,
        concat_y,
        concat_w,
        concat_h,
        "Concat + 1x1\nConv",
        r"$C/r\times(H+W)\times1$",
        fc="#eef2ff",
    )
    ax_left.text(concat_x + concat_w / 2, 1.6, "BN + ReLU", ha="center", fontsize=7, color="#333333")

    draw_box(ax_left, conv_x, conv_h_y, conv_w, conv_h, "Conv 1x1", r"$C\times H\times1$")
    draw_box(ax_left, conv_x, conv_w_y, conv_w, conv_h, "Conv 1x1", r"$C\times 1\times W$")
    ax_left.text(conv_x + conv_w / 2, conv_h_y + conv_h + 0.28, "sigmoid", fontsize=7, ha="center")
    ax_left.text(conv_x + conv_w / 2, conv_w_y - 0.45, "sigmoid", fontsize=7, ha="center")

    draw_box(ax_left, rew_x, rew_y, rew_w, rew_h, "Reweighting", r"$Y = X\times a_h\times a_w$", fc="#f5f5f5")

    draw_arrow(
        ax_left,
        input_x + input_w + 0.2,
        input_y + input_h * 0.7,
        pool_x - 0.1,
        pool_h_y + pool_h * 0.5,
        zorder=1,
    )
    draw_arrow(
        ax_left,
        input_x + input_w + 0.2,
        input_y + input_h * 0.3,
        pool_x - 0.1,
        pool_w_y + pool_h * 0.5,
        zorder=1,
    )
    draw_arrow(
        ax_left,
        pool_x + pool_w + 0.2,
        pool_h_y + pool_h * 0.5,
        concat_x - 0.1,
        concat_y + concat_h * 0.8,
        zorder=1,
    )
    draw_arrow(
        ax_left,
        pool_x + pool_w + 0.2,
        pool_w_y + pool_h * 0.5,
        concat_x - 0.1,
        concat_y + concat_h * 0.2,
        zorder=1,
    )
    draw_arrow(
        ax_left,
        concat_x + concat_w + 0.2,
        concat_y + concat_h * 0.8,
        conv_x - 0.1,
        conv_h_y + conv_h * 0.5,
        zorder=1,
    )
    draw_arrow(
        ax_left,
        concat_x + concat_w + 0.2,
        concat_y + concat_h * 0.2,
        conv_x - 0.1,
        conv_w_y + conv_h * 0.5,
        zorder=1,
    )
    draw_arrow(
        ax_left,
        conv_x + conv_w + 0.2,
        conv_h_y + conv_h * 0.5,
        rew_x - 0.1,
        rew_y + rew_h * 0.7,
        zorder=1,
    )
    draw_arrow(
        ax_left,
        conv_x + conv_w + 0.2,
        conv_w_y + conv_h * 0.5,
        rew_x - 0.1,
        rew_y + rew_h * 0.3,
        zorder=1,
    )

    ax_left.text(0.8, 5.7, "Coordinate Attention (C=512, H=W=7, r=32)", fontsize=9, fontweight="bold")

    # Right: real attention map
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.imshow(overlay)
    ax_right.axis("off")
    ax_right.set_title("Atencion espacial sobre RX", fontsize=9)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Coordinate Attention figures.")
    parser.add_argument("--output-dir", default="docs/Tesis/Figures", help="Output directory for figures.")
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pt", help="Model checkpoint path.")
    parser.add_argument("--image", default=None, help="Optional path to a specific input image.")
    parser.add_argument("--crop-warped", action="store_true", help="Crop black borders on warped images.")
    parser.add_argument("--mark-peaks", action="store_true", help="Overlay peak H/W lines on the image.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = load_model(Path(args.checkpoint), device)

    if args.image:
        img_path = Path(args.image)
    else:
        candidates = list_dataset_images()
        img_path = select_representative_image(model, candidates, device) if candidates else find_sample_image()

    img_np, tensor = preprocess_image(img_path)
    tensor = tensor.to(device)
    a_h, a_w, att2d = compute_attention_maps(model, tensor)

    att_up = cv2.resize(att2d, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    overlay = overlay_heatmap(img_np, att_up, alpha=0.45, gamma=0.65)
    if args.mark_peaks:
        overlay = overlay_peak_lines(overlay, a_h, a_w)
    if args.crop_warped:
        overlay = crop_warped_black_border(overlay)

    figure_color3d(output_dir / "coord_attention_v10_color3d.png")
    figure_mechanism_real(output_dir / "coord_attention_v10_mechanism_real.png", overlay)


if __name__ == "__main__":
    main()
