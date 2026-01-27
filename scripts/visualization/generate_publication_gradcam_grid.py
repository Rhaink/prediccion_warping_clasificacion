"""Generate publication-ready Grad-CAM grid (Original vs Warped) at 300 dpi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src_v2.models import create_classifier, get_classifier_transforms
from src_v2.visualization.gradcam import GradCAM, get_target_layer, overlay_heatmap


COVID_ID = "COVID-1059"
NORMAL_ID = "Normal-10094"
VIRAL_PNEUMONIA_ID = "Viral Pneumonia-1070"
# VIRAL_PNEUMONIA_ID = "Viral Pneumonia-1055"  # Previous candidate.

ORIG_CHECKPOINT = Path(
    "outputs/classifier_original_warped_lung_best_seed321/best_classifier_original.pt"
)
WARP_CHECKPOINT = Path(
    "outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/"
    "best_classifier.pt"
)

ORIG_TEST_DIR = Path("outputs/original/sessionXX/test")
WARP_TEST_DIR = Path("outputs/warped_lung_best/session_warping/test")

OUT_DIR = Path("results/figures/publication")

FIG_DPI = 300
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 9.0
HEATMAP_ALPHA = 0.5
HEATMAP_COLORMAP = "jet"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a serif font, falling back to default if needed."""
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_gradcam(checkpoint: Path) -> tuple[GradCAM, torch.nn.Module, dict[str, int]]:
    """Load classifier checkpoint and Grad-CAM helper."""
    device = torch.device("cpu")
    model = create_classifier(checkpoint=str(checkpoint), device=device)
    model.eval()

    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names", ["COVID", "Normal", "Viral_Pneumonia"])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    target_layer = get_target_layer(model, model.backbone_name, None)
    gradcam = GradCAM(model, target_layer)
    return gradcam, model, class_to_idx


def build_case_paths(case_id: str, class_dir: str, suffix: str) -> Path:
    """Build input path for a case."""
    if suffix == "crop0":
        return ORIG_TEST_DIR / class_dir / f"{case_id}_crop0.png"
    if suffix == "warped":
        return WARP_TEST_DIR / class_dir / f"{case_id}_warped.png"
    raise ValueError(f"Unsupported suffix: {suffix}")


def generate_overlay(
    img_path: Path,
    gradcam: GradCAM,
    class_idx: int,
    transform: torch.nn.Module,
) -> Image.Image:
    """Generate Grad-CAM overlay for a single image."""
    img_pil = Image.open(img_path).convert("RGB")
    img_array = np.array(img_pil)
    img_tensor = transform(img_pil).unsqueeze(0)
    heatmap, _, _ = gradcam(img_tensor, target_class=class_idx)
    overlay = overlay_heatmap(
        img_array,
        heatmap,
        alpha=HEATMAP_ALPHA,
        colormap=HEATMAP_COLORMAP,
    )
    return Image.fromarray(overlay)


def main() -> None:
    """Generate the figure and save it to results/figures/publication."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gradcam_orig, _, class_idx_orig = load_gradcam(ORIG_CHECKPOINT)
    gradcam_warp, _, class_idx_warp = load_gradcam(WARP_CHECKPOINT)

    transform = get_classifier_transforms(train=False, img_size=224)

    cases = [
        {
            "key": "COVID",
            "label": "COVID-19",
            "class_dir": "COVID",
            "case_id": COVID_ID,
        },
        {
            "key": "Normal",
            "label": "Normal",
            "class_dir": "Normal",
            "case_id": NORMAL_ID,
        },
        {
            "key": "Viral_Pneumonia",
            "label": "Viral Pneumonia",
            "class_dir": "Viral_Pneumonia",
            "case_id": VIRAL_PNEUMONIA_ID,
        },
    ]

    overlay_paths: dict[str, dict[str, Path]] = {"original": {}, "warped": {}}

    try:
        for case in cases:
            class_key = case["key"]
            case_id = case["case_id"]
            class_dir = case["class_dir"]

            orig_path = build_case_paths(case_id, class_dir, "crop0")
            warp_path = build_case_paths(case_id, class_dir, "warped")
            if not orig_path.exists():
                raise FileNotFoundError(orig_path)
            if not warp_path.exists():
                raise FileNotFoundError(warp_path)

            orig_overlay = generate_overlay(
                orig_path,
                gradcam_orig,
                class_idx_orig[class_key],
                transform,
            )
            warp_overlay = generate_overlay(
                warp_path,
                gradcam_warp,
                class_idx_warp[class_key],
                transform,
            )

            orig_out = OUT_DIR / f"{class_key.lower()}_original_gradcam_gt.png"
            warp_out = OUT_DIR / f"{class_key.lower()}_warped_gradcam_gt.png"
            orig_overlay.save(orig_out)
            warp_overlay.save(warp_out)
            overlay_paths["original"][class_key] = orig_out
            overlay_paths["warped"][class_key] = warp_out
    finally:
        gradcam_orig.remove_hooks()
        gradcam_warp.remove_hooks()

    width_px = int(FIG_WIDTH_IN * FIG_DPI)
    height_px = int(FIG_HEIGHT_IN * FIG_DPI)
    margin = int(0.30 * FIG_DPI)
    left_label_width = int(1.20 * FIG_DPI)
    top_label_height = int(0.45 * FIG_DPI)
    col_gap = int(0.25 * FIG_DPI)
    row_gap = int(0.25 * FIG_DPI)

    col_width = (width_px - 2 * margin - left_label_width - col_gap) // 2
    row_height = (height_px - 2 * margin - top_label_height - 2 * row_gap) // 3
    cell_size = min(col_width, row_height)

    col1_x = margin + left_label_width
    col2_x = col1_x + col_width + col_gap
    row1_y = margin + top_label_height
    row2_y = row1_y + row_height + row_gap
    row3_y = row2_y + row_height + row_gap

    canvas = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    col_font = load_font(int(0.16 * FIG_DPI), bold=True)
    row_font = load_font(int(0.14 * FIG_DPI), bold=True)

    for label, x in zip(["Original", "Warped"], [col1_x, col2_x]):
        bbox = draw.textbbox((0, 0), label, font=col_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x + (col_width - text_w) // 2
        text_y = margin + (top_label_height - text_h) // 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=col_font)

    row_positions = {
        "COVID": row1_y,
        "Normal": row2_y,
        "Viral_Pneumonia": row3_y,
    }
    for case in cases:
        label = case["label"]
        y = row_positions[case["key"]]
        bbox = draw.textbbox((0, 0), label, font=row_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = margin + left_label_width - text_w - int(0.06 * FIG_DPI)
        text_y = y + (row_height - text_h) // 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=row_font)

    col_positions = {"original": col1_x, "warped": col2_x}
    for case in cases:
        class_key = case["key"]
        for variant in ["original", "warped"]:
            img_path = overlay_paths[variant][class_key]
            img = Image.open(img_path).convert("RGB")
            img = img.resize((cell_size, cell_size), Image.LANCZOS)
            cell_x = col_positions[variant]
            cell_y = row_positions[class_key]
            x = cell_x + (col_width - cell_size) // 2
            y = cell_y + (row_height - cell_size) // 2
            canvas.paste(img, (x, y))

    final_path = OUT_DIR / "original_vs_warped_gradcam_grid_300dpi.png"
    canvas.save(final_path, dpi=(FIG_DPI, FIG_DPI))
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
