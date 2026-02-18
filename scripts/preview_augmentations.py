"""
Augmentation Preview Script — Visual Validation Gate.

Generates augmentation preview grids and SSIM metrics for each augmentation
strategy. Used as a visual gate before committing GPU time to ablation training.

For each augmentation type, forced p=1.0 is used (for preview visibility only).

Usage:
    python scripts/preview_augmentations.py \\
        --data-dir outputs/warped_cleaned/session_warping \\
        --output-dir outputs/augmentation_previews \\
        --n-samples 5

Output:
    outputs/augmentation_previews/
        elastic_alpha1_sigma30.png
        elastic_alpha20_sigma30.png
        grid_distortion_01.png
        pixel_aug.png
        mixup_preview.png
        cutmix_preview.png
        ssim_results.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import matplotlib
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Augmentation configurations (preview uses p=1.0 for forced visibility)
# ============================================================================

AUGMENTATION_CONFIGS = [
    {
        "name": "ElasticTransform(alpha=1, sigma=30)",
        "key": "elastic_alpha1_sigma30",
        "transform": lambda: A.ElasticTransform(
            alpha=1.0, sigma=30.0, p=1.0, border_mode=0, fill=0.0
        ),
        "ssim_threshold": 0.95,
        "status_type": "spatial",
    },
    {
        "name": "ElasticTransform(alpha=20, sigma=30)",
        "key": "elastic_alpha20_sigma30",
        "transform": lambda: A.ElasticTransform(
            alpha=20.0, sigma=30.0, p=1.0, border_mode=0, fill=0.0
        ),
        "ssim_threshold": 0.85,
        "status_type": "spatial",
    },
    {
        "name": "GridDistortion(distort=0.1)",
        "key": "grid_distortion_01",
        "transform": lambda: A.GridDistortion(
            num_steps=5, distort_limit=0.1, p=1.0, border_mode=0, fill=0.0, normalized=True
        ),
        "ssim_threshold": 0.95,
        "status_type": "spatial",
    },
    {
        "name": "PixelAug (brightness+noise)",
        "key": "pixel_aug",
        "transform": lambda: A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.GaussNoise(std_range=(0.01, 0.03), mean_range=(0.0, 0.0), p=1.0),
        ]),
        "ssim_threshold": 0.90,
        "status_type": "pixel",
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Augmentation preview and SSIM validation.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="outputs/warped_cleaned/session_warping",
        help="Dataset directory with train/val/test splits (ImageFolder structure)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/augmentation_previews",
        help="Output directory for preview images and SSIM results",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples per class to preview (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    return parser.parse_args()


def load_samples_per_class(
    data_dir: Path,
    n_samples: int,
    seed: int,
) -> Dict[str, List[Path]]:
    """Load N random samples per class from train split.

    Args:
        data_dir: Dataset root with train/ subdirectory (ImageFolder structure)
        n_samples: Number of samples per class
        seed: Random seed

    Returns:
        Dict mapping class_name -> list of image paths
    """
    train_dir = data_dir / "train"
    if not train_dir.exists():
        # Fall back to root if no split structure
        train_dir = data_dir
        print(f"  Warning: no train/ subdirectory, using root: {train_dir}")

    rng = random.Random(seed)
    samples = {}

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"  Error: No class subdirectories found in {train_dir}")
        sys.exit(1)

    print(f"  Found classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        image_paths = sorted(
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg"))
        )
        if not image_paths:
            print(f"  Warning: No images found in {class_dir}")
            continue

        chosen = rng.sample(image_paths, min(n_samples, len(image_paths)))
        samples[class_dir.name] = chosen
        print(f"  {class_dir.name}: {len(chosen)} samples loaded")

    return samples


def load_image_as_rgb_array(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB numpy array (H, W, 3)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img)


def apply_augmentation(img_np: np.ndarray, transform: A.BasicTransform) -> np.ndarray:
    """Apply albumentations transform to a uint8 numpy array.

    Args:
        img_np: Input image as uint8 numpy array (H, W, 3)
        transform: Albumentations transform

    Returns:
        Augmented image as uint8 numpy array
    """
    result = transform(image=img_np)
    return result["image"]


def compute_ssim_pair(orig: np.ndarray, aug: np.ndarray) -> float:
    """Compute SSIM between original and augmented images.

    Args:
        orig: Original image uint8 (H, W, 3)
        aug: Augmented image uint8 (H, W, 3)

    Returns:
        SSIM score in [0, 1]
    """
    return float(structural_similarity(orig, aug, channel_axis=-1, data_range=255))


def get_status(mean_ssim: float, threshold: float, status_type: str) -> str:
    """Get PASS/REVIEW/FAIL/INFO status for SSIM.

    Args:
        mean_ssim: Mean SSIM value
        threshold: Minimum SSIM threshold for PASS
        status_type: 'spatial', 'pixel', or 'mixing'

    Returns:
        Status string
    """
    if status_type == "mixing":
        return "INFO"
    if mean_ssim >= threshold:
        return "PASS"
    if mean_ssim >= threshold - 0.10:
        return "REVIEW"
    return "FAIL"


def save_preview_grid(
    aug_name: str,
    aug_key: str,
    orig_imgs: List[np.ndarray],
    aug_imgs: List[np.ndarray],
    class_names: List[str],
    ssim_scores: List[float],
    mean_ssim: float,
    output_dir: Path,
) -> Path:
    """Save a preview grid image.

    Rows = samples (one per class label), Columns = [Original, Augmented, Difference Map]

    Args:
        aug_name: Human-readable augmentation name
        aug_key: File-safe augmentation key
        orig_imgs: List of original images (H, W, 3) uint8
        aug_imgs: List of augmented images (H, W, 3) uint8
        class_names: Class name for each sample
        ssim_scores: SSIM score for each sample pair
        mean_ssim: Mean SSIM across all samples
        output_dir: Output directory

    Returns:
        Path to saved grid image
    """
    n_rows = len(orig_imgs)
    n_cols = 3
    fig_width = 12
    fig_height = 3 * n_rows + 0.8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Augmented", "Difference Map"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row_idx, (orig, aug, cls_name, ssim_val) in enumerate(
        zip(orig_imgs, aug_imgs, class_names, ssim_scores)
    ):
        # Difference map (absolute, amplified for visibility)
        diff = np.abs(orig.astype(np.float32) - aug.astype(np.float32))
        diff_amplified = np.clip(diff * 3.0, 0, 255).astype(np.uint8)

        axes[row_idx, 0].imshow(orig)
        axes[row_idx, 0].set_ylabel(f"{cls_name}\nSSIM={ssim_val:.4f}", fontsize=8)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        axes[row_idx, 1].imshow(aug)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        axes[row_idx, 2].imshow(diff_amplified)
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])

    fig.suptitle(
        f"{aug_name}   |   Mean SSIM: {mean_ssim:.4f}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    out_path = output_dir / f"{aug_key}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_mixing_preview(
    name: str,
    key: str,
    pairs: List[Tuple[np.ndarray, np.ndarray, str, str, np.ndarray]],
    output_dir: Path,
) -> Path:
    """Save a mixing preview grid (MixUp or CutMix).

    Args:
        name: Human-readable name
        key: File-safe key
        pairs: List of (img_a, img_b, class_a, class_b, mixed) tuples
        output_dir: Output directory

    Returns:
        Path to saved grid
    """
    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3 * n_rows + 0.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image A", "Image B", "Mixed", "Info"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row_idx, (img_a, img_b, cls_a, cls_b, mixed) in enumerate(pairs):
        axes[row_idx, 0].imshow(img_a)
        axes[row_idx, 0].set_ylabel(cls_a, fontsize=8)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        axes[row_idx, 1].imshow(img_b)
        axes[row_idx, 1].set_ylabel(cls_b, fontsize=8)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        axes[row_idx, 2].imshow(mixed)
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])

        axes[row_idx, 3].axis("off")
        info_text = f"{cls_a}\n+\n{cls_b}"
        axes[row_idx, 3].text(
            0.5, 0.5, info_text,
            ha="center", va="center", fontsize=10,
            transform=axes[row_idx, 3].transAxes,
        )

    fig.suptitle(name, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = output_dir / f"{key}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def preview_mixup(
    samples: Dict[str, List[Path]],
    output_dir: Path,
    n_pairs: int = 3,
    lam: float = 0.6,
) -> dict:
    """Generate MixUp preview.

    Args:
        samples: Dict class -> list of image paths
        output_dir: Output directory
        n_pairs: Number of cross-class pairs to preview
        lam: MixUp lambda (blending factor for image A)

    Returns:
        Info dict with key, name, and status
    """
    class_names = list(samples.keys())
    if len(class_names) < 2:
        print("  Skipping MixUp preview: need at least 2 classes")
        return {"key": "mixup_preview", "name": "MixUp", "ssim": None, "status": "SKIP"}

    pairs = []
    rng = random.Random(42)
    for _ in range(n_pairs):
        cls_a, cls_b = rng.sample(class_names, 2)
        img_a_path = rng.choice(samples[cls_a])
        img_b_path = rng.choice(samples[cls_b])
        img_a = load_image_as_rgb_array(img_a_path)
        img_b = load_image_as_rgb_array(img_b_path)
        # Resize both to same size if needed
        if img_a.shape != img_b.shape:
            h, w = img_a.shape[:2]
            img_b_pil = Image.fromarray(img_b).resize((w, h))
            img_b = np.array(img_b_pil)
        mixed = (lam * img_a.astype(np.float32) + (1.0 - lam) * img_b.astype(np.float32))
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        pairs.append((img_a, img_b, cls_a, cls_b, mixed))

    out_path = save_mixing_preview(
        f"MixUp Preview (lambda={lam}, 60% A + 40% B)",
        "mixup_preview",
        pairs,
        output_dir,
    )
    print(f"  MixUp preview saved: {out_path}")
    return {
        "key": "mixup_preview",
        "name": f"MixUp(lambda={lam})",
        "mean_ssim": None,
        "std_ssim": None,
        "status": "INFO",
        "file": str(out_path),
    }


def preview_cutmix(
    samples: Dict[str, List[Path]],
    output_dir: Path,
    n_pairs: int = 3,
    patch_fraction: float = 0.2,
) -> dict:
    """Generate CutMix preview.

    Args:
        samples: Dict class -> list of image paths
        output_dir: Output directory
        n_pairs: Number of cross-class pairs to preview
        patch_fraction: Approximate fraction of image area for the cut patch

    Returns:
        Info dict with key, name, and status
    """
    class_names = list(samples.keys())
    if len(class_names) < 2:
        print("  Skipping CutMix preview: need at least 2 classes")
        return {"key": "cutmix_preview", "name": "CutMix", "ssim": None, "status": "SKIP"}

    pairs = []
    rng = random.Random(123)
    for _ in range(n_pairs):
        cls_a, cls_b = rng.sample(class_names, 2)
        img_a_path = rng.choice(samples[cls_a])
        img_b_path = rng.choice(samples[cls_b])
        img_a = load_image_as_rgb_array(img_a_path)
        img_b = load_image_as_rgb_array(img_b_path)

        h, w = img_a.shape[:2]
        if img_a.shape != img_b.shape:
            img_b_pil = Image.fromarray(img_b).resize((w, h))
            img_b = np.array(img_b_pil)

        # Compute patch size from area fraction
        patch_h = int(h * (patch_fraction ** 0.5))
        patch_w = int(w * (patch_fraction ** 0.5))
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        x1 = max(0, cx - patch_w // 2)
        y1 = max(0, cy - patch_h // 2)
        x2 = min(w, cx + patch_w // 2)
        y2 = min(h, cy + patch_h // 2)

        mixed = img_a.copy()
        mixed[y1:y2, x1:x2] = img_b[y1:y2, x1:x2]

        pairs.append((img_a, img_b, cls_a, cls_b, mixed))

    out_path = save_mixing_preview(
        f"CutMix Preview (patch_fraction={patch_fraction}, ~{int(patch_fraction*100)}% area)",
        "cutmix_preview",
        pairs,
        output_dir,
    )
    print(f"  CutMix preview saved: {out_path}")
    return {
        "key": "cutmix_preview",
        "name": f"CutMix(patch_fraction={patch_fraction})",
        "mean_ssim": None,
        "std_ssim": None,
        "status": "INFO",
        "file": str(out_path),
    }


def run_augmentation_preview(
    aug_config: dict,
    samples: Dict[str, List[Path]],
    output_dir: Path,
) -> dict:
    """Run preview for a single augmentation config.

    Args:
        aug_config: Augmentation config dict (from AUGMENTATION_CONFIGS)
        samples: Dict class -> list of image paths
        output_dir: Output directory

    Returns:
        Result dict with name, mean_ssim, std_ssim, status, file
    """
    name = aug_config["name"]
    key = aug_config["key"]
    ssim_threshold = aug_config["ssim_threshold"]
    status_type = aug_config["status_type"]

    print(f"\n  Preview: {name}")
    transform = aug_config["transform"]()

    orig_imgs = []
    aug_imgs = []
    class_names_used = []
    ssim_scores = []

    for cls_name, img_paths in samples.items():
        img_path = img_paths[0]  # One image per class for spatial arrangement
        orig_np = load_image_as_rgb_array(img_path)
        aug_np = apply_augmentation(orig_np.copy(), transform)

        ssim_val = compute_ssim_pair(orig_np, aug_np)

        orig_imgs.append(orig_np)
        aug_imgs.append(aug_np)
        class_names_used.append(cls_name)
        ssim_scores.append(ssim_val)

    mean_ssim = float(np.mean(ssim_scores))
    std_ssim = float(np.std(ssim_scores))
    status = get_status(mean_ssim, ssim_threshold, status_type)

    grid_path = save_preview_grid(
        name, key, orig_imgs, aug_imgs, class_names_used, ssim_scores,
        mean_ssim, output_dir,
    )
    print(f"    SSIM: mean={mean_ssim:.4f}, std={std_ssim:.4f} -> {status}")
    print(f"    Grid saved: {grid_path}")

    return {
        "name": name,
        "key": key,
        "mean_ssim": mean_ssim,
        "std_ssim": std_ssim,
        "threshold": ssim_threshold,
        "status": status,
        "file": str(grid_path),
    }


def print_summary_table(results: List[dict]) -> None:
    """Print SSIM summary table to stdout."""
    header = f"{'Augmentation':<35} {'Mean SSIM':>10} {'Std SSIM':>10} {'Status':>8}"
    separator = "-" * len(header)
    print("\n" + separator)
    print("AUGMENTATION PREVIEW SUMMARY")
    print(separator)
    print(header)
    print(separator)
    for r in results:
        mean_ssim_val = r.get("mean_ssim")
        std_ssim_val = r.get("std_ssim")
        ssim_str = f"{mean_ssim_val:.4f}" if mean_ssim_val is not None else "  N/A"
        std_str = f"{std_ssim_val:.4f}" if std_ssim_val is not None else "  N/A"
        print(f"{r['name']:<35} {ssim_str:>10} {std_str:>10} {r['status']:>8}")
    print(separator)
    print()
    print("Status legend:")
    print("  PASS   = SSIM above threshold (good for training)")
    print("  REVIEW = SSIM borderline (verify visually, may need parameter tuning)")
    print("  FAIL   = SSIM below threshold (spatial transform too aggressive)")
    print("  INFO   = Mixing augmentation (SSIM not meaningful, review visually)")
    print()


def main() -> None:
    """Main entry point for augmentation preview script."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"Error: data_dir does not exist: {data_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nAugmentation Preview Generator")
    print(f"  data_dir:   {data_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  n_samples:  {args.n_samples}")
    print(f"  seed:       {args.seed}")

    # Load sample images
    print("\nLoading samples...")
    samples = load_samples_per_class(data_dir, args.n_samples, args.seed)
    if not samples:
        print("Error: No samples loaded. Check data_dir structure.")
        sys.exit(1)

    all_results = []

    # Run each spatial/pixel augmentation
    print("\nGenerating augmentation previews...")
    for aug_config in AUGMENTATION_CONFIGS:
        result = run_augmentation_preview(aug_config, samples, output_dir)
        all_results.append(result)

    # Run MixUp preview
    print("\nGenerating MixUp preview...")
    mixup_result = preview_mixup(samples, output_dir, n_pairs=min(3, len(samples)), lam=0.6)
    all_results.append(mixup_result)

    # Run CutMix preview
    print("\nGenerating CutMix preview...")
    cutmix_result = preview_cutmix(
        samples, output_dir, n_pairs=min(3, len(samples)), patch_fraction=0.2
    )
    all_results.append(cutmix_result)

    # Print summary table
    print_summary_table(all_results)

    # Save SSIM results as JSON
    ssim_json_path = output_dir / "ssim_results.json"
    with ssim_json_path.open("w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "n_samples": args.n_samples,
                "seed": args.seed,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"SSIM results saved: {ssim_json_path}")
    print(f"\nAll preview images saved to: {output_dir}")
    print("\nReview instructions:")
    print("  1. Open each PNG in the output directory")
    print("  2. Check spatial transforms preserve lung contours and diagnostic features")
    print("  3. Check pixel augmentations look clinically plausible")
    print("  4. Review MixUp and CutMix blend quality for regularization acceptability")
    print("  5. Approve or reject each augmentation type for ablation training")


if __name__ == "__main__":
    main()
