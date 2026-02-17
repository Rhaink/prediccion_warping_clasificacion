"""
Out-of-Fold (OOF) Probability Extraction from 5-Fold Classifier Checkpoints.

Reconstructs the exact StratifiedKFold(5, shuffle=True, random_state=42) splits
used during 5-fold CV training and extracts per-sample probabilities for
cleanlab label noise detection.

Usage:
    python scripts/run_oof_extraction.py
    python scripts/run_oof_extraction.py --device cpu --batch-size 64
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src_v2.models import ImageClassifier, get_classifier_transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract out-of-fold probabilities from 5-fold classifier checkpoints"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/warped_lung_best/session_warping"),
        help="Path to warped dataset with train/val/test splits",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/classifier_cv"),
        help="Directory with fold checkpoints and cross_validation_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_cleaning"),
        help="Output directory for OOF probabilities",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed — MUST match training seed for split reconstruction",
    )
    return parser.parse_args()


def verify_fold_splits(fold_results, full_samples, full_labels, n_splits=5, seed=42):
    """
    Verify that the reconstructed fold splits match stored cross_validation_results.json.

    Args:
        fold_results: List of fold metric dicts from cross_validation_results.json
        full_samples: Combined train+val samples list
        full_labels: Combined train+val labels list
        n_splits: Number of folds
        seed: Random seed for StratifiedKFold

    Returns:
        List of (train_idx, val_idx) tuples for each fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(full_samples, full_labels))

    print("\n--- Verifying Fold Split Reconstruction ---")
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        stored = fold_results[fold_idx]
        stored_val_count = stored["val_samples"]
        reconstructed_val_count = len(val_idx)

        # Verify val sample count
        if reconstructed_val_count != stored_val_count:
            raise ValueError(
                f"Fold {fold_idx + 1}: Reconstructed val count {reconstructed_val_count} "
                f"does not match stored {stored_val_count}. "
                "Split reconstruction failed — seed or dataset may differ from training."
            )

        # Verify class distribution in val set
        val_labels_array = np.array(full_labels)[val_idx]
        stored_class_dist = stored["class_distribution"]["val"]

        # Build expected counts from stored distribution
        class_names_sorted = sorted(stored_class_dist.keys())  # alphabetical = ImageFolder order
        stored_counts = {name: stored_class_dist[name] for name in class_names_sorted}

        # Get dataset class_to_idx mapping (alphabetical by ImageFolder)
        unique, counts = np.unique(val_labels_array, return_counts=True)
        actual_counts_by_label = dict(zip(unique.tolist(), counts.tolist()))

        # Map class idx to name (ImageFolder uses alphabetical: COVID=0, Normal=1, Viral_Pneumonia=2)
        idx_to_name = {0: "COVID", 1: "Normal", 2: "Viral_Pneumonia"}
        actual_named = {idx_to_name.get(k, str(k)): v for k, v in actual_counts_by_label.items()}

        mismatch = False
        for cls_name, stored_cnt in stored_counts.items():
            actual_cnt = actual_named.get(cls_name, 0)
            if actual_cnt != stored_cnt:
                print(
                    f"  WARNING Fold {fold_idx + 1} class {cls_name}: "
                    f"reconstructed={actual_cnt}, stored={stored_cnt}"
                )
                mismatch = True

        status = "MISMATCH" if mismatch else "OK"
        print(
            f"  Fold {fold_idx + 1}: val_count={reconstructed_val_count}/{stored_val_count} [{status}]"
        )

    print("--- Verification Complete ---\n")
    return splits


def extract_oof_probabilities(
    splits,
    full_samples,
    full_labels,
    checkpoint_dir: Path,
    transform,
    device: torch.device,
    batch_size: int,
    n_classes: int = 3,
):
    """
    Extract out-of-fold probabilities for each fold.

    Args:
        splits: List of (train_idx, val_idx) tuples
        full_samples: Combined train+val samples (path, label) tuples
        full_labels: Combined train+val label integers
        checkpoint_dir: Directory containing fold_NN subdirectories
        transform: Inference transforms
        device: Torch device
        batch_size: DataLoader batch size
        n_classes: Number of output classes

    Returns:
        Tuple of (pred_probs, logits_all) both shape (N, n_classes)
    """
    n_total = len(full_samples)
    pred_probs = np.zeros((n_total, n_classes), dtype=np.float32)
    logits_all = np.zeros((n_total, n_classes), dtype=np.float32)
    covered = np.zeros(n_total, dtype=bool)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        checkpoint_path = checkpoint_dir / f"fold_{fold_num:02d}" / "best_classifier.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Processing fold {fold_num}/5 — {len(val_idx)} val samples")

        # Load model
        model = ImageClassifier(num_classes=n_classes, backbone="resnet18")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Create dataset for val_idx using full_samples
        # Build a custom dataset that indexes into full_samples
        class IndexedDataset(torch.utils.data.Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                from PIL import Image
                img = Image.open(path)
                if self.transform:
                    img = self.transform(img)
                return img, label, idx

        val_samples_list = [full_samples[i] for i in val_idx]
        val_dataset = IndexedDataset(val_samples_list, transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )

        fold_logits = np.zeros((len(val_idx), n_classes), dtype=np.float32)
        fold_probs = np.zeros((len(val_idx), n_classes), dtype=np.float32)
        local_idx = 0

        with torch.no_grad():
            for imgs, labels, batch_local_indices in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)

                batch_size_actual = imgs.shape[0]
                fold_logits[local_idx:local_idx + batch_size_actual] = logits.cpu().numpy()
                fold_probs[local_idx:local_idx + batch_size_actual] = probs.cpu().numpy()
                local_idx += batch_size_actual

        # Map fold-local indices back to global full_samples indices
        pred_probs[val_idx] = fold_probs
        logits_all[val_idx] = fold_logits
        covered[val_idx] = True

        fold_acc = (fold_probs.argmax(axis=1) == np.array(full_labels)[val_idx]).mean()
        print(f"  Fold {fold_num} val accuracy: {fold_acc:.4f}")

    if not covered.all():
        n_uncovered = (~covered).sum()
        raise RuntimeError(
            f"{n_uncovered} samples were not covered by any fold's val set. "
            "This indicates an error in fold reconstruction."
        )

    return pred_probs, logits_all


def check_overconfidence_and_apply_temperature(pred_probs, logits_all, threshold=0.90, temperature=2.0):
    """
    Check if predictions are overconfident and apply temperature scaling if needed.

    Args:
        pred_probs: (N, C) probability matrix
        logits_all: (N, C) raw logits for temperature scaling
        threshold: Fraction of samples with max_prob > 0.99 to trigger temperature scaling
        temperature: Temperature value for scaling

    Returns:
        Tuple of (final_probs, applied_temperature)
    """
    max_probs = pred_probs.max(axis=1)
    frac_overconfident = (max_probs > 0.99).mean()
    print(f"\nOverconfidence check:")
    print(f"  Fraction of samples with max_prob > 0.99: {frac_overconfident:.3%}")

    if frac_overconfident > threshold:
        print(f"  WARNING: >90% overconfident — applying temperature scaling T={temperature}")
        scaled_logits = logits_all / temperature
        # Softmax of scaled logits
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        scaled_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        print(f"  After scaling: fraction max_prob > 0.99 = {(scaled_probs.max(axis=1) > 0.99).mean():.3%}")
        return scaled_probs, temperature
    else:
        print(f"  OK: Overconfidence fraction {frac_overconfident:.3%} <= {threshold:.0%} threshold")
        print(f"  Temperature scaling NOT applied (T=1.0)")
        return pred_probs, 1.0


def main():
    args = parse_args()

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Reconstruct fold splits (must exactly mirror cross_validate_classifier in cli.py)
    print("\n=== Step 1: Reconstructing fold splits ===")
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    base_train = datasets.ImageFolder(str(train_dir))
    base_val = datasets.ImageFolder(str(val_dir))

    # Combined train+val samples (CRITICAL: same order as training)
    full_samples = base_train.samples + base_val.samples   # list of (path, label)
    full_labels = base_train.targets + base_val.targets    # list of int labels

    class_names = base_train.classes  # ['COVID', 'Normal', 'Viral_Pneumonia']
    n_total = len(full_samples)
    n_classes = len(class_names)

    print(f"Total train+val samples: {n_total}")
    print(f"Classes: {class_names}")

    # Step 2: Load stored CV results for verification
    print("\n=== Step 2: Loading CV results for verification ===")
    cv_results_path = args.checkpoint_dir / "cross_validation_results.json"
    if not cv_results_path.exists():
        raise FileNotFoundError(f"CV results not found: {cv_results_path}")

    with open(cv_results_path) as f:
        cv_results = json.load(f)

    stored_train_val = cv_results.get("train_val_samples")
    if stored_train_val and stored_train_val != n_total:
        raise ValueError(
            f"Total samples mismatch: reconstructed {n_total}, stored {stored_train_val}. "
            "Dataset may differ from training."
        )

    mean_val_acc = cv_results["summary"]["val_accuracy_mean"]
    print(f"Stored mean val accuracy: {mean_val_acc:.4f}")
    print(f"Stored train_val_samples: {stored_train_val}")

    # Step 3: Verify fold splits against stored metrics
    print("\n=== Step 3: Verifying fold split reconstruction ===")
    splits = verify_fold_splits(
        fold_results=cv_results["fold_results"],
        full_samples=full_samples,
        full_labels=full_labels,
        n_splits=5,
        seed=args.seed,
    )

    # Step 4: Extract OOF probabilities
    print("\n=== Step 4: Extracting out-of-fold probabilities ===")
    transform = get_classifier_transforms(train=False)

    pred_probs, logits_all = extract_oof_probabilities(
        splits=splits,
        full_samples=full_samples,
        full_labels=full_labels,
        checkpoint_dir=args.checkpoint_dir,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        n_classes=n_classes,
    )

    # Step 5: Check for overconfidence and apply temperature scaling if needed
    print("\n=== Step 5: Overconfidence check and temperature scaling ===")
    final_probs, applied_temperature = check_overconfidence_and_apply_temperature(
        pred_probs=pred_probs,
        logits_all=logits_all,
        threshold=0.90,
        temperature=2.0,
    )

    # Step 6: Build image_names array (just filenames, not full paths)
    image_names = np.array([Path(path).name for path, _ in full_samples])
    true_labels = np.array(full_labels, dtype=np.int32)

    # Step 7: Compute OOF accuracy
    oof_preds = final_probs.argmax(axis=1)
    oof_acc = (oof_preds == true_labels).mean()

    print(f"\n=== Step 6: Verification Summary ===")
    print(f"Total OOF samples: {n_total}")
    print(f"OOF accuracy: {oof_acc:.4f}")
    print(f"Stored mean val accuracy: {mean_val_acc:.4f}")
    acc_diff = abs(oof_acc - mean_val_acc)
    print(f"Difference: {acc_diff:.4f} (threshold: 0.01)")
    if acc_diff > 0.01:
        print(f"  WARNING: OOF accuracy differs from stored mean val accuracy by > 1%")
    else:
        print(f"  OK: OOF accuracy within 1% of stored mean val accuracy")

    # Max prob distribution
    max_probs_final = final_probs.max(axis=1)
    print(f"\nMax probability distribution:")
    for thr in [0.50, 0.70, 0.90, 0.95, 0.99]:
        frac = (max_probs_final > thr).mean()
        print(f"  > {thr:.2f}: {frac:.3%} ({int(frac * n_total)})")

    print(f"\nTemperature applied: {applied_temperature}")

    # Step 8: Save OOF matrix
    print("\n=== Step 7: Saving OOF probabilities ===")
    output_path = args.output_dir / "oof_probabilities.npz"
    np.savez(
        output_path,
        pred_probs=final_probs,          # (N, 3) float32
        image_names=image_names,          # (N,) str
        true_labels=true_labels,          # (N,) int32
        class_names=np.array(class_names),  # ['COVID', 'Normal', 'Viral_Pneumonia']
        temperature=np.float32(applied_temperature),
    )
    print(f"Saved: {output_path}")
    print(f"  pred_probs shape: {final_probs.shape}")
    print(f"  image_names count: {len(image_names)}")
    print(f"  true_labels count: {len(true_labels)}")
    print(f"  class_names: {class_names}")

    print("\n=== OOF Extraction Complete ===")


if __name__ == "__main__":
    main()
