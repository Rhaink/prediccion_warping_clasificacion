"""Duplicate detection for medical imaging datasets.

This module provides dual-stage duplicate detection:
1. Stage 1: Fast perceptual hashing (PHash) for candidate pairs
2. Stage 2: CNN similarity verification for high-confidence matches

Designed for medical X-rays with conservative thresholds due to high baseline similarity.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from imagededup.methods import PHash, CNN
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_duplicates(
    image_dir: str,
    hash_threshold: int = 3,
    min_similarity_cnn: float = 0.98,
    use_cnn_verification: bool = True,
) -> Dict:
    """Detect duplicate and near-duplicate images using dual-stage approach.

    Args:
        image_dir: Directory containing images to analyze (can have subdirs)
        hash_threshold: Maximum Hamming distance for PHash duplicates (default: 3)
        min_similarity_cnn: Minimum CNN similarity score for verification (default: 0.98)
        use_cnn_verification: Whether to verify PHash candidates with CNN (default: True)

    Returns:
        Dict with keys:
            - pairs: List of tuples (img1, img2, hash_dist, cnn_similarity)
            - total_images: Number of images processed
            - total_pairs: Number of duplicate pairs found
            - candidates_rejected: Number of PHash candidates rejected by CNN (if verified)

    Notes:
        - Conservative thresholds for medical X-rays (high baseline similarity)
        - PHash is fast O(N^2) comparisons but may have false positives
        - CNN verification is slower but more accurate
        - Set use_cnn_verification=False for large datasets (>50k images)
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    # Collect all image files
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        image_files.extend(image_path.rglob(ext))

    total_images = len(image_files)
    logger.info(f"Found {total_images} images in {image_dir}")

    if total_images == 0:
        logger.warning("No images found!")
        return {"pairs": [], "total_images": 0, "total_pairs": 0, "candidates_rejected": 0}

    # Stage 1: PHash to find candidate pairs
    logger.info(f"Stage 1: Computing perceptual hashes (threshold={hash_threshold})")
    phasher = PHash()

    try:
        # Get encodings - imagededup expects string paths
        encodings = phasher.encode_images(image_dir=str(image_dir))

        # Find duplicates
        duplicates_dict = phasher.find_duplicates(
            encoding_map=encodings,
            max_distance_threshold=hash_threshold,
        )

        # Convert to pairs with hash distances
        phash_pairs = []
        seen_pairs = set()

        for img1, similar_images in duplicates_dict.items():
            if not similar_images:
                continue

            for img2 in similar_images:
                # Create canonical pair (sorted to avoid duplicates)
                pair = tuple(sorted([img1, img2]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    # Compute Hamming distance
                    hash1 = encodings[img1]
                    hash2 = encodings[img2]
                    hamming_dist = bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
                    phash_pairs.append((pair[0], pair[1], hamming_dist, None))

        logger.info(f"Stage 1 complete: {len(phash_pairs)} candidate pairs found")

    except Exception as e:
        logger.error(f"PHash failed: {e}")
        return {"pairs": [], "total_images": total_images, "total_pairs": 0,
                "candidates_rejected": 0}

    # Stage 2: CNN verification (optional)
    verified_pairs = []
    candidates_rejected = 0

    if use_cnn_verification and len(phash_pairs) > 0:
        logger.info(f"Stage 2: Verifying with CNN (threshold={min_similarity_cnn})")
        logger.warning("CNN verification is slow - this may take several minutes")

        try:
            cnn_encoder = CNN()

            # Encode all images with CNN
            cnn_encodings = cnn_encoder.encode_images(image_dir=str(image_dir))

            # Verify each PHash candidate pair
            for img1, img2, hamming_dist, _ in tqdm(phash_pairs, desc="CNN verification"):
                try:
                    # Compute cosine similarity between CNN embeddings
                    emb1 = cnn_encodings[img1]
                    emb2 = cnn_encodings[img2]

                    # Cosine similarity
                    similarity = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                    )

                    if similarity >= min_similarity_cnn:
                        verified_pairs.append((img1, img2, hamming_dist, float(similarity)))
                    else:
                        candidates_rejected += 1

                except Exception as e:
                    logger.warning(f"CNN verification failed for {img1}-{img2}: {e}")
                    continue

            logger.info(
                f"Stage 2 complete: {len(verified_pairs)} verified, "
                f"{candidates_rejected} rejected"
            )

        except Exception as e:
            logger.error(f"CNN verification failed: {e}. Using PHash results only.")
            verified_pairs = phash_pairs
    else:
        # No CNN verification - use PHash results
        verified_pairs = phash_pairs

    return {
        "pairs": verified_pairs,
        "total_images": total_images,
        "total_pairs": len(verified_pairs),
        "candidates_rejected": candidates_rejected,
    }


def classify_duplicate_types(
    pairs: List[Tuple[str, str, int, Optional[float]]],
    dataset_structure: Dict[str, Dict[str, str]],
) -> Dict:
    """Classify duplicate pairs by split and class relationships.

    Args:
        pairs: List of duplicate pairs (img1, img2, hash_dist, cnn_sim)
        dataset_structure: Map of filename -> {"split": str, "class": str}

    Returns:
        Dict with keys:
            - cross_split_cross_class: Different split AND different class (CRITICAL)
            - cross_split: Different split, same class (data leakage)
            - cross_class: Same split, different class (potential label error)
            - within_split_within_class: Same split and class (redundancy)
            - unclassified: Images not in dataset_structure
            - counts: Summary counts for each category
            - details: Detailed list for each category

    Notes:
        Priority order for flagging:
        1. cross_split_cross_class (MOST CRITICAL)
        2. cross_split (CRITICAL - data leakage)
        3. cross_class (CRITICAL - potential label error)
        4. within_split_within_class (INFO - redundancy)
    """
    results = {
        "cross_split_cross_class": [],
        "cross_split": [],
        "cross_class": [],
        "within_split_within_class": [],
        "unclassified": [],
    }

    for img1, img2, hash_dist, cnn_sim in pairs:
        # Extract filename without path
        fname1 = Path(img1).name
        fname2 = Path(img2).name

        # Get metadata
        meta1 = dataset_structure.get(fname1)
        meta2 = dataset_structure.get(fname2)

        if meta1 is None or meta2 is None:
            results["unclassified"].append((img1, img2, hash_dist, cnn_sim))
            continue

        split1, class1 = meta1["split"], meta1["class"]
        split2, class2 = meta2["split"], meta2["class"]

        # Create detail record
        detail = {
            "img1": img1,
            "img2": img2,
            "hash_distance": hash_dist,
            "cnn_similarity": cnn_sim,
            "split1": split1,
            "split2": split2,
            "class1": class1,
            "class2": class2,
        }

        # Classify by priority
        if split1 != split2 and class1 != class2:
            results["cross_split_cross_class"].append(detail)
        elif split1 != split2:
            results["cross_split"].append(detail)
        elif class1 != class2:
            results["cross_class"].append(detail)
        else:
            results["within_split_within_class"].append(detail)

    # Add summary counts
    results["counts"] = {
        key: len(results[key])
        for key in [
            "cross_split_cross_class",
            "cross_split",
            "cross_class",
            "within_split_within_class",
            "unclassified",
        ]
    }

    return results


def compare_original_vs_warped(
    original_pairs: List[Tuple[str, str, int, Optional[float]]],
    warped_pairs: List[Tuple[str, str, int, Optional[float]]],
) -> Dict:
    """Compare duplicate detection results between original and warped datasets.

    Args:
        original_pairs: Duplicate pairs from original dataset
        warped_pairs: Duplicate pairs from warped dataset

    Returns:
        Dict with keys:
            - diverged: Pairs duplicate in original but NOT in warped
            - converged: Pairs NOT duplicate in original but IS in warped
            - persistent: Pairs duplicate in both datasets
            - counts: Summary counts
            - analysis: Interpretation of results

    Notes:
        - Diverged: Warping successfully differentiated similar images (GOOD)
        - Converged: Warping made dissimilar images similar (CONCERNING)
        - Persistent: Inherent duplicates unaffected by warping
    """
    # Convert to sets of filename pairs (ignore paths for comparison)
    def normalize_pair(img1: str, img2: str) -> Tuple[str, str]:
        """Extract filenames and create canonical pair."""
        fname1 = Path(img1).stem  # Remove extension too
        fname2 = Path(img2).stem
        return tuple(sorted([fname1, fname2]))

    original_set = {normalize_pair(img1, img2) for img1, img2, _, _ in original_pairs}
    warped_set = {normalize_pair(img1, img2) for img1, img2, _, _ in warped_pairs}

    # Compute set operations
    diverged = original_set - warped_set
    converged = warped_set - original_set
    persistent = original_set & warped_set

    # Analysis
    total_original = len(original_set)
    total_warped = len(warped_set)
    diverged_pct = 100 * len(diverged) / total_original if total_original > 0 else 0
    converged_pct = 100 * len(converged) / total_warped if total_warped > 0 else 0
    persistent_pct = 100 * len(persistent) / total_original if total_original > 0 else 0

    analysis = {
        "diverged_interpretation": (
            "GOOD: Warping differentiated these similar images"
            if len(diverged) > 0
            else "No divergence detected"
        ),
        "converged_interpretation": (
            "CONCERNING: Warping made these dissimilar images similar"
            if len(converged) > 0
            else "No convergence detected"
        ),
        "persistent_interpretation": (
            "Inherent duplicates unaffected by geometric normalization"
            if len(persistent) > 0
            else "No persistent duplicates"
        ),
        "diverged_percentage": round(diverged_pct, 2),
        "converged_percentage": round(converged_pct, 2),
        "persistent_percentage": round(persistent_pct, 2),
    }

    return {
        "diverged": list(diverged),
        "converged": list(converged),
        "persistent": list(persistent),
        "counts": {
            "diverged": len(diverged),
            "converged": len(converged),
            "persistent": len(persistent),
        },
        "analysis": analysis,
    }
