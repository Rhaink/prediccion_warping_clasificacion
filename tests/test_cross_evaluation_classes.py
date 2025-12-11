"""
Test cross-evaluation class consistency.

This test validates that cross-evaluation comparisons only happen
between datasets with the SAME classes to prevent invalid comparisons.

Scientific validity: Comparing 4-class vs 3-class models is INVALID
and was the source of the erroneous "11x better generalization" claim.
"""

import json
import pytest
from pathlib import Path
from typing import Set


class TestCrossEvaluationClassConsistency:
    """Tests for class consistency in cross-evaluation."""

    def get_dataset_classes(self, dataset_path: Path) -> Set[str]:
        """Extract class names from a dataset directory structure."""
        if not dataset_path.exists():
            return set()

        # Try to get classes from test split first
        test_path = dataset_path / "test"
        if test_path.exists():
            return {d.name for d in test_path.iterdir() if d.is_dir()}

        # Otherwise from train split
        train_path = dataset_path / "train"
        if train_path.exists():
            return {d.name for d in train_path.iterdir() if d.is_dir()}

        # Direct children if no splits
        return {d.name for d in dataset_path.iterdir() if d.is_dir()}

    def test_original_3classes_matches_warped_classes(self):
        """Verify original_3_classes has same classes as warped dataset."""
        base_path = Path("outputs")

        original_3classes = base_path / "original_3_classes"
        warped_full_coverage = base_path / "full_coverage_warped_dataset"
        warped_47 = base_path / "full_warped_dataset"

        # Skip if datasets don't exist
        if not original_3classes.exists():
            pytest.skip("original_3_classes dataset not found")

        original_classes = self.get_dataset_classes(original_3classes)

        # Check against full_coverage warped
        if warped_full_coverage.exists():
            warped_classes = self.get_dataset_classes(warped_full_coverage)
            assert original_classes == warped_classes, \
                f"Class mismatch: original_3_classes has {original_classes}, " \
                f"full_coverage_warped has {warped_classes}"

        # Check against 47% warped
        if warped_47.exists():
            warped_classes = self.get_dataset_classes(warped_47)
            assert original_classes == warped_classes, \
                f"Class mismatch: original_3_classes has {original_classes}, " \
                f"warped_47 has {warped_classes}"

    def test_cross_evaluation_results_have_same_classes(self):
        """Verify cross-evaluation results use consistent classes."""
        results_path = Path("outputs/cross_evaluation_valid_3classes/cross_evaluation_results.json")

        if not results_path.exists():
            pytest.skip("Cross-evaluation results not found")

        with open(results_path) as f:
            results = json.load(f)

        # Get class names from results
        class_names = results.get("class_names", [])

        # Should have exactly 3 classes
        assert len(class_names) == 3, \
            f"Expected 3 classes, got {len(class_names)}: {class_names}"

        # Should be our standard 3 classes
        expected_classes = {"COVID", "Normal", "Viral_Pneumonia"}
        assert set(class_names) == expected_classes, \
            f"Expected classes {expected_classes}, got {set(class_names)}"

    def test_no_lung_opacity_in_3class_datasets(self):
        """Verify Lung_Opacity class is NOT present in 3-class datasets."""
        datasets_to_check = [
            Path("outputs/original_3_classes"),
            Path("outputs/full_coverage_warped_dataset"),
            Path("outputs/full_warped_dataset"),
        ]

        for dataset_path in datasets_to_check:
            if not dataset_path.exists():
                continue

            classes = self.get_dataset_classes(dataset_path)

            assert "Lung_Opacity" not in classes, \
                f"Dataset {dataset_path} should NOT contain Lung_Opacity class, " \
                f"found classes: {classes}"

    def test_confusion_matrix_dimensions_match_classes(self):
        """Verify confusion matrix dimensions match number of classes."""
        results_path = Path("outputs/cross_evaluation_valid_3classes/cross_evaluation_results.json")

        if not results_path.exists():
            pytest.skip("Cross-evaluation results not found")

        with open(results_path) as f:
            results = json.load(f)

        class_names = results.get("class_names", [])
        n_classes = len(class_names)

        # Check all confusion matrices
        for scenario in ["a_on_a", "a_on_b", "b_on_a", "b_on_b"]:
            if scenario not in results.get("results", {}):
                continue

            cm = results["results"][scenario].get("confusion_matrix", [])

            assert len(cm) == n_classes, \
                f"Confusion matrix for {scenario} has wrong rows: " \
                f"expected {n_classes}, got {len(cm)}"

            for i, row in enumerate(cm):
                assert len(row) == n_classes, \
                    f"Confusion matrix for {scenario} row {i} has wrong columns: " \
                    f"expected {n_classes}, got {len(row)}"


class TestDatasetSampleCounts:
    """Tests for sample count consistency."""

    def count_samples(self, dataset_path: Path, split: str = "test") -> int:
        """Count samples in a dataset split."""
        split_path = dataset_path / split
        if not split_path.exists():
            return 0

        count = 0
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                count += len(list(class_dir.glob("*.png")))
                count += len(list(class_dir.glob("*.jpg")))
        return count

    def test_test_splits_have_same_size(self):
        """Verify test splits have same number of samples for fair comparison.

        Note: We only compare datasets that should have the same samples:
        - original_3_classes and full_coverage_warped_dataset
        The old warped_47 dataset may have different sample counts due to
        different generation methodology.
        """
        datasets = {
            "original_3_classes": Path("outputs/original_3_classes"),
            "full_coverage_warped": Path("outputs/full_coverage_warped_dataset"),
        }

        counts = {}
        for name, path in datasets.items():
            if path.exists():
                counts[name] = self.count_samples(path, "test")

        if len(counts) < 2:
            pytest.skip("Not enough datasets found for comparison")

        # These specific datasets should have the same count
        first_count = list(counts.values())[0]
        for name, count in counts.items():
            assert count == first_count or abs(count - first_count) <= 1, \
                f"Sample count mismatch: {name} has {count}, expected ~{first_count}"

    def test_cross_eval_sample_counts_match(self):
        """Verify cross-evaluation used same sample counts."""
        results_path = Path("outputs/cross_evaluation_valid_3classes/cross_evaluation_results.json")

        if not results_path.exists():
            pytest.skip("Cross-evaluation results not found")

        with open(results_path) as f:
            results = json.load(f)

        # Get sample counts from all scenarios
        sample_counts = []
        for scenario in ["a_on_a", "a_on_b", "b_on_a", "b_on_b"]:
            if scenario in results.get("results", {}):
                n = results["results"][scenario].get("n_samples", 0)
                sample_counts.append(n)

        # All should be the same
        if sample_counts:
            first_count = sample_counts[0]
            assert all(c == first_count for c in sample_counts), \
                f"Sample counts should match across scenarios: {sample_counts}"


class TestCrossEvaluationMethodology:
    """Tests for cross-evaluation methodology correctness."""

    def test_gaps_calculated_correctly(self):
        """Verify generalization gaps are calculated correctly."""
        results_path = Path("outputs/cross_evaluation_valid_3classes/cross_evaluation_results.json")

        if not results_path.exists():
            pytest.skip("Cross-evaluation results not found")

        with open(results_path) as f:
            results = json.load(f)

        r = results.get("results", {})
        gaps = results.get("gaps", {})

        # Gap A = acc(A on A) - acc(A on B)
        expected_gap_a = r["a_on_a"]["accuracy"] - r["a_on_b"]["accuracy"]
        actual_gap_a = gaps.get("gap_a", 0)

        assert abs(expected_gap_a - actual_gap_a) < 0.01, \
            f"Gap A calculation error: expected {expected_gap_a:.2f}, got {actual_gap_a:.2f}"

        # Gap B = acc(B on B) - acc(B on A)
        expected_gap_b = r["b_on_b"]["accuracy"] - r["b_on_a"]["accuracy"]
        actual_gap_b = gaps.get("gap_b", 0)

        assert abs(expected_gap_b - actual_gap_b) < 0.01, \
            f"Gap B calculation error: expected {expected_gap_b:.2f}, got {actual_gap_b:.2f}"

    def test_ratio_indicates_correct_winner(self):
        """Verify the ratio correctly identifies which model generalizes better."""
        results_path = Path("outputs/cross_evaluation_valid_3classes/cross_evaluation_results.json")

        if not results_path.exists():
            pytest.skip("Cross-evaluation results not found")

        with open(results_path) as f:
            results = json.load(f)

        gaps = results.get("gaps", {})
        gap_a = gaps.get("gap_a", 0)
        gap_b = gaps.get("gap_b", 0)
        better = gaps.get("better_generalizer", "")

        # Lower gap = better generalization
        if gap_a < gap_b:
            assert better == "A", \
                f"Model A has lower gap ({gap_a:.2f} < {gap_b:.2f}) but {better} is marked as better"
        else:
            assert better == "B", \
                f"Model B has lower gap ({gap_b:.2f} < {gap_a:.2f}) but {better} is marked as better"
