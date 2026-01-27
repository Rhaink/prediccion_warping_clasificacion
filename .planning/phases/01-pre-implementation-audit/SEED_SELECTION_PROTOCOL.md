# Cross-Validation Seed Selection Protocol

## Document Purpose

This document formally describes the cross-validation methodology and model selection protocol used to train the 5 classifier models that form the baseline ensemble. This protocol ensures methodological rigor and prevents test set contamination.

## Cross-Validation Configuration

### K-Fold Setup
- **Number of folds**: 5 (stratified by class)
- **Total dataset**: 15,153 images
- **Training + Validation**: 13,258 images (87.5% of total)
- **Test holdout**: 1,895 images (12.5% of total, fixed across all experiments)
- **Random seed**: 42 (consistent across all experiments for reproducibility)

### Data Split Methodology
The dataset is split using stratified sampling to preserve class distribution:
- **Train**: ~80% of non-test data (approximately 10,606 images per fold)
- **Validation**: ~20% of non-test data (approximately 2,652 images per fold)
- **Test**: Fixed holdout of 1,895 images (never used during training)

**Class distribution** (from fold 1 as example):
```
Train: COVID=2,531 | Normal=7,134 | Viral_Pneumonia=941
Val:   COVID=633   | Normal=1,784 | Viral_Pneumonia=235
Test:  COVID=452   | Normal=1,274 | Viral_Pneumonia=169
```

## Model Selection Criteria

### Per-Fold Selection
- **Metric**: Best validation F1-macro score
- **Early stopping**: Patience of 10 epochs (if no improvement)
- **Checkpoint saving**: Model saved only at best validation performance
- **No cherry-picking**: All 5 folds included in ensemble (no post-hoc selection)

### Ensemble Strategy
- **Composition**: All 5 cross-validation fold models
- **Diversity source**: Different data partitions capture complementary patterns
- **No hyperparameter tuning on test set**: Test set used only for final evaluation

## Training Protocol

### Model Architecture
- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Classifier head**: Fully connected layer for 3 classes (COVID, Normal, Viral_Pneumonia)
- **Input size**: 96x96 pixels (warped and normalized lung images)

### Training Configuration
From `configs/classifier_warped_base.json`:
```json
{
  "backbone": "resnet18",
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.0001,
  "patience": 10,
  "use_class_weights": true,
  "seed": 42
}
```

### Loss Function
- **Type**: CrossEntropyLoss with class weights
- **Class weights**: Computed from training set class distribution to handle imbalance

### Data Augmentation
Standard augmentations applied during training:
- Random horizontal flip
- Random rotation (±15 degrees)
- Random affine transformations (scale, translate, shear)
- Color jitter (brightness, contrast)

No augmentation during validation or test evaluation.

## Validation Protocol

### Step 1: Train 5 Models
For each fold k=1 to 5:
1. Split non-test data into train_k and val_k
2. Train model on train_k
3. Validate on val_k after each epoch
4. Save checkpoint when validation F1-macro improves
5. Apply early stopping if no improvement for 10 epochs

### Step 2: Save Best Checkpoints
Each model checkpoint is saved when validation F1-macro achieves a new maximum:
- `outputs/classifier_cv/fold_01/best_classifier.pt`
- `outputs/classifier_cv/fold_02/best_classifier.pt`
- `outputs/classifier_cv/fold_03/best_classifier.pt`
- `outputs/classifier_cv/fold_04/best_classifier.pt`
- `outputs/classifier_cv/fold_05/best_classifier.pt`

### Step 3: Test Set Evaluation
After all training completes:
1. Load each best checkpoint
2. Evaluate on the fixed test set (1,895 images)
3. Compute per-fold test accuracy
4. Report mean ± std across 5 folds

### Step 4: Ensemble Construction
Use all 5 models for ensemble (no post-hoc selection based on test performance).

## Rationale

### Why K=5?
- **Industry standard**: 5-fold CV is widely used in medical imaging research
- **Balance**: Provides reasonable training set size while maintaining validation diversity
- **Computational efficiency**: Fewer folds than 10-fold while still capturing variance

### Why Stratified Sampling?
- **Class imbalance**: Normal class is ~67% of dataset
- **Stability**: Ensures each fold has representative class distribution
- **Reduces variance**: Prevents folds with extreme class distributions

### Why All 5 Folds in Ensemble?
- **Diversity**: Different data partitions capture complementary patterns
- **Stability**: Low variance (std < 0.2%) indicates robust methodology
- **No cherry-picking**: Using all folds prevents implicit test set leakage
- **Standard practice**: Ensembling all CV folds is methodologically sound

### Why F1-Macro for Selection?
- **Class balance**: Treats all classes equally (unlike F1-weighted)
- **Medical context**: All diseases equally important to diagnose correctly
- **Stability**: Less sensitive to class imbalance than accuracy

## Evidence and Verification

### Implementation Commits
- Commit `fb062906`: "feat: complete scientific validation with gpu-accelerated k-fold cv and rigorous documentation"
- Date: 2025-12-21

### Training Artifacts
- Configuration: `configs/classifier_warped_base.json`
- Results: `outputs/classifier_cv/cross_validation_results.json` (validation metrics)
- Results: `outputs/classifier_cv/cross_validation_test_results.json` (test metrics)
- Per-fold results: `outputs/classifier_cv/fold_*/results.json`
- Training curves: `outputs/classifier_cv/fold_*/training_history.json`

### Validation Metrics (Example from Fold 1)
- Epochs trained: 39 (stopped early due to patience=10)
- Best validation F1-macro: 0.9829
- Validation accuracy: 98.68%
- Test accuracy: 97.52%

### Cross-Fold Consistency
All 5 folds show consistent training behavior:
- Epochs trained: 34-43 (early stopping applied)
- Validation F1-macro: 0.9805-0.9874
- Test accuracy: 97.52%-97.94% (mean=97.68%, std=0.16%)

Low standard deviation across folds indicates:
1. Robust training methodology
2. Stable model architecture
3. Representative data splits
4. Minimal overfitting

## Test Set Isolation Guarantees

### Temporal Isolation
- All model checkpoints dated: 2026-01-16
- All test evaluations dated: 2026-01-27
- **Gap**: 11 days between training and test evaluation

### Methodological Isolation
- Test set never used for:
  - Model selection (used validation F1-macro)
  - Early stopping decisions (used validation loss)
  - Hyperparameter tuning (used validation metrics)
  - Checkpoint saving (triggered by validation performance)

### Evidence of Proper Usage
From `outputs/classifier_cv/fold_*/results.json`:
```json
{
  "test_metrics": null  // Test metrics not computed during training
}
```

Test metrics only appear in separate `test_results.json` files created 11 days later.

## Recommendations for Future Work

1. **Maintain seed consistency**: Continue using seed=42 for reproducibility
2. **Document any changes**: If modifying CV setup, update this protocol
3. **Preserve test set**: Never use test set for model development decisions
4. **Extend to nested CV**: Consider nested CV for hyperparameter tuning in Phase 2+

## Summary

This protocol demonstrates:
- ✅ Rigorous cross-validation methodology (stratified 5-fold)
- ✅ Proper model selection (validation F1-macro, no test set peeking)
- ✅ Ensemble diversity (all 5 folds included)
- ✅ Test set isolation (temporal and methodological)
- ✅ Reproducibility (fixed seed, documented configuration)
- ✅ Stability (low variance across folds: std=0.16%)

The baseline ensemble is methodologically sound and suitable as a foundation for Phase 2 improvements.

---
*Document created: 2026-01-27*
*Phase: 01-pre-implementation-audit*
*Plan: 01-02*
