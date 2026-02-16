# Technology Stack: Data-Centric Improvements

**Project:** COVID-19 Chest X-ray Classification
**Researched:** 2026-02-16
**Confidence:** HIGH

## Recommended Additions

### Label Noise Detection (NEW)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| cleanlab | >=2.7.1 | Confident learning, label error detection | Industry standard data-centric AI, model-agnostic, works with PyTorch via sklearn wrappers |
| cleanvision | >=0.4.0 | Image-specific quality issues | Companion to cleanlab, detects dark/bright/blurry/odd-aspect images |

**Integration:** Pass soft predictions from 5-fold CV to `cleanlab.filter.find_label_issues()`.

### Advanced Augmentation (NEW)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| albumentations | >=2.0.8 | Medical-grade augmentation pipeline | 100+ transforms, faster than torchvision, medical-specific (ElasticTransform, GridDistortion) |

**Integration:** Replace torchvision transforms with `A.Compose()`. Compatible via `ToTensorV2()`.

**Note:** AlbumentationsX uses AGPL-3.0 — stick with albumentations 2.0.8 for academic use.

### Image Quality Assessment (NEW)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pyiqa | >=0.1.13 | No-reference image quality metrics | 38+ metrics (BRISQUE, NIQE), GPU-accelerated, PyTorch-native |

**Integration:** `pyiqa.create_metric('brisque')` to score images, filter bottom 5-10%.

### Class Imbalance (NEW)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| (built-in) | — | Focal loss implementation | Simple PyTorch implementation, no extra dependency needed |
| torchsampler | >=0.1.2 | ImbalancedDatasetSampler | Automatic oversampling minority classes |

**Integration:** Replace `nn.CrossEntropyLoss(weight=...)` with focal loss. Use `ImbalancedDatasetSampler` in DataLoader.

## Core Framework (Unchanged)
- PyTorch >=2.0, torchvision >=0.15, OpenCV >=4.8, NumPy, SciPy, matplotlib, Pillow

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Label Noise | cleanlab | Manual thresholding | Cleanlab has provable guarantees |
| Augmentation | albumentations | imgaug, Kornia | imgaug maintenance uncertain, Kornia lacks medical focus |
| Quality | pyiqa | PIQ | pyiqa more comprehensive (38+ vs fewer metrics) |
| Imbalance | Focal loss (custom) | balanced-loss pkg | Simpler to implement directly, fewer dependencies |
| Oversampling | torchsampler | SMOTE | SMOTE creates unrealistic X-ray interpolations |

## What NOT to Add
- **SMOTE/imbalanced-learn** — poor PyTorch integration, unrealistic medical images
- **AlbumentationsX** — AGPL-3.0 incompatible with academic licensing
- **MONAI** — adds complexity without clear benefit for this pipeline
- **Custom label smoothing** — conflicts with confident learning

## Installation
```bash
pip install cleanlab>=2.7.1 cleanvision>=0.4.0
pip install albumentations>=2.0.8
pip install pyiqa>=0.1.13
pip install torchsampler>=0.1.2
```
