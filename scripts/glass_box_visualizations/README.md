# Glass-Box Visualization System

This directory contains scripts to generate comprehensive visualizations explaining the COVID-19 detection pipeline for non-technical audiences.

## 📁 Structure

```
glass_box_visualizations/
├── README.md                    # This file
├── utils.py                     # Common utilities (sample loading, drawing, etc.)
├── diagram_utils.py             # Diagram creation utilities (architecture diagrams)
├── block_a_pipeline.py          # ✅ Pipeline overview (A1, A2)
├── block_b_landmarks.py         # ✅ Landmark detection (B1-B6)
├── block_c_warping.py           # 🚧 Warping/GPA (C1-C5) - TO IMPLEMENT
├── block_d_classifier.py        # 🚧 Classification (D1-D4) - TO IMPLEMENT
├── block_e_comparison.py        # 🚧 Comparisons (E1-E3) - TO IMPLEMENT
└── generate_all.py              # 🚧 Orchestrator script - TO IMPLEMENT
```

## 🎯 Generated Figures

### Block A: Pipeline Overview
- **A1**: Complete flow of single image through pipeline
- **A2**: Comparison grid (Original vs Warped, 12 examples)

### Block B: Landmark Detection (✅ IMPLEMENTED)
- **B1**: ResNet-18 Feature Hierarchy (layer1 → layer4)
- **B2**: Coordinate Attention Maps
- **B3**: Regression Head Flow
- **B4**: Wing Loss Function
- **B5**: Ensemble + TTA Diagram
- **B6**: Error by Landmark Heatmap

### Block C: Geometric Normalization (🚧 TO IMPLEMENT)
- **C1**: GPA Iterative Alignment (5 steps)
- **C2**: Delaunay Triangulation
- **C3**: Piecewise Affine Transformation
- **C4**: Margin Scale Effect
- **C5**: Warping Gallery (before/after examples)

### Block D: Classification & Explainability (🚧 TO IMPLEMENT)
- **D1**: Classifier Architecture
- **D2**: Grad-CAM Heatmaps by Class
- **D3**: Failure Cases Analysis
- **D4**: Confusion Matrix with examples

### Block E: Why It Works (🚧 TO IMPLEMENT)
- **E1**: Feature Comparison (Original vs Warped)
- **E2**: CLAHE Effect Visualization
- **E3**: Artefact Elimination Examples

## 🚀 Quick Start

### Prerequisites

Ensure you have the following data ready:

```bash
# Required files
outputs/landmark_predictions/session_warping/predictions.npz
outputs/warped_lung_best/session_warping/
outputs/shape_analysis/canonical_shape.npz
outputs/classifier_warped_lung_best/best_classifier.pt
checkpoints/session10/ensemble/seed123/final_model.pt
GROUND_TRUTH.json
```

### Generate All Figures

```bash
# Once generate_all.py is implemented
python scripts/glass_box_visualizations/generate_all.py --output-dir outputs/glass_box
```

### Generate Individual Blocks

```bash
# Block A: Pipeline Overview
python scripts/glass_box_visualizations/block_a_pipeline.py

# Block B: Landmark Detection
python scripts/glass_box_visualizations/block_b_landmarks.py \
  --checkpoint checkpoints/session10/ensemble/seed123/final_model.pt \
  --image data/dataset/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png \
  --output-dir outputs/glass_box/block_b

# Block C: Warping (TO IMPLEMENT)
# python scripts/glass_box_visualizations/block_c_warping.py

# Block D: Classifier (TO IMPLEMENT)
# python scripts/glass_box_visualizations/block_d_classifier.py

# Block E: Comparisons (TO IMPLEMENT)
# python scripts/glass_box_visualizations/block_e_comparison.py
```

## 📋 Implementation Checklist

### ✅ Completed
- [x] Documentation structure (`docs/GLASS_BOX_PIPELINE.md`)
- [x] Common utilities (`utils.py`)
- [x] Diagram utilities (`diagram_utils.py`)
- [x] Block A: Pipeline overview
- [x] Block B: Landmark detection (already existed)

### 🚧 To Implement

#### Block C: Warping (`block_c_warping.py`)
```python
def generate_C1_gpa_alignment(...)
def generate_C2_delaunay_triangulation(...)
def generate_C3_piecewise_affine(...)
def generate_C4_margin_scale_effect(...)
def generate_C5_warping_gallery(...)
```

#### Block D: Classifier (`block_d_classifier.py`)
```python
def generate_D1_architecture(...)
def generate_D2_gradcam_by_class(...)
def generate_D3_failure_cases(...)
def generate_D4_confusion_matrix(...)
```

#### Block E: Comparisons (`block_e_comparison.py`)
```python
def generate_E1_feature_comparison(...)
def generate_E2_clahe_effect(...)
def generate_E3_artefact_elimination(...)
```

#### Orchestrator (`generate_all.py`)
```python
def main():
    # Run all blocks in sequence
    # Handle dependencies
    # Generate summary report
```

## 🎨 Design Guidelines

### Color Palette (defined in `utils.py`)
- **Green** (`#2ECC71`): COVID, landmarks correct
- **Blue** (`#3498DB`): Normal class
- **Yellow/Orange** (`#F39C12`): Viral Pneumonia
- **Red** (`#E74C3C`): Errors, high attention
- **Gray** (`#95A5A6`): Background, inactive

### Typography
- **Sans-serif** fonts (Arial, Helvetica)
- **Title**: 12-14pt
- **Labels**: 9-10pt
- **Annotations**: 8pt

### Figure Quality
- **DPI**: 300 (publication quality)
- **Format**: PNG (for document embedding)
- **Aspect ratios**: Consistent across similar figures
- **Margins**: Generous spacing, avoid clutter

### Annotations Style
- Use **(a)**, **(b)**, **(c)** for panel labels
- Simple language (avoid jargon)
- Spanish text (target audience)
- Include metrics from `GROUND_TRUTH.json`

## 🔧 Utility Functions

### Common Utilities (`utils.py`)

```python
# Sample loading
samples = load_representative_samples(
    predictions_path,
    n_per_class=5,
    criteria='diverse',  # or 'typical', 'difficult'
    seed=42
)

# Image drawing
img_with_landmarks = draw_landmarks_on_image(
    image, landmarks, radius=4, show_connections=True
)

# Comparison grids
fig, axes = create_comparison_grid(
    images, titles, n_cols=4, figsize=(16, 12)
)

# Save figures
save_figure(fig, output_path, dpi=300)
```

### Diagram Utilities (`diagram_utils.py`)

```python
# Architecture diagrams
builder = ArchitectureDiagramBuilder(figsize=(16, 10))
builder.add_layer_box(x, y, width, height, text, color, feature_map)
builder.add_arrow(x1, y1, x2, y2, label)
builder.save(output_path)

# Operation diagrams
create_operation_diagram('conv', output_path, example_input)
create_operation_diagram('pooling', output_path)
create_operation_diagram('relu', output_path)

# Flow diagrams
steps = [
    {'label': 'Step 1', 'description': '...', 'color': 'lightblue'},
    {'label': 'Step 2', 'description': '...', 'color': 'lightgreen'},
]
create_flow_diagram(steps, output_path)
```

## 📖 Example Usage

### Creating a Custom Visualization

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.glass_box_visualizations.utils import (
    load_representative_samples, load_image, save_figure
)
import matplotlib.pyplot as plt

def my_custom_visualization(predictions_path, output_path):
    # Load samples
    samples = load_representative_samples(
        predictions_path, n_per_class=3, criteria='diverse'
    )

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, sample in enumerate(samples[:3]):
        image = load_image(sample['image_path'])
        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(sample['category'])
        axes[idx].axis('off')

    plt.suptitle('My Custom Visualization', fontsize=14, weight='bold')
    save_figure(fig, output_path)

if __name__ == '__main__':
    my_custom_visualization(
        'outputs/landmark_predictions/session_warping/predictions.npz',
        'outputs/glass_box/my_viz.png'
    )
```

## 🧪 Testing

### Validate Metrics

All generated figures should use metrics from `GROUND_TRUTH.json`:

```python
from scripts.glass_box_visualizations.utils import load_ground_truth

gt = load_ground_truth()
landmark_error = gt['landmark_detection']['ensemble_best']['mean_error_px']
# Use: 3.61 px (validated)

classifier_accuracy = gt['classification']['warped_96']['test_accuracy']
# Use: 99.10% (validated)
```

### Check Output

```bash
# Verify all figures generated
ls -lh outputs/glass_box/block_*/

# Check figure quality (should be ~300 DPI, reasonably sized)
file outputs/glass_box/block_a/A1_complete_flow.png
```

## 📝 Next Steps

1. **Implement Block C** (`block_c_warping.py`)
   - Focus on C1 (GPA alignment) and C5 (gallery) first
   - C2-C4 are more technical, lower priority

2. **Implement Block D** (`block_d_classifier.py`)
   - D2 (Grad-CAM) is highest impact
   - Reuse existing `src_v2/visualization/gradcam.py`

3. **Implement Block E** (`block_e_comparison.py`)
   - E3 (artefact elimination) is most compelling
   - E1 and E2 can be simplified

4. **Create Orchestrator** (`generate_all.py`)
   - Run all blocks in sequence
   - Handle missing dependencies gracefully
   - Generate summary report

5. **Finalize Documentation** (`docs/GLASS_BOX_PIPELINE.md`)
   - Embed all generated figures
   - Add detailed captions
   - Review for clarity with non-technical reader

## 📚 References

- Main documentation: `docs/GLASS_BOX_PIPELINE.md`
- Project instructions: `CLAUDE.md`
- Validated metrics: `GROUND_TRUTH.json`
- Existing visualizations: `src_v2/visualization/`

## 🤝 Contributing

When adding new visualizations:
1. Follow the naming convention: `generate_XN_description()`
2. Use validated metrics from `GROUND_TRUTH.json`
3. Maintain consistent styling (colors, fonts, DPI)
4. Add docstrings with clear Args/Returns
5. Update this README with usage examples

---

**Status**: 🚧 In Progress (60% complete)
**Last Updated**: 2026-01-21
**Target Completion**: Sprint 4 (see plan in `docs/GLASS_BOX_PIPELINE.md`)
