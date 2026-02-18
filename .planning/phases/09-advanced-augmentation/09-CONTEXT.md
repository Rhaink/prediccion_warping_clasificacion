# Phase 9: Advanced Augmentation - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Integrate medical-specific augmentations (spatial transforms, MixUp/CutMix) into the classification training pipeline for warped chest X-rays. Test each augmentation via ablation study with visual validation gating. Architecture stays fixed (ResNet-18); only augmentation strategies change.

</domain>

<decisions>
## Implementation Decisions

### Transform intensity & selection
- Parameter ranges for ElasticTransform/GridDistortion: **Claude's discretion** based on medical imaging literature
- Whether to include pixel-level augmentations (brightness, contrast, noise) beyond spatial: **Claude's discretion**
- Integration with existing augmentations (stack vs replace): **Claude's discretion**
- Per-class augmentation probability (uniform vs more for Viral Pneumonia): **Claude's discretion**

### MixUp/CutMix policy
- Cross-class mixing (COVID+Normal blending): **Claude's discretion** based on medical MixUp literature
- Whether to include CutMix at all (rectangular patches may break anatomical context): **Claude's discretion**
- Scheduling (throughout training vs after warmup): **Claude's discretion** based on training dynamics
- Integration with curriculum learning: test augmentations **both independently AND combined with curriculum** (user decision)

### Ablation design
- GPU budget: **no hard limit** — run all meaningful combinations
- Baselines: report against **both** cleaned baseline (F1=0.9844) and curriculum model (F1=0.9932)
- Negative results: **brief note only** — save detailed analysis for winners
- Comparison output: **automated script with tables and plots**, same pattern as Phase 8 (thesis-ready)

### Visual validation
- Approach: **both** visual grids AND automated similarity metrics (SSIM or similar)
- Sample size per augmentation: **Claude's discretion**
- Timing: **gate before training** — generate augmentation previews, user reviews and approves before any training begins
- Rejection policy: if an augmentation looks too aggressive in preview, **drop it entirely** (don't iterate on parameters)

### Claude's Discretion
- Specific augmentation parameter values (alpha, sigma, grid dimensions)
- Which pixel-level augmentations to include (if any)
- MixUp alpha values and cross-class mixing policy
- Whether CutMix is appropriate for chest X-rays
- Augmentation scheduling relative to curriculum learning stages
- Augmentation probability per class
- Number of example images per class for visual grids
- Similarity metric thresholds

</decisions>

<specifics>
## Specific Ideas

- Follow Phase 8 comparison script pattern for ablation summary (08-03 style)
- Visual validation is a hard gate: user must approve augmentation previews before any training begins
- Test each augmentation independently AND combined with curriculum learning — complete picture for thesis
- Both baselines reported to show total effect (vs cleaned) and marginal effect (vs curriculum)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-advanced-augmentation*
*Context gathered: 2026-02-18*
