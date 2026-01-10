# Fisher Experiment (Geometric Validation)

## Summary
Validate the geometric normalization without deep learning by checking if
warped images become more linearly separable.

## Method
- 5-fold cross validation
- PCA + Fisher ratio
- k-NN classifier

## Key Results
1) Curated dataset (957 images):
   - RAW 73.96% vs WARPED 78.12% (+4.16%)
2) Large dataset (15k images with CLAHE):
   - WARPED 83.25% +/- 0.51% (k=50)
3) Warping increases explained variance by ~10%

## References
- GROUND_TRUTH.json (pfs, preprocessing)
- README.md (Geometric Validation section)
