# NeuroFeatureLab

## Overview
Short explanation of the project.

## Motivation
Inspired by MR-based imaging marker research for post-traumatic epilepsy prediction.

## What this project demonstrates
- ROI-level feature table construction
- Functional connectivity computation
- ALFF-style feature extraction
- ML model comparison
- Cross-validation
- Inference on new subject-level feature vectors
- Feature importance analysis

## Data
This project uses simulated imaging-derived subject-level features. No real patient data is used.

The simulated dataset contains three groups of features:

1. Lesion-style ROI features  
2. ALFF-style low-frequency signal features  
3. Functional connectivity-style features  

Each row represents one simulated subject. The label column indicates whether the subject has a non-PTE-like feature pattern or a PTE-like feature pattern.

This dataset is used only to demonstrate research-methods implementation, model training, evaluation, inference, and interpretability workflows.

## Important note
This project uses simulated imaging-derived features and does not use real patient data.

## Project structure
Show folder tree.

## Methods implemented
Explain lesion-style features, connectivity, ALFF, ML models.

## Phase 3: Feature Extraction Demo

This project includes a feature extraction demo using simulated fMRI-like ROI time-series data.

Implemented methods:

- ROI-to-ROI functional connectivity using Pearson correlation
- Upper-triangle flattening of connectivity matrices into model-ready features
- ALFF-style low-frequency amplitude feature extraction
- Combined subject-level feature vector construction

The implementation is located in:

- `src/connectivity.py`
- `src/alff.py`
- `notebooks/02_feature_extraction_demo.ipynb`

## Results
Show model comparison table.

## Example inference
Show command and sample output.

## How to run
Give setup commands.

## Future extensions
- Adapt to real neuroimaging data
- Add Nilearn/OpenNeuro support
- Add BIDS-compatible data loader
- Add nested cross-validation
- Add SHAP/permutation interpretation