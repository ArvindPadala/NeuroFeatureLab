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

## Phase 4: Model Training and Evaluation

This phase trains and compares multiple supervised machine learning models on the simulated imaging-derived feature dataset.

Models included:

- Logistic Regression
- Linear SVM
- RBF Kernel SVM
- Random Forest
- MLP Classifier

Evaluation metrics:

- AUC
- Accuracy
- Precision
- Recall
- F1-score
- 5-fold cross-validation AUC mean and standard deviation

Outputs:

- `outputs/model_comparison.csv`
- `outputs/roc_curve.png`
- `outputs/confusion_matrix.png`
- `models/best_model.pkl`

This phase demonstrates a reproducible research-style ML evaluation workflow. The results are based on simulated data and are not clinical predictions.

## Phase 5: Inference and Explainability

This phase demonstrates how to run inference on a new simulated subject-level feature vector and generate model-agnostic feature importance.

Implemented files:

- `src/inference.py`
- `src/explainability.py`
- `data/sample_subject_low_risk.json`
- `data/sample_subject_high_risk.json`
- `notebooks/04_inference_and_explainability.ipynb`

Example inference command:

```bash
python src/inference.py --input data/sample_subject_high_risk.json

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