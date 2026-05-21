# NeuroFeatureLab

## Overview
NeuroFeatureLab is a research-methods demonstration repository showing how to construct, train, evaluate, and interpret machine learning models for predicting post-traumatic epilepsy (PTE) patterns. The project processes simulated multi-modal neuroimaging markers (structural lesion burden, functional connectivity, and Amplitude of Low-Frequency Fluctuations) to showcase standard neuroimaging ML workflows.

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
```
NeuroFeatureLab/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/                     # Data directory (simulated features & subjects)
│   ├── simulated_pte_features.csv
│   ├── sample_subject_high_risk.json
│   └── sample_subject_low_risk.json
├── models/                   # Saved model checkpoints
│   ├── best_model.pkl        # Best pipeline (RBF Kernel SVM)
│   └── scaler.pkl            # StandardScaler state
├── notebooks/                # Jupyter Notebooks
│   ├── 01_generate_simulated_features.ipynb
│   ├── 02_feature_extraction_demo.ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   └── 04_inference_and_explainability.ipynb
├── outputs/                  # Plot outputs and comparison results
│   ├── model_comparison.csv
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── connectivity_heatmap.png
│   └── feature_importance.png
└── src/                      # Source code modules
    ├── simulate_features.py  # Synthetic data generation
    ├── connectivity.py       # Functional connectivity computation
    ├── alff.py               # ALFF feature extraction
    ├── train_models.py       # Model training & validation
    ├── explainability.py     # Permutation-based feature importance
    └── inference.py          # Subject-level inference
```

## Methods implemented
- **Lesion-style features**: Simulated volume values representing regional lesion burden across temporal, occipital, cerebellum, and parietal regions.
- **Functional Connectivity**: Pearson correlation computed from simulated BOLD fMRI signals. The upper triangle of the correlation matrix is flattened to extract unique undirected links.
- **ALFF (Amplitude of Low-Frequency Fluctuations)**: Spectral amplitude values calculated from regional time-series in the typical slow fluctuation band (0.01 - 0.10 Hz) using Fast Fourier Transform (FFT).
- **ML Models**: Comparison of Logistic Regression, Linear SVM, RBF SVM, Random Forest, and MLP Classifiers evaluated using 5-fold cross-validation.

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
```

Example Output:
```
Inference result
----------------
Prediction: PTE-like feature pattern
PTE-like pattern probability: 1.000

Note: This is a simulated research-methods demo, not a clinical tool.
```

## Results
After training the models on slightly overlapping simulated distributions, we get the following performance metrics:

| Model | Test AUC | Accuracy | Precision | Recall | F1-Score | CV AUC Mean | CV AUC Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF Kernel SVM** | **0.9956** | **0.9667** | **0.9667** | **0.9667** | **0.9667** | **0.9823** | **0.0180** |
| MLP Classifier | 0.9878 | 0.9500 | 0.9508 | 0.9500 | 0.9508 | 0.9705 | 0.0138 |
| Linear SVM | 0.9856 | 0.9667 | 0.9667 | 0.9667 | 0.9667 | 0.9701 | 0.0112 |
| Logistic Regression | 0.9800 | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9712 | 0.0119 |
| Random Forest | 0.9733 | 0.9167 | 0.9180 | 0.9167 | 0.9180 | 0.9748 | 0.0079 |

## Example inference
Run inference on a single subject JSON vector using the best-performing model checkpoint:
```bash
# High-risk subject inference
python src/inference.py --input data/sample_subject_high_risk.json

# Low-risk subject inference
python src/inference.py --input data/sample_subject_low_risk.json
```

Sample output for high-risk subject:
```
Inference result
----------------
Prediction: PTE-like feature pattern
PTE-like pattern probability: 1.000

Note: This is a simulated research-methods demo, not a clinical tool.
```

## How to run
Set up the virtual environment, install dependencies, and execute the full pipeline:
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate simulated dataset
python src/simulate_features.py

# 4. Compare models and save best model
python src/train_models.py

# 5. Generate permutation feature importances
python src/explainability.py
```

## Future extensions
- Adapt to real neuroimaging data
- Add Nilearn/OpenNeuro support
- Add BIDS-compatible data loader
- Add nested cross-validation
- Add SHAP/permutation interpretation