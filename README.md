# Predicting Cancer from Breast Cancer Single-Cell RNA Sequencing Data

Team Members: Shriya Goel, Gretchen Lam, Bryan Richins

Project goals:
- Develop machine learning classifier for cancerous cells using breast cancer single-cell RNA sequencing (scRNAseq) data
- Identify cancer biomarkers important for cancer prediction

## Model requirements
```pip install -r requirements.txt```

## Dataset overview


## Code structure
- main_functions.py: Contains helper functions used in notebooks (ex. loading data, training models)
- Notebook 0 (): Loading datasets, dataset exploration and statistics
- Notebook 1 (Baseline_CV.ipynb): Baseline model comparison on training set
- Notebook 2 (Baseline_SHAP.ipynb): Feature importance on XGBoost baseline model with all features
- Notebook 2 (differential_gene_expression.ipynb): Differential gene expression
- Notebook 3 (): Highly variable genes - metrics comparison, feature importance
- Notebook 4 (HVG_and_DGE_GridsearchCV.ipynb): Model tuning with feature selection
- Notebook 5: Testing final model with hold-out datasets
- Notebook 6: Comparison with ikarus