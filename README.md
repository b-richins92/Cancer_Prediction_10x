# Predicting Cancer from Breast Cancer Single-Cell RNA Sequencing Data
SIADS 699 Fall 2024
Team Members: Shriya Goel, Gretchen Lam, Bryan Richins

Project goals:
- Develop machine learning classifier for cancerous cells using breast cancer single-cell RNA sequencing (scRNAseq) data
- Identify cancer biomarkers important for cancer prediction

## Model requirements
Install using the following command in command prompt:
```pip install -r requirements.txt```

## Data Access Statement
The count matrices were obtained from the public, free to use [TISCH2 Database](http://tisch.comp-genomics.org/gallery/?cancer=BRCA&celltype=Malignant&species=Human), which provides access to tumor-related scRNAseq datasets.
We filtered for datasets for breast cancer containing cells labeled as "malignant" by TISCH and limited to human specimens.

Dataset names:
  * Training - BRCA_EMTAB8107
  * Test Set 1 (also referred to as "validation") - BRCA_GSE148673
  * Test Set 2 (also referred to as "test") - BRCA_GSE150660

While the datasets include the target variable, we felt that the corresponding papers of the datasets provided links to labels that were much more accurate.
...(include how to download labels)

All datasets are public and all licenses are respected.


## Dataset overview


## Code structure
- main_functions.py: Contains helper functions used in notebooks (ex. loading data, training models)
- Notebook 0 (0_create_datasets_stats.ipynb): Loading datasets, dataset exploration and statistics
- Notebook 1 (1_Baseline_CV.ipynb): Baseline model comparison on training set
- Notebook 2 (2_Baseline_SHAP.ipynb): Feature importance on XGBoost baseline model with all features
- Notebook 3 (3_differential_gene_expression.ipynb): Differential gene expression
- Notebook 4 (4_highly_variable_genes.ipynb): Highly variable genes - metrics comparison, feature importance
- Notebook 5 (5_HVG_and_DGE_GridsearchCV.ipynb): Model tuning with feature selection and testing with final model on hold-out datasets
- Notebook 6 (6_ikarus.ipynb): Comparison with ikarus
