# Predicting Cancer from Breast Cancer Single-Cell RNA Sequencing Data
SIADS 699 Fall 2024

Team Members: Shriya Goel, Gretchen Lam, Bryan Richins

Project goals:
- Develop machine learning classifier for cancerous cells using breast cancer single-cell RNA sequencing (scRNAseq) data
- Find best performing basline model, then use that model to test for optimal feature selection method and number of features for future hyperparameter tuning.
- Compare our results with ikarus, a publish pan-cancer classifier
- Identify cancer biomarkers important for cancer prediction

## Model requirements
Install using the following command in command prompt:
```pip install -r requirements.txt```

## Data Access Statement
The count matrices were obtained from the public, free to use [TISCH2 Database](http://tisch.comp-genomics.org/gallery/?cancer=BRCA&celltype=Malignant&species=Human), which provides access to tumor-related scRNAseq datasets.
We filtered for datasets for breast cancer containing cells labeled as "malignant" by TISCH and limited to human specimens.

Dataset names:
  * Training - BRCA_EMTAB8107 (filename: 'train.h5ad')
  * Test Set 1 - BRCA_GSE148673 (filename: 'test1.h5ad')
  * Test Set 2 - BRCA_GSE150660 (filename: 'test2.h5ad')

We used labels from the original datasets rather than those provided by TISCH, since the TISCH labels may include automated cell type labels and sometimes conflict with the original labels.

The links to the original labels are as follows:
* [Training](https://lambrechtslab.sites.vib.be/en/pan-cancer-blueprint-tumour-microenvironment-0) (requires creating free account to access data, stored as csv.gz)
* [Test Set 1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148673) (stored in txt.gz by patient as columns with counts)
* [Test Set 2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150660) (stored in h5ad by patient with counts)

All datasets are public and all licenses are respected.

## Code structure and How to Run:
Unfortunately, due to large file size, the datasets could not be included in the depository. Instead, please refer to links above to download the matrices and labels, and then run Notebook 0 to create the files. Then simply run the rest of the files in order.

- main_functions.py: Contains helper functions used in notebooks (ex. loading data, training models)
- Notebook 0 (0_create_datasets_stats.ipynb): Loading datasets, dataset exploration and statistics
- Notebook 1 (1_Baseline_CV.ipynb): Baseline model comparison on training set
- Notebook 2 (2_Baseline_SHAP.ipynb): Feature importance on XGBoost baseline model with all features
- Notebook 3 (3_highly_variable_genes.ipynb): Highly variable genes - metrics comparison, feature importance
- Notebook 4 (4_HVG_and_DGE_GridsearchCV.ipynb): Model tuning with feature selection and testing with final model on hold-out datasets
- Notebook 5 (5_differential_gene_expression.ipynb): Differential gene expression (Not used in main report although visualizations used in Appendix)
- Notebook 6 (6_ikarus.ipynb): Comparison with ikarus
- Notebook 7 (7_ikarus_xgboost_swapped_features.ipynb): ikarus and XGBoost models with swapped features
- Notebook 8 (8_compare_gene_sig.ipynb): Compare features from notebook 4 with known gene signatures from the literature
