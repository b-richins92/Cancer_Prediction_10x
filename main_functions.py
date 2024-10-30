import os
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score,\
                            balanced_accuracy_score, matthews_corrcoef, roc_auc_score

### Functions

# Convenience method for computing the size of objects
def print_size_in_MB(x):
  print(f"{x.__sizeof__() / 1e6:.5} MB")

# Set up AnnData object with count matrix and label
def create_adata(h5_path, label_path):
  """
    Inputs:
      - h5_path: String with path to h5 file containing normalized counts
      - label_path: String with path to tsv file containing cell type annotations
    Output: AnnData object loaded with count matrix and cell type labels
  """
  # Read in count matrix as AnnData object
  adata = sc.read_10x_h5(h5_path, gex_only = False)

  # Read in annotation file and add labels to AnnData object
  annot_df = pd.read_csv(label_path, sep='\t', index_col = 'Cell')
  annot_df['cancer_label'] = np.where(annot_df['Celltype (malignancy)'] == 'Malignant cells', 1, 0)
#  annot_df['cancer_label'] = annot_df['cancer_label'].astype('category')
  adata.obs['cancer_label'] = annot_df['cancer_label']

  # Also add metadata to AnnData object
  adata.obs['cell_type_malignancy'] = annot_df['Celltype (malignancy)']
  adata.obs['cell_type_major'] = annot_df['Celltype (major-lineage)']
  adata.obs['cell_type_minor'] = annot_df['Celltype (minor-lineage)']
  adata.obs['Patient'] = annot_df['Patient']
  adata.obs['Sample'] = annot_df['Sample']
#  adata.obs['Tissue'] = annot_df['Tissue']

  # Add patient ID column using delimiter before cell ID (string of letters or numbers)
  adata.obs['patient_id'] = adata.obs.index.str.split(r'@|_').str[0]

  # Reorder genes by alphabetical order
  adata.var = adata.var.sort_index()

  # Print number of cells and dataset size
  print(f'Number of cells: {adata.n_obs}')
  print(f'Number of features: {adata.n_vars}')
  print_size_in_MB(adata)

  return adata

# Get differentially expressed genes between cancer and normal cells in datasets
def get_diff_exp_genes(adata_obj, corr_method = 'bonferroni', pval_cutoff = 0.05, log2fc_min = 0.25):
  """
  Inputs:
    - AnnData object containing normalized counts and labels
    - corr_method - correction method to use with Wilcoxon rank sum test
    - pval_cutoff - p-value cutoff to use with Wilcoxon rank sum test (0.05 by default)
    - log2fc_min - minimum fold change to use with Wilcoxon rank sum test (0.25 by default)
  Output: Dataframe with differentially expressed genes in each group after filtering
  """

  adata_de = sc.tl.rank_genes_groups(adata_obj, groupby='cancer_label', method='wilcoxon',
                                         tie_correct = True, corr_method = corr_method,
                                         pts = True, copy = True)
  adata_de_df = sc.get.rank_genes_groups_df(adata_de, group = None,
                                                pval_cutoff = pval_cutoff, log2fc_min = log2fc_min)
  adata_de_df_filt = adata_de_df[adata_de_df['pct_nz_group'] > 0.1]
  print(f'Original shape: {adata_de_df.shape} vs after filtering < 10% expression: {adata_de_df_filt.shape}')
  display(adata_de_df_filt.head())
  return adata_de_df_filt

# Get highly variable genes from dataset


# Training function - train model with set list of features, and score test dataset with same features
def train_test_model(clf, train_df, train_labels, test_df, test_labels, features):
  """
    Inputs:
      - Classifier
      - Training dataset
      - Training labels
      - Test dataset
      - Test labels
      - Features
    Output: Trained model. Print metrics
  """

  # Train model
  clf.fit(train_df[features], train_labels)

  # Test model
  y_pred = clf.predict(test_df[features])

  recall = recall_score(test_labels, y_pred)
  precision = precision_score(test_labels, y_pred)
  accuracy = accuracy_score(test_labels, y_pred)
  f1 = f1_score(test_labels, y_pred)
  conf_matrix = confusion_matrix(test_labels, y_pred)

  print(f'# cells in training: {len(train_df)}, # cells in test: {len(test_df)}')
  print(conf_matrix)
  print(f'recall: {recall}, precision: {precision}, accuracy: {accuracy}, f1: {f1}')

  return clf