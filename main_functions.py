import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from sklearn import set_config
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score,\
                            balanced_accuracy_score, matthews_corrcoef, roc_auc_score, RocCurveDisplay, auc, average_precision_score, make_scorer
### Functions

# Convenience method for computing the size of objects
def print_size_in_MB(x):
  print(f"Size: {x.__sizeof__() / 1e6:.5} MB")

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

# Set up AnnData object for training dataset
def create_adata_train(raw_counts_path, norm_counts_path, orig_labels_path):
  """
    Creates AnnData object from training dataset using original labels
    Inputs:
      - raw_counts_path: String with path to folder with mtx files containing raw counts
      - norm_counts_path: String with path to h5 file from TISCH containing normalized counts
      - orig_labels_path: String with path to csv.gz file containing cell type annotations from original paper
    Output:
      - AnnData object with raw matrix and cancer cell annotation from original paper
      - AnnData object with normalized matrix and cancer cell annotation from original paper
  """
  # Read in normalized count matrix as AnnData object
  adata_norm = sc.read_10x_h5(norm_counts_path, gex_only = False)

  # Load in original labels/metadata
  orig_meta = pd.read_csv(orig_labels_path)
  orig_meta = orig_meta.set_index('Cell')
  orig_meta['orig_cancer_label'] = np.where(orig_meta['CellType'] == 'Cancer', 1, 0)

  # Merge original metadata with normalized AnnData from TISCH
  adata_norm.obs = pd.merge(adata_norm.obs, orig_meta, left_index = True, right_index = True, how = 'inner')
  display(adata_norm.obs.head())
 
  # Reorder genes by alphabetical order
  adata_norm.var = adata_norm.var.sort_index()

  # Load in raw counts into AnnData object - subset using TISCH cells and genes
  raw_counts_ann = sc.read_10x_mtx(raw_counts_path, gex_only = False)
  raw_counts_ann.obs['in_tisch'] = raw_counts_ann.obs.index.isin(adata_norm.obs_names)
  raw_counts_ann.var['in_tisch'] = raw_counts_ann.var.index.isin(adata_norm.var_names)
  raw_subset = raw_counts_ann[raw_counts_ann.obs['in_tisch'], raw_counts_ann.var['in_tisch']].copy()

  # Ensure shape of both normalized and raw matrices are the same
  adata_norm = adata_norm[raw_subset.obs_names, raw_subset.var_names].copy()

  # Print number of cells and dataset size
  print(f'Both datasets have {adata_norm.n_obs} cells and {adata_norm.n_vars} features')
  print(f'Size of raw dataset: {print_size_in_MB(raw_subset)}')
  print(f'Size of normalized dataset: {print_size_in_MB(adata_norm)}')

  return (raw_subset, adata_norm)

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
def get_hvgs(adata, method):
  """
    Purpose: Calculate list of highly variable genes using standard built-in scanpy methods
    Inputs:
      - adata: AnnData object containing raw and normalized counts
      - method: String indicating method to use for calculating HVGs ('seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals')
    Output:
      - Index of genes sorted by high to low variance/dispersion
  """

  # Use experimental module if 'pearson_residuals' (with raw counts), otherwise use standard method
  if method == 'pearson_residuals':
    hvg_df = sc.experimental.pp.highly_variable_genes(adata, flavor = 'pearson_residuals', n_top_genes = adata.n_vars, inplace = False)
  elif method in ['seurat_v3', 'seurat', 'cell_ranger']:
    hvg_df = sc.pp.highly_variable_genes(adata, flavor = method, n_top_genes = adata.n_vars, inplace = False)
  else:
    raise ValueError("String must be one of four values: 'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals'")

  # Sort genes by highest variance/dispersion, depending on method
  if method in ['seurat_v3', 'pearson_residuals']:
    hvg_df = hvg_df.sort_values(by = 'highly_variable_rank')
  else:
    hvg_df = hvg_df.sort_values(by = 'dispersions_norm', ascending = False)

  print(f'hvg_df.shape: {hvg_df.shape}')
  display(hvg_df.head())
  return hvg_df.index

# Pearson residual preprocessing

# Training function using cross-validation
def train_cv(clf, X, y, groups, features, metrics_dict, random_state = 0, k_fold = 5):
  """
    Inputs: 
      - clf: Classifier
      - X: Dataset
      - y: Labels
      - groups: String indicating group to split on (should be column in adata.obs)
      - features: List of features
      - metrics_dict: Dictionary of metrics to use for scoring
      - random_state: Random state to use for k-folds
      - k_fold = Number of folds to use
    Output:
      - Dataframe with metrics per fold
  """

  # 5-fold cross-validation (stratified, divided by patients) for SVM
  sgkf = StratifiedGroupKFold(n_splits=k_fold, shuffle = True, random_state = random_state)
  curr_results = cross_validate(clf, X[features], y, groups = groups, scoring = metrics_dict,
                 cv = sgkf.get_n_splits(), return_train_score = True)

  return pd.DataFrame.from_dict(curr_results)

# Function to loop through training function across features and HVG vs random selection methods
def train_feat_loop(clf, adata_raw, adata_norm, groups, num_feat_list, feat_method_list,
                    metrics_dict, random_state = 0, k_fold = 5):
  """
    Run cross-validation with different numbers of features and feature selection methods
    Inputs:
      - clf: Classifier
      - adata_raw: AnnData object containing raw count matrix and labels
      - adata_norm: AnnData object containing normalized count matrix and labels
      - groups: String indicating group to split on (should be column in adata.obs)
      - num_feat_list: List of numbers of features to use
      - feat_method_list: List of feature selection methods
        - Scanpy highly variable genes: 'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals'
        - Random selection
          - 'random_all_genes': Randomize order of all genes, then select top N genes at each number
          - 'random_per_num': Pick a new set of N random genes for each N
      - metrics_dict: Dictionary of metrics to use for scoring
      - random_state: Random state to use for k-folds
      - k_fold = Number of folds to use
    Output: 
      - Concatenated dataframe containing results for all numbers of features and feature selection methods
  """

  results_df = pd.DataFrame()

  # Loop through all feature selection methods
  for curr_method in feat_method_list:
    print(f'curr_method: {curr_method}')
    # Select features based on feature selection method
    if curr_method in ['seurat_v3', 'pearson_residuals']:
#      adata = adata_raw
      feature_order = get_hvgs(adata_raw, curr_method)
    elif curr_method in ['seurat', 'cell_ranger']:
#      adata = adata_norm
      feature_order = get_hvgs(adata_norm, curr_method)
    elif curr_method == 'random_all_genes':
      rng = np.random.default_rng(random_state)
      feature_order = rng.choice(adata_norm.var_names, size = adata_norm.n_vars, replace=False)
    elif curr_method == 'random_per_num':
      rng = np.random.default_rng(random_state)
      for curr_num_feat in num_feat_list:
        feature_order[curr_num_feat] = rng.choice(adata_norm.var_names, size = curr_num_feat, replace=False)
    else:
      raise ValueError("String must be one of these values: 'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals',\
                       'random_all_genes', 'random_per_num'")

    # Loop through all numbers of features
    for curr_num_feat in num_feat_list:
      print(f'curr_num_feat: {curr_num_feat}')
    
      # Extract top features depending on method
      if curr_method == 'random_per_num':
        curr_feat = feature_order[curr_num_feat]
      else:
        curr_feat = feature_order[:curr_num_feat]

      # Get cross-validation results and concatenate to dataframe
      curr_results = train_cv(clf, adata_norm.to_df(), adata_norm.obs['orig_cancer_label'], adata_norm.obs[groups],
                              curr_feat, metrics_dict, random_state = random_state, k_fold = k_fold)
      curr_results['feat_sel_type'] = curr_method
      curr_results['num_features'] = curr_num_feat
      results_df = pd.concat([results_df, pd.DataFrame.from_dict(curr_results)], ignore_index=True)

  return results_df


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

# Print line plots of metrics


# Calculate Jaccard coefficient overlap between feature sets