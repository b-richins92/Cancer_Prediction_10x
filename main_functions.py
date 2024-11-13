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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score,balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
import shap

### Functions

# Convenience method for computing the size of objects
def print_size_in_MB(x):
  print(f'Size: {x.__sizeof__() / 1e6:.5} MB')

# Set up AnnData object for training dataset
def create_adata_train(raw_counts_path, norm_counts_path, orig_labels_path):
  """
    Creates AnnData object from training dataset using original labels
    Inputs:
      - raw_counts_path: String with path to folder with mtx files containing raw counts
      - norm_counts_path: String with path to h5 file from TISCH containing normalized counts
      - orig_labels_path: String with path to csv.gz file containing cell type annotations from original paper
    Output:
      - AnnData object with raw and normalized matrices with cancer cell annotation from original paper
  """
  # Read in normalized count matrix as AnnData object
  adata_norm = sc.read_10x_h5(norm_counts_path, gex_only = False)

  # Load in original labels/metadata
  orig_meta = pd.read_csv(orig_labels_path)
  orig_meta = orig_meta.set_index('Cell')
  orig_meta['orig_cancer_label'] = np.where(orig_meta['CellType'] == 'Cancer', 1, 0)

  # Merge original metadata with normalized AnnData from TISCH
  adata_norm.obs = pd.merge(adata_norm.obs, orig_meta, left_index = True, right_index = True, how = 'inner')
 
  # Load in raw counts into AnnData object - subset using TISCH cells and genes
  raw_counts_ann = sc.read_10x_mtx(raw_counts_path, gex_only = False)
  raw_counts_ann.obs['in_tisch'] = raw_counts_ann.obs.index.isin(adata_norm.obs_names)
  raw_counts_ann.var['in_tisch'] = raw_counts_ann.var.index.isin(adata_norm.var_names)
  raw_subset = raw_counts_ann[raw_counts_ann.obs['in_tisch'], raw_counts_ann.var['in_tisch']].copy()

  # Add in labels to raw_subset
  raw_subset.obs = pd.merge(raw_subset.obs, orig_meta, left_index = True, right_index = True, how = 'inner')

  # Ensure shape of both normalized and raw matrices are the same
  adata_norm = adata_norm[raw_subset.obs_names, raw_subset.var_names].copy()

  # Print number of cells and dataset size
  print(f'Dataset has {raw_subset.n_obs} cells and {raw_subset.n_vars} features')
  print(f'Size of raw dataset: ')
  print_size_in_MB(raw_subset)
  print(f'Size of normalized dataset:')
  print_size_in_MB(adata_norm)

  # Create a single anndata object that contains both raw and normalized layers
  adata_all = adata_norm.copy()
  adata_all.layers['norm'] = adata_norm.X.copy()
  adata_all.layers['raw'] = raw_subset.X.copy()

  return adata_all


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

  num_genes = adata.n_vars

  # Use experimental module if 'pearson_residuals' (with raw counts), otherwise use standard method
  if method == 'pearson_residuals':
    hvg_df = sc.experimental.pp.highly_variable_genes(adata, flavor = 'pearson_residuals', n_top_genes = num_genes,
                                                      layer = 'raw', inplace = False)
  elif method == 'seurat_v3':
    hvg_df = sc.pp.highly_variable_genes(adata, flavor = method, n_top_genes = num_genes, layer = 'raw', inplace = False)
  elif method in ['seurat', 'cell_ranger']:
    hvg_df = sc.pp.highly_variable_genes(adata, flavor = method, n_top_genes = num_genes, layer = 'norm', inplace = False)
  else:
    raise ValueError("String must be one of four values: 'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals'")

  # Sort genes by highest variance/dispersion, depending on method
  if method in ['seurat_v3', 'pearson_residuals']:
    hvg_df = hvg_df.sort_values(by = 'highly_variable_rank')
  else:
    hvg_df = hvg_df.sort_values(by = 'dispersions_norm', ascending = False)

  return hvg_df.index

# Function to loop through training function across features and HVG vs random selection methods
# Applies cross validation split before feature selection
def train_feat_loop_cv(clf, adata, groups_label, num_feat_list, feat_method_list,
                       random_state = 0, k_fold = 5):
  """
    Run cross-validation with different numbers of features and feature selection methods
    Inputs:
      - clf: Classifier
      - adata: AnnData object containing raw and normalized count matrix and labels
      - groups_label: String indicating group to split on (should be column in adata.obs)
      - num_feat_list: List of numbers of features to use
      - feat_method_list: List of feature selection methods
        - Scanpy highly variable genes: 'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals'
        - Random selection
          - 'random_all_genes': Randomize order of all genes, then select top N genes at each number
          - 'random_per_num': Pick a new set of N random genes for each N
      - random_state: Random state to use for k-folds
      - k_fold = Number of folds to use
    Output: 
      - Concatenated dataframe containing results for all numbers of features and feature selection methods
  """

  results_df = pd.DataFrame()
  test_indices_dict = {}
  feat_order_dict = {}
  shap_results = {}

  # Set up X and y
  X = adata.copy()
  y = adata.obs['orig_cancer_label']
  groups_col = adata.obs[groups_label]

  # Generate cross-validation splits using StratifiedGroupKFold
  sgkf = StratifiedGroupKFold(n_splits=k_fold, shuffle = True, random_state = random_state)
  # Loop through each fold
  for i, (train_index, test_index) in enumerate(sgkf.split(X, y, groups_col)):
    print(f'i: {i}')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Store test indices in a dictionary by fold
    test_indices_dict[i] = test_index

    # Loop through all feature selection methods
    for curr_method in feat_method_list:
      print(f'curr_method: {curr_method}')
      if i == 0:
          shap_results[curr_method] = {}
      # Select features based on feature selection method
      if curr_method in ['seurat_v3', 'pearson_residuals', 'seurat', 'cell_ranger']:
        feature_order = get_hvgs(X_train, curr_method)
      elif curr_method == 'random_all_genes':
        rng = np.random.default_rng(random_state)
        feature_order = rng.choice(adata.var_names, size = adata.n_vars, replace=False)
      elif curr_method == 'random_per_num':
        rng = np.random.default_rng(random_state)
        feature_order = {}
        for curr_num_feat in num_feat_list:
          feature_order[curr_num_feat] = rng.choice(adata.var_names, size = curr_num_feat, replace=False)
      else:
        raise ValueError("String must be one of these values: 'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals', random_all_genes', 'random_per_num'")

      # Store feature order in dictionary, using largest number of features in num_feat_list
        # Except for 'random_per_num' - store dictionary of features
      if i == 0:
        feat_order_dict[curr_method] = {}
      if curr_method == 'random_per_num':
        feat_order_dict[curr_method][i] = feature_order
      else:
        feat_order_dict[curr_method][i] = feature_order[:max(num_feat_list)]
        
      # Loop through all numbers of features
      for curr_num_feat in num_feat_list:
        print(f'curr_num_feat: {curr_num_feat}')
        if i == 0:
          shap_results[curr_method][curr_num_feat] = {}
      
        # Extract top features depending on method
        if curr_method == 'random_per_num':
          curr_feat = feature_order[curr_num_feat]
        else:
          curr_feat = feature_order[:curr_num_feat]

        # Train model
        clf.fit(X_train[:, curr_feat].X, y_train)
        # Get predictions
        y_pred = clf.predict(X_test[:, curr_feat].X)

        # Calculate metrics and store in dictionary
        curr_results = {}

        curr_results['fold'] = [i]
        curr_results['feat_sel_type'] = [curr_method]
        curr_results['num_features'] = [curr_num_feat]

        curr_results['f1'] = [f1_score(y_test, y_pred)]
        curr_results['accuracy'] = [accuracy_score(y_test, y_pred)]
        curr_results['balanced_accuracy'] = [balanced_accuracy_score(y_test, y_pred)]
        curr_results['recall'] = [recall_score(y_test, y_pred)]
        curr_results['precision'] = [precision_score(y_test, y_pred)]
        curr_results['average_precision'] = [average_precision_score(y_test, y_pred)]
        curr_results['roc_auc'] = [roc_auc_score(y_test, y_pred)]
        curr_results['matthews_corrcoef'] = [matthews_corrcoef(y_test, y_pred)]
        
        # Calculate feature importance
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test[:, curr_feat].X)
        shap_results[curr_method][curr_num_feat][i] = shap_values

        # Convert values into dataframe
        results_df = pd.concat([results_df, pd.DataFrame.from_dict(curr_results)], ignore_index=True)
#        results_df.to_csv('results_df_20241112_shap.csv')

  return results_df, test_indices_dict, feat_order_dict, shap_results


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
def make_line_plots_metrics(results_df):
  """
    Generate line plots comparing metrics from cross-validation
    Inputs: 
      - results_df: Dataframe from train_test_model() with metrics in each column
    Output:
      - Return 1 figure with faceted subplots by metric
  """
  # Convert dataframe from wide to long
  results_df_tall = results_df.melt(id_vars=['feat_sel_type', 'num_features', 'fold'], var_name='metric', value_name='score')
  display(results_df_tall.head())

  # Save dataframe summarizing mean and stdev
  results_df_pivot = pd.pivot_table(results_df_tall,
                                    values=['score'],
                                    index = ['feat_sel_type', 'num_features'],
                                    columns = ['metric'],
                                    aggfunc=['mean', 'std'])
#  results_df_pivot.to_csv('results_df_pivot.csv')

  # Plot 1 figure with all test metrics versus number of features - Facet by metric. Color by feature type
  g1 = sns.catplot(
      data=results_df_tall,
      x='num_features', y='score', col='metric',
      hue = 'feat_sel_type', col_wrap = 4, kind='point', capsize = 0.2,
      sharex = False, alpha = 0.7
  )

  return g1

# Calculate Jaccard coefficient overlap between feature sets

# Generate feature importance SHAP plots for a given method and number of features across folds
def plot_feat_importance(adata, method, num_feat, feat_dict, shap_dict, test_folds_dict):
    """
    Consolidates SHAP values across folds and generates beeswarm SHAP plot for feature importance
    Inputs:
        - adata: AnnData object containing gene expression values
        - method: String of highly variable gene method used
        - num_feat: Number of features to select for
        - feat_dict: Dictionary containing order of features for each method and fold
        - shap_dict: Dictionary containing SHAP values for each method, number of features, and fold
        - test_folds_dict: Dictionary containing indices of test samples by fold
    Outputs:
        - Dataframe of SHAP values (saved)
        - SHAP plot (saved)
    """

    # Get X index and column names
    X_index = adata.obs_names
    feat_names = adata.var_names

    # Set up dataframe to store all values
    shap_vals_df = pd.DataFrame()
    
    # Loop through each fold
    for fold, index_list in test_folds_dict.items():
        print(f'fold: {fold}')
        print(f'len(index_list): {len(index_list)}')
        curr_cells = X_index[index_list]
        print(f'len(curr_cells): {len(curr_cells)}')

        # Get current set of features
        if method == 'random_per_num':
          curr_feat = feat_dict[method][fold][num_feat]
        else:
          curr_feat = feat_dict[method][fold][:num_feat]
        print(f'len(curr_feat): {len(curr_feat)}')
    
        # Create dataframe with cell indices, features, and SHAP values
        curr_shap = shap_dict[method][num_feat][fold]
        print(f'curr_shap.shape: {curr_shap.shape}')
        curr_fold_df = pd.DataFrame(data = curr_shap,
                                    index = curr_cells,
                                    columns = curr_feat)
        print(curr_fold_df.shape)
        display(curr_fold_df.head())
        # Concatenate dataframe to main dataframe - keep missing values as NaN?
        shap_vals_df = pd.concat([shap_vals_df, curr_fold_df])
    print(shap_vals_df.shape)
    display(shap_vals_df.head())
    display(shap_vals_df.tail())
    shap_vals_df.to_csv(f'shap_vals_df_{method}_features{num_feat}.csv')

    # Convert missing values to 0
#    shap_vals_df_no_na = shap_vals_df.fillna(0)

    # Calculate variance and sort shap_vals_df by variance
    shap_var_sort = shap_vals_df.var(axis = 0).sort_values(ascending = False)
    shap_vals_df = shap_vals_df[shap_var_sort.index]

    # Subset anndata to same cells and features in SHAP value frame
    adata_sub_vals = adata[shap_vals_df.index, shap_vals_df.columns].to_df()

    # Create beeswarm SHAP plot sorted by highest variance
    fig = shap.summary_plot(shap_vals_df.values,adata_sub_vals, max_display = 10, sort = False)
    fig.savefig(f'beeswarm_{method}_features{num_feat}.png', bbox_inches='tight')
    
    return shap_vals_df, fig
