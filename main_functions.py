import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import shap

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score

### Functions

# Convenience method for computing the size of objects
def print_size_in_MB(x):
    print(f'Size: {x.__sizeof__() / 1e6:.5} MB')

# Set up AnnData object - used for training and validation datasets
def create_adata(raw_counts_path, norm_counts_path, orig_labels_path):
    """
    Purpose: Creates AnnData object from training dataset using original labels
    Inputs:
      - raw_counts_path: String with path to csv file or folder with mtx files containing raw counts
      - norm_counts_path: String with path to h5 file from TISCH containing normalized counts
      - orig_labels_path: String with path to csv.gz file containing cell type annotations from original paper
    Output:
      - AnnData object with raw and normalized matrices with cancer cell annotation from original paper
    Notes about input format:
      - Assumes file for orig_labels_path contains the following columns: 
        - Cell IDs under 'Cell'
        - Cell types under 'CellType', with cancer labels using the string 'Cancer'
    """
    # Read in normalized count matrix as AnnData object
    adata_norm = sc.read_10x_h5(norm_counts_path, gex_only = False)
    
    # Load in original labels/metadata
    orig_meta = pd.read_csv(orig_labels_path)
    orig_meta = orig_meta.set_index('Cell')
    orig_meta['orig_cancer_label'] = np.where(orig_meta['CellType'] == 'Cancer', 1, 0)
    
    # Merge original metadata with normalized AnnData from TISCH
    adata_norm.obs = pd.merge(adata_norm.obs, orig_meta, left_index = True, right_index = True, how = 'inner')
    
    # Load in raw counts into AnnData object
    if (raw_counts_path.endswith('.csv.gz') | raw_counts_path.endswith('.csv')):
        raw_counts_df = pd.read_csv(raw_counts_path, index_col = 0)
        raw_counts_ann = sc.AnnData(raw_counts_df)
    else:
        raw_counts_ann = sc.read_10x_mtx(raw_counts_path, gex_only = False)

    # Subset raw counts using cells and genes from TISCH dataset
    raw_counts_ann.obs['in_tisch'] = raw_counts_ann.obs.index.isin(adata_norm.obs_names)
    raw_counts_ann.var['in_tisch'] = raw_counts_ann.var.index.isin(adata_norm.var_names)
    raw_subset = raw_counts_ann[raw_counts_ann.obs['in_tisch'], raw_counts_ann.var['in_tisch']].copy()
    raw_subset.obs = raw_subset.obs.drop(columns = ['in_tisch'])
    raw_subset.var = raw_subset.var.drop(columns = ['in_tisch'])
    
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

# Get differentially expressed genes between cancer and normal cells in dataset
def get_diff_exp_genes(adata_obj, corr_method = 'bonferroni', pval_cutoff = 0.05, log2fc_min = 0.25):
    """
    Purpose: Outputs differentially expressed genes in cancer vs normal cells, and vice versa
    Inputs:
    - AnnData object containing normalized counts and labels (same format as output from create_adata function)
    - corr_method - correction method to use with Wilcoxon rank sum test
    - pval_cutoff - p-value cutoff to use with Wilcoxon rank sum test (0.05 by default)
    - log2fc_min - minimum fold change to use with Wilcoxon rank sum test (0.25 by default)
    Output:
    - Two lists of differentially expressed genes (enriched in cancer or normal cells)
    """
    
    adata_copy = adata_obj.copy()
    adata_copy.obs['orig_cancer_label'] = pd.Categorical(adata_copy.obs['orig_cancer_label'])
    
    # Remove genes expressed in no cells to avoid error with dividing by stdev = 0 in rank_genes_groups below
    adata_copy = sc.pp.filter_genes(adata_copy, min_cells=1)

    # Calculate differentially expressed genes
    adata_deg = sc.tl.rank_genes_groups(adata_copy, groupby='orig_cancer_label', method='wilcoxon',
                                        tie_correct = True, corr_method = corr_method,
                                        pts = True, copy = True, layer = 'norm')
    adata_deg_df = sc.get.rank_genes_groups_df(adata_deg, group = None,
                                               pval_cutoff = pval_cutoff, log2fc_min = log2fc_min)
    adata_deg_df_filt = adata_deg_df[adata_deg_df['pct_nz_group'] > 0.1]
    
    # Get ordered lists of genes - one for cancer, one for normal
    adata_deg_df_filt = adata_deg_df_filt.astype({'group':'int'})
    adata_deg_df_filt = adata_deg_df_filt.set_index('names')
    adata_deg_cancer = adata_deg_df_filt[adata_deg_df_filt['group']== 0].index
    adata_deg_norm = adata_deg_df_filt[adata_deg_df_filt['group']== 1].index
    
    return adata_deg_cancer, adata_deg_norm

# Get highly variable genes from dataset
def get_hvgs(adata, method):
    """
    Purpose: Calculate list of highly variable genes using standard built-in scanpy methods
    Inputs:
      - adata: AnnData object containing raw and normalized counts (as generated from create_adata)
      - method: String indicating method to use for calculating HVGs
        - Valid values: 'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals')
    Output:
      - Index of all genes in adata sorted by high to low variance/dispersion
    """
    
    num_genes = adata.n_vars
    
    # Use experimental module if 'pearson_residuals' (with raw counts), otherwise use standard method
    if method == 'pearson_residuals':
        hvg_df = sc.experimental.pp.highly_variable_genes(adata.copy(), flavor = 'pearson_residuals',
                                                      n_top_genes = num_genes,
                                                      layer = 'raw', inplace = False)
    elif method == 'seurat_v3':
        hvg_df = sc.pp.highly_variable_genes(adata.copy(), flavor = method,
                                         n_top_genes = num_genes,
                                         layer = 'raw', inplace = False)
    elif method in ['seurat', 'cell_ranger']:
        hvg_df = sc.pp.highly_variable_genes(adata.copy(), flavor = method,
                                       #  n_top_genes = num_genes,
                                         layer = 'norm', inplace = False)
    else:
        raise ValueError("String must be one of four values: " + \
                     "'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals'")
    
    # Sort genes by highest variance/dispersion, depending on method
    if method in ['seurat_v3', 'pearson_residuals']:
        hvg_df = hvg_df.sort_values(by = 'highly_variable_rank')
    else:
        hvg_df = hvg_df.sort_values(by = 'dispersions_norm', ascending = False)

    return hvg_df.index

# Function with cross-validation applied to training dataset
# Generates evaluation metrics for lists of feature selection methods and numbers of features
def train_feat_loop_cv(clf, adata, groups_label, num_feat_list, feat_method_list,
                       random_state = 0, k_fold = 5):
    """
      Purpose: Run cross-validation with different numbers of features and feature selection methods
      Inputs:
        - clf: Classifier
        - adata: AnnData object containing raw and normalized count matrix and labels (from create_adata)
        - groups_label: String indicating group to split on (should be column in adata.obs)
        - num_feat_list: List of numbers of features to use
        - feat_method_list: List of feature selection methods
          - Scanpy highly variable genes: 'seurat_v3', 'seurat', 'cell_ranger', 'pearson_residuals'
          - Differential expression: 'dge'
          - Random selection: 'random_all_genes' (randomizes order of all genes, then select top N genes per N)
        - random_state: Random state to use for k-folds
        - k_fold = Number of folds to use
      Output: 
        - Concatenated dataframe containing results for all numbers of features and feature selection methods
    """
  
    results_df = pd.DataFrame()
    fold_indices_dict = {}
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
        print(f'Fold i: {i}')
        # Set up training and test folds
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        # Store test indices in a dictionary by fold
        fold_indices_dict[i] = {'train': train_index, 'test': test_index}
    
        # Loop through all feature selection methods
        for curr_method in feat_method_list:
            print(f'curr_method: {curr_method}')
            # For first iteration of method, create empty dictionary to store SHAP values
            if i == 0:
                shap_results[curr_method] = {}
      
            # Select features based on feature selection method
            if curr_method in ['seurat_v3', 'pearson_residuals', 'seurat', 'cell_ranger']:
                feature_order = get_hvgs(X_train, curr_method)
            elif curr_method == 'dge':
                feature_order = {}
                feature_order['cancer'], feature_order['norm'] = get_diff_exp_genes(X_train)
            elif curr_method == 'random_all_genes':
                rng = np.random.default_rng(random_state)
                feature_order = rng.choice(adata.var_names, size = adata.n_vars, replace=False)
            else:
                raise ValueError("String must be one of these values: 'seurat_v3', 'seurat', " + \
                               "'cell_ranger', 'pearson_residuals', 'dge', 'random_all_genes'")
      
            # Store feature order in dictionary, using largest number of features in num_feat_list
            if i == 0:
                feat_order_dict[curr_method] = {}
            if curr_method == 'dge':
                feat_order_dict[curr_method][i] = {'cancer': feature_order['cancer'][:max(num_feat_list)],
                                                 'norm': feature_order['norm'][:max(num_feat_list)]}
            else:
                feat_order_dict[curr_method][i] = feature_order[:max(num_feat_list)]
              
            # Loop through all numbers of features
            for curr_num_feat in num_feat_list:
                # For first iteration of number of features, create empty dictionary to store SHAP values
                if i == 0:
                    shap_results[curr_method][curr_num_feat] = {}
              
                # Extract top features depending on method
                if curr_method == 'dge':
                    # For differential gene expression, half of features will come from each list
                    num_dges = int(curr_num_feat/2)
                    curr_feat = feature_order['cancer'][:num_dges].append(feature_order['norm'][:num_dges])
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
                curr_results['recall'] = [recall_score(y_test, y_pred)]
                curr_results['precision'] = [precision_score(y_test, y_pred)]

                # Calculate feature importance using SHAP
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_test[:, curr_feat].X)
                shap_results[curr_method][curr_num_feat][i] = shap_values
        
                # Convert values into dataframe
                results_df = pd.concat([results_df, pd.DataFrame.from_dict(curr_results)],
                                       ignore_index=True)

    return results_df, fold_indices_dict, feat_order_dict, shap_results

# Trains model with list of features, and scores test dataset with same features
def train_test_model(clf, train_df, train_labels, test_df, test_labels, features):
  """
    Purpose: Trains and tests model given features and datasets
    Inputs:
      - clf: Classifier
      - train_df: Training dataset
      - train_labels: Training labels
      - test_df: Test dataset
      - test_labels: Test labels
      - features: List of features
    Output: Trained model. Print metrics
  """

  # Get subsets of train_df and test_df based on features present in 'features'
  train_df_sub = train_df[features]
  test_df_sub = test_df[test_df.columns[test_df.columns.isin(features)]]

  # Concatenate, then separate into train and test so same features are present in both
  train_test_combined = pd.concat([train_df_sub, test_df_sub]).fillna(0)

  train_df_sub_v2 = train_test_combined.iloc[0:len(train_df_sub)]
  test_df_sub_v2 = train_test_combined.iloc[len(train_df_sub):]
  
  # Train model
  clf.fit(train_df_sub_v2[features], train_labels)

  # Test model
  y_pred = clf.predict(test_df_sub_v2[features])

  recall = recall_score(test_labels, y_pred)
  precision = precision_score(test_labels, y_pred)
  accuracy = accuracy_score(test_labels, y_pred)
  f1 = f1_score(test_labels, y_pred)
  conf_matrix = confusion_matrix(test_labels, y_pred)

  metrics_df = pd.DataFrame({'recall': [recall],
                             'precision': [precision],
                             'accuracy': [accuracy],
                             'f1': [f1]})

  print(f'# cells in training: {len(train_df)}, # cells in test: {len(test_df)}')
  print(conf_matrix)
  display(metrics_df)

  return metrics_df, conf_matrix

# Generate line plots of metrics from train_feat_loop_cv
def make_line_plots_metrics(results_df):
    """
      Purpose: Generate line plots comparing metrics from cross-validation
      Inputs: 
        - results_df: Dataframe from train_feat_loop_cv() with metrics in each column
      Output:
        - Dataframe of metrics summary, grouped by feature selection method and number of features
        - Figure with faceted subplots by metric
    """
    # Convert dataframe from wide to long
    results_df_tall = results_df.melt(id_vars=['feat_sel_type', 'num_features', 'fold'],
                                      var_name='metric', value_name='score')
  
    # Create dataframe summarizing mean and stdev
    results_df_pivot = pd.pivot_table(results_df_tall,
                                      values=['score'],
                                      index = ['feat_sel_type', 'num_features'],
                                      columns = ['metric'],
                                      aggfunc=['mean', 'std'])
  
    # Plot 1 figure with all metrics versus number of features
    sns.set_theme(style='whitegrid')
    with sns.plotting_context(context = "notebook", font_scale=1.25):
        g1 = sns.catplot(
            data=results_df,
            x='num_features', y='score', col='metric',
            hue = 'feat_sel_type', col_wrap = 2, kind='point', capsize = 0.2,
            sharex = False, alpha = 0.7
        )

    g1.set_xticklabels(rotation=45)
    g1.set_axis_labels('Number of features', 'Score')

    sns.move_legend(g1, "upper right", bbox_to_anchor=(1, 1))
    g1.legend.get_title().set_text('Feature selection method')
    for text in g1.legend.texts:
        text.set_fontsize(14)

    plt.subplots_adjust(bottom=-0.01)
    plt.show()

    return results_df_pivot, g1

# Calculate Jaccard coefficient overlap between feature sets between methods
def calc_jaccard_coeff(method_list, num_feat_list, feat_dict, num_folds):
    """
    Purpose: Calculate Jaccard coefficient between feature sets of different feature selection methods
    Inputs:
        - method_list: List of feature selection methods
        - num_feat_list: List of numbers of features
        - feat_dict: Dictionary containing order of features for each method and fold
        - num_folds: Number of folds
    Outputs:
        - Dataframe with Jaccard coefficients for a given number of features
    """
    jaccard_df = pd.DataFrame()

    # Loop through folds
    for fold in range(num_folds):
        # Loop through number of features
        for curr_num_feat in num_feat_list: 
            # Loop through method 1
            for i in range(len(method_list)):
                # Get list of features for method 1
                method1 = method_list[i]
                if method1 == 'dge':
                    num_dges = int(curr_num_feat/2)
                    cancer_top_genes_1 = feat_dict[method1][fold]['cancer'][:num_dges]
                    curr_feat_1 = cancer_top_genes_1.append(feat_dict[method1][fold]['norm'][:num_dges]) 
                else:
                    curr_feat_1 = feat_dict[method1][fold][:curr_num_feat]
                method1_feat_set = set(curr_feat_1)
                # Loop through method 2
                for j in range(i):
                    # Get list of features for method 2
                    method2 = method_list[j]
                    if method2 == 'dge':
                        cancer_top_genes_2 = feat_dict[method2][fold]['cancer'][:num_dges]
                        curr_feat_2 = cancer_top_genes_2[:num_dges].append(
                                        feat_dict[method2][fold]['norm'][:num_dges]) 
                    else:
                        curr_feat_2 = feat_dict[method2][fold][:curr_num_feat]
                    method2_feat_set = set(curr_feat_2)
                    # Calculate Jaccard coefficient
                    curr_jaccard = len(method1_feat_set.intersection(method2_feat_set)) /\
                                    len(method1_feat_set.union(method2_feat_set))
                    curr_jaccard_df = pd.DataFrame(
                        {'method1': [method1], 'method2': [method2],
                         'num_features': [curr_num_feat], 'fold': [fold],
                         'jaccard_coeff': [curr_jaccard]})
                    jaccard_df = pd.concat([jaccard_df,curr_jaccard_df], ignore_index=True)

    return jaccard_df

# Generate feature importance SHAP plots for a given method and number of features across folds
def plot_feat_importance(adata, method, num_feat, feat_dict, shap_dict, folds_dict, file_prefix):
    """
    Purpose: Consolidates SHAP values across folds and generates beeswarm SHAP plot for feature importance
    Inputs:
        - adata: AnnData object containing gene expression values
        - method: String of highly variable gene method used
        - num_feat: Number of features to select for
        - feat_dict: Dictionary containing order of features for each method and fold
        - shap_dict: Dictionary containing SHAP values for each method, number of features, and fold
        - folds_dict: Dictionary containing indices of train and test samples by fold
        - file_prefix: String for file path to use for saving
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
    for fold, index_lists in folds_dict.items():
        curr_cells = X_index[index_lists['test']]

        # Get current set of features
        if method == 'dge':
            num_dges = int(curr_num_feat/2)
            curr_feat = feat_dict[method][fold]['cancer'][:num_dges].append(
                          feat_dict[method][fold]['norm'][:num_dges]) 
        else:
            curr_feat = feat_dict[method][fold][:num_feat]
    
        # Create dataframe with cell indices, features, and SHAP values
        curr_shap = shap_dict[method][num_feat][fold]
        curr_fold_df = pd.DataFrame(data = curr_shap,
                                    index = curr_cells,
                                    columns = curr_feat)
        # Concatenate dataframe to main dataframe - keep missing values as NaN?
        shap_vals_df = pd.concat([shap_vals_df, curr_fold_df])

    shap_vals_df.to_csv(f'{file_prefix}shap_vals_df_{method}_features{num_feat}.csv')

    # Convert missing values to 0
    shap_vals_df_no_na = shap_vals_df.fillna(0)

    # Subset anndata to same cells and features in SHAP value frame
    adata_sub_vals = adata[shap_vals_df.index, shap_vals_df.columns].to_df()

    # Create beeswarm SHAP plot sorted by highest absolute mean (missing values as 0s)
    fig = plt.figure() 
    shap.summary_plot(shap_vals_df_no_na.values,adata_sub_vals, max_display = 10, show = False)
    fig.suptitle(f'Top 10 Features for method {method} with {num_feat} features', fontsize=16)
    fig.ylabel('Top 10 features ordered by absolute mean', fontsize=13)
    fig.savefig(f'{file_prefix}mean_beeswarm_{method}_features{num_feat}.png', bbox_inches='tight')
    plt.close(fig)
    
    return shap_vals_df
