import time
import anndata as ad
from anndata import AnnData
import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.stats.multitest import multipletests

from typing import Union, Optional, List, Literal

from .utils import *

def wilcoxauc(
    data: Union[AnnData, np.ndarray, sp.spmatrix, pd.DataFrame], 
    # ====== general params ======
    groupby: Union[str, np.ndarray, pd.Series, None] = None, *,
    groups: Union[Literal['all'], List[str], None] = 'all',
    reference: Union[Literal['rest'], str, None] = 'rest',
    mask_var: Optional[Union[np.ndarray, str]] = None,
    n_genes: Optional[int] = 10,
    corr_method: Literal['benjamini-hochberg', 'bonferroni'] = 'benjamini-hochberg',
    # ====== anndata parameters ======
    copy: bool = False,
    use_raw: bool = False,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
    # ===== other params =====
    verbose: bool = True,
    nthreads: int = -1,
    **kwargs
    
):
    """
    Fast Wilcoxon rank sum test for single-cell data
    
    Parameters
    ----------
    Perform Wilcoxon rank-sum test for marker gene detection.
    
    Parameters
    ----------
    data : AnnData, np.ndarray, sp.spmatrix, or pd.DataFrame
        Input data. If AnnData, rows=cells, cols=genes.
        
    # === general params ===
    groupby : str or array-like, optional
        - If data is AnnData: key in adata.obs
        - If data is matrix: 1D array of group labels (length = n_cells)
    groups : 'all' or list of str, optional
        Groups to test.
    reference : str, default 'rest'
        Reference group for comparison.
    mask_var : array-like or str, optional
        Boolean mask or column name to select subset of genes.
    n_genes : int, optional
        Number of top genes to return per group.
    corr_method : Literal['benjamini-hochberg', 'bonferroni'] (default: 'benjamini-hochberg') 
        p-value correction method. Used only for 't-test', 't-test_overestim_var', and 'wilcoxon'.

    # === anndata parameters (only when data is AnnData) ===
    copy : bool, default False
        Return a copy instead of modifying in-place.
    use_raw : bool or None, optional
        Use adata.raw if available.
    layer : str, optional
        Use adata.layers[layer] instead of adata.X.
    key_added : str, optional
        Key in adata.uns to store results.
    
    # === other parameters ===
    verbose : bool (default True)
        Print progress messages.
    nthreads : int (default -1)
        Number of threads to use for computation. -1 means using all available cores.
    Returns
    -------
    AnnData or pd.DataFrame
        - AnnData if input is AnnData
        - pd.DataFrame if input is matrix-like

    """
    
    # 1. Data type and parameters check in
    if verbose:
        print("checking parameters...")
    start_time = time.time()
    is_adata = isinstance(data, AnnData)

    if not is_adata and any([copy, layer, key_added is not None, use_raw]):
        print("Warning: 'copy', 'layer', 'key_added' and 'use_raw' are ignored when data is not AnnData.")
    
    if groupby is None:
        raise ValueError("'groupby' must be specified.")
    
    end_time = time.time()
    if verbose:
        print(f"Parameter check took {end_time - start_time:.2f} seconds.")
        print("================================")

    # 2. Extract data matrix `X`, group labels `y` and gene names `var_names`
    if verbose:
        print("Extracting data matrix and group labels...")
    start_time = time.time()
    X, y, var_names = _extract_data_and_groups(
        data, groupby,
        layer=layer if is_adata else None,
        use_raw=use_raw if is_adata else None
    )
    end_time = time.time()
    if verbose:
        print(f"Data extraction took {end_time - start_time:.2f} seconds.")
        print("================================")

    # 3. Process mask_var
    if verbose:
        print("Processing gene mask...")
    start_time = time.time()
    if mask_var is not None:
        mask = _process_mask_var(mask_var, data, is_adata, X.shape[1])
        X = X[:, mask]
        var_names = np.asarray(var_names)[mask].tolist() if var_names is not None else None
    end_time = time.time()
    if verbose:
        print(f"Mask processing took {end_time - start_time:.2f} seconds.")
        print("================================")
    
    # 4. Process groups
    if verbose:
        print("Processing groups...")
    start_time = time.time()
    code_dict = _encode_groups(y ,groups)

    if reference != 'rest':
        if reference not in code_dict['label_to_code']:
            raise ValueError(f"Reference group '{reference}' not found in data.")
        else:
            reference_code = code_dict['label_to_code'][reference]
    end_time = time.time()

    if verbose:
        print(f"Group processing took {end_time - start_time:.2f} seconds.")
        print("================================")

    # 5. Core computation
    if reference == 'rest':
        core_results_ = _wilcoxauc_core(X, code_dict['y_encoded'], corr_method, nthreads, verbose=verbose)
        # slice results into only target groups
        target_indices = code_dict['target_codes']
        results_ = {
            key: val[target_indices, :] for key, val in core_results_.items()
        }
    else:
        # Specific reference group: 
        target_codes = code_dict['target_codes']
        if reference_code in target_codes:
            target_codes = target_codes[target_codes != reference_code]
            if len(target_codes) == 0:
                raise ValueError(f"No target groups remain after excluding reference group '{reference}'")

        n_target_groups = len(target_codes)
        n_total_genes = X.shape[1]

        # Initialize result containers
        results_ = {
            'avgExpr': np.zeros((n_target_groups, n_total_genes)),
            'logfoldchanges': np.zeros((n_target_groups, n_total_genes)),
            'score': np.zeros((n_target_groups, n_total_genes)),
            'auc': np.zeros((n_target_groups, n_total_genes)),
            'pval': np.zeros((n_target_groups, n_total_genes)),
            'padj': np.zeros((n_target_groups, n_total_genes)),
            'pct_1': np.zeros((n_target_groups, n_total_genes)),
            'pct_2': np.zeros((n_target_groups, n_total_genes)),
        }

        ref_mask = (code_dict['y_encoded'] == reference_code) # ndarray[bool]

        # Process each target group vs reference
        for idx, target_code in enumerate(target_codes):
            if verbose:
                print(f"Processing group '{code_dict['code_to_label'][target_code]}' vs reference '{reference}'...")
            target_mask = (code_dict['y_encoded'] == target_code)
            combined_mask = ref_mask | target_mask

            # subset data
            X_sub = X[combined_mask, :]
            y_sub = np.zeros(np.sum(combined_mask), dtype=np.int32)
            y_sub[ref_mask[combined_mask]] = 1  # reference group as 1

            core_results_sub = _wilcoxauc_core(X_sub, y_sub, corr_method, nthreads, verbose = verbose)

            # Store results
            for key in results_.keys():
                results_[key][idx, :] = core_results_sub[key][0, :]  # index 0 corresponds to target group

        code_dict['target_codes'] = target_codes

    # 6. Format results
    long_df = _format_results(results_, code_dict['target_codes'], 
                              code_dict['code_to_label'], var_names, n_genes)
    
    # 7. Return results
    if is_adata:
        if copy:
            adata = data.copy()
        else:
            adata = data

        key = key_added if key_added is not None else 'genes_groups_multi_index'
        adata.uns[key] = long_df
        
        return adata if copy else None
    else:
        return long_df

def calc_gini(
    data: Union[AnnData, np.ndarray, sp.spmatrix, pd.DataFrame], 
    # ====== general params ======
    groupby: Union[str, np.ndarray, pd.Series, None] = None, *,
    # ====== anndata parameters ======
    use_raw: bool = False,
    layer: Optional[str] = None,
    # ===== other params =====
    verbose: bool = True,
    **kwargs
    
):
    """
    Fast Wilcoxon rank sum test for single-cell data
    
    Parameters
    ----------
    Perform Wilcoxon rank-sum test for marker gene detection.
    
    Parameters
    ----------
    data : AnnData, np.ndarray, sp.spmatrix, or pd.DataFrame
        Input data. If AnnData, rows=cells, cols=genes.
        
    # === general params ===
    groupby : str or array-like, optional
        - If data is AnnData: key in adata.obs
        - If data is matrix: 1D array of group labels (length = n_cells)

    # === anndata parameters (only when data is AnnData) ===
    use_raw : bool or None, optional
        Use adata.raw if available.
    layer : str, optional
        Use adata.layers[layer] instead of adata.X.
        
    Returns
    -------
    AnnData or pd.DataFrame
        - AnnData if input is AnnData
        - pd.DataFrame if input is matrix-like

    """
    
    # 1. Data type and parameters check in
    is_adata = isinstance(data, AnnData)
    if groupby is None:
        raise ValueError("'groupby' must be specified.")

    # 2. Extract data matrix X and group labels y
    X, y, var_names = _extract_data_and_groups(
        data, groupby, 
        layer=layer if is_adata else None,
        use_raw=use_raw if is_adata else None
    )
    
    # 3. calculate gini
    code_dict = _encode_groups(y ,'all')
    gini = compute_nnz_gini(X, code_dict['y_encoded'], nthreads=-1)

    # 4. Format results
    long_df = _format_results({'gini': gini}, code_dict['target_codes'], 
                              code_dict['code_to_label'], var_names, sort = False)
    
    # 7. Return results
    return long_df

def prefilter_matrix(
    data: Union[AnnData, np.ndarray, sp.spmatrix, pd.DataFrame], 
    var_names: Optional[Union[list, np.ndarray, pd.Index]] = None,
    *,
    layer: Optional[str] = None,
    use_raw: bool = False,
    copy: bool = True,
):
    """
    Remove explicit zeros from inputs and drop all-zero gene columns.

    Parameters
    ----------
    data
        AnnData, ndarray, sparse matrix, or DataFrame. Shape must be cells x genes.
    var_names
        Required when `data` is `np.ndarray` or `sp.spmatrix`.
        Ignored for `AnnData` or `pd.DataFrame`
    layer
        AnnData layer to use. Ignored for non-AnnData inputs.
    use_raw
        If True, build the result from `adata.raw`. Ignored for non-AnnData inputs.
    copy
        If True, return a new object. If False, mutate AnnData / sparse inputs where practical.

    Returns
    -------
    If data is AnnData:
        AnnData
    If data is DataFrame:
        DataFrame
    If data is ndarray or sparse matrix:
        (X_filtered, var_names_filtered)
    """
    if isinstance(data, AnnData):
        if layer is not None and use_raw:
            raise ValueError("`layer` and `use_raw` only for AnnData inputs.")
        
        if use_raw:
            if data.raw is None:
                raise ValueError("adata.raw is not available.")
            X = data.raw.X.copy() if copy else data.raw.X
            if sp.issparse(X):
                X.eliminate_zeros()
            
            nonzero_mask = _build_nonzero_mask(X)
            X = X[:, nonzero_mask]
            raw_var = data.raw.var.loc[nonzero_mask].copy()

            result = ad.AnnData(
                X = X,
                obs = data.obs.copy(),
                var = raw_var,
            )
            result.uns = dict(data.uns)
            for key in data.obsm.keys():
                result.obsm[key] = data.obsm[key].copy()
            for key in data.obsp.keys():
                result.obsp[key] = data.obsp[key].copy()
            return result
        
        adata = data.copy() if copy else data

        if layer is not None:
            if layer not in adata.layers:
                raise KeyError(f"Layer '{layer}' not found in adata.layers")
            X = adata.layers[layer]
        else:
            X = adata.X
        
        if sp.issparse(X):
            X.eliminate_zeros()

        nonzero_mask = _build_nonzero_mask(X)
        if not np.all(nonzero_mask):
            adata._inplace_subset_var(nonzero_mask)

        return adata
    
    if isinstance(data, pd.DataFrame):
        X = data.to_numpy()
        nonzero_mask = _build_nonzero_mask(X)
        return data.loc[:, nonzero_mask]
    
    if isinstance(data, np.ndarray):
        if var_names is None:
            raise ValueError("var_names must be provided when data is np.ndarray.")
        var_names = np.asarray(var_names)
        if var_names.shape[0] != data.shape[1]:
            raise ValueError(f"Length of var_names ({var_names.shape[0]}) does not "
                             f"match number of genes ({data.shape[1]}).")
        nonzero_mask = _build_nonzero_mask(data)
        return data[:, nonzero_mask], var_names[nonzero_mask].tolist()
    
    if sp.issparse(data):
        if var_names is None:
            raise ValueError("var_names must be provided when data is sparse matrix.")
        var_names = np.asarray(var_names)
        if var_names.shape[0] != data.shape[1]:
            raise ValueError(f"Length of var_names ({var_names.shape[0]}) does not "
                             f"match number of genes ({data.shape[1]}).")
        X = data.copy() if copy else data
        X.eliminate_zeros()
        nonzero_mask = _build_nonzero_mask(X)
        return X[:, nonzero_mask], var_names[nonzero_mask].tolist()

    raise TypeError(f"Unsupported data type: {type(data)}")

def _extract_data_and_groups(data, groupby, layer = None, use_raw = None):
    """Extract data matrix X, group labels y and variable names from input data"""
    if isinstance(data, AnnData):
        return _from_anndata(data, groupby, layer, use_raw)
    
    elif isinstance(data, (np.ndarray, sp.spmatrix, pd.DataFrame)):
        try:
            groupby_arr = np.asarray(groupby)
        except Exception as e:
            raise ValueError(f"Could not convert 'groupby' to numpy array: {e}")
        
        #! Default settings: X is cells x genes, groupby is length of cells
        if groupby_arr.ndim != 1:
            raise ValueError(f"groupby must be 1-dimensional; got shape {groupby_arr.shape}")
        if len(groupby_arr) != data.shape[0]:
            raise ValueError(f"Length of 'groupby' ({len(groupby_arr)}) does not match number of samples ({data.shape[0]}).")
        
        if isinstance(data, (np.ndarray, sp.spmatrix)):
            var_names = [f"gene_{i}" for i in range(data.shape[1])]
            X = data
        else: 
            # pd.DataFrame
            var_names = data.columns.tolist()
            X = data.values
        return X, groupby_arr, var_names
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def _from_anndata(adata, groupby, layer, use_raw):
    """Extract data matrix X and group labels y from AnnData"""
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not found in adata.obs")

    # Priority: layer > use_raw > adata.X
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]
    elif use_raw and adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X

    # extract group labels
    y = adata.obs[groupby].values
    y = np.asarray(y)
    var_names = adata.var_names.tolist()

    return X, y, var_names

def _process_mask_var(mask_var, data, is_adata, n_genes):
    if isinstance(mask_var, str):
        if not is_adata:
            raise ValueError("mask_var as str only supported for AnnData")
        if mask_var not in data.var:
            raise ValueError(f"'{mask_var}' not in adata.var")
        mask = data.var[mask_var].values
    else:
        mask = np.asarray(mask_var)
        if mask.dtype != bool:
            raise ValueError("mask_var must be boolean array")
        if len(mask) != n_genes:
            raise ValueError("mask_var length must match number of genes")
    return mask

def _build_nonzero_mask(X):
    """Return a boolean mask of genes that are not all zero."""
    if sp.isspmatrix_csr(X):
        nonzero_mask = np.zeros(X.shape[1], dtype=bool)
        nonzero_mask[X.indices] = True
    elif sp.isspmatrix_csc(X):
        nonzero_mask = np.diff(X.indptr) > 0
    elif sp.issparse(X):
        nonzero_mask = np.asarray(X.getnnz(axis=0)).ravel() > 0
    else:
        nonzero_mask = np.any(X != 0, axis=0)

    if not np.any(nonzero_mask):
        raise ValueError("No genes remain after filtering all-zero genes.")

    return nonzero_mask

def _encode_groups(y, groups):
    """Encode group labels and determine target groups for comparison"""
    unique_labels = np.unique(y)
    if groups == 'all' or groups is None:
        target_labels = unique_labels
    else:
        target_labels = np.asarray(groups)
        invalid = set(target_labels) - set(unique_labels)
        if invalid:
            raise ValueError(f"Groups {sorted(invalid)} not found in data.")
    
    # Building mapping for ALL the groups (not just target groups was used in reference)
    label_to_code = {label: code for code, label in enumerate(unique_labels)}
    code_to_label = {code: label for label, code in label_to_code.items()}

    y_encoded = np.array([label_to_code[label] for label in y], dtype=np.int32)

    # Get codes for target groups
    target_codes = np.array([label_to_code[label] for label in target_labels], dtype=np.int32)
    return {
        'y_encoded': y_encoded,
        'target_codes': target_codes,
        'label_to_code': label_to_code,
        'code_to_label': code_to_label,
        'target_labels': target_labels,
    }

def _wilcoxauc_core(X, y, corr_method, nthreads, verbose):
    """
    calculate wilcoxauc statistics, including:
    avgExpr, logfoldchanges, score(norm U), 
    auc, pvals, padj, pct_1, pct_2
    """
    
    # 1. pvals/adj_pval, score, and auc calculation
    if verbose:
        print("wilcoxauc_core: computing pvals, scores, and AUC...")

    group_size = np.bincount(y)
    n_groups = len(group_size)
    n_cells = X.shape[0]
    n1n2 = group_size * (n_cells - group_size)
    n1n2 = n1n2.reshape(-1, 1)  # n1n2.ravel() in compute_pval

    start_time = time.time()
    # three type of X: dense ndarray, csr_matrix, csc_matrix
    rank_result = rank_matrix(X, nthreads=nthreads)
    # calculate pvals matrix and z-norm score matrix
    X_ranked = rank_result['X_ranked'] # sp.csr_matrix or np.ndarray
    ties_info = rank_result['ties'] # List of lists
    if verbose:
        print(f"Ranking matrix took {time.time() - start_time:.2f} seconds.")
    ustat_matrix = compute_ustats(X_ranked, y, group_size, nthreads)
    pval_matrix, z_norm_matrix = compute_pval(ustat_matrix, ties_info, n_cells, n1n2)

    # multiple testing correction
    fdr = np.full_like(pval_matrix, fill_value=np.nan, dtype=float)
    for g in range(n_groups):
        valid = ~np.isnan(pval_matrix[g, :])
        if np.any(valid):
            _, fdr[g, valid], _, _ = multipletests(
                pval_matrix[g, valid],
                alpha=0.05,
                method='fdr_bh' if corr_method == 'benjamini-hochberg' else 'bonferroni'
            )

    auc = ustat_matrix / n1n2
    
    if verbose:
        print(f"wilcoxauc_core: pvals, scores and AUC computation took {time.time() - start_time:.2f} seconds.")
        print("================================")

    # 2. pct_1, pct_2, avgExpr, logfoldchanges calculation
    if verbose:
        print("Computing expression statistics...")
    start_time = time.time()

    substep_time = time.time()
    group_sum = sum_groups(X, y, trans=False, nthreads=nthreads)
    if verbose:
        print(f"Expression stats: sum_groups took {time.time() - substep_time:.2f} seconds.")

    substep_time = time.time()
    group_nnz = nnz_groups(X, y, trans=False, nthreads=nthreads)
    if verbose:
        print(f"Expression stats: nnz_groups took {time.time() - substep_time:.2f} seconds.")

    substep_time = time.time()
    group_mean = group_sum / group_size[:, np.newaxis]

    pct_1 = (group_nnz / group_size[:, np.newaxis]) * 100
    total_nnz = np.sum(group_nnz, axis=0, keepdims=True)
    pct_2 = (
        (total_nnz - group_nnz) / 
        (n_cells - group_size[:, np.newaxis])
    ) * 100
    pct_sec = _get_second_largest(pct_1)

    epsilon = 1e-9
    rest_mean = ((np.sum(group_sum, axis=0, keepdims=True)-group_sum) / 
                                (n_cells - group_size[:, np.newaxis]))
    sec_mean = _get_second_largest(group_mean)
    
    lfc = np.log2((group_mean + epsilon) / (rest_mean + epsilon))
    lfc_sec = np.log2((group_mean + epsilon) / (sec_mean + epsilon))
    if verbose:
        print(f"Expression stats: numpy post-processing took {time.time() - substep_time:.2f} seconds.")
    if verbose:
        print(f"Expression statistics computation took {time.time() - start_time:.2f} seconds.")
        print("================================")

    return {
        'avgExpr': group_mean,
        'logfoldchanges': lfc,
        'score': z_norm_matrix,
        'auc': auc,
        'pval': pval_matrix,
        'padj': fdr,
        'pct_1': pct_1,
        'pct_2': pct_2,
        'pct_sec': pct_sec,
        'lfc_sec': lfc_sec,
    }

def _format_results(results_, target_codes, code_to_label, 
                    var_names, n_genes = None, sort = True):
    """format results into long data"""

    group_names = [code_to_label[code] for code in target_codes]

    long_df = wide2long(results_, var_names, group_names)
    if sort:
        long_df = long_df.sort_values(
            by = ['cluster','padj', 'score'],
            ascending=[True, True, False]
        ).reset_index(drop=True)

    if n_genes is not None:
        if not isinstance(n_genes, int) or n_genes <= 0:
            raise ValueError(f"n_genes must be a positive integer, got {n_genes}")
        if not sort:
            print("Warning: n_genes is applied without sorting, which may lead to unexpected results. Consider setting sort=True.")
        long_df = long_df.groupby('cluster', sort=False).head(n_genes).reset_index(drop=True)

    return long_df

def _get_second_largest(arr: np.ndarray) -> np.ndarray:
    """
    get the largest value in each column of 2D array
    apart from the origin value in the array
    """

    partitioned = np.partition(arr, -2, axis=0)
    global_max = partitioned[-1, :]
    second_max = partitioned[-2, :]

    argamx_indices = np.argmax(arr, axis=0)
    sec_array = np.zeros_like(arr) + global_max
    
    n_genes = arr.shape[1]
    rows = argamx_indices
    cols = np.arange(n_genes)
    sec_array[rows, cols] = second_max
    return sec_array
