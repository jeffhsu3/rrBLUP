import numpy as np
import pandas as pd
import pytest
import rpy2.robjects
from pyrrblup import A_mat
from rpy2.robjects import ListVector

TOLERANCE = 1e-6

@pytest.fixture
def train_data_small(): # Renamed fixture
    # :TODO used smaller test data
    try:
        df = pd.read_csv('../data/small.train.csv') # Changed filename
    except FileNotFoundError:
        df = pd.read_csv('data/small.train.csv') # Changed filename

    
    if df.columns[0] == 'label':
        train_x_np = df.iloc[:, 1:].values.astype(float)
    else: # Assuming first col is an index, and 'label' is the second, actual data starts from third
          # Or more robustly, drop 'label' by name if it exists, then take rest
        if 'label' in df.columns:
            train_x_np = df.drop('label', axis=1).values.astype(float)
        else: # if no 'label' column, assume all except potential first index are genotypes
            train_x_np = df.iloc[:, 1:].values.astype(float)

    try:
        df_check = pd.read_csv('data/small.train.csv') # Changed filename
    except FileNotFoundError:
        df_check = pd.read_csv('../data/small.train.csv') # Changed filename

    if df_check.columns[0].startswith("Unnamed"): # Handles unnamed index column
         train_x_r_input_df = df_check.iloc[:, 1:] # Skip unnamed index
    else:
         train_x_r_input_df = df_check

    if df_check.columns[0].startswith("Unnamed"): # Handles unnamed index column
         train_x_r_input_df = df_check.iloc[:, 1:] # Skip unnamed index
    else:
         train_x_r_input_df = df_check

    # Now, from this df, remove the 'label' column for R's A.mat
    if 'label' in train_x_r_input_df.columns:
        train_x_r_input_df = train_x_r_input_df.drop('label', axis=1)
    
    train_x_r = rpy2.robjects.conversion.py2rpy(train_x_r_input_df.astype(float))

    # For Python A_mat, we need {-1, 0, 1} coding.
    # The data description says "rrBLUP R package requires input genotype matrix coded as -1,0,1".
    # And "data_convertion.ipynb" handles this. So protein.train.nan-101.csv should already be in -1,0,1 format.
    # The original problem states "Convert this into a modern python package", implying the Python code
    # rrBLUP.py is the target. This Python code expects {-1,0,1}.
    # The nan-101 files are already converted according to demo.R and problem context.
    # So, train_x_np should be directly usable.

    return {"python": train_x_np, "r": train_x_r, "r_df": train_x_r_input_df}


def test_A_mat_default(train_data_small, rrblup_r_package): # Changed fixture name
    # Python execution (uses Python's default min_MAF, max_missing)
    A_py = A_mat(train_data_small["python"])

    Amat_r_obj = rrblup_r_package.A_mat(train_data_small["r"]) # Changed fixture name
    Amat_r = np.array(Amat_r_obj)

    assert A_py.shape == Amat_r.shape, "Default: Output matrices shapes differ"
    assert np.allclose(A_py, Amat_r, atol=TOLERANCE), "Default: Output matrices values differ significantly"


def test_A_mat_em_impute(train_data_small, rrblup_r_package): # Changed fixture name
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5

    A_py_em = A_mat(train_data_small["python"], 
                    min_MAF=R_MIN_MAF, 
                    max_missing=R_MAX_MISSING, 
                    impute_method='EM')

    Amat_r_em_obj = rrblup_r_package.A_mat(train_data_small["r"],  # Changed fixture name
                                   min_MAF=R_MIN_MAF, 
                                   max_missing=R_MAX_MISSING, 
                                   impute_method='EM')
    Amat_r_em = np.array(Amat_r_em_obj)

    assert A_py_em.shape == Amat_r_em.shape, "EM Impute: Output matrices shapes differ"
    assert np.allclose(A_py_em, Amat_r_em, atol=TOLERANCE), "EM Impute: Output matrices values differ significantly"


def test_A_mat_mean_impute_return_imputed(train_data_small, rrblup_r_package): # Changed fixture name
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5

    # Python execution
    A_py_mean, X_imp_py = A_mat(train_data_small["python"], 
                                min_MAF=R_MIN_MAF, 
                                max_missing=R_MAX_MISSING, 
                                impute_method='mean', 
                                return_imputed=True)

    # R execution
    # R returns a list: result$A, result$imputed
    r_result_mean = rrblup_r_package.A_mat(train_data_small["r"],  # Changed fixture name
                                   min_MAF=R_MIN_MAF, 
                                   max_missing=R_MAX_MISSING, 
                                   impute_method='mean', 
                                   return_imputed=True)
    Amat_r_mean = np.array(r_result_mean.rx2('A'))
    X_imp_r = np.array(r_result_mean.rx2('imputed'))

    assert A_py_mean.shape == Amat_r_mean.shape, "Mean Impute (A): Output matrices shapes differ"
    assert np.allclose(A_py_mean, Amat_r_mean, atol=TOLERANCE), "Mean Impute (A): Output matrices values differ significantly"
    
    assert X_imp_py.shape == X_imp_r.shape, "Mean Impute (imputed X): Output matrices shapes differ"
    assert np.allclose(X_imp_py, X_imp_r, atol=TOLERANCE), "Mean Impute (imputed X): Output matrices values differ significantly"


def test_A_mat_shrink_ej(train_data_small, rrblup_r_package): # Changed fixture name
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5

    # Python execution (shrink=True defaults to EJ)
    A_py_ej = A_mat(train_data_small["python"], 
                    min_MAF=R_MIN_MAF, 
                    max_missing=R_MAX_MISSING, 
                    shrink=True)

    # R execution (shrink=TRUE also defaults to EJ)
    Amat_r_ej_obj = rrblup_r_package.A_mat(train_data_small["r"], # Changed fixture name
                                   min_MAF=R_MIN_MAF, 
                                   max_missing=R_MAX_MISSING, 
                                   shrink=True)
    Amat_r_ej = np.array(Amat_r_ej_obj)

    assert A_py_ej.shape == Amat_r_ej.shape, "Shrink EJ: Output matrices shapes differ"
    assert np.allclose(A_py_ej, Amat_r_ej, atol=TOLERANCE), "Shrink EJ: Output matrices values differ significantly"


def test_A_mat_shrink_reg(train_data_small, rrblup_r_package): # Changed fixture name
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5
    N_QTL = 100 # As per demo.R
    N_ITER = 5  # As per demo.R

    # Calculate p for R's A.mat function
    r_marker_matrix = train_data_small["r"] # Changed fixture name
    r_X_shifted = r_marker_matrix + 1
    
    # Need to access rpy2.robjects.r for R functions
    robjects = rpy2.robjects
    
    r_freq = robjects.r['apply'](r_X_shifted, 2, lambda x: robjects.r['mean'](x, na_rm=True)) / 2
    r_maf = robjects.r['pmin'](r_freq, 1 - r_freq)
    r_col_na_means = robjects.r['colMeans'](robjects.r['is.na'](r_marker_matrix))
    
    # Convert Python floats to R Floats for comparison
    r_R_MIN_MAF = robjects.FloatVector([R_MIN_MAF])
    r_R_MAX_MISSING = robjects.FloatVector([R_MAX_MISSING])

    # Get boolean vector for conditions
    condition_maf = robjects.r['>'](r_maf, r_R_MIN_MAF[0]) # MAF >= R_MIN_MAF
    condition_missing = robjects.r['<='](r_col_na_means, r_R_MAX_MISSING[0]) # colMeans(is.na(X)) <= R_MAX_MISSING
    
    # Combine conditions
    combined_condition = robjects.r['&'](condition_maf, condition_missing)
    
    r_markers_idx = robjects.r['which'](combined_condition)
    
    # r_freq is an R S4Vector, subsetting with r_markers_idx should work directly
    # If r_freq was a numpy array converted to R, it might need robjects.FloatVector(r_freq) first
    r_p_values = r_freq.rx(r_markers_idx)

    # Python execution
    A_py_reg = A_mat(train_data_small["python"], # Changed fixture name
                     min_MAF=R_MIN_MAF, 
                     max_missing=R_MAX_MISSING, 
                     shrink="REG", 
                     n_qtl=N_QTL, 
                     n_iter=N_ITER)

    # R execution
    # R here is failing as in the demo.R
    shrink_list_r = ListVector({'method': "REG", 'n.qtl': N_QTL, 'n.iter': N_ITER})
    
    Amat_r_reg_obj = rrblup_r_package.A_mat(train_data_small["r"],  # Changed fixture name
                                    min_MAF=R_MIN_MAF, 
                                    max_missing=R_MAX_MISSING, 
                                    shrink=shrink_list_r,
                                    p=r_p_values) # Pass calculated p
    Amat_r_reg = np.array(Amat_r_reg_obj)

    assert A_py_reg.shape == Amat_r_reg.shape, "Shrink REG: Output matrices shapes differ"
    assert np.allclose(A_py_reg, Amat_r_reg, atol=TOLERANCE), "Shrink REG: Output matrices values differ significantly"
