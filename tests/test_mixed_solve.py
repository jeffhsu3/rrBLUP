import pytest
import numpy as np
import pandas as pd
import rpy2.robjects 
from pyrrblup import A_mat, mixed_solve 

# rrBLUP_r will come from conftest.py via fixture
TOLERANCE_MIXED_SOLVE = 1e-5 
R_MIN_MAF_DEFAULT = 0.05
R_MAX_MISSING_DEFAULT = 0.5

@pytest.fixture
def train_data_protein():
    # Load data from protein.train.csv
    try:
        df = pd.read_csv('data/protein.train.csv')
    except FileNotFoundError:
        df = pd.read_csv('../data/protein.train.csv')

    # Extract y (phenotypes) and X (genotypes)
    # demo.R: train_y <- as.matrix(train['label']); train_x <- as.matrix(train[-1])
    
    train_y_np = df['label'].values.astype(float).reshape(-1, 1) # Ensure it's a column vector
    
    # For train_x_np, R's train[-1] means all columns except the first one it read.
    # We need to ensure our train_x_np matches R's expectation of the genotype matrix.
    # In demo.R, train_x <- as.matrix(train[-1]) suggests that if 'label' is the first column,
    # it is excluded. If 'label' is not the first, then whatever is the first column is excluded.
    # However, the paired usage `train_y <- as.matrix(train['label'])` implies 'label' is a named column.
    # It's safer to assume train_x is everything BUT 'label'.
    if 'label' in df.columns:
        train_x_np = df.drop('label', axis=1).values.astype(float)
    else: # Fallback if 'label' column is missing, though demo.R implies it exists
        # This case should ideally not happen if data matches demo.R
        # If first column is an index, pandas might have read it as such.
        # If no 'label', assume first column is phenotype, rest are genotypes.
        train_y_np = df.iloc[:, 0].values.astype(float).reshape(-1, 1)
        train_x_np = df.iloc[:, 1:].values.astype(float)


    # Prepare R versions
    train_y_r = rpy2.robjects.conversion.py2rpy(train_y_np)
    
    try:
        df_for_r = pd.read_csv('data/protein.train.csv')
    except FileNotFoundError:
        df_for_r = pd.read_csv('../data/protein.train.csv')
    
    if 'label' in df_for_r.columns:
        train_x_r_input_df = df_for_r.drop('label', axis=1)
    else: # Should not happen based on demo.R
        train_x_r_input_df = df_for_r.iloc[:, 1:] # Fallback

    train_x_r = rpy2.robjects.conversion.py2rpy(train_x_r_input_df.astype(float))
    
    return {
        "y_py": train_y_np, "X_py": train_x_np, # X_py is markers only
        "y_r": train_y_r, "X_r": train_x_r,     # X_r is markers only, for mixed_solve Z=X_r
        "X_r_df_markers_only": train_x_r_input_df # for A.mat(X_r_df_markers_only)
    }

def compare_mixed_solve_results(py_res, r_res, tolerance):
    assert np.isclose(py_res['Vu'], r_res.rx2('Vu')[0], atol=tolerance), f"Vu differs: Py={py_res['Vu']}, R={r_res.rx2('Vu')[0]}"
    assert np.isclose(py_res['Ve'], r_res.rx2('Ve')[0], atol=tolerance), f"Ve differs: Py={py_res['Ve']}, R={r_res.rx2('Ve')[0]}"
    
    py_beta = py_res['beta'].flatten() # Ensure 1D for comparison if it's a column vector
    r_beta = np.array(r_res.rx2('beta')).flatten() # Ensure 1D
    assert py_beta.shape == r_beta.shape, f"beta shapes differ: Py={py_beta.shape}, R={r_beta.shape}"
    assert np.allclose(py_beta, r_beta, atol=tolerance), f"beta differs: Py={py_beta}, R={r_beta}"
    
    py_u = py_res['u'].flatten() # Ensure 1D for comparison
    r_u = np.array(r_res.rx2('u')).flatten() # Ensure 1D
    # R's u might be named, leading to a structured array if not careful.
    # Direct conversion to np.array should handle it, then flatten.
    assert py_u.shape == r_u.shape, f"u shapes differ: Py={py_u.shape}, R={r_u.shape}"
    assert np.allclose(py_u, r_u, atol=tolerance), f"u differs: Py={py_u}, R={r_u}"
    
    # LL can sometimes have greater discrepancies due to optimization paths or constant offsets
    assert np.isclose(py_res['LL'], r_res.rx2('LL')[0], atol=tolerance*10, rtol=tolerance*10), f"LL differs: Py={py_res['LL']}, R={r_res.rx2('LL')[0]}"


def test_mixed_solve_with_Z(train_data_protein, rrblup_r_package): # Added fixture
    # Python execution
    # mixed_solve adds an intercept by default if X is None.
    # When Z is provided, K is assumed identity. X (fixed effects) defaults to an intercept.
    py_results_Z = mixed_solve(y=train_data_protein["y_py"], Z=train_data_protein["X_py"])

    # R execution
    # mixed.solve(y, Z) also defaults to REML and adds an intercept for fixed effects.
    r_results_Z = rrblup_r_package.mixed_solve(y=train_data_protein["y_r"], Z=train_data_protein["X_r"]) # Use fixture

    compare_mixed_solve_results(py_results_Z, r_results_Z, TOLERANCE_MIXED_SOLVE)


def test_mixed_solve_with_K(train_data_protein, rrblup_r_package): # Added fixture
    K_py = A_mat(train_data_protein["X_py"], 
                 min_MAF=R_MIN_MAF_DEFAULT, 
                 max_missing=R_MAX_MISSING_DEFAULT)
    
    K_r_obj = rrblup_r_package.A_mat(train_data_protein["X_r"],  # Use fixture and X_r from fixture
                             min_MAF=R_MIN_MAF_DEFAULT, 
                             max_missing=R_MAX_MISSING_DEFAULT)
    py_results_K = mixed_solve(y=train_data_protein["y_py"], K=K_py)
    r_results_K = rrblup_r_package.mixed_solve(y=train_data_protein["y_r"], K=K_r_obj) # Use fixture
    
    compare_mixed_solve_results(py_results_K, r_results_K, TOLERANCE_MIXED_SOLVE)
