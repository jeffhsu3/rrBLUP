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
        df = pd.read_csv('data/small.train.csv')
    except FileNotFoundError:
        df = pd.read_csv('../data/small.train.csv')
    
    current_converter = rpy2.robjects.conversion.get_conversion()
    
    train_y_np = df['label'].values.astype(float).reshape(-1, 1) # Ensure it's a column vector
    
    if 'label' in df.columns:
        train_x_np = df.drop('label', axis=1).values.astype(float)
    else:
        train_y_np = df.iloc[:, 0].values.astype(float).reshape(-1, 1)
        train_x_np = df.iloc[:, 1:].values.astype(float)


    # Prepare R versions
    train_y_r = current_converter.py2rpy(train_y_np)
    
    try:
        df_for_r = pd.read_csv('data/small.train.csv')
    except FileNotFoundError:
        df_for_r = pd.read_csv('../data/small.train.csv')
    
    if 'label' in df_for_r.columns:
        train_x_r_input_df = df_for_r.drop('label', axis=1)
    else: # Should not happen based on demo.R
        train_x_r_input_df = df_for_r.iloc[:, 1:] # Fallback

    train_x_r = current_converter.py2rpy(train_x_r_input_df.astype(float))
    
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
    Amat, train_x_imp = A_mat(train_data_protein["X_py"], impute_method = 'mean', return_imputed = True)
    py_results_Z = mixed_solve(y=train_data_protein["y_py"], Z=train_x_imp)

    r_results_Z = rrblup_r_package.mixed_solve(y=train_data_protein["y_r"], Z=train_x_imp) # Use fixture

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
