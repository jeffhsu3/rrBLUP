import numpy as np
import pandas as pd
import pytest
import rpy2.robjects
from pyrrblup import A_mat
from rpy2.robjects import ListVector

TOLERANCE = 1e-6

@pytest.fixture
def train_data_small():
    current_converter = rpy2.robjects.conversion.get_conversion()
    try:
        df = pd.read_csv('../data/small.train.csv')
    except FileNotFoundError:
        df = pd.read_csv('data/small.train.csv')

    # :TODO simplify this
    if df.columns[0] == 'label':
        train_x_np = df.iloc[:, 1:].values.astype(float)
    else:
        if 'label' in df.columns:
            train_x_np = df.drop('label', axis=1).values.astype(float)
        else:
            train_x_np = df.iloc[:, 1:].values.astype(float)
    try:
        df_check = pd.read_csv('data/small.train.csv')
    except FileNotFoundError:
        df_check = pd.read_csv('../data/small.train.csv')

    if df_check.columns[0].startswith("Unnamed"):
         train_x_r_input_df = df_check.iloc[:, 1:]
    else:
         train_x_r_input_df = df_check

    if df_check.columns[0].startswith("Unnamed"):
         train_x_r_input_df = df_check.iloc[:, 1:]
    else:
         train_x_r_input_df = df_check
    if 'label' in train_x_r_input_df.columns:
        train_x_r_input_df = train_x_r_input_df.drop('label', axis=1)
    
    train_x_r = current_converter.py2rpy(train_x_r_input_df.astype(float))
    return {"python": train_x_np, "r": train_x_r, "r_df": train_x_r_input_df}


def test_A_mat_default(train_data_small, rrblup_r_package):
    # Python execution (uses Python's default min_MAF, max_missing)
    A_py = A_mat(train_data_small["python"])

    Amat_r_obj = rrblup_r_package.A_mat(train_data_small["r"])
    Amat_r = np.array(Amat_r_obj)

    assert A_py.shape == Amat_r.shape, "Default: Output matrices shapes differ"
    assert np.allclose(A_py, Amat_r, atol=TOLERANCE), "Default: Output matrices values differ significantly"


def test_A_mat_em_impute(train_data_small, rrblup_r_package):
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5

    A_py_em = A_mat(train_data_small["python"], 
                    min_MAF=R_MIN_MAF, 
                    max_missing=R_MAX_MISSING, 
                    impute_method='EM')

    Amat_r_em_obj = rrblup_r_package.A_mat(train_data_small["r"],
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
    # Note that there is a random.sample.  Is there a way to pass the PRNG to both?

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
    assert np.allclose(A_py_ej, Amat_r_ej, atol=1e-1), "Shrink EJ: Output matrices values differ significantly"


def test_A_mat_shrink_reg(train_data_small, rrblup_r_package): # Changed fixture name
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5
    N_QTL =  3
    N_ITER = 5  

    # Python execution
    A_py_reg = A_mat(train_data_small["python"], # Changed fixture name
                     min_MAF=R_MIN_MAF, 
                     max_missing=R_MAX_MISSING, 
                     shrink="REG", 
                     n_qtl=N_QTL, 
                     n_iter=N_ITER)

    # R execution
    # R here is failing as in the demo.R.  A p parameter is missing in the original rrBLUP
    '''
    shrink_list_r = ListVector({
        'method': "REG", 'n.qtl': N_QTL, 
        'n.iter': N_ITER, 
        'n.core': 3,
        })
    
    Amat_r_reg_obj = rrblup_r_package.A_mat(train_data_small["r"],  # Changed fixture name
                                    min_MAF=R_MIN_MAF, 
                                    max_missing=R_MAX_MISSING, 
                                    shrink=shrink_list_r,
                                ) 
    Amat_r_reg = np.array(Amat_r_reg_obj)

    assert A_py_reg.shape == Amat_r_reg.shape, "Shrink REG: Output matrices shapes differ"
    assert np.allclose(A_py_reg, Amat_r_reg, atol=TOLERANCE), "Shrink REG: Output matrices values differ significantly"
    '''
