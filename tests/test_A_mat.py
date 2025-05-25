import pytest
import numpy as np
import pandas as pd
# from rpy2.robjects.packages import importr # Removed
# import rpy2.robjects.numpy2ri # Removed
# from rpy2.robjects import pandas2ri # Removed
from rpy2.robjects import ListVector # Keep if used directly for creating R objects
import rpy2.robjects # Required for rpy2.robjects.conversion.py2rpy
from pyrrblup import A_mat # Assuming pyrrblup is installed or in PYTHONPATH

# R_RRBLUP_AVAILABLE and rrBLUP_r will come from conftest.py via fixture
# Define a tolerance for floating point comparisons
TOLERANCE = 1e-6

@pytest.fixture
def train_data_nan_101():
    # Load data, ensuring it's loaded relative to the project root or tests directory
    # Adjust path as necessary when running tests.
    # For now, assume 'data' is at the same level as 'tests' and 'pyrrblup'
    try:
        df = pd.read_csv('data/protein.train.nan-101.csv')
    except FileNotFoundError:
        # Fallback for environments where PWD is tests/
        df = pd.read_csv('../data/protein.train.nan-101.csv')
    
    # Genotypes are all columns except the first one (label)
    # The R code uses train_x <- as.matrix(train[-1])
    # The first column in the CSV is an unnamed index column if index=False was used during saving.
    # If the first column is 'label', then it's train[-1].
    # Let's assume the first column is NOT 'label' and needs to be dropped if it's an index,
    # OR, if 'label' is the first column, then it's df.iloc[:, 1:].
    # Based on demo.R: train <- read.csv(...); train_x <- as.matrix(train[-1])
    # This implies 'label' is the first *named* column.
    # Pandas read_csv by default makes the first column the index if it's unnamed.
    # Let's be explicit:
    if df.columns[0] == 'label':
        train_x_np = df.iloc[:, 1:].values.astype(float)
    else: # Assuming first col is an index, and 'label' is the second, actual data starts from third
          # Or more robustly, drop 'label' by name if it exists, then take rest
        if 'label' in df.columns:
            train_x_np = df.drop('label', axis=1).values.astype(float)
        else: # if no 'label' column, assume all except potential first index are genotypes
            train_x_np = df.iloc[:, 1:].values.astype(float)


    # Convert to R matrix for R functions
    # R's as.matrix will handle this. For rpy2, convert numpy array.
    # R by default reads dataframes with stringsAsFactors=TRUE, but for numeric data it's fine.
    # The -1 in R's train[-1] refers to removing the first column by position.
    # So if protein.train.nan-101.csv has 'label' as the first column,
    # R's train[-1] means "all columns except 'label'".
    # If the CSV has an unnamed index, then 'label' is the second, and R's train[-1] will remove the unnamed index.
    # Let's ensure our train_x_np matches R's expectation.
    # demo.R: train <- read.csv('data/protein.train.nan-101.csv'); train_x <- as.matrix(train[-1])
    # This means R takes all columns *except the first one it reads*.
    # If pandas read_csv makes the first CSV column an index, then df.values would be all data cols.
    # If the first column is 'Unnamed: 0', pandas might use it as index.
    # Let's re-read the CSV to be sure of the structure.
    try:
        df_check = pd.read_csv('data/protein.train.nan-101.csv')
    except FileNotFoundError:
        df_check = pd.read_csv('../data/protein.train.nan-101.csv')

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


def test_A_mat_default(train_data_nan_101, rrblup_r_package): # Added fixture
    # Python execution (uses Python's default min_MAF, max_missing)
    A_py = A_mat(train_data_nan_101["python"])

    # R execution (uses R's default min.MAF=0.05, max.missing=0.5)
    # This test compares behavior when both use their respective, potentially different, defaults for filtering.
    Amat_r_obj = rrblup_r_package.A_mat(train_data_nan_101["r"]) # Use fixture
    Amat_r = np.array(Amat_r_obj)

    assert A_py.shape == Amat_r.shape, "Default: Output matrices shapes differ"
    # Note: This assertion might fail if default filtering differs significantly.
    # The purpose is to characterize the Python default against R's default.
    # For closer comparisons of specific algorithms (EM, REG etc.), MAF/missing will be aligned in other tests.
    assert np.allclose(A_py, Amat_r, atol=TOLERANCE), "Default: Output matrices values differ significantly"


def test_A_mat_em_impute(train_data_nan_101, rrblup_r_package): # Added fixture
    # Align min_MAF and max_missing to R's defaults for this specific test
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5

    # Python execution
    A_py_em = A_mat(train_data_nan_101["python"], 
                    min_MAF=R_MIN_MAF, 
                    max_missing=R_MAX_MISSING, 
                    impute_method='EM')

    # R execution
    Amat_r_em_obj = rrblup_r_package.A_mat(train_data_nan_101["r"],  # Use fixture
                                   min_MAF=R_MIN_MAF, 
                                   max_missing=R_MAX_MISSING, 
                                   impute_method='EM')
    Amat_r_em = np.array(Amat_r_em_obj)

    assert A_py_em.shape == Amat_r_em.shape, "EM Impute: Output matrices shapes differ"
    assert np.allclose(A_py_em, Amat_r_em, atol=TOLERANCE), "EM Impute: Output matrices values differ significantly"


def test_A_mat_mean_impute_return_imputed(train_data_nan_101, rrblup_r_package): # Added fixture
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5

    # Python execution
    A_py_mean, X_imp_py = A_mat(train_data_nan_101["python"], 
                                min_MAF=R_MIN_MAF, 
                                max_missing=R_MAX_MISSING, 
                                impute_method='mean', 
                                return_imputed=True)

    # R execution
    # R returns a list: result$A, result$imputed
    r_result_mean = rrblup_r_package.A_mat(train_data_nan_101["r"],  # Use fixture
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


def test_A_mat_shrink_ej(train_data_nan_101, rrblup_r_package): # Added fixture
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5

    # Python execution (shrink=True defaults to EJ)
    A_py_ej = A_mat(train_data_nan_101["python"], 
                    min_MAF=R_MIN_MAF, 
                    max_missing=R_MAX_MISSING, 
                    shrink=True)

    # R execution (shrink=TRUE also defaults to EJ)
    Amat_r_ej_obj = rrblup_r_package.A_mat(train_data_nan_101["r"],  # Use fixture
                                   min_MAF=R_MIN_MAF, 
                                   max_missing=R_MAX_MISSING, 
                                   shrink=True)
    Amat_r_ej = np.array(Amat_r_ej_obj)

    assert A_py_ej.shape == Amat_r_ej.shape, "Shrink EJ: Output matrices shapes differ"
    assert np.allclose(A_py_ej, Amat_r_ej, atol=TOLERANCE), "Shrink EJ: Output matrices values differ significantly"


def test_A_mat_shrink_reg(train_data_nan_101, rrblup_r_package): # Added fixture
    R_MIN_MAF = 0.05
    R_MAX_MISSING = 0.5
    N_QTL = 100 # As per demo.R
    N_ITER = 5  # As per demo.R

    # Python execution
    A_py_reg = A_mat(train_data_nan_101["python"], 
                     min_MAF=R_MIN_MAF, 
                     max_missing=R_MAX_MISSING, 
                     shrink="REG", 
                     n_qtl=N_QTL, 
                     n_iter=N_ITER)

    # R execution
    shrink_list_r = ListVector({'method': "REG", 'n.qtl': N_QTL, 'n.iter': N_ITER})
    
    Amat_r_reg_obj = rrblup_r_package.A_mat(train_data_nan_101["r"],  # Use fixture
                                    min_MAF=R_MIN_MAF, 
                                    max_missing=R_MAX_MISSING, 
                                    shrink=shrink_list_r)
    Amat_r_reg = np.array(Amat_r_reg_obj)

    assert A_py_reg.shape == Amat_r_reg.shape, "Shrink REG: Output matrices shapes differ"
    assert np.allclose(A_py_reg, Amat_r_reg, atol=TOLERANCE), "Shrink REG: Output matrices values differ significantly"
