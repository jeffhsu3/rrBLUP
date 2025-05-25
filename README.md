# pyrrblup Python Package

rrBLUP is a R package used for genomic prediction with the rrBLUP linear mixed model ([Endelman 2011](https://acsess.onlinelibrary.wiley.com/doi/full/10.3835/plantgenome2011.08.0024)). This repository contains a Python implementation of its core functions.

This project is structured as a Python package, allowing for easy installation and use in other Python projects.

## Requirements

### Python Dependencies

The package requires Python 3.7 or higher. The main Python dependencies are:

- numpy==1.24.0
- scikit-learn==1.2.0
- pandas==1.5.2
- scipy==1.9.3
- rpy2==3.5.6

A full list of Python dependencies can be found in the `pyproject.toml` file.

### System Dependencies: R and rrBLUP R Package

**Important**: This package, particularly its testing suite and certain functionalities (like `mixed_solve` which relies on R for its core computation via `rpy2`), has a system dependency on **R** and the R package **`rrBLUP`**.

You must have R installed on your system. You can download R from [CRAN (The Comprehensive R Archive Network)](https://cran.r-project.org/).

Once R is installed, you need to install the `rrBLUP` package within your R environment. You can do this by opening an R console and running the following command:

```R
install.packages("rrBLUP")
```

Please ensure R is correctly installed and the `rrBLUP` package is available in the R environment that `rpy2` will interact with, especially if you are using virtual environments or multiple R versions.

## Installation

To install the `pyrrblup` Python package, navigate to the root directory of this repository (where `pyproject.toml` is located) and run:

```bash
pip install .
```

For development, you can install it in editable mode:

```bash
pip install -e .
```

We suggest creating a new virtual environment for a clean installation:

```bash
conda create -n pyrrblup python=3.8 # Or your preferred Python version >= 3.7
conda activate pyrrblup
# Ensure R and the R package rrBLUP are installed as described in "System Dependencies"
pip install .
```

## Data

We use the genomic dataset with protein as the trait from SoyNAM ([soybase.org/SoyNAM/index.php](https://soybase.org/SoyNAM/index.php)), found in `data` folder. The default coding format in SoyNAM is : **-1** for the missing value, **0** for 0/0 genotype, **1** for 0/1 genotype and **2** for 1/1 genotype. However, rrBLUP R package requires input genotype matrix coded as **-1**,**0**,**1** for 0/0, 0/1, 1/1 genotype respectively. Therefore, we convert the data coding format in `data/data_convertion.ipynb`.

## Usage

The package provides two main functions: `A_mat` and `mixed_solve`, available from the `pyrrblup` module.

```python
from pyrrblup import A_mat, mixed_solve
import numpy as np

# --- A_mat Usage Example ---

# Example Genotype Data (n_individuals x n_markers)
# Markers coded as -1, 0, 1
# Replace with your actual data
X_genotypes = np.array([
    [1, 0, 1, -1],
    [0, 1, 1, 0],
    [-1, 1, 0, 1],
    [1, -1, -1, 0],
    [0, 0, 1, 1]
])

# Calculate Additive Relationship Matrix (A)
# Using default parameters (mean imputation, no shrinkage)
A = A_mat(X_genotypes)
print("Additive Relationship Matrix (A):")
print(A)

# Example with EM imputation and returning imputed genotypes
A_em, X_imputed_em = A_mat(X_genotypes, impute_method='EM', return_imputed=True)
print("\nAdditive Relationship Matrix (A) with EM imputation:")
print(A_em)
print("\nImputed Genotype Matrix (X_imputed_em):")
print(X_imputed_em)


# --- mixed_solve Usage Example ---

# Example Phenotype Data (n_individuals x 1)
# Replace with your actual data
y_phenotypes = np.array([10.2, 11.5, 9.8, 12.1, 10.5]).reshape(-1, 1)

# Example Genotype Data (n_individuals x n_markers) for Z matrix
X_genotypes_ms = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [-1, 1, 0],
    [1, -1, -1],
    [0, 0, 1]
])

# Ensure y_phenotypes and X_genotypes_ms have same number of individuals (rows)
if y_phenotypes.shape[0] != X_genotypes_ms.shape[0]:
    raise ValueError("Phenotypes and Genotypes must have data for the same number of individuals.")

# Solve mixed model using Z matrix (marker effects as random)
# This assumes K is an identity matrix scaled by Vu
results_with_Z = mixed_solve(y=y_phenotypes, Z=X_genotypes_ms)
print("Results using Z matrix:")
print(f"  Vu (genetic variance): {results_with_Z['Vu']:.4f}")
print(f"  Ve (residual variance): {results_with_Z['Ve']:.4f}")
print(f"  Beta (fixed effects - intercept): {results_with_Z['beta']}")
# print(f"  u (random marker effects): {results_with_Z['u']}") # Can be long

# Solve mixed model using K matrix (e.g., from A_mat)
# K_matrix = A_mat(X_genotypes_ms) # Using the same genotype data for consistency in example
# For this example, let's use a simplified K if X_genotypes_ms is small / has issues with A_mat defaults
# Or, use a pre-computed A from previous example if dimensions match
# For a robust example, let's use the X_genotypes from A_mat example if suitable dimensions
X_for_K = np.array([
    [1, 0, 1, -1], [0, 1, 1, 0], [-1, 1, 0, 1], [1, -1, -1, 0], [0, 0, 1, 1]
]) # 5 individuals
if y_phenotypes.shape[0] == X_for_K.shape[0]:
    K_matrix = A_mat(X_for_K)
    results_with_K = mixed_solve(y=y_phenotypes, K=K_matrix)
    print("\nResults using K matrix:")
    print(f"  Vu (genetic variance): {results_with_K['Vu']:.4f}")
    print(f"  Ve (residual variance): {results_with_K['Ve']:.4f}")
    print(f"  Beta (fixed effects - intercept): {results_with_K['beta']}")
    # print(f"  u (random individual effects): {results_with_K['u']}") # Can be long
else:
    print("\nSkipping mixed_solve with K example as dimensions do not match for provided sample data.")

```

Detailed descriptions of the function parameters are provided below.

### `A_mat(X, min_MAF=None, max_missing=None, impute_method="mean", tol=0.02, shrink=False, n_qtl=100, n_iter=5, return_imputed=False)`

Parameters:
-----------
X [array]:
    Matrix of unphased genotypes for n lines and m biallelic markers, coded as {-1,0,1}
    
min_MAF [float, default = None]:
    Minimum minor allele frequency, default removes monomorphic markers
    
max_missing [float, default = None]:
    Maximum proportion of missing data, default removes completely missing markers
    
impute_method [str("mean" or "EM"), default = "mean"]:
    Method of genotype imputation, there are only two options, "mean" imputes with the mean of each marker and
    "EM" imputes with an EM algorithm
    
tol [float, default = 0.02]:
    Convergence criterion for the EM algorithm
    
shrink [Union[bool, str("EJ" or "REG")], default = False]:
    Method of shrinkage estimation, default disable shrinkage estimation; If string, there are only two options,
    "EJ" uses EJ algorithm described in Endelman and Jannink (2012) and "REG" uses REG algorithm described in
    Muller et al. (2015); If True, uses EJ algorithm

n_qtl [int, default = 100]:
    Number of simulated QTL for the REG algorithm

n_iter [int, default = 5]:
    Number of iterations for the REG algorithm

return_imputed [bool, default = False]:
    Whether to return the imputed marker matrix


Returns:
--------
A [array]:
    Additive genomic relationship matrix (n * n)

(When return_imputed = True)
imputed [array]:
    Imputed X matrix

### `mixed_solve(y, Z=None, K=None, X=None, method="REML", bounds=[1e-09, 1e+09], SE=False, return_Hinv=False)`

Parameters:
-----------
y [array]:
    Vector of observations for n lines and 1 observation

Z [array, default = None]:
    Design matrix of the random effects for n lines and m random effects, default to be the identity matrix

K [array, default = None]:
    Covariance matrix of the random effects, if not passed, assumed to be the identity matrix

X [array, default = None]:
    Design matrix of the fixed effects for n lines and p fixed effects, which should be full column rank,
    default to be a vector of 1's

method [str("ML" or "REML"), default = "REML"]:
    Method of maximum-likelihood used in algorithm, there are only two options, "ML" uses full maximum-likelihood
    method and "REML" uses restricted maximum-likelihood method

bounds [list, default = [1e-09, 1e+09]]:
    Lower and upper bound for the ridge parameter

SE [bool, default = False]:
    whether to calculate and return standard errors

return_Hinv [bool, default = False]:  # Corrected from return.Hinv
    whether to return the inverse of H = Z*K*Z' + \lambda*I, which is useful for GWAS


Returns:
--------
Vu [float]:
    Estimator for the marker variance \sigma^2_u

Ve [float]:
    Estimator for the residual variance \sigma^2_e

beta [array]:
    BLUE for the fixed effects \beta

u [array]:
    BLUP for the random effects u

LL [float]:
    maximized log-likelihood

(When SE = True)
beta_SE [float]: # Corrected from beta.SE
    Standard error for the fixed effects \beta

u_SE [float]: # Corrected from u.SE
    Standard error for the random effects u

(When return_Hinv = True)
Hinv [array]:
    Inverse of H = Z*K*Z' + \lambda*I

## License

This project is released under the MIT License. See the `pyproject.toml` file for details (or a separate `LICENSE` file if added).

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` file (if available) or open an issue on the project's bug tracker.
(Placeholder for project URLs will be updated in `pyproject.toml`)
