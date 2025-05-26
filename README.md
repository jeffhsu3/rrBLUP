# pyrrblup Python Package

rrBLUP is a R package used for genomic prediction with the rrBLUP linear mixed model ([Endelman 2011](https://acsess.onlinelibrary.wiley.com/doi/full/10.3835/plantgenome2011.08.0024)). This repository contains a Python implementation of its core functions.

This project is structured as a Python package, allowing for easy installation and use in other Python projects.

## Requirements

### Python Dependencies

The package requires Python 3.7 or higher. The main Python dependencies are:

- numpy
- scikit-learn
- pandas
- scipy

For testing:
- rpy2

A full list of Python dependencies can be found in the `pyproject.toml` file.

### Result comparison with R: R, rrBLUP and asrmle R Package

You must have R installed on your system. You can download R from [CRAN (The Comprehensive R Archive Network)](https://cran.r-project.org/).

Once R is installed, you need to install the `rrBLUP` package within your R environment. You can do this by opening an R console and running the following command:

```R
install.packages("rrBLUP")
install.packages("asreml")
install.packages("tidyverse")
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
conda create -n pyrrblup -c conda-forge python=3.12 r-base rpy2 numpy scikit-learn r-rrblup pandas scipy
conda activate pyrrblup
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

## License

This project is released under the Apace License Version 2.0. See the `pyproject.toml` file for details and `LICENSE`.

## TODO

- [ ] Improve tolerance comparison for `test_A_mat_shrink_ej` test to better handle numerical precision differences
- [ ] Fix and implement `reg` shrinkage method in `A_mat` function to match R rrBLUP behavior

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md`.
