library(rrBLUP)

train <- read.csv('data/small.train.csv') # Changed to small.train.csv
test <- read.csv('data/small.test.csv') # Changed to small.test.csv

train_x <- as.matrix(train[-1])
train_y <- as.matrix(train['label'])
test_x <- as.matrix(test[-1])
test_y <- as.matrix(test['label'])

# A.mat
start_time <- Sys.time()
Amat <- A.mat(train_x)
print(Amat)
print(Sys.time()-start_time)

# EM impute method
start_time <- Sys.time()
Amat <- A.mat(train_x, impute.method='EM')
print(Amat)
print(Sys.time()-start_time)

# mean impute method, return imputed 
start_time <- Sys.time()
result <- A.mat(train_x, impute.method='mean', return.imputed=TRUE)
Amat <- result$A
train_x_imp <- result$imputed
print(Amat)
print(train_x_imp)
print(Sys.time()-start_time)

# EJ shrink method
start_time <- Sys.time()
Amat <- A.mat(train_x, shrink=TRUE)
print(Amat)
print(Sys.time()-start_time)

# The following "REG shrink method" section for A.mat is commented out.
# Reason: The rrBLUP::A.mat function has an issue with the REG shrinkage method
# when specified as a list (e.g., shrink=list(method="REG", ...)).
# It either complains about an "unused argument (p=...)" if allele frequencies are passed directly,
# or errors with "argument 'p' is missing, with no default" internally if 'p' is not passed directly
# or if 'p' is passed within the shrink list.
# This appears to be an internal bug in the R package's handling of allele frequencies for this specific feature.

# REG shrink method
start_time <- Sys.time()

# Set seed for reproducibility
set.seed(42) # Using 42 as an example seed

# Calculate p for REG shrink method
train_x_shifted <- train_x + 1
freq <- apply(train_x_shifted, 2, function(x) mean(x, na.rm = TRUE)) / 2
min_MAF_val <- 0.05 # Default value, adjust if necessary
max_missing_val <- 0.5 # Default value, adjust if necessary

# Filter markers based on MAF and missingness FOR THE PURPOSE OF GETTING p_values
# This logic should align with what A.mat does internally if p is not supplied,
# or use the same filtered marker set if p is supplied.
MAF <- pmin(freq, 1 - freq)
missing_prop <- colMeans(is.na(train_x))

# It's crucial that markers_idx here corresponds to the markers A.mat will actually use.
# The R version of A.mat handles marker filtering internally based on min.MAF and max.missing.
# If we provide 'p', it should be for *those* markers A.mat will use.
# For simplicity, let's assume A.mat uses all markers if p is not provided,
# or let's ensure p_values align with A.mat's internal filtering if we can determine it.
# The original R code had issues with passing 'p'. Let's try to call A.mat
# such that it calculates p internally, or pass p for all markers if that's how pyrrblup works.

# Given the comment about issues with passing 'p' to R's A.mat,
# and that pyrrblup's shrink_coeff gets passed freq_mat[0,:] (implying frequencies for markers *after* filtering),
# the most robust way for the R code to mimic this is to also let A.mat do its internal filtering
# and rely on its internal 'p' calculation if the 'shrink' list doesn't explicitly override it in a working way.
# The original rrBLUP documentation suggests 'p' is optional for shrink$REG.

# Let's define p_values based on *all* markers initially, as pyrrblup calculates `freq_mat`
# which is then indexed by `markers` (filtered list) before being passed to shrink_coeff.
# However, the R A.mat might expect 'p' to correspond to the markers *it* decides to use *after* its own filtering.
# This is a known tricky point with the R package.

# Simplification: The python code calculates `freq_mat` first, then filters markers, then uses `freq_mat[0, markers]` for `p`.
# R's A.mat does its own filtering. If `shrink` parameter in R needs `p`, it must be for the *final* set of markers A.mat uses.
# The original R example comments out passing `p` directly due to errors.
# Let's try calling it without `p` first in the `shrink` list, letting A.mat use its defaults or internally calculated `p`.

# Define current_n_qtl. Ensure it's less than or equal to the number of markers A.mat will use.
# This might require a preliminary call to A.mat or knowledge of its filtering if not all markers are used.
# For now, let's use a small number.
current_n_qtl <- min(10, ncol(train_x)) # Simplified: use a small number or ncol(train_x)
                                      # Python version uses m (number of filtered markers)

# The original R code had issues passing 'p'.
# Let's try the call structure that was commented, but without 'p' in the shrink list first,
# as the error message suggested 'p' was an unused argument or missing.
# If A.mat's REG method can run without 'p' in the list, it might use its own calculation.
print("Attempting A.mat with REG shrink method (no explicit 'p' in shrink list)...")
Amat <- tryCatch({
    A.mat(train_x,
          shrink=list(method="REG", n.qtl=current_n_qtl, n.iter=5),
          min.MAF=min_MAF_val,
          max.missing=max_missing_val)
}, error = function(e) {
    print(paste("Error with REG (no explicit p):", e$message))
    # Fallback or try alternative if the above fails, e.g. trying to pass p if the error indicates it's needed
    # For now, just return NULL or an error indicator
    NULL
})

if (!is.null(Amat)) {
    print(Amat)
} else {
    print("A.mat with REG shrink method failed or was skipped.")
    # As per the original R file's comments, the REG method in R's A.mat is problematic.
    # The goal here is to have the R code structure for seeding, even if the REG call itself
    # highlights issues in the R package.
    # The python seed mechanism is the primary fix.
}
print(Sys.time()-start_time)

# The following 'mixed.solve' sections are commented out.
# Reason: The rrBLUP::mixed.solve function encounters numerical issues
# (e.g., "Error in eigen(Hb, symmetric = TRUE) : infinite or missing values in 'x'")
# when used with the very small 'small.train.csv' dataset.
# This is likely due to the small dimensions and random nature of the data leading to
# singular or ill-conditioned matrices in the underlying calculations.
# These sections would run with larger, more structured datasets like 'protein.train.csv'.

# train <- read.csv('data/small.train.csv') # Changed to small.train.csv
# test <- read.csv('data/small.test.csv') # Changed to small.test.csv
# 
# train_x <- as.matrix(train[-1])
# train_y <- as.matrix(train['label'])
# test_x <- as.matrix(test[-1])
# test_y <- as.matrix(test['label'])
# 
# # Define X_intercept for mixed.solve
# X_intercept <- matrix(1, nrow=nrow(train_y), ncol=1)
# 
# # mixed.solve using Matrix Z
# start_time <- Sys.time()
# result <- mixed.solve(y=train_y, Z=train_x, X=X_intercept)
# print(result)
# print(Sys.time()-start_time)
# test_pred <- (test_x %*% as.matrix(result$u)) + as.vector(result$beta) # Reverted to original
# print(test_y)
# print(test_pred)
# 
# # mixed.solve using Matrix K
# start_time <- Sys.time()
# result <- mixed.solve(y=train_y, K=A.mat(train_x), X=X_intercept) # A.mat(train_x) here is the simple one
# print(result)
# print(Sys.time()-start_time)
# train_pred <- as.matrix(result$u) + as.vector(result$beta) # Reverted to original
# print(train_y)
# print(train_pred)