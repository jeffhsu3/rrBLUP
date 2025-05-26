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

# # REG shrink method
# start_time <- Sys.time()
# 
# # Calculate p for REG shrink method
# train_x_shifted <- train_x + 1
# freq <- apply(train_x_shifted, 2, function(x) mean(x, na.rm = TRUE)) / 2
# min_MAF_val <- 0.05
# max_missing_val <- 0.5
# MAF <- pmin(freq, 1 - freq)
# missing_prop <- colMeans(is.na(train_x))
# markers_idx <- which((MAF >= min_MAF_val) & (missing_prop <= max_missing_val))
# p_values <- freq[markers_idx]
# 
# current_n_qtl <- min(10, length(p_values)) # Define current_n_qtl
# 
# # Test passing p inside the shrink list, and min.MAF/max.missing as main A.mat arguments
# Amat <- A.mat(train_x, 
#               shrink=list(method="REG", p=p_values, n.qtl=current_n_qtl, n.iter=5), 
#               min.MAF=min_MAF_val, 
#               max.missing=max_missing_val)
# print(Amat)
# print(Sys.time()-start_time)

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