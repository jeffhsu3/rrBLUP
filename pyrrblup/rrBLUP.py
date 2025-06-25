import os
import random

import numpy as np
import pandas as pd
from einops import rearrange
from scipy.optimize import fminbound
from sklearn.preprocessing import scale


def A_mat(
    X,
    min_MAF=None,
    max_missing=None,
    impute_method="mean",
    tol=0.02,
    shrink=False,
    n_qtl=100,
    n_iter=5,
    return_imputed=False,
    seed=None,
):
    """
    Calculates the realized additive relationship matrix.

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
    """

    def substitude_missing(A, B, missing_index):
        for i in range(len(missing_index)):
            for j in range(len(missing_index)):
                A[missing_index[i], missing_index[j]] = B[i, j]
        return A

    def shrink_coeff(i, W, n_qtl, p, seed=None):
        if seed is not None:
            random.seed(seed)
        m = W.shape[1]
        n = W.shape[0]
        qtl = np.array(random.sample(range(m), n_qtl))
        reqtl = np.setdiff1d(range(m), qtl)
        A_mark = (W[:, reqtl] @ W[:, reqtl].T) / np.sum(2 * p[reqtl] * (1 - p[reqtl]))
        A_qtl = (W[:, qtl] @ W[:, qtl].T) / np.sum(2 * p[qtl] * (1 - p[qtl]))
        x = A_mark - np.mean(np.diag(A_mark)) * np.eye(n)
        y = A_qtl - np.mean(np.diag(A_qtl)) * np.eye(n)
        x = rearrange(x, "i j -> (i j)")
        y = rearrange(y, "i j -> (i j)")
        return 1 - np.cov(y, x) / np.var(x, ddof=1)

    def impute_EM(W, cov_mat, mean_vec):
        n = W.shape[0]
        m = W.shape[1]
        S = np.zeros((n, n))
        for i in range(m):
            Wi = W[:, i].reshape(-1, 1)
            missing_index = np.argwhere(np.isnan(Wi))[:, 0]
            if len(missing_index) > 0:
                not_NA = np.setdiff1d(range(n), missing_index)
                Bt = np.linalg.solve(
                    cov_mat.take(not_NA, 0).take(not_NA, 1),
                    cov_mat.take(not_NA, 0).take(missing_index, 1),
                )
                Wi[missing_index] = mean_vec[missing_index] + (
                    Bt.T @ (Wi[not_NA] - mean_vec[not_NA])
                )
                C = cov_mat.take(missing_index, 0).take(missing_index, 1) - (
                    cov_mat.take(not_NA, 0).take(missing_index, 1).T @ Bt
                )
                D = Wi @ Wi.T
                tmp = D.take(missing_index, 0).take(missing_index, 1) + C
                D = substitude_missing(D, tmp, missing_index)
                W[:, i] = Wi.reshape(-1)
            else:
                D = Wi @ Wi.T
            S = S + D
        W_imp = W
        return [S, W_imp]

    def cov_W_shrink(W):
        n = W.shape[0]
        m = W.shape[1]
        Z = np.transpose(scale(np.transpose(W), with_std=False))
        Z2 = np.multiply(Z, Z)
        S = Z @ Z.T / m
        target = np.mean(np.diag(S)) * np.eye(n)
        var_S = (Z2 @ Z2.T) / (m * m) - (S * S) / m
        b2 = np.sum(var_S)
        d2 = np.sum((S - target) * (S - target))
        delta = max(0, min(1, np.min(b2 / d2)))
        print("Shrinkage intensity:", format(delta, ".2f"))
        return target * delta + (1 - delta) * S

    if impute_method not in ["mean", "EM"]:
        print("Invalid imputation method.")
        return

    if type(shrink) != str and type(shrink) != bool:
        print("Invalid shrinkage method.")
        return
    elif type(shrink) == str:
        shrink_method = shrink
        shrink = True
        if shrink_method != "REG" and shrink_method != "EJ":
            print("Invalid shrinkage method.")
            return
    else:
        if shrink:
            shrink_method = "EJ"

    n = X.shape[0]
    m = X.shape[1]
    X_shifted = X + 1
    frac_missing = np.zeros(m)
    freq = np.zeros(m)
    MAF = np.zeros(m)
    # :TODO simplify this
    for i in range(m):
        frac_missing[i] = np.sum(np.isnan(X)[:, i]) / n
        freq[i] = np.nanmean(X_shifted[:, i]) / 2
        MAF[i] = min(freq[i], 1 - freq[i])
    missing = max(frac_missing) > 0
    if not min_MAF:
        min_MAF = 1 / (2 * n)
    if not max_missing:
        max_missing = 1 - 1 / (2 * n)
    # :TODO simplify this
    markers = np.intersect1d(
        np.where(MAF >= min_MAF)[0], np.where(frac_missing <= max_missing)[0]
    )
    m = len(markers)
    var_A = 2 * np.mean(freq[markers] * (1 - freq[markers]))
    ones_vector = np.ones((n, 1))
    monomorphic_markers = np.where(freq * (1 - freq) == 0)
    freqmono = freq[monomorphic_markers[0]]
    freqmarkers = freq[markers]
    X[:, monomorphic_markers[0]] = 2 * (ones_vector @ freqmono.reshape(-1, 1).T) - 1
    freq_mat = ones_vector @ freqmarkers.reshape(-1, 1).T
    W = X[:, markers] + 1 - 2 * freq_mat
    if not missing:
        if shrink:
            if shrink_method == "EJ":
                W_mean = np.nanmean(W, axis=1)
                cov_W = cov_W_shrink(W)
                A = (cov_W + (W_mean @ W_mean.T)) / var_A
            else:
                delta = []
                for i in range(n_iter):
                    delta.append(shrink_coeff(i, W, n_qtl, freq_mat[0, :], seed=seed))
                delta = np.nanmean(np.array(delta))
                print("Shrinkage intensity:", format(delta, ".2f"))
                A = (W @ W.T) / var_A / m
                A = (1 - delta) * A + delta * np.mean(np.diag(A)) * np.eye(n)
        else:
            A = (W @ W.T) / var_A / m
        if return_imputed:
            return A, X
        else:
            return A
    else:
        is_nan = np.argwhere(np.isnan(W))
        for i in range(len(is_nan)):
            W[is_nan[i][0], is_nan[i][1]] = 0
        if impute_method.upper() == "EM":
            if m < n:
                print(
                    "Linear dependency among the lines: imputing with mean instead of EM algorithm."
                )
            else:
                mean_vec_new = np.mean(W, axis=1).reshape(-1, 1)
                cov_mat_new = np.cov(np.transpose(W), rowvar=False)
                if np.linalg.matrix_rank(cov_mat_new) < cov_mat_new.shape[0] - 1:
                    print(
                        "Linear dependency among the lines: imputing with mean instead of EM algorithm."
                    )
                else:
                    for i in range(len(is_nan)):
                        W[is_nan[i][0], is_nan[i][1]] = np.nan
                    A_new = (cov_mat_new + (mean_vec_new @ mean_vec_new.T)) / var_A
                    err = tol + 1
                    print("A_mat converging:")
                    while err >= tol:
                        A_old = A_new
                        cov_mat_old = cov_mat_new
                        mean_vec_old = mean_vec_new
                        S, W_imp = impute_EM(W, cov_mat_old, mean_vec_old)
                        mean_vec_new = np.mean(W_imp, axis=1).reshape(-1, 1)
                        cov_mat_new = (S - (mean_vec_new @ mean_vec_new.T) * m) / (
                            m - 1
                        )
                        A_new = (cov_mat_new + (mean_vec_new @ mean_vec_new.T)) / var_A
                        err = np.linalg.norm(A_old - A_new) / n
                        print("{:.3}".format(err))
                    if return_imputed:
                        Ximp = W_imp - 1 + 2 * freq_mat
                        return A_new, Ximp
                    else:
                        return A_new
        if shrink:
            if shrink_method == "EJ":
                W_mean = np.mean(W, axis=1)
                cov_W = cov_W_shrink(W)
                A = (cov_W + W_mean @ W_mean.T) / var_A
            else:
                delta = []
                for i in range(n_iter):
                    delta.append(shrink_coeff(i, W, n_qtl, freq_mat[0, :], seed=seed))
                delta = np.nanmean(np.array(delta))
                print("Shrinkage intensity:", format(delta, ".2f"))
                A = (W @ W.T) / var_A / m
                A = (1 - delta) * A + delta * np.mean(np.diag(A)) * np.eye(n)
        else:
            A = W @ W.T / var_A / m
        if return_imputed:
            Ximp = W - 1 + 2 * freq_mat
            return A, Ximp
        else:
            return A


def mixed_solve(
    y,
    Z=None,
    K=None,
    X=None,
    method="REML",
    bounds=[1e-09, 1e09],
    SE=False,
    return_Hinv=False,
):
    """
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

    return.Hinv [bool, default = False]:
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
    beta.SE [float]:
        Standard error for the fixed effects \beta

    u.SE [float]:
        Standard error for the random effects u

    (When return_Hinv = True)
    Hinv [array]:
        Inverse of H = Z*K*Z' + \lambda*I
    """

    if method not in ["ML", "REML"]:
        print("Invalid maximum-likelihood method.")
        return

    pi = 3.14159
    n = len(y)
    not_nan = np.argwhere(y)[:, 0]
    if X is None:
        p = 1
        X = np.ones((n, 1))
    p = X.shape[1]
    if not p:
        p = 1
        X = X.reshape(-1, 1)
    if Z is None:
        Z = np.eye(n)
    m = Z.shape[1]
    if not m:
        m = 1
        Z = Z.reshape(-1, 1)
    if Z.shape[0] != n:
        print("ERROR: Z.shape[0] != n")
        return
    if X.shape[0] != n:
        print("ERROR: X.shape[0] != n")
        return
    if K is not None:
        if K.shape[0] != m:
            print("ERROR: K.shape[0] != m")
            return
        if K.shape[1] != m:
            print("ERROR: K.shape[1] != m")
            return
    Z = Z[not_nan, :]
    X = X[not_nan, :]
    n = len(not_nan)
    y = y[not_nan, :]
    XtX = X.T @ X
    rank_X = np.linalg.matrix_rank(XtX)
    if rank_X < p:
        print("ERROR: X not full rank")
        return
    XtXinv = np.linalg.inv(XtX)
    S = np.eye(n) - (np.dot(X, XtXinv) @ X.T)
    if n <= m + p:
        spectral_method = "eigen"
    else:
        spectral_method = "cholesky"
        if K is not None:
            K += np.diag([1e-06] * m)
            try:
                B = np.transpose(np.linalg.cholesky(K))
            except:
                print("ERROR: K not positive semi-definite")
                return
    if spectral_method == "cholesky":
        if K is None:
            ZBt = Z
        else:
            ZBt = Z @ B.T
        u, svd_ZBt_d, v = np.linalg.svd(Z, full_matrices=0)
        phi = np.append(svd_ZBt_d * svd_ZBt_d, np.zeros(n - m))
        SZBt = np.dot(S, ZBt)
        try:
            svd_SZBt_u, svd_SZBt_d, v = np.linalg.svd(SZBt, full_matrices=0)
        except:
            svd_SZBt_u, svd_SZBt_d, v = np.linalg.svd(SZBt + 1e-10, full_matrices=0)
        matrix_to_decompose = np.hstack((X, svd_SZBt_u))
        Q_numpy, R_numpy = np.linalg.qr(matrix_to_decompose, mode="complete")
        Q = Q_numpy[:, p:n]
        R = R_numpy[p:m, p:m]
        try:
            ans = np.linalg.solve(np.transpose(R * R), svd_SZBt_d * svd_SZBt_d)
            theta = np.append(ans, np.zeros(n - p - m))
        except:
            spectral_method = "eigen"
    if spectral_method == "eigen":
        offset = np.sqrt(n)
        if K is None:
            Hb = (Z @ Z.T) + offset * np.eye(n)
        else:
            Hb = (np.dot(Z, K) @ Z.T) + offset * np.eye(n)

        eigenvalues, eigenvectors = np.linalg.eig(Hb)

        # Sort the eigenvalues in descending order, and sort the eigenvectors to match
        idx = eigenvalues.argsort()[::-1]
        Hb_system_values = eigenvalues[idx]
        Hb_system_vectors = eigenvectors[:, idx]

        phi = Hb_system_values - offset
        if np.nanmin(phi) < -1e-06:
            print("K not positive semi-definite.")
            return
        U = Hb_system_vectors
        SHbS = np.dot(np.dot(S, Hb), S)
        eigenvalues, eigenvectors = np.linalg.eig(SHbS)

        # Sort the eigenvalues in descending order, and sort the eigenvectors to match
        idx = eigenvalues.argsort()[::-1]
        SHbS_system_values = eigenvalues[idx]
        SHbS_system_vectors = eigenvectors[:, idx]

        theta = SHbS_system_values[0 : (n - p)] - offset
        Q = SHbS_system_vectors[:, 0 : (n - p)]
    omega = Q.T @ y
    omega_sq = omega * omega
    if method == "ML":
        df = n

        def f_ML(Lambda):
            return n * np.log(
                np.nansum(omega_sq.reshape(-1) / (theta + Lambda))
            ) + np.nansum(np.log(phi + Lambda))

        lambda_opt = fminbound(f_ML, bounds[0], bounds[1])
        objective = f_ML(lambda_opt)
        os.system("rm bounds.csv df.csv phi.csv theta.csv omega_sq.csv")
    else:
        df = n - p

        def f_REML(Lambda):
            return (n - p) * np.log(
                np.nansum(omega_sq.reshape(-1) / (theta + Lambda))
            ) + np.nansum(np.log(theta + Lambda))

        lambda_opt = fminbound(f_REML, bounds[0], bounds[1])
        objective = f_REML(lambda_opt)
    Vu_opt = np.nansum(omega_sq.reshape(-1) / (theta + lambda_opt)) / df
    Ve_opt = lambda_opt * Vu_opt
    Hinv = np.dot(U, (np.transpose(U) / (phi + lambda_opt).reshape(-1, 1)))
    W = X.T @ np.dot(Hinv, X)
    beta = np.linalg.solve(W, (X.T @ np.dot(Hinv, y)))
    if K is None:
        KZt = np.transpose(Z)
    else:
        KZt = K @ Z.T
    KZt_Hinv = np.dot(KZt, Hinv)
    u = np.dot(KZt_Hinv, (y - np.dot(X, beta)))
    LL = -0.5 * (objective + df + df * np.log(2 * pi / df))
    if not SE:
        if return_Hinv:
            result = {
                "Vu": Vu_opt,
                "Ve": Ve_opt,
                "beta": beta,
                "u": u,
                "LL": LL,
                "Hinv": Hinv,
            }
            return result
        else:
            result = {"Vu": Vu_opt, "Ve": Ve_opt, "beta": beta, "u": u, "LL": LL}
            return result
    else:
        Winv = np.linalg.inv(W)
        beta_SE = np.sqrt(Vu_opt * np.diag(Winv))
        WW = KZt_Hinv @ KZt.T
        WWW = np.dot(KZt_Hinv, X)
        if K is None:
            u_SE = np.sqrt(
                Vu_opt
                * (np.ones(m) - np.diag(WW) + np.diag((np.dot(WWW, Winv) @ WWW.T)))
            )
        else:
            u_SE = np.sqrt(
                Vu_opt
                * (np.diag(K) - np.diag(WW) + np.diag((np.dot(WWW, Winv) @ WWW.T)))
            )
        if return_Hinv:
            result = {
                "Vu": Vu_opt,
                "Ve": Ve_opt,
                "beta": beta,
                "beta_SE": beta_SE,
                "u": u,
                "u_SE": u_SE,
                "LL": LL,
                "Hinv": Hinv,
            }
            return result
        else:
            result = {
                "Vu": Vu_opt,
                "Ve": Ve_opt,
                "beta": beta,
                "beta_SE": beta_SE,
                "u": u,
                "u_SE": u_SE,
                "LL": LL,
            }
            return result
