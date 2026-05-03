"""Linear voltage sensitivity models from data: regression with optional excitation filtering."""

import numpy as np
from numba import jit, njit  # noqa: F401  (jit kept for d_diag/fill_matrix decorators below)


def solve_lsq_filtered(X, v, delta=None, lambd=None, absolute_value=True, filter_var="p"):
    """Least squares fit `s = pinv(X) @ v`; optionally drop low-excitation rows below `delta`."""
    if delta is not None:
        X, v = excitation_filter(X, v, delta, absolute_value=absolute_value, filter_var=filter_var)
    if X.shape[0] < X.shape[1]:
        raise ValueError(
            f"least-squares fit is under-determined: {X.shape[0]} rows, {X.shape[1]} features"
        )
    s = pinv(X, lambd) @ v
    # numba's np.linalg.inv returns NaN/inf instead of raising on singular X^T X.
    if not np.isfinite(s).all():
        raise np.linalg.LinAlgError(
            "non-finite least-squares solution; X may be rank-deficient (try lambd > 0)"
        )
    return s


@njit
def pinv(X, lambd=None):
    """Left pseudoinverse `(X^T X + lambd I)^-1 X^T`; lambd=None means plain pinv."""
    X_pinv = gram(X, lambd) @ X.T
    return X_pinv


@njit
def gram(X, lambd=None):
    """Inverse Gram matrix with optional l2 regularization."""
    if lambd is not None:
        return np.linalg.inv(X.T @ X + lambd * np.eye(X.shape[1]))
    else:
        return np.linalg.inv(X.T @ X)


def excitation_filter(
    X, v, quantile, absolute_value=False, filter_var="p", return_idx_filtered=False
):
    """Drop rows of (X, v) whose excitation falls at/below `quantile` of `filter_var` ('p'/'x'/'v')."""
    if filter_var not in ("p", "x", "v"):
        raise ValueError(f"filter_var={filter_var!r} must be one of 'p', 'x', 'v'.")

    X_test = X[:, 0] if X.ndim > 1 else X
    if filter_var in ("p", "x"):
        target = np.abs(X_test) if absolute_value else X_test
        threshold = np.quantile(a=target, q=quantile)
        idx_filtered = np.arange(X.shape[0])[target > threshold]
    else:
        target = np.abs(v) if absolute_value else v
        threshold = np.quantile(a=target, q=quantile)
        idx_filtered = np.arange(v.shape[0])[target > threshold]
    X_filtered = X[idx_filtered]
    v_filtered = v[idx_filtered]
    if return_idx_filtered:
        return X_filtered, v_filtered, idx_filtered
    else:
        return X_filtered, v_filtered


def make_study_slice(study_idx, slice_range, n_nodes=1379):
    """Centered neighborhood slice of width `2*slice_range+1` around `study_idx` in [0, n_nodes)."""
    if slice_range == 0:
        study_slice = study_idx
        local_sp_idx = 0
        return study_slice, local_sp_idx
    else:
        if study_idx == 0:
            study_slice = np.s_[0 : 2 * slice_range + 1]
            local_sp_idx = 0
        elif study_idx <= slice_range and slice_range > 1:
            study_slice = np.s_[0 : 2 * slice_range + 1]
            local_sp_idx = study_idx
        elif study_idx == n_nodes - 1:
            study_slice = np.s_[n_nodes - 2 * slice_range : n_nodes]
            local_sp_idx = study_idx - (n_nodes - 2 * slice_range)
        elif study_idx >= n_nodes - slice_range:
            study_slice = np.s_[n_nodes - 2 * slice_range : n_nodes]
            local_sp_idx = study_idx - (study_idx - slice_range)
        else:
            study_slice = np.s_[study_idx - slice_range : study_idx + slice_range + 1]
            local_sp_idx = slice_range
        return study_slice, local_sp_idx


@jit
def d_diag(X, V, d=3, lambd=2.5e-5, fill_S_matrix=False, fit_offset=False):
    """Structured d-diagonal sensitivity coefficients (default tri-diagonal)."""
    d_M, d_N = X.shape

    # Lower diagonal (alpha), diagonal (beta) and upper diagonal (gamma) coefficients
    # Follow convention that goes lower->diag->upper
    coef = np.zeros((d, d_N))

    # alpha_1 = gamma_n = 0upper and lower diagonals are zero
    coef[0, 0] = 0
    coef[2, -1] = 0
    for i, v_i in enumerate(V.T):
        x_i = X[:, i]
        if not fit_offset:
            if i == 0:  # At beginning, only contain upper tri diagonal
                coef[1:, i] = pinv(np.vstack((x_i.T, X[:, i + 1].T)).T, lambd) @ v_i
            elif i == d_N - 1:  # At the end, only contain lower tri diagonal
                coef[:2, i] = pinv(np.vstack((X[:, i - 1].T, x_i.T)).T, lambd) @ v_i
            else:  # contain all
                coef[:, i] = pinv(np.vstack((X[:, i - 1].T, x_i.T, X[:, i + 1].T)).T, lambd) @ v_i
        elif fit_offset:
            if i == 0:  # At beginning, only contain upper tri diagonal
                coef_i = pinv(np.vstack([x_i.T, X[:, i + 1].T, np.ones(len(x_i)).T]).T, lambd) @ v_i
                print(coef_i.shape)
                coef[1:, i] = coef_i[:-1]
            elif i == d_N - 1:  # At the end, only contain lower tri diagonal
                coef_i = pinv(np.vstack([X[:, i - 1].T, x_i.T, np.ones(len(x_i)).T]).T, lambd) @ v_i
                coef[:2, i] = coef_i[:-1]
            else:  # contain all
                coef_i = (
                    pinv(
                        np.vstack([X[:, i - 1].T, x_i.T, X[:, i + 1].T, np.ones(len(x_i)).T]).T,
                        lambd,
                    )
                    @ v_i
                )
                coef[:, i] = coef_i[:-1]

    if fill_S_matrix:
        return fill_matrix(coef)
    else:
        return coef


@jit
def fill_matrix(S_diag):
    """Assemble a tri-diagonal matrix from `S_diag` rows (lower, main, upper)."""
    return np.diag(S_diag[1, :]) + np.diag(S_diag[2, :-1], 1) + np.diag(S_diag[0, 1:], -1)
