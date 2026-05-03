#!/usr/bin/env python
"""
Methods for constructing and manipulating linear approximations of the power flow equations via voltage magnitude sensitivity models.
"""

__author__ = "Samuel Talkington"
__contact__ = "talkington@pm.me"
__copyright__ = "Copyright (c) 2021-Present Samuel Talkington, All Rights Reserved."
__license__ = "MIT"
__date__ = "2022/03/28"
__version__ = "0.0.1"

import numpy as np
from numba import jit, njit


@jit(forceobj=True)
def solve_lsq_filtered(X, v, delta=None, lambd=None, absolute_value=True, filter_var="p"):
    """
    Least squares sensitivity model with percentile filtering

    Parameters:
        X: (m_samples,n_features) array of deviations of power
        v: (m_samples,n_features) Deviations of voltage magnitudes
        delta (float): Absolute excitation percentile filter
        lambd (float): l2 regularization
    Returns:
        S: Linear voltage sensitivity model

    """
    if delta is not None:
        X, v = excitation_filter(X, v, delta, absolute_value=absolute_value, filter_var=filter_var)
    s = pinv(X, lambd) @ v
    return s


@njit
def pinv(X, lambd=None):
    """
    Psuedoinverse with optional l2 regularization

    X_pinv = (X^T X + lambd*I)^{-1} X^T

    """
    X_pinv = gram(X, lambd) @ X.T
    return X_pinv


@njit
def gram(X, lambd=None):
    """
    Gram matrix with optional l2 regularization
    """
    if lambd is not None:
        return np.linalg.inv(X.T @ X + lambd * np.eye(X.shape[1]))
    else:
        return np.linalg.inv(X.T @ X)


@jit(forceobj=True)
def excitation_filter(
    X, v, quantile, absolute_value=False, filter_var="p", return_idx_filtered=False
):
    """
    Filter an AMI power array X (P or Q) and voltage target array v according to power excitation quantiles
    Params:
        X (ndarray): P, Q, or P/Q load array
        v (ndarray): Voltage magnitude array
        quantile (float): Quantile of power injections/perturbations below which observations are discarded.
        filter_var (str): "p"/"x" or "v", whether to filter according to voltage quantiles (v) or power quantiles (X)
        absolute_value (bool): Whether to find the quantile w.r.t. to absolute value of the filter_var or not.
        return_idx_filtered (bool): Whether to return the selected indeces returned from the filter.
    """
    if X.ndim > 1:
        X_test = X[:, 0]
    else:
        X_test = X
    if absolute_value:
        if filter_var == "p" or filter_var == "x":
            X_quantile = np.quantile(a=np.abs(X_test), q=quantile)
            idx_filtered = np.arange(X.shape[0])[np.abs(X_test) > X_quantile]
        elif filter_var == "v":
            v_q = np.quantile(a=np.abs(v), q=quantile)
            idx_filtered = np.arange(v.shape[0])[np.abs(v) > v_q]
    else:
        if filter_var == "p" or filter_var == "x":
            X_quantile = np.quantile(a=X_test, q=quantile)
            idx_filtered = np.arange(X.shape[0])[X_test > X_quantile]
        elif filter_var == "v":
            v_q = np.quantile(a=v, q=quantile)
            idx_filtered = np.arange(v.shape[0])[v > v_q]
    X_filtered = X[idx_filtered]
    v_filtered = v[idx_filtered]
    if return_idx_filtered:
        return X_filtered, v_filtered, idx_filtered
    else:
        return X_filtered, v_filtered


@jit(forceobj=True)
def make_study_slice(study_idx, slice_range, n_nodes=1379):
    """
    Generates a neighborhood slice object over the indeces 1,...,n_nodes for a given node study_idx and width slice_range
    Params:
        study_idx (int): The index of the node under study
        n_nodes (int): The number of nodes in the AMI dataset
        slice_range (int): The width of the neighborhood slice centered around study_idx
    """
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
    """
    Generate structured d-diagonal dynamic mode decomposition/sensitivity matrix. (Default tri-diagonal)
    """
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
    S = np.diag(S_diag[1, :]) + np.diag(S_diag[2, :-1], 1) + np.diag(S_diag[0, 1:], -1)
    return S
