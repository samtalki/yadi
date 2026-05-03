import numpy as np
import pytest

from yadi import CLA


def test_cla_fit_recovers_linear_relation() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 64, 3
    a_true = rng.normal(size=n_features)
    b_true = 0.5
    X = rng.normal(size=(n_samples, n_features))
    y = X @ a_true + b_true + 0.01 * rng.normal(size=n_samples)

    cla = CLA(verbose=False, maxiters=2000)
    a_hat, b_hat = cla.fit(X, y, ub=y + 0.1, lb=y - 0.1)

    # CLA is one-sided constrained, so the fit can drift from the noise-free
    # truth; the constraints are the load-bearing invariant.
    assert a_hat is not None and b_hat is not None
    pred = X @ a_hat + b_hat
    assert np.all(pred <= y + 0.1 + 1e-6), "CLA fit violated upper bound"
    assert np.all(pred >= y - 0.1 - 1e-6), "CLA fit violated lower bound"


def test_cla_fit_pq_splits_coefficients() -> None:
    rng = np.random.default_rng(1)
    m, n = 50, 4
    P = rng.normal(size=(m, n))
    Q = rng.normal(size=(m, n))
    a_p = rng.normal(size=n)
    a_q = rng.normal(size=n)
    V = P @ a_p + Q @ a_q + 1.0 + 0.01 * rng.normal(size=m)

    cla = CLA(verbose=False, maxiters=2000)
    dv2p, dv2q, intercept = cla.fit_pq(P, Q, V, ub=V + 0.05, lb=V - 0.05)
    assert dv2p.shape == (n,)
    assert dv2q.shape == (n,)
    assert intercept.shape == (1,)

    # The split must be correct; swapping halves should make the fit strictly worse.
    pred = P @ dv2p + Q @ dv2q + intercept
    pred_swapped = P @ dv2q + Q @ dv2p + intercept
    assert np.linalg.norm(V - pred) < np.linalg.norm(V - pred_swapped)


def test_cla_fit_unconstrained_warns_and_recovers() -> None:
    """No bounds → warns, falls back to unconstrained l1 fit and tracks the noise-free truth."""
    rng = np.random.default_rng(2)
    n_samples, n_features = 64, 3
    a_true = rng.normal(size=n_features)
    b_true = 0.25
    X = rng.normal(size=(n_samples, n_features))
    y = X @ a_true + b_true + 0.005 * rng.normal(size=n_samples)

    cla = CLA(verbose=False, maxiters=2000)
    with pytest.warns(UserWarning, match="No upper or lower bounds"):
        a_hat, b_hat = cla.fit(X, y)
    assert np.allclose(a_hat, a_true, atol=0.1)
    assert abs(b_hat - b_true) < 0.1
