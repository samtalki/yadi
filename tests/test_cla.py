import numpy as np

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

    assert a_hat is not None and b_hat is not None
    assert np.allclose(a_hat, a_true, atol=0.1)
    assert abs(b_hat[0] - b_true) < 0.1
