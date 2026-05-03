"""Conservative linear approximation via l1 regression with optional inequality constraints."""

import warnings

import cvxpy as cp
import numpy as np


class CLA:
    def __init__(
        self,
        verbose: bool = True,
        maxiters: int = 1000,
        solver: str = "SCS",
    ) -> None:
        self.verbose = verbose
        self.maxiters = maxiters
        self.solver = solver
        self.params: tuple[np.ndarray | None, np.ndarray | None] = (None, None)
        self.intercept: np.ndarray | None = None
        self.dvp: np.ndarray | None = None
        self.dvq: np.ndarray | None = None
        self.dv2p: np.ndarray | None = None
        self.dv2q: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        ub: np.ndarray | None = None,
        lb: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve `min ||Y - (X a + b)||_1` subject to `lb <= X a + b <= ub` and return `(a, b)`."""
        m, n = X.shape
        assert m == Y.shape[0], "X and Y must have the same number of rows"

        a = cp.Variable(n)
        b = cp.Variable(1)

        constraints = []
        if ub is not None or lb is not None:
            if ub is not None:
                constraints.append(X @ a + b <= ub)
            if lb is not None:
                constraints.append(X @ a + b >= lb)
        else:
            warnings.warn("No upper or lower bounds specified. Using unconstrained approximation.")

        obj = cp.norm(Y - (X @ a + b), 1)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=self.solver, verbose=self.verbose, max_iters=self.maxiters)

        if a.value is None or b.value is None:
            raise RuntimeError(f"CLA solver did not return a value (status: {prob.status}).")
        return a.value, b.value

    def fit_pq(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        V: np.ndarray,
        ub: np.ndarray | None = None,
        lb: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit V against `[P | Q]` and split the coefficients into `(dv2p, dv2q, intercept)`."""
        m, n = P.shape
        mQ, _ = Q.shape
        assert m == mQ, "P and Q must have the same number of rows"
        assert m == V.shape[0], "V must have the same number of rows as P and Q"

        a, b = self.fit(np.hstack((P, Q)), V, ub=ub, lb=lb)
        self.dv2p = a[:n]
        self.dv2q = a[n:]
        self.intercept = b
        return self.dv2p, self.dv2q, self.intercept
