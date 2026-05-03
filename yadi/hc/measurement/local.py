"""Localized hosting capacity from linear voltage sensitivity models."""

import numpy as np

import yadi.sens.data_linear as linear
import yadi.utils as utils


def pq_fit(p, q, delta, lambd, absolute_value, filter_var):
    """Explain reactive power deviations via active power deviations; returns [q/p, q0]."""
    X = np.vstack((p.T, np.ones(p.shape[0]).T)).T
    return linear.solve_lsq_filtered(
        X, q, delta=delta, lambd=lambd, absolute_value=absolute_value, filter_var=filter_var
    )


def estimate_pf(p, q, delta=None, lambd=None, absolute_value=None, filter_var=None):
    """Estimate power factor as cos(arctan(dq/dp))."""
    s_coef = pq_fit(p, q, delta, lambd, absolute_value=absolute_value, filter_var=filter_var)
    return np.cos(np.arctan(s_coef[0]))


def analyze_hosting_capacity(
    V_test, V_train, Q, P, delta, lambd, filter_var, absolute_value, pf_inj=None
):
    """Per-node hosting capacity via excitation filtering and least squares regression."""
    HC, v_S, pq_S, pf, X_filtered, v_filtered = [], [], [], [], [], []

    for study_idx, (p, q, v) in enumerate(zip(P.T, Q.T, V_train.T)):
        X = np.vstack((p.T, q.T, np.ones(p.shape[0]).T)).T
        X, v = utils.drop_nan_rows(X, v)
        DX, dv = utils.diff(X, v)
        DX, dv = linear.excitation_filter(
            DX, dv, quantile=delta, absolute_value=absolute_value, filter_var=filter_var
        )
        v_filtered.append(dv)
        X_filtered.append(DX)

        v_S.append(linear.pinv(DX, lambd) @ dv)
        pq_S.append(
            pq_fit(
                p=DX[:, 0],
                q=DX[:, 1],
                delta=delta,
                lambd=lambd,
                absolute_value=absolute_value,
                filter_var=filter_var,
            )
        )
        pf.append(
            estimate_pf(
                p=DX[:, 0],
                q=DX[:, 1],
                delta=delta,
                lambd=lambd,
                absolute_value=absolute_value,
                filter_var=filter_var,
            )
        )

        k_max = np.argmax(V_test[:, study_idx])
        v_k_max, q_k_max = V_test[k_max, study_idx], q[k_max]
        pf_hc = pf[-1] if pf_inj is None else pf_inj

        hc_node = compute_hc(
            v_k_max=v_k_max,
            q_k=q_k_max,
            dvdp=v_S[-1][0],
            dvdq=v_S[-1][1],
            dqdp=pq_S[-1][0],
            pf=pf_hc,
        )
        HC.append([max(h, 0) for h in hc_node])

    return np.asarray(HC), v_S, pq_S, X_filtered, v_filtered, pf


def _safe_div(num, denom, eps=1e-12):
    """Return num/denom, or NaN when |denom| < eps so callers can drop or impute."""
    if not np.isfinite(num) or not np.isfinite(denom) or abs(denom) < eps:
        return np.nan
    return num / denom


def compute_hc(v_k_max, dvdp, dvdq, dqdp, pf, q_k, v_lim=1.05):
    """Five candidate hosting capacity formulas; emits NaN where a denominator is ~0."""
    headroom = v_lim - v_k_max
    qp_combined = dvdp + dvdq * dqdp
    signed_combined = dvdp - np.sign(q_k) * dvdq * dqdp
    return [
        _safe_div(headroom, qp_combined),
        _safe_div(headroom, signed_combined),
        _safe_div(_safe_div(headroom, signed_combined), pf),
        _safe_div(_safe_div(headroom, qp_combined), 0.7),
        _safe_div(headroom, dvdp),
    ]
