"""Local hosting capacity from synthetic AMI data using regression sensitivities."""

import numpy as np

from yadi.hc.measurement.local import analyze_hosting_capacity


def main() -> None:
    rng = np.random.default_rng(0)
    M_train, M_test, N = 200, 50, 6
    P_train = rng.normal(scale=10.0, size=(M_train, N))
    Q_train = 0.3 * P_train + rng.normal(scale=2.0, size=(M_train, N))
    V_train = 1.0 + 0.001 * P_train + rng.normal(scale=1e-4, size=(M_train, N))
    V_test = 1.04 + rng.normal(scale=1e-3, size=(M_test, N))

    HC, v_S, pq_S, X_filt, v_filt, pf = analyze_hosting_capacity(
        V_test=V_test,
        V_train=V_train,
        Q=Q_train,
        P=P_train,
        delta=0.5,
        lambd=1e-3,
        filter_var="p",
        absolute_value=True,
    )
    print(f"HC shape: {HC.shape}")
    print(f"power factors: {[round(x, 3) for x in pf]}")


if __name__ == "__main__":
    main()
