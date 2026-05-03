import numpy as np

from yadi import DSS_Sensitivities


def test_get_svp_returns_finite_matrix(case3_balanced: str) -> None:
    sens = DSS_Sensitivities(case3_balanced, verbose=False, per_unit=True)
    res = sens.get_svp()
    assert res["matrix"].shape == (9, 9)
    assert np.isfinite(res["matrix"]).all()
    assert len(res["nodes"]) == 9


def test_get_svq_returns_finite_matrix(case3_balanced: str) -> None:
    sens = DSS_Sensitivities(case3_balanced, verbose=False, per_unit=True)
    res = sens.get_svq()
    assert res["matrix"].shape == (9, 9)
    assert np.isfinite(res["matrix"]).all()
