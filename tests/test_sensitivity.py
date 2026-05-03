import numpy as np
import pytest

from yadi import DSS_Sensitivities


@pytest.fixture(scope="module")
def sens(case3_balanced: str) -> DSS_Sensitivities:
    return DSS_Sensitivities(case3_balanced, verbose=False, per_unit=True)


@pytest.mark.parametrize("method", ["get_svp", "get_svq", "get_sthp", "get_sthq"])
def test_sensitivity_matrices_finite_and_shaped(sens: DSS_Sensitivities, method: str) -> None:
    res = getattr(sens, method)()
    assert res["matrix"].shape == (9, 9)
    assert np.isfinite(res["matrix"]).all()
    assert len(res["nodes"]) == 9


def test_svp_self_sensitivity_is_positive(sens: DSS_Sensitivities) -> None:
    # kw_inj=-100 is generation; dV/d(-load) is positive at the perturbed node.
    spv = sens.get_svp()["matrix"]
    diag = np.diag(spv)
    assert (diag >= 0).all(), f"Svp diagonal should be non-negative, got {diag}"
    assert diag.max() > 0, "Svp diagonal is identically zero — perturbation had no effect"
