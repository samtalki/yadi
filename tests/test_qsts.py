import numpy as np
import pandas as pd
import pytest

from yadi import DSS_Timeseries


@pytest.fixture(scope="module")
def qsts(case3_balanced: str) -> DSS_Timeseries:
    sim = DSS_Timeseries(
        case3_balanced,
        time_step=3600,
        simulation_steps=24,
        verbose=False,
    )
    sim.run()
    return sim


def test_qsts_runs_24_steps(qsts: DSS_Timeseries) -> None:
    assert qsts.voltages_mvts.shape == (24, 9)
    assert qsts.vmags_pu_mvts.shape == (24, 9)
    assert qsts.complex_powers_mvts.shape == (24, 9)
    assert np.isfinite(qsts.vmags_pu_mvts).all()
    assert ((qsts.vmags_pu_mvts > 0.8) & (qsts.vmags_pu_mvts < 1.2)).all(), (
        "QSTS voltages drifted outside the [0.8, 1.2] pu sanity band"
    )


def test_qsts_line_currents_are_finite(qsts: DSS_Timeseries) -> None:
    assert qsts.line_currents_mvts.shape[0] == 24
    assert np.isfinite(qsts.line_currents_mvts).all()
    assert not hasattr(qsts, "xfmr_currents_mvts") or qsts.xfmr_currents_mvts is None, (
        "xfmr_currents_mvts is not populated by the python QSTS path; should not be allocated"
    )


def test_qsts_node_dataframes_are_populated(qsts: DSS_Timeseries) -> None:
    nodes = qsts.dss.Circuit.YNodeOrder()
    df = qsts.get_node_qsts_df(nodes[0])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24
    assert {"netloadV", "netloadP", "netloadQ"} == set(df.columns)


def test_get_system_deviations_uses_time_step(qsts: DSS_Timeseries) -> None:
    """Default granularity should pick up self.time_step rather than the old 900s constant."""
    devs = qsts.get_system_deviations()
    assert devs["deltaV"].shape == (9, 23)
    explicit = qsts.get_system_deviations(granularity=qsts.time_step)
    assert np.allclose(devs["deltaV"], explicit["deltaV"])
