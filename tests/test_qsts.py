import numpy as np

from yadi import DSS_Timeseries


def test_qsts_runs_24_steps(case3_balanced: str) -> None:
    sim = DSS_Timeseries(
        case3_balanced,
        time_step=3600,
        simulation_steps=24,
        verbose=False,
    )
    sim.run()
    assert sim.voltages_mvts.shape == (24, 9)
    assert sim.vmags_pu_mvts.shape == (24, 9)
    assert sim.complex_powers_mvts.shape == (24, 9)
    assert np.isfinite(sim.vmags_pu_mvts).all()
    assert ((sim.vmags_pu_mvts > 0.8) & (sim.vmags_pu_mvts < 1.2)).all(), (
        "QSTS voltages drifted outside the [0.8, 1.2] pu sanity band"
    )
