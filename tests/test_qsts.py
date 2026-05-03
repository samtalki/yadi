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
