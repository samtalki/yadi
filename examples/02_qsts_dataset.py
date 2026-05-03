"""Run a 24-step QSTS simulation and save the voltage MVTS to disk."""

from pathlib import Path

import numpy as np

from yadi import DSS_Timeseries

CASE = Path(__file__).resolve().parents[1] / "test_cases" / "case3_balanced.dss"
OUT = Path(__file__).resolve().parent / "qsts_output.npz"


def main() -> None:
    sim = DSS_Timeseries(
        str(CASE),
        time_step=3600,
        simulation_steps=24,
        verbose=False,
    )
    sim.run()
    print(f"voltages_mvts shape: {sim.voltages_mvts.shape}")
    np.savez(
        OUT,
        voltages_mvts=sim.voltages_mvts,
        vmags_pu_mvts=sim.vmags_pu_mvts,
        complex_powers_mvts=sim.complex_powers_mvts,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
