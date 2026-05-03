"""Compile the IEEE 13-bus benchmark and print its base voltages."""

from pathlib import Path

from yadi import DSS_Data

CASE = Path(__file__).resolve().parents[1] / "test_cases" / "13Bus" / "IEEE13Nodeckt.dss"


def main() -> None:
    d = DSS_Data(str(CASE), verbose=False)
    print(f"compiled circuit: {d.dss.Circuit.Name()}")
    print(f"converged:        {d.dss.Solution.Converged()}")
    voltages = d.dss.Circuit.AllBusMagPu()
    print(f"min vpu:          {min(voltages):.4f}")
    print(f"max vpu:          {max(voltages):.4f}")


if __name__ == "__main__":
    main()
