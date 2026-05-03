"""Compute the perturb-and-observe Svp matrix on case3_balanced and plot it."""

from pathlib import Path

import matplotlib.pyplot as plt

from yadi import DSS_Sensitivities

CASE = Path(__file__).resolve().parents[1] / "test_cases" / "case3_balanced.dss"
OUT = Path(__file__).resolve().parent / "svp.png"


def main() -> None:
    sens = DSS_Sensitivities(str(CASE), verbose=False, per_unit=True)
    res = sens.get_svp()
    print(f"Svp shape: {res['matrix'].shape}")
    print(f"nodes:     {res['nodes']}")

    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(res["matrix"], cmap="viridis")
    ax.set_title(r"$S_{vp}$ on case3_balanced")
    ax.set_xlabel("perturbed node index")
    ax.set_ylabel("observed node index")
    fig.colorbar(im, ax=ax)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
