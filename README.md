![yadi](assets/logo/yadi_logo_clear_purple_tilt.png)

# yadi

Research workflows for distribution network analysis on top of [OpenDSSDirect.py](https://github.com/dss-extensions/OpenDSSDirect.py). yadi adds sensitivity analysis, approximation, and dataset generation on top of a cross platform OpenDSS binding.

## Install

```bash
uv sync                       # install runtime + dev deps
uv sync --extra research      # add the optional JAX/optax stack used by experimental modules
```

`uv` ([install instructions](https://docs.astral.sh/uv/getting-started/installation/)) is the supported tool. `pip install -e ".[dev,research]"` also works for users who don't want uv.

## Quickstart

```python
from yadi import DSS_Data

d = DSS_Data("test_cases/13Bus/IEEE13Nodeckt.dss", verbose=False)
print(d.dss.Circuit.Name())
print(d.dss.Circuit.AllBusMagPu())
```

For more, see [`examples/`](examples/) (quickstart, QSTS dataset, perturb and observe sensitivities, local hosting capacity).

## What yadi adds on top of OpenDSSDirect.py

- `yadi.dss.DSS_Sensitivities` — perturb and observe sensitivity matrices `S_vp`, `S_vq`, `S_θp`, `S_θq`.
- `yadi.dss.DSS_Timeseries` — quasi-static time series (QSTS) dataset orchestration: voltages, complex powers, currents, line flows.
- `yadi.hc.DSS_VC_HCA` — voltage constrained hosting capacity (model based).
- `yadi.hc.measurement.local.analyze_hosting_capacity` — AMI based hosting capacity using regression sensitivities.
- `yadi.sens.CLA` — conservative linear approximation via constrained l1 regression.
- `yadi.data.AMIData` — HDF5 ingestion for finite difference workflows on real meter data.

## Where yadi sits in the ecosystem

yadi is a research workflow layer on top of an OpenDSS Python binding. See [`yadi/dss/_binding.py`](yadi/dss/_binding.py). The current binding ecosystem we are using is the [dss-extensions](https://github.com/dss-extensions) community fork of OpenDSS. It ships three Python packages:

- **`OpenDSSDirect.py`** — function-call API; what yadi currently uses.
- **`DSS-Python`** — COM-style API on the same engine; notably exposes a programmatic `YMatrix` (system admittance matrix).
- **`AltDSS-Python`** — newer object-based API combining both.

The other relevant, official EPRI binding that we plan to support is **EPRI's `py_dss_interface`**.

## Test cases

Sample feeders live in [`test_cases/`](test_cases/) — small balanced and unbalanced 3-bus networks, the IEEE 13-bus benchmark, and several transformer test cases.

## Status

- **Stable:** `yadi.dss`, `yadi.sens`, `yadi.hc`, `yadi.data`. Tested in CI on macOS 14 and Ubuntu, Python 3.11/3.12.
- **Experimental:** `yadi.experimental` — incomplete research code (JAX sensitivity model, group HC, secondary network solver). API may change.

## Citation

```bibtex
@misc{talkington2024yadi,
  author = {Samuel Talkington and Alejandro Owen and Alex Reyna and Jorge Fernandez},
  title  = {yadi: research workflows for distribution network analysis},
  year   = {2024},
  note   = {https://github.com/samtalki/yadi}
}
```

## Acknowledgement

This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE-2039655. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
