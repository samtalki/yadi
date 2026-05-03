# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`yadi` ("yet another DSS interface") is a Python wrapper around OpenDSS for distribution network research, focused on generating datasets for machine learning and sensitivity work. The OpenDSS bridge is `opendssdirect.py`; everything else is built on top of it.

## Environment

Poetry manages the package. Python 3.9+.

- `poetry install` — install dependencies declared in `pyproject.toml`.
- `poetry shell` — activate the venv.
- `poetry run python -c "import yadi; print('ok')"` — import smoke test.

`requirements.txt` lists `jax`, `jaxopt`, `optax`, `scikit-learn`, `cvxpylayers`, `dss_python`, and `fire` — none of which are in `pyproject.toml`. After a plain `poetry install` the JAX-using modules (`yadi/sens/linear_sensitivity_model.py`, `yadi/sens/ev.py`, `yadi/hc/measurement/group.py`, `yadi/hc/model/ev.py`) will fail at import. Pip-install whatever you need from `requirements.txt` by hand.

There is no test runner. Validate changes by exercising the affected module against a relevant `test_cases/*.dss` file or the smallest notebook that imports it.

## Architecture

### Namespace package, no `__init__.py`

The package has no `__init__.py` files anywhere — it's an implicit namespace package (PEP 420). Always import via dotted paths (`import yadi.dss.qsts as qsts`); `from yadi import ...` will not work for sibling modules.

### `yadi/dss/` is a linear inheritance chain

Each module in `yadi/dss/` subclasses the leaf of the previous one, building a single combined OpenDSS data class through composition by inheritance rather than mixins:

```
DSS_Data (model.py)
└── DSS_Monitor (monitor.py)
    └── DSS_LoadShape (load_shape.py)
        └── DSS_Bus (bus.py)
            └── DSS_Line (line.py)
                └── DSS_Transformer (transformer.py)
                    └── DSS_Shunt (shunt.py)
                        └── DSS_Load (load.py)
                            └── DSS_LineCode (line_code.py)
                                └── DSS_VoltageSource (voltage_source.py)
                                    └── DSS_Timeseries (qsts.py)
```

`DSS_Timeseries` is the leaf and the main entry point for QSTS simulations — it carries every method from the chain. Adding a new element type means inserting another link: import the current leaf, subclass it, add methods, and update whatever currently sat on top of the old leaf to subclass the new one.

`DSS_Sensitivities` (`yadi/dss/sensitivity.py`) extends `DSS_Data` directly and sits outside the chain. `DSS_VC_HCA` (`yadi/hc/model/vc.py`) and `DSS_Secondaries` (`yadi/data/secondary.py`) similarly branch off `DSS_Data` / `DSS_Sensitivities`.

### Public API in practice

The notebooks in `notebooks/` consistently import the same three modules: `yadi.dss.model`, `yadi.dss.sensitivity`, `yadi.dss.qsts`. Those plus `yadi.hc.model.vc` are the de-facto user-facing surface — the rest of the chain (bus, line, transformer, etc.) is invoked indirectly through the leaf class.

### Sub-packages outside `dss/`

- `yadi/sens/` — regression-based sensitivity models. NumPy + numba in `data_linear.py` and `cla.py`; JAX in `linear_sensitivity_model.py` and `ev.py`.
- `yadi/hc/` — hosting capacity analysis. `model/` is model-driven (`vc.DSS_VC_HCA` extends `DSS_Data`); `measurement/` is data-driven (`local.py`, `group.py`).
- `yadi/data/` — `ami.AMIData` reads HDF5 AMI datasets; `secondary.py` does secondary network sensitivity work.
- `yadi/vis/` — plotting helpers.
- `yadi/utils.py` — finite-difference helpers and analysis plots. Depends on `yadi.data.ami`.

## Broken / dead modules to avoid

- `yadi/sens/model_perturb.py` defines `OpenDSS_Sensitivities(model.OpenDSS_Data)`. There is no `OpenDSS_Data` in `yadi/dss/model.py` (the class is `DSS_Data`), so this file fails on import.
- `yadi/dss/ev.py` then tries to subclass `perturb.DSS_Sensitivities`, which the broken `model_perturb` module never defines. Also dead on import.
- The right sensitivity entry point is `yadi/dss/sensitivity.py:DSS_Sensitivities`.
- `yadi/hc/measurement/{tc_meas,vc_meas}.py` and `yadi/hc/model/tc.py` are empty placeholder files. The directory layout implies they exist; they don't.

## Test cases

`test_cases/` holds dozens of `.dss` networks (3-bus through IEEE 13-bus, plus a `secondary_test_network/`) and CSV/DBL load profile fixtures. Reuse these rather than adding new ones. `case3_balanced.dss` and `13Bus/IEEE13Nodeckt.dss` are the obvious starting points for smoke tests.

## Conventions and contributions

See `AGENTS.md` for code style (Python 3.9 syntax, snake_case, public API names like `DSS_Data` are stable) and PR conventions (short present-tense commit subjects, optional area prefix like `dss: ...`).
