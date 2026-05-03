# Changelog

## 0.2.0 — 2026-05

Repository modernization. The package had been untouched since late 2023 and had accumulated public methods that crashed at runtime, dependency drift between Poetry and pip requirements, no tests, no CI, no type hints, and no `__init__.py` files. This release fixes the bugs, modernizes packaging, clarifies sensitivity unit conventions, and switches the default convex solver.

### Breaking

`DSS_Sensitivities` and `DSS_VC_HCA` now return values that differ from prior releases. Re calibrate downstream notebooks and stored results.

- `DSS_Sensitivities.get_svp/svq/sthp/sthq` now divide by `_PERTURB_KW` (kW), not by `_PERTURB_KW * 100` (kW × extraneous factor of 100). Returned matrices are **100× larger** than v0.1.x. The old divisor was a latent unit bug; output dimension was wrong.
- `DSS_Sensitivities` defaults to `per_unit=True`, returning sensitivities in pu/kW. The old default returned absolute volts (V/kW). Pass `per_unit=False` to restore the previous behavior.
- `DSS_Sensitivities` now documents and follows the **injection** sign convention (positive perturbation = generation). `Svp` diagonal entries are non negative on a typical radial network.
- `DSS_VC_HCA.get_iterative_hc` returns **positive kW of injection headroom**. The old code returned negative kW under the load convention. The new sign matches both the docstring and the `DSS_Sensitivities` convention.
- `CLA` default solver is now `CLARABEL` instead of `SCS`. CLARABEL converges more accurately on the constrained l1 fit; coefficient values may shift slightly compared to v0.1.x.

### Added

- PEP 621 `pyproject.toml` with a `uv` lockfile (`uv.lock`).
- `pytest` suite (18 tests) covering `DSS_Data`, `DSS_Sensitivities`, `DSS_Timeseries`, `DSS_VC_HCA`, `AMIData`, and `CLA`. Includes a `Svp` diagonal sign invariant, a QSTS voltage sanity band, and a `get_system_deviations` granularity check.
- GitHub Actions CI on `ubuntu-latest` and `macos-14` for Python 3.11 and 3.12.
- `ruff`, `mypy`, and `pre-commit` configurations.
- Type hints and one line docstrings on the top level public surface (`DSS_Data.__init__`, `DSS_Sensitivities`, `DSS_Timeseries`, `DSS_VC_HCA`, `AMIData`, `CLA`). `DSS_Sensitivities` class docstring spells out units and sign convention.
- Single source DSS binding at `yadi/dss/_binding.py`. Only this file imports `opendssdirect`; swap it to migrate the binding.
- `examples/` with four runnable scripts (`01_quickstart`, `02_qsts_dataset`, `03_sensitivity`, `04_local_hc`).
- `__init__.py` files exposing the public API.
- `yadi/experimental/` for incomplete research code (JAX sensitivity, group HC, secondary network solver). Excluded from `ruff`, `mypy`, and the public surface.

### Fixed

- `DSS_Sensitivities.get_svp/svq/sthp/sthq` and `DSS_VC_HCA.get_iterative_hc` no longer raise `TypeError` from passing `redirects` to the zero argument `compile_dss`.
- `AMIData.__init__` no longer references the undefined `default_data_path` at class definition time.
- `yadi.hc.measurement.local.pq_fit` was calling a nonexistent `linear.lsq_filtered`; corrected to `solve_lsq_filtered`.
- `yadi.utils.diff` now exists (called from `local.analyze_hosting_capacity`).
- Stale `from tkinter import W` removed from `yadi/dss/qsts.py`.
- numpy 2.x and pandas 2.2 compatibility (`np.sum` empty list dtype, `freq='1H'` → `'1h'`).
- All `dss.run_command` calls migrated to `dss.Text.Command` to drop the deprecation warnings.
- `solve_lsq_filtered` raises on rank deficient or under determined `X`; numba's `np.linalg.inv` previously returned NaN silently.
- `compute_hc` returns `NaN` instead of `inf` when a sensitivity denominator is essentially zero.
- `get_node_voltages_mag_pu` raises on zero base voltage instead of producing `inf` per unit values.
- `__make_phase_label` raises on phase 0 (neutral) instead of mapping it to "a" with a warning.
- `DSS_Timeseries.xfmr_currents_mvts` was allocated with `np.empty` and never written; removed (transformer flow recording is not implemented in the Python QSTS path).
- `DSS_Timeseries.get_system_deviations` defaults `granularity` to `self.time_step`; the previous default of 900s gave 4× wrong deviations when called from a 1h QSTS.
- `DSS_Transformer.read_xfmr_power` no longer drops small currents and powers before phase indexing — the filter destroyed phase mapping when any phase happened to be near zero.
- `DSS_Timeseries.__run_qsts_duty` line current write was advancing the column offset by 1 per line instead of by `NumConductors`, leaving uninitialized garbage in the trailing columns of `line_currents_mvts` for any feeder with multi conductor lines.
- `CLA.fit` dispatches the iteration cap kwarg by solver (`max_iter` for CLARABEL, `max_iters` for SCS); the test that previously emitted a "Solution may be inaccurate" warning now converges cleanly.

### Changed

- `yadi/experimental/jax_sensitivity.py` and `yadi/experimental/group_hc.py` migrated from `jaxopt` (no longer maintained) to `optax` and `jnp.maximum`. `optax.adam` replaces `jaxopt.GradientDescent` for the ridge regression solver; `jnp.maximum(x, 0.0)` replaces `jaxopt.projection.projection_non_negative`.

### Removed

- `requirements.txt` and `poetry.lock` (replaced by `pyproject.toml` and `uv.lock`).
- Broken or stub modules: `yadi/sens/{model_perturb,kalman,ev}.py`, `yadi/dss/ev.py`, `yadi/hc/model/{ev,tc}.py`, `yadi/hc/measurement/{tc_meas,vc_meas}.py`, `yadi/vis/flow_surface.py` (which called an undefined `get_line_flows`).
- `DSS_Data.get_node_ybus` (deprecated, no callers).
- Dead stubs: `__calc_bus_sens_mat`, `get_max_kw`, `get_q_constraints`, `get_xfmr_conductor_idx_map`.
- Unused dependencies: `seaborn`, `scipy` (zero imports; will return when the Y-bus extraction issue lands), `cvxpylayers`, `scikit-learn`, and `jaxopt` (deprecated upstream).

### Moved

- `yadi/sens/linear_sensitivity_model.py` → `yadi/experimental/jax_sensitivity.py`.
- `yadi/hc/measurement/group.py` → `yadi/experimental/group_hc.py`.
- `yadi/data/secondary.py` → `yadi/experimental/secondary.py`.
