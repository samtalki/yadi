# Changelog

## 0.2.0 — 2026-05

Repository modernization.

### Added
- PEP 621 `pyproject.toml` with `uv` lockfile (`uv.lock`).
- `pytest` test suite covering `DSS_Data`, `DSS_Sensitivities`, `DSS_Timeseries`, `DSS_VC_HCA`, and `CLA`.
- GitHub Actions CI matrix on Ubuntu and macOS-14 for Python 3.11 and 3.12.
- `ruff`, `mypy`, and `pre-commit` configurations.
- Type hints and one-line docstrings on the top-level public surface (`DSS_Data.__init__`, `DSS_Sensitivities`, `DSS_Timeseries`, `DSS_VC_HCA`, `AMIData`, `CLA`); legacy `DSS_Data.get_*` methods kept their existing docstrings.
- Single-source DSS binding indirection at `yadi/dss/_binding.py`.
- `examples/` with four runnable scripts.
- `__init__.py` files exposing the public API: `DSS_Data`, `DSS_Sensitivities`, `DSS_Timeseries`, `DSS_VC_HCA`, `AMIData`, `CLA`.
- `yadi/experimental/` for incomplete research code (JAX sensitivity, group HC, secondary network solver).

### Fixed
- `DSS_Sensitivities.get_svp/svq/sthp/sthq` and `DSS_VC_HCA.get_iterative_hc` no longer raise `TypeError` from passing redirects to the no-arg `compile_dss`.
- `AMIData.__init__` no longer references the undefined `default_data_path` at class-definition time.
- Stale `from tkinter import W` removed from `yadi/dss/qsts.py`.
- `numpy 2.x` and `pandas 2.x` compatibility (`np.sum` empty-list dtype, `freq='1H'` → `'1h'`).
- All `dss.run_command` calls migrated to `dss.Text.Command` to drop the deprecation warnings.
- `yadi.hc.measurement.local.pq_fit` was calling a nonexistent `linear.lsq_filtered`; corrected to `solve_lsq_filtered`.
- `yadi.utils.diff` now exists (called from `local.analyze_hosting_capacity`).
- Silent numerical failures: `solve_lsq_filtered` now raises on rank-deficient or under-determined `X` (numba's `np.linalg.inv` returned NaN before); `compute_hc` returns `NaN` instead of `inf` on near-zero sensitivities; `get_node_voltages_mag_pu` raises on zero base voltage instead of producing `inf` per-unit values.
- `__make_phase_label` no longer maps neutral (phase 0) to "a" with a warning; raises.
- `DSS_Sensitivities` perturbation magnitude pulled into named constants so the `kw_inj`/divisor pair stays consistent.
- `DSS_Timeseries.xfmr_currents_mvts` was allocated with `np.empty` and never written; removed (transformer-flow recording in the Python QSTS path is not yet implemented).
- `DSS_Timeseries.get_system_deviations` now defaults `granularity` to `self.time_step`; previous default of 900s gave 4× wrong deviations when called from a 1h QSTS.
- `DSS_Transformer.read_xfmr_power` no longer drops near-zero currents/powers before phase indexing — the filter destroyed phase mapping when any phase happened to be ~0.
- `DSS_Timeseries.__run_qsts_duty` line-current write was advancing the column offset by 1 per line instead of by `NumConductors`, leaving uninitialized garbage in the trailing columns of `line_currents_mvts` for any feeder with multi-conductor lines.

### Changed
- Migrated `yadi/experimental/jax_sensitivity.py` and `yadi/experimental/group_hc.py` from `jaxopt` (no longer maintained) to `optax` and `jnp.maximum`. `optax.adam` replaces `jaxopt.GradientDescent` for the ridge-regression solver; `jnp.maximum(x, 0.0)` replaces `jaxopt.projection.projection_non_negative`.

### Removed
- `requirements.txt` and `poetry.lock` (replaced by `pyproject.toml` and `uv.lock`).
- Broken/stub modules: `yadi/sens/{model_perturb,kalman,ev}.py`, `yadi/dss/ev.py`, `yadi/hc/model/{ev,tc}.py`, `yadi/hc/measurement/{tc_meas,vc_meas}.py`, `yadi/vis/flow_surface.py` (called undefined `get_line_flows`).
- `DSS_Data.get_node_ybus` (self-deprecated, no callers).
- Dead stubs: `__calc_bus_sens_mat`, `get_max_kw`, `get_q_constraints`, `get_xfmr_conductor_idx_map`.
- Unused dependencies: `seaborn`, `scipy` (zero imports — `scipy` will return when the Y-bus extraction issue lands), `cvxpylayers`, `scikit-learn` (zero imports), and `jaxopt` (deprecated upstream — see Changed).

### Moved
- `yadi/sens/linear_sensitivity_model.py` → `yadi/experimental/jax_sensitivity.py`.
- `yadi/hc/measurement/group.py` → `yadi/experimental/group_hc.py`.
- `yadi/data/secondary.py` → `yadi/experimental/secondary.py`.
