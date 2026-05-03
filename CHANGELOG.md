# Changelog

## 0.2.0 — 2026-05

Repository modernization.

### Added
- PEP 621 `pyproject.toml` with `uv` lockfile (`uv.lock`).
- `pytest` test suite covering `DSS_Data`, `DSS_Sensitivities`, `DSS_Timeseries`, `DSS_VC_HCA`, and `CLA`.
- GitHub Actions CI matrix on Ubuntu and macOS-14 for Python 3.11 and 3.12.
- `ruff`, `mypy`, and `pre-commit` configurations.
- Type hints and one-line docstrings on the public API.
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

### Removed
- `requirements.txt` and `poetry.lock` (replaced by `pyproject.toml` and `uv.lock`).
- Broken/stub modules: `yadi/sens/{model_perturb,kalman,ev}.py`, `yadi/dss/ev.py`, `yadi/hc/model/{ev,tc}.py`, `yadi/hc/measurement/{tc_meas,vc_meas}.py`.
- `DSS_Data.get_node_ybus` (self-deprecated, no callers).
- Dead stubs: `__calc_bus_sens_mat`, `get_max_kw`, `get_q_constraints`, `get_xfmr_conductor_idx_map`.

### Moved
- `yadi/sens/linear_sensitivity_model.py` → `yadi/experimental/jax_sensitivity.py`.
- `yadi/hc/measurement/group.py` → `yadi/experimental/group_hc.py`.
- `yadi/data/secondary.py` → `yadi/experimental/secondary.py`.
