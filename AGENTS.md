# Repository Guidelines

## Project Structure & Module Organization

`yadi/` contains the Python package. The main areas are `yadi/dss/` for OpenDSS wrappers and model objects, `yadi/sens/` for sensitivity and linearization code, `yadi/hc/` for hosting-capacity models and measurements, `yadi/data/` for data helpers, and `yadi/vis/` for plotting. `test_cases/` stores OpenDSS `.dss` models and related CSV/DBL fixtures used for examples and validation. `notebooks/` holds exploratory and tutorial notebooks; keep generated figures in `notebooks/figures/`. Branding assets live in `assets/logo/`.

## Build, Test, and Development Commands

- `poetry install`: install the package and dependencies from `pyproject.toml` and `poetry.lock`.
- `poetry shell`: activate the project virtual environment for interactive work.
- `poetry run python -m pip show yadi`: confirm the local package is installed.
- `poetry run python -c "import yadi; print('ok')"`: quick import smoke test after package changes.
- `poetry build`: create source and wheel distributions for release checks.

There is no configured test runner in `pyproject.toml` yet. When adding one, prefer `pytest` and document any new command here.

## Coding Style & Naming Conventions

Use Python 3.9-compatible syntax. Follow the existing four-space indentation style and keep imports grouped as standard library, third-party, then local modules. Use `snake_case` for functions, variables, and modules; use uppercase names for constants such as `ELEMENT_CLASSES`. Preserve existing public API names like `DSS_Data` unless performing a coordinated compatibility update. Add concise docstrings to public classes and methods, especially when wrapping OpenDSS behavior or returning pandas/numpy objects.

## Testing Guidelines

For new behavior, add focused tests under a future `tests/` directory using filenames like `test_dss_model.py` and functions like `test_get_node_voltages_handles_missing_base_voltage`. Reuse small fixtures from `test_cases/` rather than duplicating DSS networks. Validate notebook-facing changes with the smallest relevant notebook or a script that imports the affected module and loads a representative `.dss` file.

## Commit & Pull Request Guidelines

Recent commits use short, present-tense summaries such as `adding real and imaginary components of voltage` and `correcting yadi.yadi imports`. Keep messages concise and mention the affected area when useful, for example `dss: fix transformer formatting`.

Pull requests should include a brief purpose statement, the commands or notebooks used for validation, and any changed assumptions about OpenDSS inputs. Include screenshots or regenerated PDFs when visualization output changes. Link related issues when available and call out large fixture or notebook-output changes explicitly.
