# Contributing to yadi

Thanks for considering a contribution. yadi is a research codebase, so the bar for new code is "clear, typed, tested" rather than "production-grade across all axes."

## Setup

```bash
uv sync --extra dev --extra research
uv run pre-commit install
```

## Day-to-day commands

| Task | Command |
| --- | --- |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check yadi tests` |
| Format | `uv run ruff format yadi tests` |
| Type-check | `uv run mypy yadi` |
| Strip notebook outputs | `uv run nbstripout notebooks/*.ipynb` |

CI runs all of these on every PR against `main`.

## Conventions

- **Public API in `yadi.dss`, `yadi.sens`, `yadi.hc`, `yadi.data`.** Type-hint everything in those modules.
- **Experimental code goes in `yadi.experimental`** with a banner comment at the top of each module. mypy is relaxed on that subtree.
- **Power-systems variable names** like `V`, `P`, `Q`, `Svp` are intentional and excluded from naming-convention lints.
- **No dead code.** Don't leave `pass`-only stubs; either implement or delete.
- **The DSS binding is imported from `yadi/dss/_binding.py` only.** Don't `import opendssdirect` directly — that file documents why.

## Adding a test case

Drop the `.dss` files in `test_cases/`, then add a fixture in `tests/conftest.py` and a smoke test that compiles the circuit and asserts a basic invariant.
