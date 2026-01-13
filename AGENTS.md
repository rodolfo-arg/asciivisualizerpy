# Repository Guidelines

This guide defines the contribution conventions for asciivisualizerpy. The repository is currently a scaffold; update these sections as new tooling or directories are added.

## Project Structure & Module Organization
- Preferred layout: `src/asciivisualizerpy/` for library code, `tests/` for pytest suites, `assets/` for sample inputs/outputs, and `scripts/` for developer utilities.
- CLI or demo entry points should live in `src/asciivisualizerpy/cli.py` (or an equivalent `__main__.py`) so `python -m asciivisualizerpy` can run.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` to create and activate a local virtualenv.
- `python -m pip install -e .[dev]` when a `pyproject.toml` exists; otherwise use `python -m pip install -r requirements.txt`.
- `python -m pytest` to run the test suite.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer type hints on public APIs and short, focused functions.
- Format with `black` and lint with `ruff` once tooling is added; keep files sorted with `isort` if imports grow.

## Testing Guidelines
- Use `pytest` with tests in `tests/` and files named `test_*.py`.
- Add regression tests for bug fixes and unit tests for new features; keep fixtures in `tests/fixtures/` or `tests/data/`.
- No coverage gate yet; target the core rendering/transform logic first.

## Commit & Pull Request Guidelines
- No established commit history yet; use short, imperative subjects (<=72 chars), e.g., `Add ascii frame scaler`.
- PRs should include: purpose summary, tests run, and sample output or screenshots for visual changes.
- Keep PRs focused and update docs or examples when behavior changes.

## Configuration & Assets
- Do not commit secrets. Store local settings in `.env` or `config.local.json` and add them to `.gitignore`.
- Place large media or generated output in `assets/` and keep the repo lean.
