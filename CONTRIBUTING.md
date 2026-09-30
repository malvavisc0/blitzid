# Contributing to BlitzID

## Development setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/malvavisc0/blitzid.git
cd blitzid
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

## Running checks locally

```bash
# Lint
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/

# Type checking
uv run mypy src/ scripts/

# Tests
uv run pytest tests/ -v

# Complexity + dead code (see docs/coding-standard.md)
uvx radon cc src scripts tests -s | grep -E '\-\s(C|D|E|F)\s'   # must be empty
uvx vulture src/ tests/ scripts/ --min-confidence 60
```

## Pull request guidelines

1. Fork the repo and create a feature branch from `main`.
2. Ensure all checks pass (`ruff`, `mypy`, `pytest`).
3. Add or update tests for any new functionality.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a PR with a clear description of the change.
