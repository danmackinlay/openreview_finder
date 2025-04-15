# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `uv pip install -e .` 
- Run: `openreview-finder index` or `openreview-finder search "query"`

## Development Commands
- Format: `ruff format .`
- Lint: `ruff check`
- Test: `pytest` or `pytest tests/test_specific.py::test_function`

## Code Style
- Use Python 3.9+ features and type hints
- Follow PEP 8 conventions
- Imports: stdlib first, then third-party, then local modules
- Prefer f-strings over `.format()` or `%` formatting
- Use logging instead of print statements
- Class names: PascalCase
- Function/variable names: snake_case
- Constants: UPPER_SNAKE_CASE
- Error handling: Use specific exceptions with helpful messages
- Document functions with docstrings