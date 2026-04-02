# Agent Loop

![CI](https://github.com/nekobola/Agent_loop_sample/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/nekobola/Agent_loop_sample/branch/master/graph/badge.svg)](https://codecov.io/gh/nekobola/Agent_loop_sample)

Agent loop implementation for workspace-coder.

## Setup

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/ -v --cov=agent --cov-report=term-missing
```

## Linting

```bash
ruff check agent/ tests/
```
