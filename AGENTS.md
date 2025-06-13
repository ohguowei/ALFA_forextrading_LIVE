# Repo Guidance for Codex Agent

## Style Guide
- Follow PEP 8 with 4 spaces for indentation.
- Keep lines under 79 characters where possible.
- Include docstrings for public classes and functions.

## Programmatic Checks
- After making code changes, run:
  ```bash
  python -m py_compile $(git ls-files '*.py')
  ```
  to ensure all Python files compile.

## Reinforcement Learning Specifics
- Keep `SimulatedOandaForexEnv` rewards consistent with the live environment:
  * Return realized profit when a position closes.
  * If a position is open, return mark-to-market profit.
  * Otherwise return `0.0`.
- Compute the policy advantage as:
  ```python
  advantage = reward_t + gamma * next_value - value
  ```
  where `gamma` defaults to `0.99`.
- Accumulate multi-step returns only when `accumulate_returns` is enabled.

