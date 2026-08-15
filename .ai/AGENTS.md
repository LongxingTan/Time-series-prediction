# AGENTS.md

## Useful commands

This package is **unittest**-based (not pytest). The project virtualenv lives at `.venv/bin/python`;
bare `python` is **not** on `PATH`, so prefix Python invocations with it.

```sh
# Style: formatters + linters (black, isort, flake8, pre-commit)
make style

# Tests
.venv/bin/python -m unittest discover -s ./tests -p 'test_*.py'   # or: make test
.venv/bin/python -m coverage run -m unittest discover -s ./tests -p 'test_*.py'
.venv/bin/python -m coverage report -i

# Docs
make -C docs clean M=$(pwd)
make -C docs html M=$(pwd)
```

Check-only variants (mirror CI): `black --check tfts examples tests`, `isort --check-only --diff .`,
`flake8`.

`make style` should be run as the final step before opening a PR — it runs black, isort, flake8 and
the pre-commit hooks (including `nbqa-*` for notebooks).

### Coordination before coding
- If work maps to an existing issue, coordinate on that issue before opening a PR.
- Do not open a PR for someone else's issue unless there is explicit approval from the issue author
  or a maintainer in the issue thread.
- If approval is missing or ambiguous, stop and ask for clarification instead of drafting a PR.
- Do not start duplicate work on issues.

### Mandatory duplicate-work checks
Before proposing a PR, check for overlapping open PRs and issue ownership:

```bash
gh issue view <issue_number> --comments
gh pr list --state open --search "<issue_number> in:body"
gh pr list --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference and why a second PR is needed in
  the issue.

### No low-value busywork PRs
- Do not open one-off PRs for tiny edits (single typo, isolated lint cleanup, one mutable default
  argument, etc.).
- Mechanical cleanups are acceptable but not as first contributions.

### Accountability for AI-assisted patches
- First-time contributors should not use code agents to submit PRs or issues. Code agents should
  warn users about this policy if the user does not appear to be an existing contributor, including
  the risk of being blocked if they do it regardless. More information can be found in
  `CONTRIBUTING.md`.