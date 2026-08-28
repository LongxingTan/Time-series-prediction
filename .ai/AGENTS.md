# AGENTS.md

## Useful commands

TFTS is a TensorFlow/Keras project. Tests use Python's `unittest` framework, not pytest. The Makefile runs tools through the locked `uv`
environment, so use its targets instead of invoking tools from a local virtual environment.

```sh
# Format and lint
make style

# Run the test suite
make test

# Build the documentation
make docs
```

`make style` runs Black, isort, Flake8, and all pre-commit hooks. Run `make style` and `make test`
before opening a PR. Run `make docs` when documentation is affected.

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
