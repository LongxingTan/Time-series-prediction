# AGENTS.md

## Principles

Do what is right for the long term. Do what is right for the long term. Do what is right for the long term.

- Think long term and design for maintainability. Every new change should reduce future development costs; do not introduce technical debt.
- Be alert to code that could create architectural forks. Follow best practices and converge on one coherent structure.
- Maintain a single source of truth: provide one authoritative write path, and derive or reuse everything else from it.
- Solve structural problems through refactoring. Do not bypass them with patches or special cases.
- Follow the best practical, modern practices available.
- Read the existing code before writing code. First look for code within the project that can be reused or extended.
- Prefer iterative replacement over additive layering. Remove obsolete code after a rewrite or refactor.
- Follow KISS. Choose the shortest path that is also best for the long term; if the architecture creates redundancy, simplify the architecture.
- Use strong typing to the degree appropriate, and keep a single authoritative definition for each shared type.
- Do not create cosmetic or misleading fixes. Changes must solve the real problem.
- Do not trade solution quality for implementation convenience. AI implementation is not human labor; always choose the best solution.
- Build deeply encapsulated components: keep complexity inside, expose only the few lifecycle hooks and APIs that callers need, and design for caller ergonomics.
- Avoid short-term hacks and patches.
- Make precise, surgical changes; do not add excessive fallback behavior.

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
