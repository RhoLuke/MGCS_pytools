# AGENTS.md — Producer Guidelines & Quality Contract

This file provides authoritative guidelines for AI coding agents (Google Jules, subagents, and local LLMs) working in this repository.

## 1. Code Architecture & Elegance
- **Clarity over Cleverness:** Write idiomatic, self-explanatory code. Avoid premature abstractions, unnecessary design patterns, and unneeded dependencies.
- **Small, Focused Units:** Keep functions cohesive and short (10–40 lines).
- **Explicit Typing:** All new functions and public methods must have complete type signatures.

## 2. Mandatory Verification Gates (Before Every Commit / PR)
You must NOT declare a task completed, create a pull request, or commit changes without running the repository's verification checks and verifying that all exit codes are 0:

### Python Standards:
```bash
# 1. Format check
uv run ruff format --check .

# 2. Linting
uv run ruff check .

# 3. Type Checking
uv run mypy .

# 4. Tests
uv run pytest -v
```

### Safety & Test Integrity Rules:
- **Never weaken tests:** Do not remove assertions, skip tests (`@pytest.mark.skip`), or delete test files to make CI green.
- **Never suppress types blindly:** Do not introduce `# type: ignore` or `typing.Any` without an explicit comment explaining why.

## 3. Git Commit Hygiene
- Make small, atomic commits addressing one task at a time.
- Follow Conventional Commits format:
  - `feat(scope): add new capability`
  - `fix(scope): resolve issue description`
  - `test(scope): add test coverage for module`
  - `refactor(scope): simplify implementation without behavior change`

## 4. Operational Boundaries
- Modify ONLY the files required for the task described in the prompt.
- Do NOT touch CI/CD pipelines, repository secrets, credentials, or destructive git commands (`git reset --hard`, `git clean -fd`).
