# Repository Guidelines

## Project Structure & Module Organization

- `src/factor_demo/` contains the Python CLI, FastAPI service, NOOA agent harness, isolated execution runtime, dataset tools, and evidence validation. Keep older functionality under `legacy/`.
- `src/factor_demo/knowledge/` contains domain and statistics skills.
- `frontend/` contains the Vue 3 and TypeScript interface.
- `tests/` contains Python regression tests; `scripts/` contains verification and documentation utilities.
- `configs/` holds runtime and capability settings. `docs/sphinx/` is the maintained documentation source; `deploy/` contains deployment resources.
- Generated datasets and research artifacts belong in ignored `data/` and `runs/` directories.

## Build, Test, and Development Commands

Use Python 3.12 or 3.13, uv, and Node 24.

- `uv sync --python 3.12`: install Python dependencies.
- `npm ci --prefix frontend`: install locked frontend dependencies.
- `uv run --env-file .env factor-demo serve --host 127.0.0.1 --port 8765`: start the local service.
- `npm run dev --prefix frontend`: start Vite development mode.
- `make test`: run pytest and Ruff checks.
- `npm test --prefix frontend`: run Vitest tests.
- `npm run build --prefix frontend`: type-check and build the interface.
- `npm ci --prefix docs` followed by `make docs-check`: build and validate Sphinx documentation.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and modules, and `PascalCase` classes. Follow the configured Ruff checks and 100-character line length. Prefer existing scientific libraries and shared project utilities over custom replacements. Keep modules focused and match adjacent Vue/TypeScript conventions.

本项目新增或修改的代码注释与文档字符串统一使用中文，包括模块、类、方法、参数、返回值和异常说明。标识符、接口字段名及必要的技术术语保留英文；文档字符串保留 Google 风格结构。此规则优先于通用技能中要求使用英文注释的约定，不要求批量翻译未涉及的现有代码。

When designing UI, HTML, or slides, avoid unnecessary secondary labels and descriptions; let layout and labels explain the interface.

## Testing Guidelines

Use pytest with `tests/test_*.py` filenames and `test_*` functions. Run focused checks with `uv run pytest tests/test_persistent_sessions.py -q`. Add regression coverage for changed behavior, especially execution isolation, cancellation, persistence, and evidence provenance. No numeric coverage threshold is configured. Report checks actually run; distinguish mocked tests from real-model or deployment validation.

## Commit & Pull Request Guidelines

Follow existing prefixes: `feat:`, `fix:`, `docs:`, and `chore:`. Use concise, imperative subjects. PRs should describe the problem, resulting behavior, relevant issues, and validation performed. Include screenshots for visible interface changes and update documentation when behavior changes.

## Security & Configuration

Copy `.env.example` to `.env` for local credentials. Never commit secrets, private conversations, research databases, traces, or generated datasets. Preserve credential-free execution boundaries and hidden-truth isolation. Evidence checks and model review do not establish physical causality.
