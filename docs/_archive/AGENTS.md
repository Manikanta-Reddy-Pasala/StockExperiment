# Repository Guidelines

## Project Structure & Module Organization
Application code sits in `src/`, with `main.py` orchestrating schedulers and wiring. Domain modules stay in `src/services/` (brokers, market data, portfolio, ML), persistence in `src/models/`, shared helpers in `src/utils/`, and external APIs in `src/integrations/`. The Flask web layer—including templates and static files—lives under `src/web/`. Defaults live in `config.py`, tunable filters in `config/stock_filters.yaml`, and bootstrap SQL in `init-scripts/`. Keep runtime artefacts in `logs/`.

## Build, Test, and Development Commands
- `./run.sh dev` — start the Docker stack with hot reload, Postgres, and Redis.
- `./run.sh start` — launch the production-like stack; pair with `./run.sh stop|logs|status` for lifecycle tasks.
- `python -m venv venv && source venv/bin/activate` — create/enter the virtualenv for local-only runs.
- `pip install -r requirements.txt && python3 run.py` — install dependencies and start Flask without Docker.
- `pytest src/tests` — execute the Python test suite; narrow with `-k` for focused runs.
- `black src api run.py` / `flake8 src api` — format and lint before pushing.

## Coding Style & Naming Conventions
Target Python 3.10+ with 4-space indentation and type hints on public APIs. Keep imports explicit and avoid module-level side effects. Use snake_case for modules, functions, and variables; PascalCase for classes; CONSTANT_CASE for configuration keys. Extract reusable helpers into `src/utils/` and leave feature-specific logic inside the relevant service package.

## Testing Guidelines
Pytest is the standard runner. Mirror the service layout (`src/tests/services/test_market_gateway.py`) and group fixtures in `src/tests/conftest.py` or `src/tests/data/` as they appear. Every new domain service or API contract should land with success and failure-path tests, and bug fixes require regression coverage before merge. Run `pytest` locally or through CI prior to opening a PR.

## Commit & Pull Request Guidelines
Recent history leans on short imperative titles (`Update stock_initialization_service.py`, `updated`). Keep the first line ≤72 characters and prefer scope-prefixed verbs (`services: add order retry`) to clarify intent. Include behavioural context, validation steps, and rollback notes in the PR body, link issues, and attach screenshots or cURL transcripts for UI/API changes. Confirm tests and linters in the PR checklist before requesting review.

## Environment & Secrets
Copy the team `.env` template before running Docker; populate broker tokens, the Postgres DSN, and Redis URL. Keep `.env`, generated logs, and credentials out of git. Coordinate edits to `config/stock_filters.yaml` through reviewed PRs so live trading rules remain auditable. Use `README.md` for a feature overview and `DEVELOPMENT.md` for detailed Docker workflow guidance.
