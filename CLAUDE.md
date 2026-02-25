# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RESI is a Bittensor subnet (Subnet 46) that incentivizes accurate real estate price prediction models. Miners train ONNX models and submit them on-chain; validators download, evaluate in Docker sandboxes against fresh sales data, and set weights. Winner-takes-all (99%) with a threshold mechanism to reward innovation over copying.

## Common Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run unit tests only (what CI runs)
uv run pytest real_estate/tests/unit/ -v --tb=short

# Run integration tests (requires Docker image built first)
docker build -t resi-onnx-runner:latest real_estate/evaluation/
uv run pytest real_estate/tests/integration/ -v --tb=short

# Run a single test file
uv run pytest real_estate/tests/unit/evaluation/test_metrics.py -v

# Run a single test
uv run pytest real_estate/tests/unit/evaluation/test_metrics.py::test_name -v

# Linting and formatting
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .

# Type checking (currently set to continue-on-error in CI)
uv run mypy real_estate neurons --ignore-missing-imports

# Start validator (auto-updating wrapper)
uv run python scripts/start_validator.py [args]

# Miner CLI
uv run miner-cli evaluate --model.path ./model.onnx
uv run miner-cli submit --model.path ./model.onnx --hf.repo_id user/repo --wallet.name miner
```

## Architecture

### Two Roles

- **Validator** (`real_estate/validator/`, `scripts/start_validator.py`): Downloads miner models, evaluates them in Docker, scores, and sets weights on-chain.
- **Miner** (`real_estate/miner_cli/`): CLI tool to validate ONNX models locally and submit commitments on-chain.

### Validator Flow (3 concurrent async loops)

1. **Pre-Download Loop**: Starts ~3 hours before evaluation. Fetches chain commitments via Pylon, spreads model downloads across the window, runs a 30-minute catch-up retry phase.
2. **Evaluation Loop**: Triggered daily when validation data arrives (fetched at 18:00 UTC via `ValidationDatasetClient` cron). Encodes features → runs Docker inference → detects duplicates → selects winner → distributes weights.
3. **Weight Setting Loop**: Every 60s checks if enough blocks have elapsed (`epoch_length=361`), then normalizes scores, applies burn allocation, and sets weights on-chain via Pylon.

### Key Subsystems

- **`real_estate/chain/`** — Pylon client wrapper. Pylon is a Docker sidecar service that handles all Bittensor chain interactions (metagraph, commitments, weight setting). Runs at `localhost:8000`.
- **`real_estate/models/`** — Model lifecycle: `ModelDownloadScheduler` orchestrates downloads with circuit breaker logic, `ModelVerifier` checks license/size/hash, `ModelCache` stores downloaded ONNX files.
- **`real_estate/evaluation/`** — Docker-sandboxed ONNX inference. `DockerRunner` creates isolated containers (no network, 2GB mem, 300s timeout). `inference_script.py` runs inside the container. `EvaluationOrchestrator` runs models in parallel (max 4 concurrent). Metrics: MAPE-based score = `1 - MAPE`.
- **`real_estate/data/`** — `FeatureEncoder` converts property JSON to numpy arrays using YAML-based feature config with pluggable transforms. `ValidationDatasetClient` fetches daily evaluation data from dashboard API.
- **`real_estate/duplicate_detector/`** — Groups predictions by similarity (threshold 1e-6), identifies pioneers (earliest committer) vs copiers. Copiers get zero weight.
- **`real_estate/incentives/`** — `WinnerSelector` picks winner set within `score_threshold` of best, elects by earliest commit time. `IncentiveDistributor` does 99/1 split.
- **`real_estate/observability/`** — WandB logging for evaluation metrics.
- **`real_estate/orchestration/`** — `ValidationOrchestrator` ties together feature encoding → evaluation → duplicate detection → winner selection → weight distribution.

### Design Patterns

- **Factory methods**: `ValidationOrchestrator.create()`, `create_model_scheduler()`, `create_eval_orchestrator()` wire up dependencies.
- **Frozen dataclasses**: Core data models (`ChainModelMetadata`, `Metagraph`, `Neuron`, etc.) are immutable.
- **Circuit breaker**: Model downloader tracks consecutive failures, opens for 5 minutes after 5 failures to avoid hammering HuggingFace.
- **Async concurrency**: Validator uses `asyncio.gather()` for 3 loops, `asyncio.Lock` for metagraph access, semaphores for parallel Docker evaluation.

### Infrastructure

- **Docker Compose**: Runs Pylon sidecar. On Apple Silicon: `DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up -d`.
- **Docker evaluation image**: Built from `real_estate/evaluation/Dockerfile` (python:3.11-slim + numpy + onnxruntime). Must be tagged `resi-onnx-runner:latest`.

## CI

GitHub Actions runs on PRs to `main`: ruff lint + format check, mypy (continue-on-error), unit tests (Python 3.11), integration tests (requires Docker image build). All must pass for the `ci-success` gate.

## Code Style

- Python 3.11+, managed with `uv`
- Ruff for linting and formatting (line length 88, double quotes)
- `asyncio_mode = "auto"` for pytest-asyncio (no need for `@pytest.mark.asyncio`)
- Tests mirror source structure under `real_estate/tests/unit/` and `real_estate/tests/integration/`
- Ruff excludes `neurons/`, `scripts/`, `tests/` (top-level), `contrib/` directories
