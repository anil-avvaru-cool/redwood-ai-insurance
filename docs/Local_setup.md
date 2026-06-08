# Local Setup

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| Docker + Docker Compose | v2+ | [docs.docker.com](https://docs.docker.com/get-docker/) |
| UV | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Git | any | system package manager |

## Required Directory Layout

All repos must be cloned as siblings under the same parent directory. The
platform-level `docker-compose.yml` uses relative paths (`../repo-name`) to
reference each sub-project.

```
your-workspace/
├── redwood-ai-insurance/          ← platform repo (docker-compose.yml lives here)
├── insurance-data-platform/
├── intelligent-underwriting-platform/
├── ai-fraud-detection-platform/
├── fnol-claims-multi-agent-system/
└── enterprise-rag-platform/
```

Clone all repos into the same parent:

```bash
git clone https://github.com/anil-avvaru-cool/redwood-ai-insurance
git clone https://github.com/anil-avvaru-cool/insurance-data-platform
git clone https://github.com/anil-avvaru-cool/intelligent-underwriting-platform
git clone https://github.com/anil-avvaru-cool/ai-fraud-detection-platform
git clone https://github.com/anil-avvaru-cool/fnol-claims-multi-agent-system
git clone https://github.com/anil-avvaru-cool/enterprise-rag-platform
```

## Environment Setup

All configuration is driven by environment variables. Start from the shared template:

```bash
cd redwood-ai-insurance
cp example.env .env
```

Edit `.env` and set at minimum:

```bash
ANTHROPIC_API_KEY=sk-ant-...   # required for underwriting + FNOL agents
NEO4J_PASSWORD=your-password   # must match what you set in Neo4j
```

On Linux, set your host user IDs to avoid root-owned volume files:

```bash
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env
```

## Running with Docker Compose

All commands run from the `redwood-ai-insurance/` directory.

### Start shared infrastructure only (Redis + Neo4j)

```bash
docker compose up
```

This is the minimum required before running any sub-project directly.

### Start a specific sub-project

```bash
docker compose --profile data up          # insurance-data-platform
docker compose --profile underwriting up  # intelligent-underwriting-platform (port 8001)
docker compose --profile fraud up         # ai-fraud-detection-platform (port 8002)
docker compose --profile rag up           # enterprise-rag-platform (port 8003)
docker compose --profile fnol up          # fnol-claims + rag (port 8000)
```

### Start everything

```bash
docker compose --profile all up
```

### Service port map

| Service | Port | Profile |
|---------|------|---------|
| FNOL claims API | 8000 | `fnol` |
| Underwriting API | 8001 | `underwriting` |
| Fraud detection API | 8002 | `fraud` |
| RAG API | 8003 | `rag`, `fnol` |
| Redis | 6379 | always |
| Neo4j Bolt | 7687 | always |
| Neo4j Browser | 7474 | always |

> **Note:** underwriting, fraud, fnol, and rag app services require a `Dockerfile`
> in each sub-project repo. Only `insurance-data-platform` has one today. Add
> Dockerfiles to the other repos before using those profiles.

## Running Sub-projects Directly (without Docker)

Use this for active development on a single sub-project. Start shared infra
via Docker Compose first, then run the app locally.

```bash
# 1. Start shared infra
cd redwood-ai-insurance
docker compose up -d

# 2. In a separate terminal, run the sub-project
cd ../insurance-data-platform
cp ../redwood-ai-insurance/.env .env
uv run python main.py

# FastAPI sub-projects (underwriting, fraud, rag)
cd ../intelligent-underwriting-platform
uv run uvicorn api.main:app --reload --port 8001
```

### Installing sub-project dependencies

Each sub-project manages its own dependencies via `pyproject.toml` and UV:

```bash
cd <sub-project>
uv sync
```

> `requirements.txt` files in some sub-projects are legacy — `pyproject.toml`
> is the source of truth per project standards.

## First-Run Steps

### 1. Generate synthetic data

```bash
cd insurance-data-platform
uv run python main.py generate
```

This writes raw parquet files to `insurance-data-platform/data/raw/` which
underwriting and fraud detection read at training time.

### 2. Run entity resolution

```bash
uv run python main.py resolve
```

Entity resolution must complete before graph construction. See
[DEC-011](../docs/DECISION_LOG.md) for why.

### 3. Seed the feature store

```bash
uv run python main.py seed-features
```

Writes initial feature vectors to Redis. Required for the fraud sync path
(<100ms) to operate.

### 4. Train models (optional for local dev)

```bash
# Underwriting hurdle model
cd ../intelligent-underwriting-platform
uv run python main.py train

# Fraud ensemble
cd ../ai-fraud-detection-platform
uv run python main.py train
```

Pre-trained artifacts are checked into `models/artifacts/` in each repo, so
this step is only needed after data changes.

## Verifying the Setup

With infra running and features seeded:

```bash
# Neo4j browser
open http://localhost:7474

# Underwriting API docs
open http://localhost:8001/docs

# Fraud API docs
open http://localhost:8002/docs

# RAG API docs
open http://localhost:8003/docs
```

## Troubleshooting

See [docs/Troubleshooting.md](./Troubleshooting.md) for common issues.
