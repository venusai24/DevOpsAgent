.DEFAULT_GOAL := help

# ─── Colours ──────────────────────────────────────────────────────────────────
CYAN  := \033[36m
RESET := \033[0m

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'

# ─── Infrastructure ───────────────────────────────────────────────────────────
infra: ## Start Temporal, Qdrant, Redis via Docker Compose
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@docker compose ps

infra-down: ## Stop all infrastructure containers
	docker compose down

infra-logs: ## Tail infrastructure logs
	docker compose logs -f

infra-clean: ## Remove all containers AND volumes (destructive!)
	docker compose down -v

# ─── Python Environment ───────────────────────────────────────────────────────
install: ## Install AIRS and dev dependencies
	pip install -e ".[dev]"

install-prod: ## Install AIRS production dependencies only
	pip install -e .

# ─── Worker ───────────────────────────────────────────────────────────────────
workers: ## Start the unified AIRS worker
	python -m airs.workers.unified_worker

# ─── Knowledge Corpus ─────────────────────────────────────────────────────────
ingest: ## Ingest all corpus documents into Qdrant
	python scripts/ingest_corpus.py

ingest-verbose: ## Ingest with verbose output
	python scripts/ingest_corpus.py --verbose

# ─── Testing ──────────────────────────────────────────────────────────────────
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests (requires running infra)
	pytest tests/integration/ -v --timeout=120

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src/airs --cov-report=html --cov-report=term-missing

# ─── Linting ──────────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check src/ tests/ scripts/

lint-fix: ## Auto-fix ruff lint errors
	ruff check --fix src/ tests/ scripts/

format: ## Format code with ruff
	ruff format src/ tests/ scripts/

typecheck: ## Run mypy type checking
	mypy src/airs/

# ─── Trigger ──────────────────────────────────────────────────────────────────
trigger: ## Trigger a test investigation (usage: make trigger ALERT=path/to/alert.json)
	python scripts/trigger_investigation.py $(ALERT)

.PHONY: help infra infra-down infra-logs infra-clean install install-prod \
        workers ingest ingest-verbose test test-unit test-integration test-cov \
        lint lint-fix format typecheck trigger
