.PHONY: help setup build-index api docker-up docker-down docker-build-index

# Default target — show available commands
help:
	@echo ""
	@echo "  Job Data RAG Pipeline — Makefile"
	@echo ""
	@echo "  Local development:"
	@echo "    make setup              Create venv and install backend dependencies"
	@echo "    make build-index        Build Pinecone + BM25 index (runs locally)"
	@echo "    make api                Start FastAPI with hot-reload on :8000"
	@echo ""
	@echo "  Docker Compose:"
	@echo "    make docker-up          Build images and start all services"
	@echo "    make docker-build-index Build the vector index inside Docker"
	@echo "    make docker-down        Stop and remove all containers"
	@echo ""

# ── Local ──────────────────────────────────────────────────────────────────────

setup:
	uv venv
	. .venv/bin/activate && uv pip install -e backend

build-index:
	PYTHONPATH=backend python backend/scripts/build_index.py

api:
	PYTHONPATH=backend uvicorn app.main:app --reload

# ── Docker Compose ─────────────────────────────────────────────────────────────

docker-up:
	docker compose up --build

docker-build-index:
	docker compose run --rm api python backend/scripts/build_index.py

docker-down:
	docker compose down
