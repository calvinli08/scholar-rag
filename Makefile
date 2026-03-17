# ScholarRAG — development commands
# Usage: make <target>

.PHONY: help db-up db-down db-reset db-shell db-logs pgadmin test lint ingest

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ScholarRAG — available commands"
	@echo ""
	@echo "  Database"
	@echo "    make db-up       Start PostgreSQL container (detached)"
	@echo "    make db-down     Stop and remove the container"
	@echo "    make db-reset    Wipe all data and restart fresh"
	@echo "    make db-shell    Open a psql session inside the container"
	@echo "    make db-logs     Tail container logs"
	@echo "    make pgadmin     Start pgAdmin at http://localhost:5050"
	@echo ""
	@echo "  Development"
	@echo "    make test        Run the test suite"
	@echo "    make lint        Lint with ruff"
	@echo "    make ingest      Ingest all PDFs in ./papers/"
	@echo ""

# ── Database ──────────────────────────────────────────────────────────────────
db-up:
	docker compose up -d db
	@echo "Waiting for DB to be healthy..."
	@docker compose exec db pg_isready -U scholarrag -d scholarrag || sleep 3

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d db

db-shell:
	docker compose exec db psql -U scholarrag -d scholarrag

db-logs:
	docker compose logs -f db

pgadmin:
	docker compose --profile tools up -d pgadmin
	@echo "pgAdmin available at http://localhost:5050"
	@echo "  Email   : admin@scholarrag.local"
	@echo "  Password: admin"

# ── Dev ───────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check . --fix

frontend:
	python -m streamlit run ui/app.py

backend:
	python -m fastapi dev main.py