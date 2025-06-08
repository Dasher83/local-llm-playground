SHELL := /bin/bash
VENV = .venv
ACTIVATE = source $(VENV)/bin/activate

.PHONY: compile-dev-dependencies install-dev-dependencies upgrade-dev-dependencies pre-commit-update \
	compile-backend compile-frontend upgrade-backend upgrade-frontend \
	up down restart logs logs-backend logs-frontend logs-ollama \
	ollama-list ollama-pull ollama-remove ollama-show ollama-cleanup

# Install pip-tools if not yet installed
install-dev-dependencies:
	$(VENV)/bin/pip install -r dev.requirements.txt

# Upgrade dev dependencies
upgrade-dev-dependencies:
	$(VENV)/bin/pip-compile --upgrade dev.requirements.in

# Update pre-commit hooks to their latest versions and run it
pre-commit-update:
	$(VENV)/bin/pre-commit autoupdate && $(VENV)/bin/pre-commit run

# Compile backend requirements.txt from requirements.in
compile-backend:
	$(VENV)/bin/pip-compile backend/requirements.in

# Compile frontend requirements.txt from requirements.in
compile-frontend:
	$(VENV)/bin/pip-compile frontend/requirements.in

# Upgrade all backend packages
upgrade-backend:
	$(VENV)/bin/pip-compile --upgrade backend/requirements.in

# Upgrade all frontend packages
upgrade-frontend:
	$(VENV)/bin/pip-compile --upgrade frontend/requirements.in

# Docker Compose commands
up:
	docker compose up --build

down:
	docker compose down

restart:
	docker compose down && docker compose up --build

logs:
	docker compose logs -f

logs-backend:
	docker compose logs -f backend

logs-frontend:
	docker compose logs -f frontend

logs-ollama:
	docker compose logs -f ollama

# Ollama model management
ollama-list:
	docker compose exec ollama ollama list

ollama-pull:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make ollama-pull MODEL='modelname'"; \
		echo "Example: make ollama-pull MODEL='llama3.1:70b'"; \
		exit 1; \
	fi
	docker compose exec ollama ollama pull "$(MODEL)"

ollama-remove:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make ollama-remove MODEL='modelname'"; \
		echo "Example: make ollama-remove MODEL='llama3.1:70b'"; \
		exit 1; \
	fi
	docker compose exec ollama ollama rm "$(MODEL)"

ollama-show:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make ollama-show MODEL='modelname'"; \
		echo "Example: make ollama-show MODEL='llama3.1:70b'"; \
		exit 1; \
	fi
	docker compose exec ollama ollama show "$(MODEL)"

ollama-cleanup:
	docker compose exec ollama ollama prune
