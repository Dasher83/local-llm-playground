SHELL := /bin/bash
VENV = .venv
ACTIVATE = source $(VENV)/bin/activate

.PHONY: compile-dev-dependencies install-dev-dependencies upgrade-dev-dependencies pre-commit-update \
	compile-backend compile-frontend upgrade-backend upgrade-frontend

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
