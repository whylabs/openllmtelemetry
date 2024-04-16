.PHONY: help lint lint-fix format format-fix fix install test integ clean all

lint: ## Check for type issues with pyright
	@{ echo "Running pyright\n"; poetry run pyright; PYRIGHT_EXIT_CODE=$$?; } ; \
	{ echo "\nRunning ruff check\n"; poetry run ruff check; RUFF_EXIT_CODE=$$?; } ; \
	exit $$(($$PYRIGHT_EXIT_CODE + $$RUFF_EXIT_CODE))

lint-fix:
	poetry run ruff check --fix

format: ## Check for formatting issues
	poetry run ruff format --check

format-fix: ## Fix formatting issues
	poetry run ruff format

fix: lint-fix format-fix ## Fix all linting and formatting issues

install: ## Install dependencies with poetry
	poetry install -E "openai"

test: ## Run unit tests
	poetry run pytest -vvv -s -o log_level=INFO -o log_cli=true tests/

integ: ## Run integration tests
	poetry run pytest -vvv -o log_level=INFO -o log_cli=true integ/

dist: ## Build the distribution
	poetry build

clean: ## remove build artifacts
	rm -rf ./dist/*

bump-patch: ## Bump the patch version (_._.X) everywhere it appears in the project
	@$(call i, Bumping the patch number)
	poetry run bumpversion patch --allow-dirty

bump-minor: ## Bump the minor version (_.X._) everywhere it appears in the project
	@$(call i, Bumping the minor number)
	poetry run bumpversion minor --allow-dirty

bump-major: ## Bump the major version (X._._) everywhere it appears in the project
	@$(call i, Bumping the major number)
	poetry run bumpversion major --allow-dirty

bump-release: ## Convert the version into a release variant (_._._) everywhere it appears in the project
	@$(call i, Removing the dev build suffix)
	poetry run bumpversion release --allow-dirty

bump-build: ## Bump the build number (_._._-____XX) everywhere it appears in the project
	@$(call i, Bumping the build number)
	poetry run bumpversion build --allow-dirty

help: ## Show this help message.
	@echo 'usage: make [target] ...'
	@echo
	@echo 'targets:'
	@egrep '^(.+)\:(.*) ##\ (.+)' ${MAKEFILE_LIST} | sed -s 's/:\(.*\)##/: ##/' | column -t -c 2 -s ':#'

