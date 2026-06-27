.PHONY: lint lint-fix format format-check

lint:
	poetry run ruff check .

lint-fix:
	poetry run ruff check --fix .

format:
	poetry run ruff check --select I --fix .
	poetry run ruff format .

format-check:
	poetry run ruff check --select I .
	poetry run ruff format --check .
