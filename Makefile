.PHONY: dev down test

dev:
	docker compose up --build

down:
	docker compose down -v

test:
	pytest -q
