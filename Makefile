.PHONY: up down psql load-data promote-data install

up:
	docker compose up -d

down:
	docker compose down

psql:
	docker exec -it kaiser_postgres psql -U derkaiser677 -d kaiser_db

load-data:
	python3.12 src/data/load.py

promote-data:
	python3.12 src/data/promote_to_base.py

install:
	pip install -r requirements.txt

run-sql:
	docker exec -i kaiser_postgres psql -U derkaiser677 -d kaiser_db < $(FILE)

test:
	pytest tests/

api:
	uvicorn src.api.main:app --reload --port 8000