.PHONY: install train test dashboard api docker
install:
	pip install -r requirements-dev.txt
train:
	python ml_nids.py
test:
	python -m unittest discover -s tests -t . -v
dashboard:
	streamlit run app.py
api:
	uvicorn api:app --host 0.0.0.0 --port 8000
docker:
	docker compose --profile train run --rm train && docker compose up --build
