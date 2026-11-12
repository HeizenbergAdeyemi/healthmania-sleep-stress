.PHONY: install test lint demo app

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests app

demo:
	python -m healthmania.train --data data/sample_synthetic.csv

app:
	streamlit run app/streamlit_app.py
