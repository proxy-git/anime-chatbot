.PHONY: start
start:
# uvicorn main:app --host 0.0.0.0 --port 10000 On Render
	uvicorn src.main:app --reload --port 9000
.PHONY: format
format:
	black .
	isort .