.PHONY: install check test
install:
	python -m pip install -e '.[ml,dev]'
check:
	ruff check .
	ruff format --check .
test:
	pytest -q
