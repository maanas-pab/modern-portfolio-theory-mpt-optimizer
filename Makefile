.PHONY: test lint run figures clean

test:            ## run the pytest suite
	python3 -m pytest tests/

lint:            ## lint with ruff
	python3 -m ruff check mpt tests

