.PHONY: test lint run figures clean

test:            ## run the pytest suite
	python3 -m pytest tests/

lint:            ## lint with ruff
	python3 -m ruff check mpt tests

run:             ## full pipeline from the cached data
	python3 -m mpt.cli

figures:         ## regenerate every chart in the 30-chart gallery
	python3 -m mpt.cli --rebalance monthly --cost 0.001 --walk-forward --black-litterman

