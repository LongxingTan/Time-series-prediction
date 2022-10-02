.PHONY: style test pre-release

# run checks on all files and potentially modifies some of them

style:
	black --check .
	isort --check-only --diff .
	flake8

# run tests for the library

test:
	python -m unittest


# run tests for the docs

testdoc:
	python -m unittest


# release

# pre-release:
# 	python utils/release.py
