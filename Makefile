.PHONY: style test docs pre-release help

# Directories to run style checks on
CHECK_DIRS := tfts examples tests

## Format code and run linting tools
style:  ## Run formatters and linters (black, isort, flake8, pre-commit)
	black $(CHECK_DIRS)
	isort $(CHECK_DIRS)
	flake8 $(check_dirs)
	pre-commit run --all-files

## Run all unit tests
test:  ## Run unit tests using unittest
	python -m unittest discover

## Build the documentation
docs:  ## Build HTML documentation using Sphinx
	make -C docs clean M=$(shell pwd)
	make -C docs html M=$(shell pwd)

## Display help for make targets
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[33m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z\/_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
