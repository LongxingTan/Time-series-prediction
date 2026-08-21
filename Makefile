.PHONY: style test test-warnings docs pre-release help

# Directories to run style checks on
CHECK_DIRS := tfts examples tests

## Format code and run linting tools
style:  ## Run formatters and linters (black, isort, flake8, pre-commit)
	black $(CHECK_DIRS)
	isort $(CHECK_DIRS)
	flake8 $(CHECK_DIRS)
	pre-commit run --all-files

## Keep the default test output focused on failures. Set these before Python starts
TEST_ENV := CUDA_VISIBLE_DEVICES=0 TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore

## Run all unit tests
test:  ## Run unit tests without routine TensorFlow/Python warnings
	$(TEST_ENV) python -m unittest discover

test-warnings:  ## Run unit tests with all warnings and TensorFlow info logs
	CUDA_VISIBLE_DEVICES=0 python -m unittest discover

## Build the documentation
docs:  ## Build HTML documentation using Sphinx
	make -C docs clean M=$(shell pwd)
	make -C docs html M=$(shell pwd)

## Display help for make targets
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[33m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z\/_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
