# RAGiCamp Makefile - Common commands

.PHONY: help install test lint format clean run-gemma2b

help:
	@echo "RAGiCamp - Available commands:"
	@echo ""
	@echo "  make install          - Install dependencies with uv"
	@echo "  make install-all      - Install with all optional dependencies"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Run linting"
	@echo "  make format           - Format code"
	@echo "  make clean            - Clean generated files"
	@echo ""
	@echo "  make run-gemma2b      - Run Gemma 2B baseline (quick test, filtered)"
	@echo "  make run-gemma2b-full - Run Gemma 2B baseline (100 examples, filtered)"
	@echo "  make run-baseline     - Run DirectLLM baseline"
	@echo ""
	@echo "Note: Gemma 2B commands filter out questions without explicit answers"
	@echo ""

install:
	uv sync

install-all:
	uv sync --extra dev --extra metrics --extra viz

test:
	uv run pytest tests/

lint:
	uv run flake8 src/ tests/ || true
	uv run mypy src/ || true

format:
	uv run black src/ tests/ experiments/ -l 99
	uv run isort src/ tests/ experiments/ --profile black

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache dist/ build/ *.egg-info
	rm -rf outputs/*.json

# Gemma 2B evaluation commands
run-gemma2b:
	@echo "Running Gemma 2B baseline (10 examples, quick test)..."
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 10 \
		--device cuda \
		--filter-no-answer

run-gemma2b-full:
	@echo "Running Gemma 2B baseline (100 examples)..."
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 100 \
		--device cuda \
		--filter-no-answer

run-gemma2b-cpu:
	@echo "Running Gemma 2B baseline on CPU (10 examples)..."
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 10 \
		--device cpu

run-gemma2b-8bit:
	@echo "Running Gemma 2B baseline with 8-bit quantization..."
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 100 \
		--load-in-8bit

# Other baselines
run-baseline:
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/baseline_direct.yaml \
		--mode eval

compare-baselines:
	uv run python experiments/scripts/compare_baselines.py

