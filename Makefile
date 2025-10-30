# RAGiCamp Makefile - Common commands

.PHONY: help install setup test lint format clean

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║                    RAGiCamp - Commands                       ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 SETUP & INSTALLATION"
	@echo "  make install              - Install base dependencies"
	@echo "  make install-dev          - Install with dev tools"
	@echo "  make install-metrics      - Install with metric dependencies"
	@echo "  make install-all          - Install everything"
	@echo "  make setup-bleurt         - Download BLEURT checkpoint"
	@echo "  make setup                - Full setup (install + BLEURT)"
	@echo ""
	@echo "🧪 DEVELOPMENT"
	@echo "  make test                 - Run tests"
	@echo "  make lint                 - Run linting"
	@echo "  make format               - Format code (black + isort)"
	@echo "  make clean                - Clean generated files"
	@echo ""
	@echo "🚀 QUICK START"
	@echo "  make run-gemma2b          - Quick test (10 examples, EM + F1)"
	@echo "  make run-gemma2b-full     - Full eval (100 examples, EM + F1)"
	@echo ""
	@echo "📊 ADVANCED METRICS"
	@echo "  make run-bertscore        - With BERTScore"
	@echo "  make run-bleurt           - With BLEURT (requires setup-bleurt)"
	@echo "  make run-all-metrics      - With all metrics (EM, F1, BERT, BLEURT)"
	@echo ""
	@echo "⚙️  OPTIONS"
	@echo "  make run-gemma2b-cpu      - Run on CPU"
	@echo "  make run-gemma2b-8bit     - Run with 8-bit quantization"
	@echo ""
	@echo "📝 TIPS"
	@echo "  - First time? Run: make setup"
	@echo "  - For BLEURT: make setup-bleurt (downloads ~500MB checkpoint)"
	@echo "  - GPU required for reasonable speed"
	@echo ""

# ============================================================================
# Setup & Installation
# ============================================================================

install:
	@echo "📦 Installing base dependencies..."
	uv sync

install-dev:
	@echo "📦 Installing with dev tools..."
	uv sync --extra dev

install-metrics:
	@echo "📦 Installing with metrics (BERTScore, BLEURT)..."
	uv sync --extra metrics

install-viz:
	@echo "📦 Installing with visualization tools..."
	uv sync --extra viz

install-all:
	@echo "📦 Installing all dependencies..."
	uv sync --extra dev --extra metrics --extra viz

setup-bleurt:
	@echo "📥 Downloading BLEURT checkpoint..."
	@echo "This will download ~500MB. Please wait..."
	@mkdir -p ~/.cache/bleurt
	@uv run python -c "from bleurt import score; scorer = score.BleurtScorer('BLEURT-20')" || \
		(echo "⚠️  BLEURT checkpoint download failed. This is normal if you haven't installed metrics." && \
		 echo "Run: make install-metrics && make setup-bleurt")
	@echo "✅ BLEURT checkpoint ready!"

setup: install-metrics setup-bleurt
	@echo ""
	@echo "✅ Setup complete! You can now run:"
	@echo "   make run-gemma2b          - Quick test"
	@echo "   make run-all-metrics      - Full evaluation with all metrics"
	@echo ""

# ============================================================================
# Development
# ============================================================================

test:
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v

test-coverage:
	@echo "🧪 Running tests with coverage..."
	uv run pytest tests/ --cov=src/ragicamp --cov-report=html --cov-report=term

lint:
	@echo "🔍 Running linters..."
	@uv run flake8 src/ tests/ || true
	@uv run mypy src/ || true

format:
	@echo "✨ Formatting code..."
	uv run black src/ tests/ experiments/ --line-length 100
	uv run isort src/ tests/ experiments/ --profile black
	@echo "✅ Code formatted!"

clean:
	@echo "🧹 Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache dist/ build/ *.egg-info
	@echo "✅ Cleaned!"

clean-outputs:
	@echo "🧹 Cleaning output files..."
	rm -rf outputs/*.json
	@echo "✅ Outputs cleaned!"

clean-all: clean clean-outputs
	@echo "✅ Everything cleaned!"

# ============================================================================
# Evaluation - Quick Start
# ============================================================================

run-gemma2b:
	@echo "🚀 Running Gemma 2B baseline (10 examples, EM + F1)..."
	@echo "⏱️  This should take ~2-3 minutes on GPU"
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 10 \
		--device cuda \
		--filter-no-answer \
		--metrics exact_match f1

run-gemma2b-full:
	@echo "🚀 Running Gemma 2B baseline (100 examples, EM + F1)..."
	@echo "⏱️  This will take ~15-20 minutes on GPU"
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 100 \
		--device cuda \
		--filter-no-answer \
		--metrics exact_match f1

# ============================================================================
# Evaluation - Advanced Metrics
# ============================================================================

run-bertscore:
	@echo "🚀 Running with BERTScore metric..."
	@echo "⏱️  This will take ~3-4 minutes on GPU (10 examples)"
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 10 \
		--device cuda \
		--filter-no-answer \
		--metrics exact_match f1 bertscore

run-bleurt:
	@echo "🚀 Running with BLEURT metric..."
	@echo "⚠️  Make sure you've run: make setup-bleurt"
	@echo "⏱️  This will take ~4-5 minutes on GPU (10 examples)"
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 10 \
		--device cuda \
		--filter-no-answer \
		--metrics exact_match f1 bleurt

run-all-metrics:
	@echo "🚀 Running with ALL metrics (EM, F1, BERTScore, BLEURT)..."
	@echo "⚠️  Make sure you've run: make setup-bleurt"
	@echo "⏱️  This will take ~5-6 minutes on GPU (10 examples)"
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 10 \
		--device cuda \
		--filter-no-answer \
		--metrics exact_match f1 bertscore bleurt

# ============================================================================
# Evaluation - Options
# ============================================================================

run-gemma2b-cpu:
	@echo "🚀 Running on CPU (10 examples)..."
	@echo "⚠️  This will be SLOW (~30+ minutes)"
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 10 \
		--device cpu \
		--filter-no-answer \
		--metrics exact_match f1

run-gemma2b-8bit:
	@echo "🚀 Running with 8-bit quantization..."
	@echo "💾 Uses less memory (~3GB instead of ~8GB)"
	uv run python experiments/scripts/run_gemma2b_baseline.py \
		--dataset natural_questions \
		--num-examples 100 \
		--device cuda \
		--load-in-8bit \
		--filter-no-answer \
		--metrics exact_match f1

# ============================================================================
# Other Baselines
# ============================================================================

run-baseline:
	@echo "🚀 Running DirectLLM baseline..."
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/baseline_direct.yaml \
		--mode eval

compare-baselines:
	@echo "📊 Comparing baselines..."
	uv run python experiments/scripts/compare_baselines.py

# ============================================================================
# Analysis
# ============================================================================

analyze-results:
	@echo "📊 Analyzing per-question results..."
	@if [ -f outputs/gemma_2b_baseline_predictions.json ]; then \
		uv run python examples/analyze_per_question_metrics.py \
			outputs/gemma_2b_baseline_predictions.json; \
	else \
		echo "⚠️  No results found. Run an evaluation first!"; \
	fi

