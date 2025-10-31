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
	@echo "🏋️  TRAINING & INDEXING"
	@echo "  make train-fixed-rag      - Train FixedRAG (index docs + save)"
	@echo "  make train-fixed-rag-small- Quick test (1000 docs)"
	@echo "  make index-wikipedia      - Index Wikipedia for NQ"
	@echo "  make index-wikipedia-small- Quick test (1000 docs)"
	@echo "  make list-artifacts       - List saved artifacts"
	@echo "  make clean-artifacts      - Remove all artifacts"
	@echo ""
	@echo "🚀 EVALUATION (Baseline LLM)"
	@echo "  make run-gemma2b          - Quick test (10 examples, EM + F1)"
	@echo "  make run-gemma2b-full     - Full eval (100 examples, EM + F1)"
	@echo ""
	@echo "📊 ADVANCED METRICS"
	@echo "  make run-bertscore        - With BERTScore"
	@echo "  make run-bleurt           - With BLEURT (requires setup-bleurt)"
	@echo "  make run-all-metrics      - With all metrics (EM, F1, BERT, BLEURT)"
	@echo ""
	@echo "🧪 DEVELOPMENT"
	@echo "  make test                 - Run tests"
	@echo "  make lint                 - Run linting"
	@echo "  make format               - Format code (black + isort)"
	@echo "  make clean                - Clean generated files"
	@echo ""
	@echo "📝 TIPS"
	@echo "  - First time? Run: make setup"
	@echo "  - Train RAG: make train-fixed-rag-small (quick test)"
	@echo "  - For BLEURT: make setup-bleurt (downloads ~500MB checkpoint)"
	@echo "  - GPU recommended for training"
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
# Training & Indexing
# ============================================================================

train-fixed-rag:
	@echo "🏋️  Training FixedRAG agent (indexing documents)..."
	@echo "This will:"
	@echo "  1. Load Natural Questions dataset"
	@echo "  2. Create document index with embeddings"
	@echo "  3. Save retriever and agent artifacts"
	@echo ""
	uv run python experiments/scripts/train_fixed_rag.py \
		--agent-name fixed_rag_nq_v1 \
		--retriever-name wikipedia_nq_v1 \
		--top-k 5

train-fixed-rag-small:
	@echo "🏋️  Training FixedRAG (small test version - 1000 docs)..."
	uv run python experiments/scripts/train_fixed_rag.py \
		--agent-name fixed_rag_nq_small \
		--retriever-name wikipedia_nq_small \
		--num-docs 1000 \
		--top-k 3

index-wikipedia:
	@echo "📚 Indexing Wikipedia for Natural Questions..."
	uv run python experiments/scripts/index_wikipedia_for_nq.py \
		--artifact-name wikipedia_nq_v1 \
		--embedding-model all-MiniLM-L6-v2

index-wikipedia-small:
	@echo "📚 Indexing Wikipedia (small test - 1000 docs)..."
	uv run python experiments/scripts/index_wikipedia_for_nq.py \
		--artifact-name wikipedia_nq_small \
		--embedding-model all-MiniLM-L6-v2 \
		--num-docs 1000

list-artifacts:
	@echo "📦 Saved artifacts:"
	@echo ""
	@if [ -d artifacts/retrievers ]; then \
		echo "Retrievers:"; \
		ls -1 artifacts/retrievers/ 2>/dev/null || echo "  (none)"; \
		echo ""; \
	fi
	@if [ -d artifacts/agents ]; then \
		echo "Agents:"; \
		ls -1 artifacts/agents/ 2>/dev/null || echo "  (none)"; \
	fi

clean-artifacts:
	@echo "🧹 Cleaning artifacts..."
	rm -rf artifacts/
	@echo "✓ Artifacts removed"

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

