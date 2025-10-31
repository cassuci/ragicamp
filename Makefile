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
	@echo "🏋️  INDEXING & CORPUS"
	@echo "  make index-corpus         - Index corpus (new architecture)"
	@echo "  make index-wiki-simple    - Index Simple Wikipedia (~200k articles)"
	@echo "  make index-wiki-small     - Quick test (10k articles)"
	@echo "  make list-artifacts       - List saved artifacts"
	@echo "  make clean-artifacts      - Remove all artifacts"
	@echo ""
	@echo "📋 EXPERIMENT MANAGEMENT"
	@echo "  make demo-architecture    - Demo new architecture"
	@echo "  make list-experiments     - List all experiments"
	@echo "  make compare-experiments  - Compare experiment results"
	@echo ""
	@echo "🚀 EVALUATION (Baseline LLM)"
	@echo "  make run-gemma2b          - Quick test (10 examples, EM + F1)"
	@echo "  make run-gemma2b-full     - Full eval (100 examples, EM + F1)"
	@echo ""
	@echo "🔍 EVALUATION (FixedRAG with Retrieval)"
	@echo "  make run-fixed-rag        - Quick test (10 examples)"
	@echo "  make run-fixed-rag-full   - Full eval (100 examples)"
	@echo "  make run-fixed-rag-bertscore - With BERTScore"
	@echo ""
	@echo "📊 ADVANCED METRICS (Baseline)"
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
	uv run black src/ tests/ experiments/ --line-length 99
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
# RAG Evaluation (with trained retriever)
# ============================================================================

run-fixed-rag:
	@echo "🔍 Running FixedRAG evaluation (quick test - 10 examples)..."
	@if [ ! -d artifacts/retrievers/wikipedia_small ]; then \
		echo "⚠️  Wikipedia index not found. Indexing first..."; \
		$(MAKE) index-wiki-small; \
	fi
	uv run python experiments/scripts/run_fixed_rag_eval.py \
		--retriever-artifact wikipedia_small \
		--top-k 3 \
		--dataset natural_questions \
		--num-examples 10 \
		--filter-no-answer \
		--metrics exact_match f1 bertscore bleurt \
		--load-in-8bit \
		--output outputs/fixed_rag_small_results.json

run-fixed-rag-full:
	@echo "🔍 Running FixedRAG evaluation (full - 100 examples)..."
	@if [ ! -d artifacts/retrievers/wikipedia_small ]; then \
		echo "⚠️  Wikipedia index not found. Indexing first..."; \
		$(MAKE) index-wiki-small; \
	fi
	uv run python experiments/scripts/run_fixed_rag_eval.py \
		--retriever-artifact wikipedia_small \
		--top-k 5 \
		--dataset natural_questions \
		--num-examples 100 \
		--filter-no-answer \
		--metrics exact_match f1 \
		--load-in-8bit \
		--output outputs/fixed_rag_full_results.json

# ============================================================================
# Training & Indexing
# ============================================================================

index-wiki:
	@echo "📚 Indexing Full English Wikipedia..."
	@echo "This will download and index ~6M Wikipedia articles"
	@echo "⚠️  First run will download several GB of data"
	@echo "⚠️  This will take HOURS to complete"
	@echo ""
	uv run python experiments/scripts/index_corpus.py \
		--corpus-name wikipedia_en \
		--corpus-version 20231101.en \
		--embedding-model all-MiniLM-L6-v2 \
		--artifact-name wikipedia_en_full

index-wiki-simple:
	@echo "📚 Indexing Simple English Wikipedia (full ~200k articles)..."
	@echo "⚠️  This will take 30-60 minutes"
	@echo ""
	uv run python experiments/scripts/index_corpus.py \
		--corpus-name wikipedia_simple \
		--corpus-version 20231101.simple \
		--embedding-model all-MiniLM-L6-v2 \
		--artifact-name wikipedia_simple_full

index-wiki-small:
	@echo "📚 Quick test: Indexing 10k Simple Wikipedia articles..."
	@echo "⚠️  For TESTING ONLY - use index-wiki-simple for evaluation"
	@echo ""
	uv run python experiments/scripts/index_corpus.py \
		--corpus-name wikipedia_simple \
		--corpus-version 20231101.simple \
		--embedding-model all-MiniLM-L6-v2 \
		--max-docs 10000 \
		--artifact-name wikipedia_small

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
# New Architecture: Corpus, Config, OutputManager
# ============================================================================

demo-architecture:
	@echo "🎭 Running architecture demo..."
	uv run python experiments/scripts/demo_new_architecture.py

index-corpus:
	@echo "📚 Indexing corpus (new architecture)..."
	uv run python experiments/scripts/index_corpus.py \
		--corpus-name wikipedia_simple \
		--corpus-version 20231101.simple \
		--max-docs 10000 \
		--artifact-name wikipedia_corpus_v1

list-experiments:
	@echo "📋 Listing all experiments..."
	@uv run python -c "from ragicamp.output import OutputManager; \
		mgr = OutputManager(); \
		exps = mgr.list_experiments(limit=20); \
		print(f'Found {len(exps)} experiments:'); \
		for exp in exps: print(f\"  {exp['experiment_name']:40} | {exp.get('dataset','N/A'):15} | {exp.get('timestamp','N/A')[:19]}\")"

compare-experiments:
	@echo "📊 Comparing experiments..."
	@echo "Usage: make compare-experiments EXPERIMENTS='exp1 exp2 exp3'"
	@if [ -z "$(EXPERIMENTS)" ]; then \
		echo "⚠️  Please specify EXPERIMENTS variable"; \
		echo "Example: make compare-experiments EXPERIMENTS='fixed_rag_v1 baseline_v1'"; \
		exit 1; \
	fi
	@uv run python -c "from ragicamp.output import OutputManager; \
		mgr = OutputManager(); \
		mgr.print_comparison('$(EXPERIMENTS)'.split())"

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

