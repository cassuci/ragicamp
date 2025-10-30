# Changelog

All notable changes to RAGiCamp will be documented in this file.

## [Unreleased]

### Added
- **Expected answer in results**: JSON output now includes `expected_answer` (primary) and `all_acceptable_answers` fields
- **Answer filtering**: New `--filter-no-answer` flag to filter out questions without explicit answers
- **Dataset filtering methods**: Added `filter_with_answers()` and `get_examples_with_answers()` to QADataset base class

### Changed
- Results JSON now has three answer fields:
  - `expected_answer`: Primary/first expected answer (NEW)
  - `all_acceptable_answers`: All acceptable answers (NEW)
  - `references`: All acceptable answers (kept for backward compatibility)
- Makefile commands now use `--filter-no-answer` by default for cleaner evaluations

## [0.1.0] - 2024-10-29

### Added
- Initial RAGiCamp framework with modular architecture
- Agents: DirectLLM, FixedRAG, BanditRAG, MDPRAG
- Models: HuggingFace and OpenAI interfaces
- Retrievers: Dense (FAISS) and Sparse (TF-IDF)
- Datasets: Natural Questions, HotpotQA, TriviaQA
- Metrics: Exact Match, F1, BERTScore, LLM-as-judge
- Policies: Epsilon-Greedy, UCB, Q-Learning, Random
- Training and evaluation utilities
- Configuration-driven experiment system
- Comprehensive documentation

### Added (Package Manager Migration)
- Migrated from pip to uv package manager
- Added uv.lock for reproducible builds
- Added .python-version file
- Updated pyproject.toml with hatchling build backend

### Added (Gemma 2B Baseline)
- Dedicated Gemma 2B evaluation script
- Gemma 2B configuration file
- CPU/GPU support with 8-bit quantization option
- Multiple dataset support (NQ, HotpotQA, TriviaQA)
- Comprehensive documentation (GEMMA2B_QUICKSTART.md, QUICK_START_GEMMA.md)
- Makefile with convenient commands

