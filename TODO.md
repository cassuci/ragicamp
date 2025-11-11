# RAGiCamp TODO

This file tracks features, improvements, and development roadmap for RAGiCamp.

---

## ✅ Recently Completed

### Metrics & Evaluation
- [x] BERTScore metric implementation
- [x] BLEURT metric implementation  
- [x] **LLM-as-a-judge metric (binary & ternary)**
- [x] Per-question metrics tracking
- [x] Statistical summaries for metrics
- [x] Text normalization (SQuAD-style)

### Configuration & Usability
- [x] Config-driven experiment system
- [x] Ready-to-use config files for common scenarios
- [x] Makefile commands for all evaluation types
- [x] CPU evaluation support
- [x] 8-bit quantization support

### Documentation
- [x] Comprehensive metrics guide
- [x] Config-based evaluation guide
- [x] LLM-as-a-judge guide
- [x] Quick reference cheat sheet
- [x] Baseline evaluation guide
- [x] **Documentation reorganization and cleanup**

---

## 🎯 High Priority

### Core Features
- [ ] Implement document corpus loading utilities
- [ ] Add hybrid/reranking retrievers
- [ ] Query reformulation strategies
- [ ] Caching for retrieval results

### Training & RL
- [ ] Improve MDP state representation
- [ ] Add more bandit algorithms (Thompson Sampling, UCB variants)
- [ ] Implement PPO/A2C for MDP agents
- [ ] Better reward shaping utilities

### Evaluation
- [ ] Add ROUGE metrics
- [ ] Faithfulness metrics (answer grounded in context)
- [ ] Calibration metrics (confidence vs accuracy)
- [ ] Cross-encoder reranking evaluation

---

## 📊 Medium Priority

### Features
- [ ] Multi-turn conversation support
- [ ] Support for custom document corpora
- [ ] More dataset loaders (MS MARCO, SQuAD 2.0)
- [ ] Integration with vector databases (Pinecone, Weaviate, Milvus)
- [ ] Experiment tracking (Weights & Biases, MLflow)
- [ ] Visualization utilities for results

### Models
- [ ] Support for more LLM providers (Anthropic, Cohere)
- [ ] Local LLM optimization (vLLM, TGI)
- [ ] Model comparison utilities

### Retrieval
- [ ] Ensemble retrieval strategies
- [ ] Late interaction models (ColBERT)
- [ ] Learned sparse retrieval (SPLADE)

---

## 🔬 Research Features (Low Priority)

- [ ] Distributed training support
- [ ] Active learning for RAG
- [ ] Few-shot prompt optimization
- [ ] Retrieval-free generation (parametric memory)
- [ ] Multimodal retrieval support
- [ ] Chain-of-thought RAG

---

## 🏗️ Infrastructure

### Deployment
- [ ] API server for deploying agents
- [ ] Docker containers for reproducibility
- [ ] Model serving optimization
- [ ] Batch inference utilities

### UI/Tools
- [ ] Web UI for experiments
- [ ] Interactive evaluation tool
- [ ] Result analysis dashboard
- [ ] Annotation tool for eval

---

## 📚 Documentation

### Completed
- [x] README with overview
- [x] Architecture documentation
- [x] Agents guide
- [x] Usage guide
- [x] Metrics guide
- [x] Config guide
- [x] LLM judge guide
- [x] Quick reference
- [x] Documentation index

### Todo
- [ ] API reference (auto-generated)
- [ ] Tutorial notebooks
- [ ] Example use cases / recipes
- [ ] Paper reproduction guides
- [ ] Video tutorials
- [ ] Blog posts

---

## 🧪 Testing & Quality

### Testing
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] End-to-end experiment tests
- [ ] Performance benchmarks
- [ ] CI/CD pipeline

### Code Quality
- [ ] Complete type hints throughout
- [ ] Full docstring coverage
- [ ] Linting configuration (flake8, mypy)
- [x] Code formatting setup (black, isort)
- [ ] Pre-commit hooks
- [ ] Security scanning

---

## 💡 Ideas / Future Exploration

- Automatic hyperparameter tuning
- Meta-learning for RAG
- Explainability tools (why did it retrieve this?)
- A/B testing framework
- Cost optimization tools
- Prompt engineering utilities
- Knowledge graph integration
- Structured output generation

---

## 📝 Notes

### Priorities
1. **Core functionality** - Make existing features robust
2. **Usability** - Config-driven, well-documented
3. **Research** - Enable experimentation with new approaches
4. **Production** - Deployment-ready features

### Guidelines
- Keep codebase modular and extensible
- Prioritize documentation alongside features
- Config-driven over code changes
- Production-ready defaults

---

**Last Updated:** 2025-11-11

**Recent Focus:** Metrics expansion (LLM judge), config system, documentation cleanup
