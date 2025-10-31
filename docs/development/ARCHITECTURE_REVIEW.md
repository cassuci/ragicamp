# Architecture Review & Improvement Proposals

**Date:** 2024-10-31  
**Context:** After fixing data leakage and implementing memory-efficient evaluation

## Current State Analysis

### ✅ **What's Working Well**

1. **Clear Separation of Core Concepts**
   - `agents/` - Question answering strategies
   - `models/` - LLM wrappers
   - `retrievers/` - Document retrieval
   - `datasets/` - QA datasets
   - `metrics/` - Evaluation metrics

2. **Good Base Classes**
   - `RAGAgent` (base for all agents)
   - `Retriever` (base for all retrievers)
   - `LanguageModel` (base for all models)
   - `Metric` (base for all metrics)
   - LSP compliance is good

3. **Modular Evaluation**
   - `Evaluator` class separates evaluation from agents
   - Metrics are composable

### ❌ **Issues Discovered Today**

1. **Document Corpus vs Dataset Confusion**
   - **Problem:** We were indexing training questions instead of Wikipedia articles
   - **Root cause:** No clear separation between "documents for retrieval" and "QA examples"
   - **Impact:** Massive data leakage

2. **No Document Source Abstraction**
   - **Problem:** Hard-coded Wikipedia loading in scripts
   - **Should have:** `DocumentCorpus` base class
   - **Missing:** Clean way to load/index different corpora

3. **Artifact Management is Ad-hoc**
   - **Problem:** Save/load logic scattered in retrievers and agents
   - **Should have:** Centralized `ArtifactRegistry` or `CheckpointManager`
   - **Missing:** Versioning, metadata tracking

4. **Memory Management is Implicit**
   - **Problem:** OOM errors required manual two-pass workaround
   - **Should have:** `ResourceManager` or memory-aware pipeline
   - **Missing:** Automatic memory profiling and model swapping

5. **Evaluation Pipeline is Monolithic**
   - **Problem:** `run_fixed_rag_eval.py` does everything: load, retrieve, generate, evaluate
   - **Should have:** Composable pipeline stages
   - **Missing:** Caching between stages, partial re-runs

6. **Configuration is Scattered**
   - **Problem:** Some in YAML, some in argparse, some hard-coded
   - **Should have:** Unified config system (Hydra/OmegaConf)
   - **Missing:** Config validation, defaults management

## Proposed New Entities

### 1. **Document Corpus Abstraction**

```python
# ragicamp/corpus/base.py
class DocumentCorpus(ABC):
    """Source of documents for retrieval (NOT QA pairs)."""
    
    @abstractmethod
    def load(self) -> Iterator[Document]:
        """Yield documents without answer information."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return corpus metadata (size, source, version)."""
        pass

# ragicamp/corpus/wikipedia.py
class WikipediaCorpus(DocumentCorpus):
    def __init__(self, config: str = "20231101.simple"):
        self.config = config
    
    def load(self) -> Iterator[Document]:
        dataset = load_dataset("wikimedia/wikipedia", self.config)
        for article in dataset:
            yield Document(
                id=f"wiki_{article['id']}",
                text=article['text'],
                metadata={"title": article['title'], "source": "wikipedia"}
            )
```

**Benefits:**
- Clear separation: `DocumentCorpus` for retrieval, `QADataset` for evaluation
- Easy to add new corpora (PubMed, arXiv, etc.)
- No more confusion about what gets indexed

### 2. **Artifact Registry**

```python
# ragicamp/registry/artifacts.py
class ArtifactRegistry:
    """Centralized artifact management with versioning."""
    
    def save_retriever(self, retriever: Retriever, name: str, 
                      metadata: Dict) -> str:
        """Save retriever with automatic versioning."""
        pass
    
    def load_retriever(self, name: str, version: Optional[str] = None) -> Retriever:
        """Load retriever by name (latest or specific version)."""
        pass
    
    def list_artifacts(self, type: str) -> List[ArtifactInfo]:
        """List all saved artifacts of a type."""
        pass
    
    def get_lineage(self, artifact: str) -> Dict:
        """Get provenance: what corpus, config, date."""
        pass
```

**Benefits:**
- Track what corpus was used for each retriever
- Version control for reproducibility
- Easy artifact discovery and management

### 3. **Evaluation Pipeline**

```python
# ragicamp/pipeline/stages.py
class PipelineStage(ABC):
    """Single stage in evaluation pipeline."""
    
    @abstractmethod
    def execute(self, input_data: Any, cache_dir: Path) -> Any:
        """Execute stage, using cache if available."""
        pass
    
    @abstractmethod
    def can_use_cache(self, cache_dir: Path) -> bool:
        """Check if cached output exists and is valid."""
        pass

class RetrievalStage(PipelineStage):
    """Stage 1: Retrieve contexts."""
    def execute(self, queries, cache_dir):
        # Load retriever, retrieve, save, unload
        pass

class GenerationStage(PipelineStage):
    """Stage 2: Generate answers."""
    def execute(self, contexts, cache_dir):
        # Load LLM, generate, save, unload
        pass

class MetricsStage(PipelineStage):
    """Stage 3: Compute metrics."""
    def execute(self, predictions, cache_dir):
        # Compute all metrics
        pass

# ragicamp/pipeline/runner.py
class PipelineRunner:
    """Execute multi-stage pipeline with caching."""
    
    def run(self, stages: List[PipelineStage], 
            initial_input: Any) -> Dict:
        """Run pipeline, caching intermediate results."""
        pass
```

**Benefits:**
- Each stage independent and testable
- Automatic caching between stages
- Can re-run from any stage (e.g., re-compute metrics without re-generating)
- Memory-efficient by design

### 4. **Resource Manager**

```python
# ragicamp/resources/manager.py
class ResourceManager:
    """Manage model loading/unloading for memory efficiency."""
    
    def __init__(self, max_gpu_memory: float = 0.9):
        self.max_gpu_memory = max_gpu_memory
        self.loaded_models = {}
    
    def load_model(self, model_spec: Dict) -> Any:
        """Load model, unloading others if needed."""
        if self.get_memory_usage() > self.max_gpu_memory:
            self.unload_least_recently_used()
        return self._load(model_spec)
    
    def with_model(self, model_spec: Dict):
        """Context manager for automatic cleanup."""
        model = self.load_model(model_spec)
        try:
            yield model
        finally:
            self.unload_model(model)
```

**Benefits:**
- Automatic memory management
- No manual unload logic in scripts
- Configurable memory budget
- LRU caching for model reuse

### 5. **Index Builder**

```python
# ragicamp/indexing/builder.py
class IndexBuilder:
    """Build retriever indices from document corpora."""
    
    def __init__(self, corpus: DocumentCorpus, 
                 retriever_type: str = "dense"):
        self.corpus = corpus
        self.retriever_type = retriever_type
    
    def build(self, output_name: str, 
             config: Dict) -> str:
        """Build index and save as artifact."""
        retriever = self._create_retriever(config)
        
        # Index documents from corpus
        for doc_batch in self.corpus.load_batched(batch_size=1000):
            retriever.index_documents(doc_batch)
        
        # Save with metadata
        registry.save_retriever(
            retriever, 
            output_name,
            metadata={
                "corpus": self.corpus.get_metadata(),
                "config": config,
                "timestamp": datetime.now()
            }
        )
```

**Benefits:**
- Clear indexing workflow
- Separates corpus loading from indexing
- Automatic metadata tracking

## Proposed Directory Structure

```
ragicamp/
├── src/ragicamp/
│   ├── agents/           # Question answering strategies
│   ├── models/           # LLM wrappers
│   ├── retrievers/       # Document retrieval
│   ├── datasets/         # QA evaluation datasets
│   ├── metrics/          # Evaluation metrics
│   │
│   ├── corpus/           # NEW: Document corpora (no answers!)
│   │   ├── base.py
│   │   ├── wikipedia.py
│   │   ├── pubmed.py
│   │   └── custom.py
│   │
│   ├── indexing/         # NEW: Index building
│   │   ├── builder.py
│   │   └── strategies.py
│   │
│   ├── pipeline/         # NEW: Evaluation pipelines
│   │   ├── stages.py
│   │   ├── runner.py
│   │   └── cache.py
│   │
│   ├── registry/         # NEW: Artifact management
│   │   ├── artifacts.py
│   │   └── versioning.py
│   │
│   ├── resources/        # NEW: Memory/resource management
│   │   ├── manager.py
│   │   └── profiler.py
│   │
│   └── config/           # NEW: Configuration management
│       ├── schema.py
│       └── defaults.py
│
├── experiments/
│   ├── configs/          # Experiment configs (Hydra)
│   ├── scripts/          # High-level experiment runners
│   └── notebooks/        # Analysis notebooks
│
└── artifacts/            # Saved models, indices, results
    ├── retrievers/
    ├── corpora/          # NEW: Corpus metadata
    └── experiments/      # NEW: Full experiment artifacts
```

## Implementation Priority

### Phase 1: Critical Fixes (Week 1)
1. ✅ **DocumentCorpus abstraction** - Prevent future data leakage
2. ✅ **IndexBuilder** - Clean indexing workflow
3. **ArtifactRegistry** - Track what corpus was used

### Phase 2: Pipeline Improvements (Week 2)
4. **PipelineStage** - Modular evaluation
5. **ResourceManager** - Automatic memory management
6. **Cache system** - Reusable intermediate results

### Phase 3: Configuration (Week 3)
7. **Unified config** - Hydra integration
8. **Config validation** - Pydantic schemas
9. **Experiment tracking** - MLflow/Weights&Biases

## Migration Strategy

### Don't Break Existing Code
- Keep old scripts working with deprecation warnings
- Provide migration guide for each new abstraction
- Add new entities alongside old ones

### Example Migration

**Old way (current):**
```python
# Hard-coded in script
dataset = load_dataset("wikimedia/wikipedia", "20231101.simple")
retriever.index_documents(extract_docs(dataset))
```

**New way (proposed):**
```python
# Clean separation
corpus = WikipediaCorpus(config="20231101.simple")
builder = IndexBuilder(corpus, retriever_type="dense")
artifact_path = builder.build("wikipedia_v1", config={...})
```

## Benefits Summary

1. **Prevent Data Leakage**
   - Clear corpus/dataset separation
   - Automatic validation

2. **Improve Modularity**
   - Pipeline stages independent
   - Easy to add new corpora/retrievers

3. **Better Memory Management**
   - Automatic model loading/unloading
   - Configurable resource limits

4. **Reproducibility**
   - Artifact versioning
   - Full provenance tracking

5. **Easier Experimentation**
   - Reusable pipeline components
   - Caching between stages

## Discussion Points

1. **Too much abstraction?** 
   - Risk of overengineering
   - Balance with simplicity

2. **Backward compatibility?**
   - Keep old scripts working?
   - Migration timeline?

3. **External dependencies?**
   - Hydra for configs?
   - MLflow for tracking?

4. **Testing strategy?**
   - Unit tests for each new entity
   - Integration tests for pipelines

## Next Steps

1. **Review this proposal** - Get feedback
2. **Prototype DocumentCorpus** - Validate design
3. **Implement Phase 1** - Critical fixes first
4. **Iterate based on usage** - Don't overbuild upfront

