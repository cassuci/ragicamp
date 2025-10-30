"""Base classes for datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QAExample:
    """Represents a question-answering example.
    
    Attributes:
        id: Unique example identifier
        question: The question text
        answers: List of acceptable answers
        context: Optional context/passage (for reading comprehension)
        metadata: Additional example metadata
    """
    id: str
    question: str
    answers: List[str]
    context: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QADataset(ABC):
    """Base class for QA datasets.
    
    Provides a unified interface for loading and accessing different
    QA datasets (NQ, HotpotQA, TriviaQA, etc.).
    """
    
    def __init__(self, name: str, split: str = "train", **kwargs: Any):
        """Initialize the dataset.
        
        Args:
            name: Dataset identifier
            split: Dataset split (train/validation/test)
            **kwargs: Dataset-specific configuration
        """
        self.name = name
        self.split = split
        self.config = kwargs
        self.examples: List[QAExample] = []
    
    @abstractmethod
    def load(self) -> None:
        """Load the dataset from source."""
        pass
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> QAExample:
        """Get example by index."""
        return self.examples[idx]
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)
    
    def get_subset(self, n: int, seed: Optional[int] = None) -> List[QAExample]:
        """Get a random subset of examples.
        
        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled examples
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        n = min(n, len(self.examples))
        return random.sample(self.examples, n)
    
    def filter_with_answers(self) -> None:
        """Filter dataset to only include examples with explicit answers.
        
        Removes examples where answers list is empty or contains only empty strings.
        Updates self.examples in-place.
        """
        original_count = len(self.examples)
        self.examples = [
            ex for ex in self.examples
            if ex.answers and any(answer.strip() for answer in ex.answers)
        ]
        filtered_count = original_count - len(self.examples)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} examples without explicit answers")
            print(f"Remaining: {len(self.examples)} examples")
    
    def get_examples_with_answers(self, n: Optional[int] = None) -> List[QAExample]:
        """Get examples that have explicit answers.
        
        Args:
            n: Optional maximum number of examples to return
            
        Returns:
            List of examples with non-empty answers
        """
        filtered = [
            ex for ex in self.examples
            if ex.answers and any(answer.strip() for answer in ex.answers)
        ]
        
        if n is not None:
            filtered = filtered[:n]
        
        return filtered
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', split='{self.split}', size={len(self)})"

