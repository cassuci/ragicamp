"""Natural Questions dataset loader."""

from typing import Any

from datasets import load_dataset

from ragicamp.datasets.base import QADataset, QAExample


class NaturalQuestionsDataset(QADataset):
    """Loader for Google's Natural Questions dataset.
    
    Natural Questions contains real Google search queries with
    answers from Wikipedia articles.
    """
    
    def __init__(self, split: str = "train", **kwargs: Any):
        """Initialize NQ dataset.
        
        Args:
            split: Dataset split (train/validation)
            **kwargs: Additional configuration
        """
        super().__init__(name="natural_questions", split=split, **kwargs)
        self.load()
    
    def load(self) -> None:
        """Load Natural Questions from HuggingFace datasets."""
        # Load from HuggingFace
        dataset = load_dataset("nq_open", split=self.split)
        
        # Convert to our format
        for i, item in enumerate(dataset):
            example = QAExample(
                id=f"nq_{self.split}_{i}",
                question=item["question"],
                answers=item["answer"],  # NQ has multiple acceptable answers
                context=None,  # Open-domain version doesn't include context
                metadata={"source": "natural_questions"}
            )
            self.examples.append(example)

