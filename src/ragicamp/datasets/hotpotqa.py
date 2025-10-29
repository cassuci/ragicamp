"""HotpotQA dataset loader."""

from typing import Any

from datasets import load_dataset

from ragicamp.datasets.base import QADataset, QAExample


class HotpotQADataset(QADataset):
    """Loader for HotpotQA dataset.
    
    HotpotQA requires reasoning over multiple Wikipedia passages
    to answer questions (multi-hop reasoning).
    """
    
    def __init__(self, split: str = "train", distractor: bool = True, **kwargs: Any):
        """Initialize HotpotQA dataset.
        
        Args:
            split: Dataset split (train/validation)
            distractor: Whether to use distractor setting (with irrelevant contexts)
            **kwargs: Additional configuration
        """
        super().__init__(name="hotpotqa", split=split, **kwargs)
        self.distractor = distractor
        self.load()
    
    def load(self) -> None:
        """Load HotpotQA from HuggingFace datasets."""
        # Map split names
        hf_split = "train" if self.split == "train" else "validation"
        subset = "distractor" if self.distractor else "fullwiki"
        
        # Load from HuggingFace
        dataset = load_dataset("hotpot_qa", subset, split=hf_split)
        
        # Convert to our format
        for item in dataset:
            # Combine context passages
            context_parts = []
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                context_parts.append(f"{title}: {' '.join(sentences)}")
            
            example = QAExample(
                id=item["id"],
                question=item["question"],
                answers=[item["answer"]],
                context="\n\n".join(context_parts),
                metadata={
                    "source": "hotpotqa",
                    "level": item.get("level", ""),
                    "type": item.get("type", "")
                }
            )
            self.examples.append(example)

