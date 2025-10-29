"""Exact match and F1 metrics."""

import re
import string
from collections import Counter
from typing import Any, Dict, List, Union

from ragicamp.metrics.base import Metric


class ExactMatchMetric(Metric):
    """Exact match metric with normalization."""
    
    def __init__(self, normalize: bool = True, **kwargs: Any):
        """Initialize exact match metric.
        
        Args:
            normalize: Whether to normalize text before comparison
            **kwargs: Additional configuration
        """
        super().__init__(name="exact_match", **kwargs)
        self.normalize = normalize
    
    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        **kwargs: Any
    ) -> float:
        """Compute exact match score."""
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Handle multiple references
            refs = [ref] if isinstance(ref, str) else ref
            
            if self.normalize:
                pred_norm = self._normalize(pred)
                refs_norm = [self._normalize(r) for r in refs]
            else:
                pred_norm = pred
                refs_norm = refs
            
            # Check if prediction matches any reference
            score = 1.0 if any(pred_norm == r for r in refs_norm) else 0.0
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text


class F1Metric(Metric):
    """Token-level F1 metric."""
    
    def __init__(self, **kwargs: Any):
        """Initialize F1 metric."""
        super().__init__(name="f1", **kwargs)
    
    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        **kwargs: Any
    ) -> float:
        """Compute F1 score."""
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Handle multiple references - take max F1
            refs = [ref] if isinstance(ref, str) else ref
            
            f1_scores = [self._compute_f1(pred, r) for r in refs]
            scores.append(max(f1_scores))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _compute_f1(self, prediction: str, reference: str) -> float:
        """Compute F1 between prediction and single reference."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # Compute token overlap
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1

