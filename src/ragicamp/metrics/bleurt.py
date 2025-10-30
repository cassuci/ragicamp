"""BLEURT metric implementation."""

from typing import Any, Dict, List, Union

from ragicamp.metrics.base import Metric


class BLEURTMetric(Metric):
    """BLEURT (Bilingual Evaluation Understudy with Representations from Transformers).
    
    A learned metric for natural language generation that correlates well with
    human judgments.
    """
    
    def __init__(self, checkpoint: str = "BLEURT-20", **kwargs: Any):
        """Initialize BLEURT metric.
        
        Args:
            checkpoint: BLEURT checkpoint to use (e.g., 'BLEURT-20', 'bleurt-large-512')
            **kwargs: Additional configuration
        """
        super().__init__(name="bleurt", **kwargs)
        self.checkpoint = checkpoint
        
        # Lazy import to avoid requiring bleurt unless used
        try:
            from bleurt import score as bleurt_score
            self.scorer = bleurt_score.BleurtScorer(checkpoint)
        except ImportError:
            raise ImportError(
                "BLEURT is required for BLEURTMetric. "
                "Install with: pip install git+https://github.com/google-research/bleurt.git"
            )
        except Exception as e:
            # Handle checkpoint download issues
            raise RuntimeError(
                f"Failed to load BLEURT checkpoint '{checkpoint}'. "
                f"Error: {e}\n"
                "You may need to download the checkpoint first."
            )
    
    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        **kwargs: Any
    ) -> Dict[str, float]:
        """Compute BLEURT scores.
        
        Returns:
            Dict with average BLEURT score
        """
        # Handle multiple references - take first one for now
        refs = []
        for ref in references:
            if isinstance(ref, list):
                refs.append(ref[0] if ref else "")
            else:
                refs.append(ref)
        
        # Compute BLEURT scores
        scores = self.scorer.score(references=refs, candidates=predictions)
        
        return {
            "bleurt": float(sum(scores) / len(scores)) if scores else 0.0,
            "bleurt_scores": [float(s) for s in scores]  # Individual scores
        }

