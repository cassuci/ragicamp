"""Evaluator for RAG systems."""

import json
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ragicamp.agents.base import RAGAgent
from ragicamp.datasets.base import QADataset
from ragicamp.metrics.base import Metric


class Evaluator:
    """Evaluator for RAG agents.
    
    Runs evaluation on a dataset and computes multiple metrics.
    """
    
    def __init__(
        self,
        agent: RAGAgent,
        dataset: QADataset,
        metrics: List[Metric],
        **kwargs: Any
    ):
        """Initialize evaluator.
        
        Args:
            agent: The RAG agent to evaluate
            dataset: Evaluation dataset
            metrics: List of metrics to compute
            **kwargs: Additional configuration
        """
        self.agent = agent
        self.dataset = dataset
        self.metrics = metrics
        self.config = kwargs
    
    def evaluate(
        self,
        num_examples: Optional[int] = None,
        save_predictions: bool = False,
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate the agent on the dataset.
        
        Args:
            num_examples: Evaluate on first N examples (None = all)
            save_predictions: Whether to save predictions
            output_path: Path to save results
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary with metric scores and statistics
        """
        # Prepare examples
        examples = list(self.dataset)
        if num_examples is not None:
            examples = examples[:num_examples]
        
        # Generate predictions
        predictions = []
        references = []
        questions = []
        responses = []
        
        print(f"Evaluating on {len(examples)} examples...")
        for example in tqdm(examples, desc="Generating answers"):
            # Generate answer
            response = self.agent.answer(example.question)
            
            # Store
            predictions.append(response.answer)
            references.append(example.answers)
            questions.append(example.question)
            responses.append(response)
        
        # Compute metrics
        print("\nComputing metrics...")
        results = {}
        
        for metric in self.metrics:
            print(f"  - {metric.name}")
            
            # Handle metrics that need questions
            if metric.name == "llm_judge":
                score = metric.compute(
                    predictions=predictions,
                    references=references,
                    questions=questions
                )
            else:
                score = metric.compute(
                    predictions=predictions,
                    references=references
                )
            
            # Store score(s)
            if isinstance(score, dict):
                results.update(score)
            else:
                results[metric.name] = score
        
        # Add statistics
        results["num_examples"] = len(examples)
        results["agent_name"] = self.agent.name
        results["dataset_name"] = self.dataset.name
        
        # Save if requested
        if save_predictions and output_path:
            self._save_results(
                examples=examples,
                predictions=predictions,
                responses=responses,
                results=results,
                output_path=output_path
            )
        
        return results
    
    def _save_results(
        self,
        examples: List[Any],
        predictions: List[str],
        responses: List[Any],
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Save evaluation results to file.
        
        Args:
            examples: Dataset examples
            predictions: Generated predictions
            responses: Full agent responses
            results: Metric scores
            output_path: Path to save results
        """
        output = {
            "results": results,
            "predictions": [
                {
                    "id": ex.id,
                    "question": ex.question,
                    "prediction": pred,
                    "references": ex.answers,
                    "metadata": resp.metadata if hasattr(resp, "metadata") else {}
                }
                for ex, pred, resp in zip(examples, predictions, responses)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def compare_agents(
        self,
        agents: List[RAGAgent],
        num_examples: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple agents on the same dataset.
        
        Args:
            agents: List of agents to compare
            num_examples: Number of examples to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping agent names to their results
        """
        all_results = {}
        
        for agent in agents:
            print(f"\n{'='*60}")
            print(f"Evaluating agent: {agent.name}")
            print(f"{'='*60}")
            
            # Temporarily set agent
            original_agent = self.agent
            self.agent = agent
            
            # Evaluate
            results = self.evaluate(num_examples=num_examples, **kwargs)
            all_results[agent.name] = results
            
            # Restore original agent
            self.agent = original_agent
        
        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        for agent_name, results in all_results.items():
            print(f"\n{agent_name}:")
            for metric_name, score in results.items():
                if metric_name not in ["num_examples", "agent_name", "dataset_name"]:
                    print(f"  {metric_name}: {score:.4f}" if isinstance(score, float) else f"  {metric_name}: {score}")
        
        return all_results

