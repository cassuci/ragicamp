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
        
        # Compute per-question metrics
        per_question_metrics = self._compute_per_question_metrics(
            predictions, references, questions
        )
        
        # Save if requested
        if save_predictions and output_path:
            self._save_results(
                examples=examples,
                predictions=predictions,
                responses=responses,
                results=results,
                per_question_metrics=per_question_metrics,
                output_path=output_path
            )
        
        return results
    
    def _compute_per_question_metrics(
        self,
        predictions: List[str],
        references: List[Any],
        questions: List[str]
    ) -> List[Dict[str, Any]]:
        """Compute metrics for each individual question.
        
        Args:
            predictions: List of predictions
            references: List of references
            questions: List of questions
            
        Returns:
            List of per-question metric scores
        """
        per_question = []
        
        for i, (pred, ref, q) in enumerate(zip(predictions, references, questions)):
            question_metrics = {"question_index": i, "question": q}
            
            # Compute each metric individually for this question
            for metric in self.metrics:
                try:
                    score = metric.compute_single(pred, ref)
                    if isinstance(score, dict):
                        # Handle metrics that return multiple scores (like BERTScore)
                        for key, value in score.items():
                            question_metrics[key] = value
                    else:
                        question_metrics[metric.name] = score
                except Exception as e:
                    # If metric fails for this question, record None
                    question_metrics[metric.name] = None
            
            per_question.append(question_metrics)
        
        return per_question
    
    def _save_results(
        self,
        examples: List[Any],
        predictions: List[str],
        responses: List[Any],
        results: Dict[str, Any],
        per_question_metrics: List[Dict[str, Any]],
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
        import os
        from datetime import datetime
        
        output = {
            "results": results,
            "predictions": [
                {
                    "id": ex.id,
                    "question": ex.question,
                    "prediction": pred,
                    "expected_answer": ex.answers[0] if ex.answers else None,  # Primary expected answer
                    "all_acceptable_answers": ex.answers,  # All acceptable answers
                    "references": ex.answers,  # Keep for backward compatibility
                    "metadata": resp.metadata if hasattr(resp, "metadata") else {}
                }
                for ex, pred, resp in zip(examples, predictions, responses)
            ]
        }
        
        # Save main results file
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        # Save separate metrics summary file
        metrics_path = output_path.replace('.json', '_metrics.json')
        metrics_summary = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": results.get("agent_name", "unknown"),
            "dataset_name": results.get("dataset_name", "unknown"),
            "num_examples": results.get("num_examples", 0),
            "metrics": {
                k: v for k, v in results.items() 
                if k not in ["num_examples", "agent_name", "dataset_name"]
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        print(f"Metrics summary saved to {metrics_path}")
        
        # Also save a simple text summary
        txt_path = output_path.replace('.json', '_metrics.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EVALUATION METRICS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {metrics_summary['timestamp']}\n")
            f.write(f"Agent: {metrics_summary['agent_name']}\n")
            f.write(f"Dataset: {metrics_summary['dataset_name']}\n")
            f.write(f"Examples: {metrics_summary['num_examples']}\n\n")
            f.write("Metrics:\n")
            f.write("-" * 70 + "\n")
            for metric, value in metrics_summary['metrics'].items():
                if isinstance(value, float):
                    f.write(f"  {metric:30s}: {value:.4f}\n")
                else:
                    f.write(f"  {metric:30s}: {value}\n")
            f.write("=" * 70 + "\n")
        
        print(f"Metrics text summary saved to {txt_path}")
        
        # Save per-question metrics
        per_question_path = output_path.replace('.json', '_per_question.json')
        per_question_output = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": results.get("agent_name", "unknown"),
            "dataset_name": results.get("dataset_name", "unknown"),
            "num_examples": len(per_question_metrics),
            "per_question_metrics": per_question_metrics
        }
        
        with open(per_question_path, 'w') as f:
            json.dump(per_question_output, f, indent=2)
        
        print(f"Per-question metrics saved to {per_question_path}")
        
        # Also save a CSV for easy analysis in spreadsheets
        csv_path = output_path.replace('.json', '_per_question.csv')
        try:
            import csv
            if per_question_metrics:
                with open(csv_path, 'w', newline='') as f:
                    # Get all unique field names
                    fieldnames = set()
                    for item in per_question_metrics:
                        fieldnames.update(item.keys())
                    fieldnames = sorted(list(fieldnames))
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(per_question_metrics)
                
                print(f"Per-question CSV saved to {csv_path}")
        except Exception as e:
            print(f"Note: Could not save CSV format: {e}")
    
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

