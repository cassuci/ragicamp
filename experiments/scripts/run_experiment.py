#!/usr/bin/env python3
"""Main experiment runner script."""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.agents.bandit_rag import BanditRAGAgent
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.agents.mdp_rag import MDPRAGAgent
from ragicamp.datasets.hotpotqa import HotpotQADataset
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.triviaqa import TriviaQADataset
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.policies.bandits import EpsilonGreedyBandit, UCBBandit
from ragicamp.policies.mdp import QLearningMDPPolicy, RandomMDPPolicy
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.retrievers.sparse import SparseRetriever
from ragicamp.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: dict):
    """Create model from configuration."""
    model_type = config.get("type", "huggingface")
    
    if model_type == "huggingface":
        return HuggingFaceModel(
            model_name=config["model_name"],
            device=config.get("device", "cpu")
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_retriever(config: dict):
    """Create retriever from configuration."""
    retriever_type = config.get("type", "dense")
    
    if retriever_type == "dense":
        return DenseRetriever(
            name=config.get("name", "dense_retriever"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            index_type=config.get("index_type", "flat")
        )
    elif retriever_type == "sparse":
        return SparseRetriever(
            name=config.get("name", "sparse_retriever"),
            max_features=config.get("max_features", 10000)
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def create_policy(config: dict):
    """Create policy from configuration."""
    policy_type = config.get("type", "random")
    
    if policy_type == "epsilon_greedy":
        return EpsilonGreedyBandit(
            name=config.get("name", "epsilon_greedy"),
            actions=config.get("actions", []),
            epsilon=config.get("epsilon", 0.1)
        )
    elif policy_type == "ucb":
        return UCBBandit(
            name=config.get("name", "ucb"),
            actions=config.get("actions", []),
            c=config.get("c", 2.0)
        )
    elif policy_type == "qlearning":
        return QLearningMDPPolicy(
            name=config.get("name", "qlearning"),
            action_types=config.get("action_types", ["retrieve", "reformulate", "generate"]),
            learning_rate=config.get("learning_rate", 0.1),
            discount_factor=config.get("discount_factor", 0.95),
            epsilon=config.get("epsilon", 0.1)
        )
    elif policy_type == "random":
        return RandomMDPPolicy(
            name=config.get("name", "random"),
            action_types=config.get("action_types", ["retrieve", "reformulate", "generate"])
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def create_agent(config: dict, model, retriever=None, policy=None):
    """Create agent from configuration."""
    agent_type = config.get("type", "direct_llm")
    
    if agent_type == "direct_llm":
        return DirectLLMAgent(
            name=config.get("name", "direct_llm"),
            model=model,
            system_prompt=config.get("system_prompt", "")
        )
    elif agent_type == "fixed_rag":
        return FixedRAGAgent(
            name=config.get("name", "fixed_rag"),
            model=model,
            retriever=retriever,
            top_k=config.get("top_k", 5),
            system_prompt=config.get("system_prompt", "")
        )
    elif agent_type == "bandit_rag":
        return BanditRAGAgent(
            name=config.get("name", "bandit_rag"),
            model=model,
            retriever=retriever,
            policy=policy,
            system_prompt=config.get("system_prompt", "")
        )
    elif agent_type == "mdp_rag":
        return MDPRAGAgent(
            name=config.get("name", "mdp_rag"),
            model=model,
            retriever=retriever,
            policy=policy,
            max_steps=config.get("max_steps", 5),
            system_prompt=config.get("system_prompt", "")
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_dataset(config: dict):
    """Create dataset from configuration."""
    dataset_name = config.get("name", "natural_questions")
    split = config.get("split", "validation")
    
    if dataset_name == "natural_questions":
        dataset = NaturalQuestionsDataset(split=split)
    elif dataset_name == "hotpotqa":
        dataset = HotpotQADataset(split=split)
    elif dataset_name == "triviaqa":
        dataset = TriviaQADataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit number of examples if specified
    num_examples = config.get("num_examples")
    if num_examples and num_examples < len(dataset):
        dataset.examples = dataset.examples[:num_examples]
    
    return dataset


def create_metrics(config: list):
    """Create metrics from configuration."""
    metrics = []
    
    for metric_name in config:
        if metric_name == "exact_match":
            metrics.append(ExactMatchMetric())
        elif metric_name == "f1":
            metrics.append(F1Metric())
        # Add more metrics as needed
    
    return metrics


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run RAGiCamp experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="eval",
        help="Run mode: train or eval"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create components
    print("Creating model...")
    model = create_model(config["model"])
    
    retriever = None
    if "retriever" in config:
        print("Creating retriever...")
        retriever = create_retriever(config["retriever"])
        # TODO: Index documents from corpus
    
    policy = None
    if "policy" in config:
        print("Creating policy...")
        policy = create_policy(config["policy"])
    
    print("Creating agent...")
    agent = create_agent(config["agent"], model, retriever, policy)
    
    print("Loading dataset...")
    dataset = create_dataset(config["dataset"])
    print(f"Dataset size: {len(dataset)}")
    
    print("Creating metrics...")
    metrics = create_metrics(config["metrics"])
    
    # Run experiment
    if args.mode == "train":
        print("\n=== Training Mode ===")
        trainer = Trainer(
            agent=agent,
            dataset=dataset,
            metrics=metrics,
            reward_metric="f1"
        )
        
        training_config = config.get("training", {})
        results = trainer.train(
            num_epochs=training_config.get("num_epochs", 1),
            eval_interval=training_config.get("eval_interval", 100)
        )
        
        # Save policy if applicable
        output_config = config.get("output", {})
        if output_config.get("save_policy") and policy:
            policy_path = output_config.get("policy_path", "outputs/policy.json")
            os.makedirs(os.path.dirname(policy_path), exist_ok=True)
            policy.save(policy_path)
            print(f"Policy saved to {policy_path}")
    
    else:  # eval mode
        print("\n=== Evaluation Mode ===")
        evaluator = Evaluator(
            agent=agent,
            dataset=dataset,
            metrics=metrics
        )
        
        output_config = config.get("output", {})
        results = evaluator.evaluate(
            save_predictions=output_config.get("save_predictions", False),
            output_path=output_config.get("output_path")
        )
        
        # Print results
        print("\n=== Results ===")
        for metric_name, score in results.items():
            if metric_name not in ["num_examples", "agent_name", "dataset_name"]:
                if isinstance(score, float):
                    print(f"{metric_name}: {score:.4f}")
                else:
                    print(f"{metric_name}: {score}")


if __name__ == "__main__":
    main()

