"""Bandit-based RAG agent - Adaptive parameter selection using bandit algorithms."""

from typing import Any, Dict

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.models.base import LanguageModel
from ragicamp.policies.base import Policy
from ragicamp.retrievers.base import Retriever


class BanditRAGAgent(RAGAgent):
    """RAG agent that uses bandit policies to adaptively select parameters.
    
    This agent uses bandit algorithms to dynamically choose:
    - Number of documents to retrieve (top_k)
    - Retrieval strategy
    - Context formatting approach
    - Other RAG hyperparameters
    
    The policy learns which parameter configurations work best over time.
    """
    
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        policy: Policy,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.",
        **kwargs: Any
    ):
        """Initialize the bandit-based RAG agent.
        
        Args:
            name: Agent identifier
            model: The language model to use
            retriever: The retriever for finding relevant documents
            policy: Bandit policy for parameter selection
            system_prompt: System prompt for the LLM
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.policy = policy
        self.system_prompt = system_prompt
    
    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer using bandit-selected parameters.
        
        Args:
            query: The input question
            **kwargs: Additional generation parameters
            
        Returns:
            RAGResponse with the answer and selected parameters
        """
        # Use policy to select RAG parameters
        rag_params = self.policy.select_action(query=query)
        
        # Retrieve documents with selected parameters
        top_k = rag_params.get("top_k", 5)
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k, **rag_params)
        
        # Create context
        context = RAGContext(
            query=query,
            retrieved_docs=retrieved_docs,
            metadata={"selected_params": rag_params}
        )
        
        # Format and generate
        context_text = self._format_context(retrieved_docs, rag_params)
        prompt = self._build_prompt(query, context_text)
        answer = self.model.generate(prompt, **kwargs)
        
        # Return response with selected parameters
        return RAGResponse(
            answer=answer,
            context=context,
            metadata={
                "agent_type": "bandit_rag",
                "selected_params": rag_params,
                "num_docs_used": len(retrieved_docs)
            }
        )
    
    def update_policy(self, query: str, params: Dict[str, Any], reward: float) -> None:
        """Update the bandit policy based on observed reward.
        
        Args:
            query: The query that was answered
            params: The parameters that were selected
            reward: The reward/score achieved
        """
        self.policy.update(query=query, action=params, reward=reward)
    
    def _build_prompt(self, query: str, context_text: str) -> str:
        """Build the prompt for the LLM."""
        return f"{self.system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    
    def _format_context(self, docs: list, params: Dict[str, Any]) -> str:
        """Format documents based on selected parameters."""
        if not docs:
            return "No relevant context found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            text = doc.get("text", doc.get("content", ""))
            formatted.append(f"[{i}] {text}")
        
        return "\n\n".join(formatted)

