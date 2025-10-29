"""Fixed RAG agent - Baseline 2: Standard RAG with fixed parameters."""

from typing import Any, List

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Retriever


class FixedRAGAgent(RAGAgent):
    """Baseline RAG agent with fixed retrieval parameters.
    
    This agent implements the standard RAG pipeline:
    1. Retrieve top-k documents
    2. Format context with retrieved documents
    3. Generate answer using LLM with context
    """
    
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        top_k: int = 5,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.",
        context_template: str = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        **kwargs: Any
    ):
        """Initialize the fixed RAG agent.
        
        Args:
            name: Agent identifier
            model: The language model to use
            retriever: The retriever for finding relevant documents
            top_k: Number of documents to retrieve
            system_prompt: System prompt for the LLM
            context_template: Template for formatting retrieved context
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.context_template = context_template
    
    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer using fixed RAG pipeline.
        
        Args:
            query: The input question
            **kwargs: Additional generation parameters
            
        Returns:
            RAGResponse with the answer and retrieved context
        """
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k)
        
        # Create context
        context = RAGContext(
            query=query,
            retrieved_docs=retrieved_docs,
            metadata={"top_k": self.top_k}
        )
        
        # Format context for LLM
        context_text = self._format_context(retrieved_docs)
        prompt = f"{self.system_prompt}\n\n{self.context_template.format(context=context_text, query=query)}"
        
        # Generate answer
        answer = self.model.generate(prompt, **kwargs)
        
        # Return response
        return RAGResponse(
            answer=answer,
            context=context,
            metadata={"agent_type": "fixed_rag", "num_docs_used": len(retrieved_docs)}
        )
    
    def _format_context(self, docs: List[dict]) -> str:
        """Format retrieved documents into a context string.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not docs:
            return "No relevant context found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            text = doc.get("text", doc.get("content", ""))
            formatted.append(f"[{i}] {text}")
        
        return "\n\n".join(formatted)

