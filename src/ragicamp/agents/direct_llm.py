"""Direct LLM agent - Baseline 1: No retrieval, just ask the LLM."""

from typing import Any

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.models.base import LanguageModel
from ragicamp.utils.prompts import PromptBuilder


class DirectLLMAgent(RAGAgent):
    """Baseline agent that directly queries an LLM without retrieval.
    
    This is the simplest baseline: just prompt the LLM with the question
    and return its answer.
    """
    
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        system_prompt: str = "You are a helpful assistant. Answer questions accurately and concisely.",
        **kwargs: Any
    ):
        """Initialize the direct LLM agent.
        
        Args:
            name: Agent identifier
            model: The language model to use
            system_prompt: System prompt for the LLM
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.prompt_builder = PromptBuilder(system_prompt=system_prompt)
    
    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer by directly querying the LLM.
        
        Args:
            query: The input question
            **kwargs: Additional generation parameters
            
        Returns:
            RAGResponse with the LLM's answer
        """
        # Create context (no retrieval for this baseline)
        context = RAGContext(query=query)
        
        # Build prompt using utility
        prompt = self.prompt_builder.build_direct_prompt(query)
        
        # Generate answer
        answer = self.model.generate(prompt, **kwargs)
        
        # Return response
        return RAGResponse(
            answer=answer,
            context=context,
            metadata={"agent_type": "direct_llm"}
        )

