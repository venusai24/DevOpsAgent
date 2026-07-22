"""LLM Factory Provider for Orchestration Agents."""

import logging
import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load the keys from .env
load_dotenv()

logger = logging.getLogger(__name__)

class FallbackModelWrapper:
    """Wrapper that seamlessly applies LangChain fallbacks to a primary and multiple secondary models, including structured output."""
    def __init__(self, primary: BaseChatModel, fallbacks: list[BaseChatModel]):
        self.primary = primary
        self.fallbacks = fallbacks
        
    def with_structured_output(self, schema: Any, **kwargs: Any):
        primary_structured = self.primary.with_structured_output(schema, **kwargs)
        
        fallback_kwargs = kwargs.copy()
        # Open-source models on OpenRouter often fail with 400 if strict=True is sent
        fallback_kwargs["strict"] = False
        
        fallback_structured_list = []
        for f in self.fallbacks:
            try:
                if getattr(f, "__class__", None).__name__ == "ChatOpenAI":
                    fallback_structured_list.append(f.with_structured_output(schema, **fallback_kwargs))
                else:
                    fallback_structured_list.append(f.with_structured_output(schema, **kwargs))
            except Exception:
                fallback_structured_list.append(f.with_structured_output(schema, **kwargs))
                
        return primary_structured.with_fallbacks(fallback_structured_list)
        
    def bind_tools(self, tools: list, **kwargs):
        primary_bound = self.primary.bind_tools(tools, **kwargs)
        
        fallback_kwargs = kwargs.copy()
        fallback_kwargs["strict"] = False
        
        fallback_bound_list = []
        for f in self.fallbacks:
            try:
                if getattr(f, "__class__", None).__name__ == "ChatOpenAI":
                    fallback_bound_list.append(f.bind_tools(tools, **fallback_kwargs))
                else:
                    fallback_bound_list.append(f.bind_tools(tools, **kwargs))
            except Exception:
                fallback_bound_list.append(f.bind_tools(tools, **kwargs))
                
        return FallbackModelWrapper(primary_bound, fallback_bound_list)

    def invoke(self, *args, **kwargs):
        return self.primary.with_fallbacks(self.fallbacks).invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.primary.with_fallbacks(self.fallbacks).ainvoke(*args, **kwargs)

class LLMFactory:
    """Factory to instantiate specialized LLMs per agent role."""
    
    @staticmethod
    def get_llm(role: Literal["triage", "evidence", "reasoning", "report", "verifier", "rca"]) -> BaseChatModel:
        """
        Returns a configured Langchain ChatModel based on the agent's role.
        """
        # The user requested Gemma for testing. 
        # Note: If the specific model name differs in the Google API, update it here.
        model_name = os.environ.get("GOOGLE_MODEL", "gemma-4-31b-it")
        fallback_model = "gemma-4-26b-a4b"
        
        # We fetch the Gemini API key you provided in the .env file
        google_api_key = os.environ.get("GEMINI_API_KEY")
        openrouter_api_key = os.environ.get("OPENROUTER_KEY")
        
        if not google_api_key:
            logger.warning("GEMINI_API_KEY is missing from environment variables!")
            
        logger.debug(f"Instantiating Google API LLM '{model_name}' for role '{role}'")
        
        primary_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.0, # Zero temperature for strict deterministic RCA
            max_retries=2,   # Fail quickly on 503 to trigger fallback
            timeout=60,      # Force timeout if server hangs
        )
        
        secondary_llm = ChatGoogleGenerativeAI(
            model=fallback_model,
            google_api_key=google_api_key,
            temperature=0.0,
            max_retries=2,
            timeout=60,
        )
        
        llama_llm = ChatOpenAI(
            model="meta-llama/llama-3.1-70b-instruct",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_retries=2,
            timeout=60,
        )
        
        mistral_llm = ChatOpenAI(
            model="mistralai/mistral-nemo",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_retries=2,
            timeout=60,
        )
        
        return FallbackModelWrapper(primary_llm, [secondary_llm, llama_llm, mistral_llm])
