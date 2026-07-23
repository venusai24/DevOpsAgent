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
        fallback_structured_list = [f.with_structured_output(schema, **kwargs) for f in self.fallbacks]
        return primary_structured.with_fallbacks(fallback_structured_list, exceptions_to_handle=(Exception,))
        
    def bind_tools(self, tools: list, **kwargs):
        primary_bound = self.primary.bind_tools(tools, **kwargs)
        fallback_bound_list = [f.bind_tools(tools, **kwargs) for f in self.fallbacks]
        return FallbackModelWrapper(primary_bound, fallback_bound_list)

    def invoke(self, *args, **kwargs):
        return self.primary.with_fallbacks(self.fallbacks, exceptions_to_handle=(Exception,)).invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.primary.with_fallbacks(self.fallbacks, exceptions_to_handle=(Exception,)).ainvoke(*args, **kwargs)

class LLMFactory:
    """Factory to instantiate specialized LLMs per agent role."""
    
    @staticmethod
    def get_llm(role: Literal["triage", "evidence", "reasoning", "report"]) -> BaseChatModel:
        """
        Returns a configured Langchain ChatModel using Gemma via Google AI Studio as primary,
        and a lightweight fallback model on OpenRouter.
        """
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        google_model = os.environ.get("GOOGLE_MODEL", "gemma-2-27b-it")
        openrouter_api_key = os.environ.get("OPENROUTER_KEY")
        
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY is missing from environment variables!")
        if not openrouter_api_key:
            logger.warning("OPENROUTER_KEY is missing from environment variables!")
            
        logger.debug(f"Instantiating Gemini/Google Studio model '{google_model}' for role '{role}'")
        
        # Primary Model: Gemma via Google AI Studio
        primary_llm = ChatGoogleGenerativeAI(
            model=google_model,
            google_api_key=gemini_api_key,
            temperature=0.0,
            max_retries=3,
            timeout=60,
        )
        
        # Fallback Model 1: OLMo 3 32B Think via OpenRouter
        olmo_llm = ChatOpenAI(
            model="allenai/olmo-3-32b-think",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_retries=1,
            timeout=30,
        )
        
        # Fallback Model 2: Lightweight Llama 3.1 8B Instruct via OpenRouter
        lightweight_fallback_llm = ChatOpenAI(
            model="meta-llama/llama-3.1-8b-instruct",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_retries=3,
            timeout=60,
        )
        
        return FallbackModelWrapper(primary_llm, [olmo_llm, lightweight_fallback_llm])
