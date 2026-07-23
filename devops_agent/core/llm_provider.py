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
        Returns a configured Langchain ChatModel based on the agent's role.
        """
        google_model = os.environ.get("GOOGLE_MODEL", "gemma-2-27b-it")
        fallback_model = "gemma-2-9b-it"
        
        # We fetch the Gemini API keys provided in the .env file
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        gemini_api_key_backup = os.environ.get("GEMINI_API_KEY_BACKUP")
        
        openrouter_api_key = os.environ.get("OPENROUTER_KEY")
        openrouter_api_key_backup = os.environ.get("OPENROUTER_KEY_BACKUP")
        openrouter_api_key_backup_2 = os.environ.get("OPENROUTER_KEY_BACKUP_2")
        
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY is missing from environment variables!")
        if not openrouter_api_key:
            logger.warning("OPENROUTER_KEY is missing from environment variables!")
            
        logger.debug(f"Instantiating Gemini/Google Studio model '{google_model}' for role '{role}'")
        
        # Primary Model: Gemma via Google AI Studio
        primary_llm = ChatGoogleGenerativeAI(
            model=google_model,
            google_api_key=gemini_api_key,
            temperature=0.0, # Zero temperature for strict deterministic RCA
            max_retries=3,   # Fail quickly on 503 to trigger fallback
            timeout=240,     # Increased timeout for large contexts
        )
        
        fallbacks = []
        
        # Add primary model with backup key
        if gemini_api_key_backup:
            fallbacks.append(ChatGoogleGenerativeAI(
                model=google_model,
                google_api_key=gemini_api_key_backup,
                temperature=0.0,
                max_retries=2,
                timeout=240,
            ))
            
        # Add secondary Gemini model with primary key
        if gemini_api_key:
            fallbacks.append(ChatGoogleGenerativeAI(
                model=fallback_model,
                google_api_key=gemini_api_key,
                temperature=0.0,
                max_retries=2,
                timeout=240,
            ))
            
        # Add secondary Gemini model with backup key
        if gemini_api_key_backup:
            fallbacks.append(ChatGoogleGenerativeAI(
                model=fallback_model,
                google_api_key=gemini_api_key_backup,
                temperature=0.0,
                max_retries=2,
                timeout=240,
            ))
            
        # Add OpenRouter fallback models for each available key
        openrouter_keys = [k for k in [openrouter_api_key, openrouter_api_key_backup, openrouter_api_key_backup_2] if k]
        for or_key in openrouter_keys:
            # Fallback Model 1: Llama 3.1 70B via OpenRouter (supports parallel tool calls)
            fallbacks.append(ChatOpenAI(
                model="meta-llama/llama-3.1-70b-instruct",
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_retries=2,
                timeout=240,
            ))
            # Fallback Model 2: GPT-OSS 20B Free via OpenRouter
            fallbacks.append(ChatOpenAI(
                model="openai/gpt-oss-20b:free",
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_retries=2,
                timeout=240,
            ))
            # Fallback Model 3: Mistral Nemo via OpenRouter
            fallbacks.append(ChatOpenAI(
                model="mistralai/mistral-nemo",
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_retries=2,
                timeout=240,
            ))
        
        return FallbackModelWrapper(primary_llm, fallbacks)
