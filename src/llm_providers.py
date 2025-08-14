"""
LLM Provider Management for Voice Agent API
Supports OpenAI, Anthropic, Google, and Groq providers with easy switching
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.config import (
    LLM_PROVIDER_TYPE, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    LLM_ENDPOINT_URL, LLM_ENDPOINT_HEADERS,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY
)

@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider"""
    provider_type: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 150
    endpoint_url: Optional[str] = None
    endpoint_headers: Optional[Dict[str, str]] = None
    api_key_required: bool = True
    pricing_tier: str = "Standard"

class LLMProviderManager:
    """Manages LLM provider configurations and switching"""
    
    # Supported models from the documentation
    SUPPORTED_MODELS = {
        "open_ai": {
            "gpt-4.1": {"pricing_tier": "Advanced"},
            "gpt-4.1-mini": {"pricing_tier": "Standard"},
            "gpt-4.1-nano": {"pricing_tier": "Standard"},
            "gpt-4o": {"pricing_tier": "Advanced"},
            "gpt-4o-mini": {"pricing_tier": "Standard"},
        },
        "anthropic": {
            "claude-3-5-haiku-latest": {"pricing_tier": "Standard"},
            "claude-sonnet-4-20250514": {"pricing_tier": "Advanced"},
        },
        "google": {
            "gemini-2.5-flash": {"pricing_tier": "Standard"},
            "gemini-2.0-flash": {"pricing_tier": "Standard"},
            "gemini-2.0-flash-lite": {"pricing_tier": "Standard"},
        },
        "groq": {
            "openai/gpt-oss-20b": {"pricing_tier": "Standard"},
        }
    }
    
    def __init__(self):
        self.current_config = self._load_current_config()
    
    def _load_current_config(self) -> LLMProviderConfig:
        """Load current configuration from environment"""
        return LLMProviderConfig(
            provider_type=LLM_PROVIDER_TYPE,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            endpoint_url=LLM_ENDPOINT_URL,
            endpoint_headers=LLM_ENDPOINT_HEADERS,
        )
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available providers with their API key status"""
        providers = {}
        
        # OpenAI
        providers["open_ai"] = {
            "name": "OpenAI",
            "api_key_available": bool(OPENAI_API_KEY),
            "models": list(self.SUPPORTED_MODELS["open_ai"].keys()),
            "endpoint_required": False,
        }
        
        # Anthropic
        providers["anthropic"] = {
            "name": "Anthropic",
            "api_key_available": bool(ANTHROPIC_API_KEY),
            "models": list(self.SUPPORTED_MODELS["anthropic"].keys()),
            "endpoint_required": False,
        }
        
        # Google
        providers["google"] = {
            "name": "Google",
            "api_key_available": bool(GEMINI_API_KEY),
            "models": list(self.SUPPORTED_MODELS["google"].keys()),
            "endpoint_required": True,
        }
        
        # Groq
        providers["groq"] = {
            "name": "Groq",
            "api_key_available": bool(GROQ_API_KEY),
            "models": list(self.SUPPORTED_MODELS["groq"].keys()),
            "endpoint_required": True,
        }
        
        return providers
    
    def get_recommended_models(self, provider_type: str, pricing_preference: str = "Standard") -> List[str]:
        """Get recommended models for a provider based on pricing preference"""
        if provider_type not in self.SUPPORTED_MODELS:
            return []
        
        models = []
        for model, config in self.SUPPORTED_MODELS[provider_type].items():
            if config.get("pricing_tier", "Standard") == pricing_preference:
                models.append(model)
        
        return models
    
    def validate_configuration(self, provider_type: str, model: str) -> Dict[str, Any]:
        """Validate a provider/model configuration"""
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "requirements": []
        }
        
        # Check if provider is supported
        if provider_type not in self.SUPPORTED_MODELS:
            result["errors"].append(f"Unsupported provider: {provider_type}")
            return result
        
        # Check if model is supported
        if model not in self.SUPPORTED_MODELS[provider_type]:
            result["warnings"].append(f"Model {model} not in supported list - using custom model")
        
        # Check API key requirements
        api_key_map = {
            "open_ai": OPENAI_API_KEY,
            "anthropic": ANTHROPIC_API_KEY,
            "google": GEMINI_API_KEY,
            "groq": GROQ_API_KEY,
        }
        
        if provider_type in api_key_map:
            if not api_key_map[provider_type]:
                result["errors"].append(f"API key required for {provider_type}")
                result["requirements"].append(f"Set {provider_type.upper()}_API_KEY environment variable")
        
        # Check endpoint requirements
        endpoint_required_providers = ["google", "groq"]
        if provider_type in endpoint_required_providers and not LLM_ENDPOINT_URL:
            result["requirements"].append(f"Custom endpoint URL required for {provider_type}")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def generate_voice_agent_config(self, provider_type: str, model: str, 
                                  temperature: float = 0.7, endpoint_url: str = None,
                                  endpoint_headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Generate Voice Agent API configuration for the specified provider"""
        config = {
            "think": {
                "provider": {
                    "type": provider_type,
                    "model": model,
                    "temperature": temperature
                }
            }
        }
        
        # Add endpoint configuration if provided
        if endpoint_url or endpoint_headers:
            config["think"]["endpoint"] = {}
            if endpoint_url:
                config["think"]["endpoint"]["url"] = endpoint_url
            if endpoint_headers:
                config["think"]["endpoint"]["headers"] = endpoint_headers
        
        return config
    
    def get_provider_examples(self) -> Dict[str, Dict[str, Any]]:
        """Get example configurations for each provider"""
        examples = {}
        
        # OpenAI example
        examples["open_ai"] = {
            "description": "OpenAI GPT models with managed endpoint",
            "config": self.generate_voice_agent_config("open_ai", "gpt-4o-mini"),
            "env_vars": ["OPENAI_API_KEY"]
        }
        
        # Anthropic example
        examples["anthropic"] = {
            "description": "Anthropic Claude models with managed endpoint",
            "config": self.generate_voice_agent_config("anthropic", "claude-3-5-haiku-latest"),
            "env_vars": ["ANTHROPIC_API_KEY"]
        }
        
        # Google example
        examples["google"] = {
            "description": "Google Gemini models with custom endpoint",
            "config": self.generate_voice_agent_config(
                "google", 
                "gemini-2.0-flash",
                endpoint_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                endpoint_headers={"authorization": "Bearer {{token}}"}
            ),
            "env_vars": ["GEMINI_API_KEY", "LLM_ENDPOINT_URL"]
        }
        
        # Groq example
        examples["groq"] = {
            "description": "Groq models with custom endpoint",
            "config": self.generate_voice_agent_config(
                "groq",
                "openai/gpt-oss-20b",
                endpoint_url="https://api.groq.com/openai/v1/chat/completions",
                endpoint_headers={"authorization": "Bearer {{token}}"}
            ),
            "env_vars": ["GROQ_API_KEY", "LLM_ENDPOINT_URL"]
        }
        
        return examples

def print_provider_status():
    """Print current provider status and available options"""
    manager = LLMProviderManager()
    
    print("=== LLM Provider Status ===")
    print(f"Current Provider: {manager.current_config.provider_type}")
    print(f"Current Model: {manager.current_config.model}")
    print(f"Temperature: {manager.current_config.temperature}")
    print(f"Max Tokens: {manager.current_config.max_tokens}")
    
    if manager.current_config.endpoint_url:
        print(f"Custom Endpoint: {manager.current_config.endpoint_url}")
    
    print("\n=== Available Providers ===")
    providers = manager.get_available_providers()
    
    for provider_id, info in providers.items():
        status = "✅" if info["api_key_available"] else "❌"
        print(f"{status} {info['name']} ({provider_id})")
        print(f"   Models: {', '.join(info['models'])}")
        print(f"   API Key: {'Available' if info['api_key_available'] else 'Missing'}")
        print(f"   Endpoint Required: {'Yes' if info['endpoint_required'] else 'No'}")
        print()

if __name__ == "__main__":
    print_provider_status()