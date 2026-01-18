import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str

def get_llm_config() -> LLMConfig:
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
        
    return LLMConfig(api_key, base_url, model)
