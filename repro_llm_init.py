import sys
import os
import logging

# Configure logging to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock settings
class MockSettings:
    local_llm_model = "qwen2.5:1.5b"
    ollama_base_url = "http://localhost:11434"

# Paste the class logic directly to avoid import issues or dependency on config
import openai

class LocalLLMAnalyzer:
    def __init__(self):
        settings = MockSettings()
        self.model = settings.local_llm_model
        self.base_url = settings.ollama_base_url
        
        print(f"DEBUG: Initializing with model={self.model}, base_url={self.base_url}")
        
        try:
            self.client = openai.OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="ollama",
                timeout=240.0
            )
            
            print("DEBUG: Client created. Listing models...")
            models = self.client.models.list()
            # print(f"DEBUG: Raw models response: {models}")
            
            model_names = [m.id for m in models.data]
            print(f"DEBUG: Available models: {model_names}")
            
            if self.model not in model_names:
                print(f"WARNING: Model {self.model} not found in {model_names}")
            else:
                print(f"SUCCESS: Model {self.model} found.")
                
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        analyzer = LocalLLMAnalyzer()
    except Exception as e:
        print(f"Fatal error: {e}")
