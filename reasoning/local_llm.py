"""
Interview Intelligence System - Local LLM Analyzer (Ollama)

Runs a local LLM via Ollama for descriptive analysis of interview videos.
"""

import logging
from typing import Dict, List, Optional
import openai
from config import settings

logger = logging.getLogger(__name__)

class LocalLLMAnalyzer:
    """
    Local LLM analyzer using Ollama API.
    """
    
    def __init__(self):
        """Initialize local LLM"""
        self.model = settings.local_llm_model
        self.base_url = settings.ollama_base_url
        
        logger.info(f"Initializing Local LLM ({self.model}) via Ollama at {self.base_url}...")
        
        try:
            # Initialize OpenAI client for Ollama
            self.client = openai.OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="ollama",  # Ollama doesn't manage keys
                timeout=240.0  # 240 second timeout for slow local generation
            )
            
            # Verify connection by listing models
            models = self.client.models.list()
            model_names = [m.id for m in models.data]
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found in Ollama. Available: {model_names}")
                # Fallback to first available if specific model missing, or let it fail?
                # For now, just warn. Ollama might auto-pull or generic error.
            
            logger.info("Local LLM initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Local LLM: {e}")
            self.client = None
            raise

    def generate_description(self, signals: Dict) -> str:
        """
        Generate a descriptive analysis of the interview segment.
        
        Args:
            signals: Dictionary containing aggregated signals (eye contact, emotion, etc.)
            
        Returns:
            Generated text description
        """
        if self.client is None:
            return "Local LLM not initialized."

        # Extract signals
        avg_eye_contact = signals.get('avg_eye_contact', 0.5)
        blink_rate = signals.get('blink_rate', 0.0)
        emotions = signals.get('emotions', {})
        # Get top emotion safely
        top_emotion = "neutral"
        if emotions:
            try:
                top_emotion = max(emotions, key=emotions.get)
            except:
                pass
                
        transcript = signals.get('transcript', "")
        visual_context = signals.get('visual_context', {})
        language = signals.get('language', 'en')
        
        # Format Visual Context
        visual_str = "Visual Context not available."
        if visual_context:
            visual_str = f"Room Type: {visual_context.get('room_type', 'unknown')}\n"
            visual_str += f"Attire: {visual_context.get('attire', 'unknown')}\n"
            visual_str += f"Occupancy: {visual_context.get('occupancy', 'unknown')}"

        prompt = f"""
You are an expert Interview Intelligence Analyst. Analyze the following 2-minute segment of an interview video.

CONTEXT:
{visual_str}
LANGUAGE DETECTED: {language}

BEHAVIORAL SIGNALS:
- Avg Eye Contact: {avg_eye_contact:.2f} (Target > 0.7)
- Blink Rate: {blink_rate:.1f} blinks/min (Normal ~15-30)
- Dominant Emotion: {top_emotion}

TRANSCRIPT (Spoken Dialogue):
"{transcript}"

TASK:
1. First, Describe the scene based on the Visual Context (e.g., "The candidate is wearing [Attire] in [Room Type] with [Occupancy]").
2. Then, Reconstruct the dialogue in a Q&A format. Based on the text, identify who is the Interviewer and who is the Candidate.
   - If the transcript is in a non-English language (e.g., Hindi), you may quote the original text but provide the analysis in English.
   Format as:
   **Interviewer**: [Question]
   **Candidate**: [Response]
   (If specific speakers are unclear, summarize the topic discussed).
3. Finally, Analyze the candidate's behavior. Are they nervous (high blink rate, low eye contact)? Are they confident?

Output a concise but detailed analysis. Do not hallucinate information not present in signals or transcript.
"""
        
        # Generate
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for analyzing interview videos. You are precise and observational."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens if hasattr(self, 'max_tokens') else 400,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Analysis failed: {e}"
