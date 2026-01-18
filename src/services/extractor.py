import json
from openai import OpenAI
from src.services.llm import get_llm_config

class QuestionExtractor:
    @staticmethod
    def extract(markdown_content: str) -> list:
        config = get_llm_config()
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        
        system_prompt = """You are a professional exam paper analysis assistant. Your task is to extract all question information from the provided exam paper content.

Please carefully analyze the exam paper content and extract the following information for each question:
1. Question number (e.g., "1.", "Question 1", etc.)
2. Complete question text content (if multiple choice, include all options)
3. Related image file names (if any, though in this text-only mode there may be none, return empty list)

For multiple choice questions, please merge the stem and all options into one complete question text.

Please return results in JSON format as follows:
{
    "questions": [
        {
            "question_number": "1",
            "question_text": "Complete question content...",
            "images": []
        }
    ]
}
"""
        user_prompt = f"Exam paper content:\n\n{markdown_content[:20000]}" # Limit context if too large

        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("questions", [])
            
        except Exception as e:
            print(f"Extraction error: {e}")
            return []
