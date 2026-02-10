import json
import asyncio
from openai import OpenAI
from src.services.llm import get_llm_config

class QuestionGenerator:
    @staticmethod
    async def generate(reference_questions: list) -> list:
        config = get_llm_config()
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        
        generated_questions = []
        
        # Parallel generation could be better but sticking to async loop for simplicity
        # or use asyncio.gather if we want speed. Let's do a simple loop for stability first layer
        # as requested "process... generate... give pdf".
        
        print(f"Generating mimic questions for {len(reference_questions)} references...")

        for index, ref_q in enumerate(reference_questions):
            q_text = ref_q.get("question_text", "")
            print(f"Generating mimic for question {index+1}...")
            
            prompt = f"""Reference question:
{q_text}

Requirements:
1. Keep a similar difficulty level.
2. Identify the core knowledge concept(s) of the reference and keep them EXACTLY the same.
3. Change the scenario/objects/geometry; do not simply replace numbers or symbols.
4. Alter at least one part of the reasoning process or add a new sub-question.
5. Keep the problem entirely within the same mathematical scope.
6. Ensure the prompt is rigorous, precise, and self-contained.

Please output the new question in JSON format:
{{
    "question_text": "...",
    "solution": "...",
    "explanation": "..."
}}
"""
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                
                content = response.choices[0].message.content
                mimic_q = json.loads(content)
                
                generated_questions.append({
                    "reference_id": ref_q.get("question_number"),
                    "mimic_question": mimic_q
                })
                
            except Exception as e:
                print(f"Generation failed for q{index}: {e}")
                generated_questions.append({
                    "reference_id": ref_q.get("question_number"),
                    "error": str(e)
                })
                
        return generated_questions
