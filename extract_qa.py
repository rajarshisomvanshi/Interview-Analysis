import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from reasoning.analyzer import InterviewAnalyzer
from core.schemas import QuestionAnswerPair, SpeakerLabel
from core.storage import storage
from config import settings

def extract_qa(session_id: str):
    session_data = storage.load_session_data(session_id)
    
    if not session_data:
        print(f"Error: Session {session_id} not found.")
        return

    slice_path = Path(f"data/sessions/{session_id}/2-minute-slices.txt")
    if not slice_path.exists():
        print(f"Error: {slice_path} not found.")
        return

    with open(slice_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract all Transcript sections
    transcript_blocks = []
    import re
    blocks = re.split(r'={50,}', content)
    for block in blocks:
        if "Transcript:" in block:
            parts = block.split("Transcript:")
            if len(parts) > 1:
                # Remove LLM Analysis section if present in the block
                transcript_part = parts[1].split("LLM Analysis:")[0].strip()
                if transcript_part:
                    transcript_blocks.append(transcript_part)

    full_transcript = "\n".join(transcript_blocks)
    if not full_transcript:
        print("Error: No transcript content found.")
        return

    print(f"Found {len(transcript_blocks)} transcript blocks. Total length: {len(full_transcript)} chars.")

    analyzer = InterviewAnalyzer()
    
    prompt = f"""You are an expert interview analyzer. Extract all question and answer pairs from the following interview transcript.
    
    Return the result as a JSON list of objects. Each object MUST have:
    - "question": The exact or summarized question text.
    - "answer": The exact or summarized answer text.
    - "speaker": "interviewer" or "interviewee" (who asked the question).
    
    TRANSCRIPT:
    ---
    {full_transcript}
    ---
    
    JSON RESPONSE:"""

    print("Calling LLM to extract Q&A pairs...")
    try:
        response = analyzer._call_llm(prompt)
        print("Raw response received.")
        
        # Robustly find JSON content
        import re
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if not json_match:
            # Fallback: try to find anything between [ and ]
            json_match = re.search(r'(\[.*\])', response, re.DOTALL)
            
        if not json_match:
            print("Error: Could not find JSON list in response.")
            print("Response:", response[:500])
            return
            
        json_str = json_match.group(1)
        qa_list = json.loads(json_str)
        print(f"Parsed {len(qa_list)} Q&A pairs.")

        # Load session data dict directly to avoid Pydantic issues
        data_path = settings.get_session_data_path(session_id)
        with open(data_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            
        # Convert to schema-compatible dicts
        updated_qa = []
        for i, item in enumerate(qa_list):
            updated_qa.append({
                "qa_index": i,
                "question_text": item.get("question") or item.get("question_text") or "N/A",
                "response_text": item.get("answer") or item.get("response_text") or "N/A",
                "question_start_ms": 0,
                "question_end_ms": 0,
                "response_start_ms": 0,
                "response_end_ms": 0,
                "response_latency_ms": 0,
                "face_signals_summary": {},
                "body_signals_summary": {},
                "audio_signals_summary": {}
            })
            
        data_dict["question_answer_pairs"] = updated_qa
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully updated {data_path} with {len(updated_qa)} pairs.")

    except Exception as e:
        print(f"Error during LLM extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_qa("fb3c4459-7243-4050-bfe2-9d86683a82ca")
