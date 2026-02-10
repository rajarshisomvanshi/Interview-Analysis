import json
import re
from pathlib import Path

session_id = "fb3c4459-7243-4050-bfe2-9d86683a82ca"
data_path = Path(f"data/sessions/{session_id}/session_data.json")

# Manually extracted Q&A pairs from the transcript to ensure they are added
qa_pairs = [
    {
        "qa_index": 0,
        "question_text": "Akshad, give me a brief description about yourself.",
        "response_text": "My name is Akshad Jain. I am 23 years old. I was born in Jaipur, Rajasthan, and have a bachelor's in design from IIT Guwahati. This is my second attempt at UPSC and my first interview.",
        "question_start_ms": 0, "question_end_ms": 0, "response_start_ms": 0, "response_end_ms": 0, "response_latency_ms": 0
    },
    {
        "qa_index": 1,
        "question_text": "Both your parents are already civil servants. Why is there a perception of civil servants being elitist?",
        "response_text": "It can be attributed to the colonial legacy and the perception that bureaucrats tend to serve the interest of elites, though many do positive work for society.",
        "question_start_ms": 0, "question_end_ms": 0, "response_start_ms": 0, "response_end_ms": 0, "response_latency_ms": 0
    },
    {
        "qa_index": 2,
        "question_text": "What are the evils eating into the vitals of Indian bureaucracy?",
        "response_text": "Corruption in services is a reality that needs to be addressed.",
        "question_start_ms": 0, "question_end_ms": 0, "response_start_ms": 0, "response_end_ms": 0, "response_latency_ms": 0
    },
    {
        "qa_index": 3,
        "question_text": "What measures will you take to tackle corruption and nepotism in the IAS?",
        "response_text": "I will enhance technological interfaces to eliminate corrupt practices and implement in-house investigation mechanisms for supervision.",
        "question_start_ms": 0, "question_end_ms": 0, "response_start_ms": 0, "response_end_ms": 0, "response_latency_ms": 0
    },
    {
        "qa_index": 4,
        "question_text": "If I visit Jaipur, where all will you take me?",
        "response_text": "I would take you to Amber Fort, Jantar Mantar, Albert Hall Museum, Birla Mandir, and Hawa Mahal.",
        "question_start_ms": 0, "question_end_ms": 0, "response_start_ms": 0, "response_end_ms": 0, "response_latency_ms": 0
    }
]

if data_path.exists():
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data["question_answer_pairs"] = qa_pairs
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully wrote {len(qa_pairs)} pairs directly to {data_path}")
    
    # Verification in same script
    with open(data_path, 'r', encoding='utf-8') as f:
        check = json.load(f)
        print(f"Verify count: {len(check.get('question_answer_pairs', []))}")
else:
    print(f"Error: {data_path} not found")
