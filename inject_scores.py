import json
import os
import random
from pathlib import Path

def inject():
    sessions_dir = Path("data/sessions")
    sessions = sorted([d for d in os.listdir(sessions_dir) if (sessions_dir / d).is_dir()], 
                      key=lambda x: os.path.getmtime(sessions_dir / x))
    
    if not sessions:
        print("No sessions found.")
        return
        
    latest = sessions[-1]
    data_path = sessions_dir / latest / "session_data.json"
    
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    if "analysis" not in data or data["analysis"] is None:
        from datetime import datetime
        data["analysis"] = {
            "session_id": latest,
            "analyzed_at": datetime.now().isoformat(),
            "question_analyses": [],
            "overall_trends": "Demonstration metrics injected.",
            "communication_patterns": "Dynamic delivery observed.",
            "behavioral_patterns": "High engagement detected.",
            "executive_summary": "Injected scores for visualization.",
            "integrity_score": 85.0,
            "confidence_score": 88.0,
            "risk_score": 12.0,
            "slices": []
        }
    
    slices = data["analysis"].get("slices", [])
    if not slices:
        # Create some dummy slices if none exist
        for i in range(5):
            score = float(random.randint(75, 92))
            slices.append({
                "start_ms": i * 60000,
                "end_ms": (i + 1) * 60000,
                "insight": "High engagement and clear articulation.",
                "score": score,
                "summary": "Consistent professional demeanor.",
                "fluency": score + random.randint(-5, 5),
                "confidence": score + random.randint(-5, 5),
                "attitude": score + random.randint(-5, 5)
            })
    else:
        for s in slices:
            if s.get("score", 0) == 0:
                score = float(random.randint(75, 92))
                s["score"] = score
                s["fluency"] = score + random.randint(-5, 5)
                s["confidence"] = score + random.randint(-5, 5)
                s["attitude"] = score + random.randint(-5, 5)
                s["insight"] = s.get("insight") or "High engagement and clear articulation."
                s["summary"] = s.get("summary") or "Consistent professional demeanor."

    data["analysis"]["slices"] = slices
    
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully injected scores into {latest}")

if __name__ == "__main__":
    inject()
