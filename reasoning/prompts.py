"""
Interview Intelligence System - LLM Prompt Templates

Defines prompt templates for behavioral analysis with cautious language.
"""

from typing import Dict
from core.schemas import QuestionAnswerPair


class PromptTemplates:
    """
    LLM prompt templates for interview analysis.
    """
    
    @staticmethod
    def per_question_analysis_prompt(qa_pair: QuestionAnswerPair) -> str:
        """
        Generate prompt for per-question behavioral analysis.
        
        Args:
            qa_pair: QuestionAnswerPair with aggregated signals
            
        Returns:
            Formatted prompt string
        """
        # Extract key metrics
        face_summary = qa_pair.face_signals_summary
        body_summary = qa_pair.body_signals_summary
        audio_summary = qa_pair.audio_signals_summary
        
        prompt = f"""You are analyzing behavioral signals from an interview question-response pair. Your role is to provide objective observations based on measurable data, using cautious language.

**CRITICAL CONSTRAINTS:**
- Use ONLY behavioral observations (e.g., "increased blink rate", "reduced eye contact")
- NEVER make emotion claims (avoid "nervous", "anxious", "lying", "confident")
- Use cautious language: "may indicate", "suggests", "observable pattern"
- Base observations on data provided, not assumptions

---

**Question {qa_pair.qa_index + 1}:**
"{qa_pair.question_text}"

**Response:**
"{qa_pair.response_text}"

**Response Latency:** {qa_pair.response_latency_ms}ms

---

**Facial Behavior & Weak Priors:**
- Eye contact: {face_summary.get('eye_contact_mean', 'N/A'):.2f} (1.0 = direct)
- Gaze towards Interviewer: {face_summary.get('gaze_towards_interviewer_mean', 'N/A'):.2f} (1.0 = looking at interviewer)
- Blink rate: {face_summary.get('blink_rate_per_min', 'N/A'):.1f} blinks/min
- Action Units: {', '.join([f"{k}={v:.2f}" for k, v in face_summary.items() if k.startswith('AU')])}
- Emotion Weak Priors: {', '.join([f"{k}={v:.2f}" for k, v in face_summary.items() if k in ['happy', 'sad', 'surprised', 'neutral']])}

**Body Movement Signals:**
- Hand movement: {body_summary.get('hand_movement_mean', 'N/A'):.2f} (mean intensity)
- Posture shifts: {body_summary.get('posture_shift_count', 0)} shifts detected
- Leg movement: {body_summary.get('leg_movement_mean', 'N/A'):.2f} (mean intensity)

**Audio Signals:**
- Speech rate: {audio_summary.get('speech_rate_mean', 'N/A'):.1f} words/min
- Filler words: {qa_pair.filler_word_count} ({', '.join(qa_pair.filler_words[:5]) if qa_pair.filler_words else 'none'})
- Pitch stability: {audio_summary.get('pitch_stability_mean', 'N/A'):.2f}
- Fluency score: {audio_summary.get('fluency_score', 'N/A'):.2f}
- Pause count: {audio_summary.get('pause_count', 0)}

---

**Task:** Provide a behavioral analysis with the following structure:

1. **Communication Clarity** (2-3 sentences):
   - Assess response structure and coherence based on speech rate, pauses, and filler words
   - Note any observable patterns in verbal delivery

2. **Observable Confidence Indicators** (2-3 sentences):
   - Describe behavioral patterns that may relate to confidence (e.g., eye contact, response latency, speech fluency)
   - Use cautious language: "may suggest", "could indicate"

3. **Stress-Related Behavioral Patterns** (2-3 sentences):
   - Note any observable stress-related behaviors (e.g., increased blink rate, hand movement, pitch variation)
   - Avoid definitive claims; use "observable increase in...", "pattern suggests..."

4. **Summary** (1-2 sentences):
   - Brief overall observation of behavioral patterns during this response

**Analysis Confidence:** Rate your confidence in this analysis as "high", "medium", or "low" based on data quality and consistency.

Provide your analysis now:"""
        
        return prompt
        
    @staticmethod
    def session_summary_prompt(
        qa_analyses: list,
        session_duration_ms: int,
        total_questions: int
    ) -> str:
        """
        Generate prompt for session-level summary.
        
        Args:
            qa_analyses: List of per-question analyses
            session_duration_ms: Total session duration
            total_questions: Number of questions
            
        Returns:
            Formatted prompt string
        """
        duration_min = session_duration_ms / 60000.0
        
        prompt = f"""You are analyzing behavioral trends across an entire interview session.

**Session Overview:**
- Duration: {duration_min:.1f} minutes
- Total questions: {total_questions}

**Per-Question Analyses:**

"""
        
        for i, analysis in enumerate(qa_analyses):
            prompt += f"""
**Question {i + 1}:**
- Communication clarity: {analysis.communication_clarity}
- Confidence indicators: {analysis.confidence_indicators}
- Stress indicators: {analysis.stress_indicators}

"""
        
        prompt += """
---

**Task:** Provide a session-level summary with the following structure:

1. **Overall Trends** (3-4 sentences):
   - Identify patterns across the interview (e.g., changes in behavioral signals over time)
   - Note any progression or regression in observable metrics

2. **Communication Patterns** (2-3 sentences):
   - Summarize overall communication style and clarity
   - Note consistency or variability in verbal delivery

3. **Behavioral Patterns** (2-3 sentences):
   - Summarize observable behavioral patterns (facial, body, audio)
   - Identify any notable trends or changes throughout the session

4. **Executive Summary** (2-3 sentences):
   - High-level summary of interview performance based on behavioral observations
   - Use cautious, evidence-based language

5. **Quantitative Scoring** (CRITICAL: Provide actual numbers 0-100, do NOT use placeholders):
   - Integrity Score: [0-100] (Likelihood of honesty based on consistency)
   - Confidence Score: [0-100] (Based on fluency, eye contact, latency)
   - Risk Score: [0-100] (Based on stress indicators and inconsistencies)

**REMEMBER:** Use only behavioral observations, no emotion or deception claims.

Provide your session summary now:"""
        
        return prompt
        
    @staticmethod
    def q_a_extraction_prompt(transcript_text: str) -> str:
        """
        Generate prompt for extracting Q&A pairs from transcript.
        """
        prompt = f"""You are an expert interview analyst. Your task is to extract all significant question and answer pairs from the provided interview transcript.
        
        **CRITICAL INSTRUCTIONS:**
        1. Identify clear questions asked (likely by an interviewer/interviewer role).
        2. Identify the corresponding answers given.
        3. Maintain the core meaning and tone of both the question and the answer.
        4. Summarize long responses without losing critical substance.
        5. Return the result as a strict JSON list of objects.
        
        Required JSON format:
        [
          {{
            "question": "The question asked",
            "answer": "The answer provided"
          }},
          ...
        ]
        
        **TRANSCRIPT:**
        ---
        {transcript_text}
        ---
        
        JSON RESPONSE:"""
        return prompt

    @staticmethod
    def format_analysis_response(response_text: str) -> Dict[str, str]:
        """
        Parse LLM response into structured format.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Dictionary with parsed sections
        """
        # Simple parsing - in production, use more robust parsing
        sections = {
            'communication_clarity': '',
            'confidence_indicators': '',
            'stress_indicators': '',
            'summary': '',
            'analysis_confidence': 'medium'
        }
        
        # Extract sections (placeholder - implement proper parsing)
        lines = response_text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'communication clarity' in line_lower:
                current_section = 'communication_clarity'
            elif 'confidence indicator' in line_lower:
                current_section = 'confidence_indicators'
            elif 'stress' in line_lower and 'behavioral' in line_lower:
                current_section = 'stress_indicators'
            elif 'summary' in line_lower:
                current_section = 'summary'
            elif 'analysis confidence' in line_lower:
                # Extract confidence level
                if 'high' in line_lower:
                    sections['analysis_confidence'] = 'high'
                elif 'low' in line_lower:
                    sections['analysis_confidence'] = 'low'
                else:
                    sections['analysis_confidence'] = 'medium'
            elif current_section and line.strip() and not line.startswith('**'):
                sections[current_section] += line.strip() + ' '
                
        # Clean up
        for key in sections:
            if isinstance(sections[key], str):
                sections[key] = sections[key].strip()
                
        return sections
