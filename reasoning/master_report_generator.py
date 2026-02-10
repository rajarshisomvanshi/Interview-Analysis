"""
Master Report Generator

Generates comprehensive interview analysis reports by:
1. Analyzing interim 2-minute slices
2. Evaluating Q&A pairs with suggested improvements
3. Identifying behavioral patterns (fumbles, confidence, nervousness)
4. Providing actionable recommendations
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re

from reasoning.local_llm import LocalLLMAnalyzer

logger = logging.getLogger(__name__)


class MasterReportGenerator:
    """Generates comprehensive master reports from interview analysis data."""
    
    def __init__(self):
        """Initialize the master report generator."""
        self.llm = LocalLLMAnalyzer()
        logger.info("Initialized MasterReportGenerator")
    
    def load_interim_slices(self, slice_file_path: Path) -> List[Dict]:
        """
        Load and parse interim slices from 2-minute-slices.txt
        
        Args:
            slice_file_path: Path to the slices file
            
        Returns:
            List of slice dictionaries with timestamp, metrics, and analysis
        """
        slices = []
        
        if not slice_file_path.exists():
            logger.warning(f"Slice file not found: {slice_file_path}")
            return slices
        
        with open(slice_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by slice separators
        slice_blocks = content.split('=' * 80)
        
        for block in slice_blocks:
            if not block.strip():
                continue
                
            slice_data = {}
            
            # Extract timestamp
            timestamp_match = re.search(r'\[(.+?) - (.+?)\]', block)
            if timestamp_match:
                slice_data['start_time'] = timestamp_match.group(1)
                slice_data['end_time'] = timestamp_match.group(2)
            
            # Extract metrics
            eye_contact_match = re.search(r'Average Eye Contact: ([\d.]+)', block)
            if eye_contact_match:
                slice_data['eye_contact'] = float(eye_contact_match.group(1))
            
            emotion_match = re.search(r'Dominant Emotion: (.+)', block)
            if emotion_match:
                slice_data['emotion'] = emotion_match.group(1).strip()
            
            blink_match = re.search(r'Blink Rate: ([\d.]+)', block)
            if blink_match:
                slice_data['blink_rate'] = float(blink_match.group(1))
            
            # Extract transcript
            transcript_match = re.search(r'Transcript:\n(.+?)(?=\n\nLLM Analysis:|\n={80}|$)', block, re.DOTALL)
            if transcript_match:
                slice_data['transcript'] = transcript_match.group(1).strip()
            
            # Extract LLM analysis
            llm_match = re.search(r'LLM Analysis:\n(.+?)(?=\n={80}|$)', block, re.DOTALL)
            if llm_match:
                slice_data['llm_analysis'] = llm_match.group(1).strip()
            
            if slice_data:
                slices.append(slice_data)
        
        logger.info(f"Loaded {len(slices)} interim slices")
        return slices
    
    def generate_performance_summary(self, slices: List[Dict]) -> str:
        """
        Generate overall performance summary from slices.
        
        Args:
            slices: List of interim slice data
            
        Returns:
            Performance summary text
        """
        if not slices:
            return "No data available for performance summary."
        
        # Calculate averages
        avg_eye_contact = sum(s.get('eye_contact', 0) for s in slices) / len(slices)
        avg_blink_rate = sum(s.get('blink_rate', 0) for s in slices) / len(slices)
        
        # Collect all emotions
        emotions = [s.get('emotion', 'N/A') for s in slices if s.get('emotion')]
        
        # Prepare data for LLM
        prompt = f"""
You are an expert interview analyst. Analyze the overall performance of this interview.

**Interview Statistics**:
- Duration: {len(slices) * 2} minutes ({len(slices)} segments)
- Average Eye Contact: {avg_eye_contact:.2f} (Target > 0.7)
- Average Blink Rate: {avg_blink_rate:.1f} blinks/min (Normal 15-30)
- Emotions Observed: {', '.join(set(emotions))}

**Segment-by-Segment Analysis**:
{self._format_slices_for_prompt(slices)}

**Task**:
Provide a comprehensive performance summary covering:
1. **What Went Right**: Strengths demonstrated throughout the interview
2. **What Went Wrong**: Areas of concern or weakness
3. **Overall Impression**: Professional assessment of the candidate's performance

Be specific and reference behavioral signals and transcript content.
"""
        
        try:
            summary = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            ).choices[0].message.content
            
            return summary
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return "Performance summary generation failed."
    
    def generate_qa_analysis(self, slices: List[Dict]) -> str:
        """
        Generate Q&A analysis with suggested better answers.
        
        Args:
            slices: List of interim slice data
            
        Returns:
            Q&A analysis text
        """
        qa_sections = []
        
        for i, slice_data in enumerate(slices):
            transcript = slice_data.get('transcript', '')
            if not transcript:
                continue
            
            # Extract Q&A from transcript using LLM
            prompt = f"""
Analyze this interview segment and extract Question-Answer pairs.

**Segment**: [{slice_data.get('start_time', 'N/A')} - {slice_data.get('end_time', 'N/A')}]

**Transcript**:
{transcript}

**Behavioral Signals**:
- Eye Contact: {slice_data.get('eye_contact', 'N/A')}
- Blink Rate: {slice_data.get('blink_rate', 'N/A')} blinks/min
- Emotion: {slice_data.get('emotion', 'N/A')}

**Task**:
1. Identify the question(s) asked by the interviewer
2. Identify the candidate's answer(s)
3. Suggest a BETTER answer the candidate could have given
4. Explain WHY your suggested answer is better (structure, content, delivery, relevance)

Format your response as:
**Question**: [Question text]
**Candidate's Answer**: [Actual answer]
**Suggested Better Answer**: [Your improved answer]
**Reasoning**: [Why this is better]

If no clear Q&A is present, state "No distinct Q&A identified in this segment."
"""
            
            try:
                qa_analysis = self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                ).choices[0].message.content
                
                qa_sections.append(f"### Segment {i+1}: [{slice_data.get('start_time', 'N/A')} - {slice_data.get('end_time', 'N/A')}]\n\n{qa_analysis}\n")
            except Exception as e:
                logger.error(f"Failed to generate Q&A analysis for segment {i+1}: {e}")
                qa_sections.append(f"### Segment {i+1}: Analysis failed\n")
        
        return "\n".join(qa_sections)
    
    def generate_behavioral_analysis(self, slices: List[Dict]) -> str:
        """
        Generate behavioral pattern analysis (fumbles, confidence, nervousness).
        
        Args:
            slices: List of interim slice data
            
        Returns:
            Behavioral analysis text
        """
        # Identify fumble moments (low eye contact, high blink rate, negative emotions)
        fumble_moments = []
        confident_moments = []
        
        for slice_data in slices:
            eye_contact = slice_data.get('eye_contact', 0)
            blink_rate = slice_data.get('blink_rate', 0)
            emotion = slice_data.get('emotion', '')
            
            # Fumble indicators
            if eye_contact < 0.5 or blink_rate > 35:
                fumble_moments.append(slice_data)
            # Confidence indicators
            elif eye_contact > 0.7 and blink_rate < 25 and emotion in ['happy', 'neutral']:
                confident_moments.append(slice_data)
        
        # Prepare prompt for LLM
        prompt = f"""
You are an expert behavioral analyst. Analyze the candidate's behavioral patterns.

**Fumble Moments** ({len(fumble_moments)} segments):
{self._format_slices_for_prompt(fumble_moments)}

**Confident Moments** ({len(confident_moments)} segments):
{self._format_slices_for_prompt(confident_moments)}

**Task**:
Provide a detailed behavioral analysis covering:

1. **Confidence Patterns**:
   - Topics/questions where candidate showed high confidence
   - Evidence from behavioral signals and transcript

2. **Fumble/Nervousness Moments**:
   - Specific timestamps and topics where candidate struggled
   - Behavioral indicators (eye contact, blink rate, emotions)

3. **Nervousness Triggers**:
   - What types of questions make the candidate nervous?
   - Patterns in question difficulty, topic, or format

4. **Improvement Recommendations**:
   - Specific, actionable advice based on observed patterns
   - Focus on areas where candidate can improve

Be specific and reference actual data.
"""
        
        try:
            behavioral_analysis = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            ).choices[0].message.content
            
            return behavioral_analysis
        except Exception as e:
            logger.error(f"Failed to generate behavioral analysis: {e}")
            return "Behavioral analysis generation failed."
    
    def generate_master_report(
        self,
        session_id: str,
        candidate_name: str,
        slice_file_path: Path,
        output_path: Path
    ) -> bool:
        """
        Generate the complete master report.
        
        Args:
            session_id: Session identifier
            candidate_name: Name of the candidate
            slice_file_path: Path to 2-minute-slices.txt
            output_path: Path to save the master report
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating master report for session {session_id}")
        
        # Load slices
        slices = self.load_interim_slices(slice_file_path)
        if not slices:
            logger.error("No slices loaded, cannot generate report")
            return False
        
        # Generate sections
        logger.info("Generating performance summary...")
        performance_summary = self.generate_performance_summary(slices)
        
        logger.info("Generating Q&A analysis...")
        qa_analysis = self.generate_qa_analysis(slices)
        
        logger.info("Generating behavioral analysis...")
        behavioral_analysis = self.generate_behavioral_analysis(slices)
        
        # Compile report
        report = f"""# Interview Analysis Report - {candidate_name}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Session ID**: {session_id}
**Duration**: {len(slices) * 2} minutes
**Segments Analyzed**: {len(slices)}

---

## Executive Summary

{performance_summary}

---

## Question & Answer Analysis

{qa_analysis}

---

## Behavioral Analysis

{behavioral_analysis}

---

## Timeline Analysis

Below is the detailed breakdown of each 2-minute segment:

"""
        
        # Append timeline
        for i, slice_data in enumerate(slices):
            report += f"""
### Segment {i+1}: [{slice_data.get('start_time', 'N/A')} - {slice_data.get('end_time', 'N/A')}]

**Metrics**:
- Eye Contact: {slice_data.get('eye_contact', 'N/A')}
- Blink Rate: {slice_data.get('blink_rate', 'N/A')} blinks/min
- Emotion: {slice_data.get('emotion', 'N/A')}

**Transcript**:
{slice_data.get('transcript', 'N/A')}

**Analysis**:
{slice_data.get('llm_analysis', 'N/A')}

---
"""
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Master report saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save master report: {e}")
            return False
    
    def _format_slices_for_prompt(self, slices: List[Dict]) -> str:
        """Helper to format slices for LLM prompts."""
        if not slices:
            return "No data available."
        
        formatted = []
        for i, s in enumerate(slices):
            formatted.append(f"""
Segment {i+1} [{s.get('start_time', 'N/A')} - {s.get('end_time', 'N/A')}]:
- Eye Contact: {s.get('eye_contact', 'N/A')}
- Blink Rate: {s.get('blink_rate', 'N/A')} blinks/min
- Emotion: {s.get('emotion', 'N/A')}
- Transcript: {s.get('transcript', 'N/A')[:200]}...
""")
        
        return "\n".join(formatted)
