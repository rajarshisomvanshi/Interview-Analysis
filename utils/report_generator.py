"""
Interview Intelligence System - Report Generator

Formats session analysis into a structured Markdown report with timestamps.
"""

from typing import Dict, List
from pathlib import Path
from datetime import datetime
from core.schemas import SessionData, QuestionAnswerPair, QuestionAnalysis

class InterviewReportGenerator:
    """
    Generates structured Markdown reports from interview session data.
    """
    
    def __init__(self, session_data: SessionData):
        self.session_data = session_data
        self.metadata = session_data.metadata
        self.analysis = session_data.analysis
        
    def generate_markdown(self) -> str:
        """
        Produce a full Markdown report.
        """
        if not self.analysis:
            return "# Interview Analysis Report\n\n*Analysis not yet completed for this session.*"
            
        report = []
        
        # Header
        report.append(f"# Interview Analysis Report: {self.metadata.interviewee_name or 'Candidate'}")
        report.append(f"**Session ID:** {self.metadata.session_id}")
        report.append(f"**Date:** {self.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        # Add summary stats
        duration_ms = self.session_data.metadata.duration_ms or 0
        report.append(f"**Duration:** {duration_ms / 1000 / 60:.1f} minutes")
        report.append(f"**Interviewee:** {self.session_data.metadata.interviewee_name}")
        report.append("\n---")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(self.analysis.executive_summary)
        report.append("\n---")
        
        # Timeline Analysis
        report.append("## full Video Analysis Timeline")
        report.append("Detailed breakdown of what happened at what time during the interview.")
        report.append("\n| Timestamp | Event | Actor | Description |")
        report.append("| :--- | :--- | :--- | :--- |")
        
        # Interleave questions and behavioral insights
        for i, qa in enumerate(self.session_data.question_answer_pairs):
            q_time = self._format_ms(qa.question_start_ms)
            r_time = self._format_ms(qa.response_start_ms)
            
            # Question
            report.append(f"| {q_time} | Question {i+1} | Interviewer | {qa.question_text} |")
            
            # Answer
            report.append(f"| {r_time} | Response {i+1} | Interviewee | {qa.response_text} |")
            
            # Insight (if available)
            if i < len(self.analysis.question_analyses):
                insight = self.analysis.question_analyses[i]
                report.append(f"| {r_time} | Behavioral Insight | AI Model | {insight.summary} |")
        
        report.append("\n---")
        
        # Detailed Patterns
        report.append("## Behavioral & Communication Patterns")
        
        report.append("### Overall Trends")
        report.append(self.analysis.overall_trends)
        
        report.append("### Communication Patterns")
        report.append(self.analysis.communication_patterns)
        
        report.append("### Behavioral Patterns")
        report.append(self.analysis.behavioral_patterns)
        
        report.append("\n---")
        report.append("*Disclaimer: This report uses objective behavioral observations and 'weak priors' from AI models. Interpret with context-aware judgment.*")
        
        return "\n".join(report)
        
    def _format_ms(self, ms: int) -> str:
        """Format milliseconds to MM:SS"""
        seconds = (ms // 1000) % 60
        minutes = (ms // (1000 * 60))
        return f"{minutes:02d}:{seconds:02d}"

    def save_report(self, output_path: Path):
        """Save the generated report to a file."""
        markdown_content = self.generate_markdown()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return output_path
