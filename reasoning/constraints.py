"""
Interview Intelligence System - Language Safety Constraints

Ensures LLM outputs use cautious, behavioral language.
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class LanguageConstraints:
    """
    Enforces cautious language in LLM outputs.
    """
    
    # Prohibited words/phrases (emotion and deception claims)
    PROHIBITED_TERMS = [
        'nervous', 'anxious', 'stressed', 'worried', 'scared', 'afraid',
        'confident', 'excited', 'happy', 'sad', 'angry', 'frustrated',
        'lying', 'deceptive', 'dishonest', 'truthful', 'honest',
        'guilty', 'innocent', 'suspicious',
        'is nervous', 'is anxious', 'is confident', 'is lying'
    ]
    
    # Recommended cautious phrases
    CAUTIOUS_PHRASES = [
        'may indicate', 'may suggest', 'could indicate', 'could suggest',
        'observable pattern', 'behavioral pattern', 'data suggests',
        'appears to show', 'tendency toward', 'consistent with',
        'increased', 'decreased', 'elevated', 'reduced',
        'higher than baseline', 'lower than baseline'
    ]
    
    def __init__(self):
        """Initialize language constraints"""
        logger.info("Initialized LanguageConstraints")
        
    def check_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text contains prohibited terms.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []
        text_lower = text.lower()
        
        for term in self.PROHIBITED_TERMS:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                violations.append(term)
                
        is_valid = len(violations) == 0
        
        if not is_valid:
            logger.warning(f"Language violations found: {violations}")
            
        return is_valid, violations
        
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text by replacing prohibited terms with cautious language.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        sanitized = text
        
        # Replace emotion claims with behavioral observations
        replacements = {
            r'\bis nervous\b': 'shows behavioral patterns that may indicate nervousness',
            r'\bis anxious\b': 'displays patterns that may suggest anxiety',
            r'\bis confident\b': 'displays patterns that may suggest confidence',
            r'\bis lying\b': 'shows inconsistent behavioral patterns',
            r'\bnervous\b': 'behavioral patterns',
            r'\banxious\b': 'stress-related patterns',
            r'\bconfident\b': 'confidence-related patterns',
        }
        
        for pattern, replacement in replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
            
        return sanitized
        
    def add_cautious_framing(self, text: str) -> str:
        """
        Add cautious framing to text.
        
        Args:
            text: Text to frame
            
        Returns:
            Text with cautious framing
        """
        # Add disclaimer if not present
        disclaimer = "Note: This analysis is based on observable behavioral patterns and should not be interpreted as definitive psychological assessment."
        
        if disclaimer.lower() not in text.lower():
            text = text + "\n\n" + disclaimer
            
        return text
        
    def validate_analysis(self, analysis_text: str) -> Tuple[bool, str]:
        """
        Validate analysis text for language compliance.
        
        Args:
            analysis_text: Analysis text to validate
            
        Returns:
            Tuple of (is_valid, sanitized_text)
        """
        # Check for violations
        is_valid, violations = self.check_text(analysis_text)
        
        # Sanitize if violations found
        if not is_valid:
            logger.warning(f"Sanitizing analysis due to violations: {violations}")
            sanitized = self.sanitize_text(analysis_text)
        else:
            sanitized = analysis_text
            
        # Add cautious framing
        sanitized = self.add_cautious_framing(sanitized)
        
        return is_valid, sanitized


# Reasoning package initialization
__all__ = ["LanguageConstraints"]
