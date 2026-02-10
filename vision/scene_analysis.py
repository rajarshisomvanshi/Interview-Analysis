"""
Interview Intelligence System - Scene Analysis

Uses CLIP (Contrastive Language-Image Pre-Training) for zero-shot scene classification.
Classifies room type, attire, and occupancy.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import logging
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

class SceneAnalyzer:
    """
    Analyzes visual scene context using CLIP.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize Scene Analyzer.
        
        Args:
            model_name: HuggingFace model hub name
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SceneAnalyzer with {model_name} on {self.device}...")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logger.info("SceneAnalyzer initialized successfully")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.model_loaded = False
            
        # Define candidate labels for classification
        self.labels = {
            "room_type": [
                "an office setting",
                "a conference room", 
                "a home office",
                "a living room",
                "a bedroom",
                "a kitchen",
                "a generic room"
            ],
            "attire": [
                "formal business suit",
                "business casual attire",
                "casual t-shirt and jeans",
                "formal blazer and tie",
                "casual hoodie",
                "traditional clothing"
            ],
            "occupancy": [
                "a single person",
                "two people",
                "a group of people",
                "an empty room"
            ]
        }

    def analyze_scene(self, frame: np.ndarray) -> Dict[str, str]:
        """
        Analyze scene context from a single frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary of detected attributes (room_type, attire, occupancy)
        """
        if not self.model_loaded:
            return {}
            
        # Convert BGR to PIL Image (RGB)
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            return {}
            
        results = {}
        
        # We can batch these queries if needed, but for now we do them per category
        # to get specific best matches.
        
        for category, candidates in self.labels.items():
            try:
                inputs = self.processor(
                    text=candidates, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Get probabilities
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                # Get top label
                top_idx = probs.argmax().item()
                top_label = candidates[top_idx]
                top_prob = probs[0][top_idx].item()
                
                # Only return if confidence is reasonable (e.g., > 0.3 for multi-class)
                # CLIP logits can be close, so we just take the max for now.
                results[category] = top_label
                
            except Exception as e:
                logger.error(f"CLIP inference failed for {category}: {e}")
                
        return results
