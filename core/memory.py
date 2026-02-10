"""
Interview Intelligence System - Cognitive Memory Layer

Interfaces with the Vestige MCP server to provide long-term behavioral memory
and forensic pattern retrieval across sessions.
"""

import json
import logging
import subprocess
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryClient:
    """
    Client for interacting with the Vestige Cognitive Memory server.
    
    Uses FSRS-6 and spreading activation to manage forensic behavioral patterns.
    """
    
    def __init__(self, mcp_command: str = "vestige"):
        self.mcp_command = mcp_command

    def _run_command(self, args: List[str]) -> Optional[str]:
        """Execute a vestige CLI command and return output"""
        try:
            result = subprocess.run(
                [self.mcp_command] + args,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except FileNotFoundError:
            logger.warning("Vestige CLI not found. Operating in mock/passive memory mode.")
            return None
        except Exception as e:
            logger.error(f"Error running Vestige command: {e}")
            return None

    def ingest_session(self, session_id: str, data: Dict):
        """
        Ingest session analysis into cognitive memory.
        
        Args:
            session_id: Unique identifier for the interview
            data: Dictionary containing insights, scores, and signals
        """
        content = f"Interview Session {session_id} summary: {data.get('summary', '')}\n"
        content += f"Behavioral Patterns: {data.get('patterns', 'None detected')}\n"
        content += f"Inferred Stress Level: {data.get('stress_index', 'Unknown')}"
        
        # We use 'smart_ingest' via CLI if available
        # Mapping to Vestige tags
        tags = ["upsc-forensic", f"session-{session_id}"]
        if data.get('risk_score', 0) > 70:
            tags.append("high-risk")
            
        logger.info(f"Ingesting memory for session {session_id} into Vestige")
        self._run_command(["smart_ingest", "--content", content, "--tags", ",".join(tags)])

    def search_patterns(self, query: str) -> List[Dict]:
        """
        Search for relevant past behavioral patterns.
        
        Args:
            query: Natural language search query
            
        Returns:
            List of relevant memory nodes
        """
        output = self._run_command(["search", query])
        if not output:
            return []
            
        try:
            return json.loads(output)
        except:
            return [{"content": output}]

    def get_candidate_evolution(self, candidate_name: str) -> str:
        """
        Retrieves a summary of how a candidate's behavior has changed over time.
        """
        query = f"behavior of {candidate_name} across multiple interviews improvements stress"
        results = self.search_patterns(query)
        
        if not results:
            return "No previous behavioral history found for this candidate."
            
        context = "\n".join([r.get('content', '') for r in results])
        return f"Historical Behavioral Context for {candidate_name}:\n{context}"

# Global instance
memory_client = MemoryClient()
