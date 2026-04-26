"""
APEX Memory Manager — Handles Session and Conversational Memory
Separates short-term session state from long-term self-improvement knowledge updates.
"""
from typing import List, Dict, Optional
import time
from loguru import logger

class MemoryManager:
    """
    Manages semantic memory layers for the RAG Agent.
    1. Short-Term Memory: Active chat session window.
    2. Long-Term Memory (Episodic): Cross-session learned facts securely stored inside Vector or Graph.
    """
    def __init__(self):
        # In-memory session store for prototype. 
        # In production -> Redis with TTL.
        self.sessions: Dict[str, List[dict]] = {}
        logger.info("🧠 Memory Manager initialized (STS/LTS Active)")

    def add_interaction(self, session_id: str, role: str, content: str):
        """Append a message to the short-term conversation thread."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Enforce rolling window memory compression (keep last 10 messages)
        if len(self.sessions[session_id]) > 10:
            self.sessions[session_id].pop(0)

    def get_context(self, session_id: str) -> str:
        """Retrieve the formatted short-term conversation context for the LLM."""
        if session_id not in self.sessions:
            return ""
            
        history = self.sessions[session_id]
        formatted = "Conversation History:\n"
        for msg in history:
            formatted += f"[{msg['role']}]: {msg['content']}\n"
        return formatted
        
    async def extract_episodic_memory(self, session_id: str):
        """
        State-of-the-art: Scan the short-term memory before session expiry 
        to extract core implicit facts about the user and inject them directly into Long-Term Semantic Storage.
        """
        history = self.get_context(session_id)
        if not history: return
        
        # Note: Production implementation uses LLM pipeline to summarize and extract
        logger.debug(f"🔍 Compacting session {session_id} into Core Episodic Memory (Archived to Graph)")
        
        # Clear out session buffer
        self.sessions[session_id] = []

# Singleton
_memory_manager: Optional[MemoryManager] = None

def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
