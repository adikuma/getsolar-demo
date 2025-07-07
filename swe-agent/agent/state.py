from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import BaseMessage

class ConversationMemory:
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.exchanges: List[Dict[str, str]] = []
       
    def add_exchange(self, user_input: str, agent_response: str) -> None:
        exchange = {
            "user": user_input,
            "agent": agent_response,
            "timestamp": datetime.now().isoformat()
        }
       
        self.exchanges.append(exchange)
       
        # maintain max history limit
        if len(self.exchanges) > self.max_history:
            self.exchanges = self.exchanges[-self.max_history:]
           
    def update_last_response(self, agent_response: str) -> None:
        if self.exchanges:
            self.exchanges[-1]["agent"] = agent_response
           
    def get_history(self) -> List[Dict[str, str]]:
        return self.exchanges.copy()
       
    def clear(self) -> None:
        self.exchanges.clear()
       
    def get_context_summary(self) -> str:
        if not self.exchanges:
            return "no conversation history"
           
        recent = self.exchanges[-3:]  # last 3 exchanges
        summary = []
       
        for i, exchange in enumerate(recent, 1):
            user_preview = exchange["user"][:100] + "..." if len(exchange["user"]) > 100 else exchange["user"]
            summary.append(f"{i}. user: {user_preview}")
           
        return "\n".join(summary)