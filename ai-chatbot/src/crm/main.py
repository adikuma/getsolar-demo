import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class LeadScoring(BaseModel):
    lead_quality: str  # "hot", "warm", "cold"
    interest_level: int  # 1-10 scale
    readiness_to_buy: str  # "immediate", "short_term", "long_term", "research_only"
    reasoning: str  # why this scoring was given

class CRMDatabase:
    def __init__(self, db_path: str = "crm_leads.db"):
        self.db_path = db_path
        self.init_database()
        
        self.scoring_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1,
        )
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT,
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    lead_quality TEXT,
                    interest_level INTEGER,
                    readiness_to_buy TEXT,
                    scoring_reasoning TEXT,
                    conversation_summary TEXT,
                    full_conversation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def analyze_conversation(self, conversation_messages: List, contact_info: Dict) -> LeadScoring:
        conversation_text = ""
        for msg in conversation_messages:
            if hasattr(msg, 'type') and msg.type == "human":
                conversation_text += f"Customer: {msg.content}\n"
            elif hasattr(msg, 'type') and msg.type == "ai":
                conversation_text += f"Sunny: {msg.content}\n"
        
        scoring_prompt = f"""
        You are an expert lead scoring analyst for GetSolar, a solar panel company in Singapore.
        
        Analyze this conversation and score the lead based on these criteria:
        
        **LEAD QUALITY SCORING:**
        - HOT: Ready to buy soon, asked about pricing/consultation, has budget, mentioned timeline
        - WARM: Interested and gathering information, some budget signals, considering solar
        - COLD: Just browsing, early research phase, no clear buying signalse
        
        **CONVERSATION TO ANALYZE:**
        {conversation_text}
        
        **CONTACT INFO:**
        Name: {contact_info.get('name', 'Unknown')}
        Email: {contact_info.get('email', 'Unknown')}
        Phone: {contact_info.get('phone', 'Unknown')}
        
        Based on this conversation, provide a detailed lead scoring analysis.
        Pay attention to:
        1. Did they ask about pricing, consultation, or next steps?
        2. Did they express urgency or timeline?
        3. What problems/pain points did they mention?
        4. Any budget or financial capability indicators?
        """
        
        try:
            structured_llm = self.scoring_llm.with_structured_output(LeadScoring)
            
            messages = [
                SystemMessage(content="You are an expert lead scoring analyst. Analyze conversations and provide accurate lead quality scores."),
                HumanMessage(content=scoring_prompt)
            ]
            
            scoring_result = structured_llm.invoke(messages)
            return scoring_result
            
        except Exception as e:
            print(f"[CRM] Error in lead scoring: {e}")

            return LeadScoring(
                lead_quality="warm",
                interest_level=5,
                readiness_to_buy="unknown",
                reasoning=f"Automatic scoring due to analysis error: {str(e)}"
            )
    
    def save_lead(self, thread_id: str, contact_info: Dict, conversation_messages: List, conversation_summary: str = ""):
        print(f"[CRM] Analyzing and saving lead for {contact_info.get('name', 'Unknown')}")
        
        # analyze conversation for lead scoring
        scoring = self.analyze_conversation(conversation_messages, contact_info)
        
        # prepare conversation data
        full_conversation = json.dumps([
            {"type": msg.type, "content": msg.content} 
            for msg in conversation_messages 
            if hasattr(msg, 'type') and hasattr(msg, 'content')
        ])
        
        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO leads (
                    thread_id, name, email, phone, lead_quality, interest_level,
                    scoring_reasoning, conversation_summary, full_conversation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                thread_id,
                contact_info.get('name', ''),
                contact_info.get('email', ''),
                contact_info.get('phone', ''),
                scoring.lead_quality,
                scoring.interest_level,
                scoring.reasoning,
                conversation_summary,
                full_conversation
            ))
            conn.commit()
        
        print(f"[CRM] Saved {scoring.lead_quality.upper()} lead: {contact_info.get('name')} ({scoring.interest_level}/10)")
        print(f"[CRM] Score: {scoring.readiness_to_buy} buyer")
        
        return scoring
    
    def get_lead_by_email(self, email: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM leads WHERE email = ? ORDER BY created_at DESC LIMIT 1", (email,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_leads_by_quality(self, quality: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM leads WHERE lead_quality = ? ORDER BY created_at DESC", 
                (quality,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_leads(self, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM leads ORDER BY created_at DESC LIMIT ?", 
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_lead_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    lead_quality,
                    COUNT(*) as count,
                    AVG(interest_level) as avg_interest
                FROM leads 
                GROUP BY lead_quality
            """)
            
            stats = {}
            total_leads = 0
            for row in cursor.fetchall():
                quality, count, avg_interest = row
                stats[quality] = {
                    "count": count,
                    "avg_interest": round(avg_interest, 1)
                }
                total_leads += count
            
            stats["total"] = total_leads
            return stats

from langchain_core.tools import tool
crm = CRMDatabase()

@tool
def save_lead_to_crm(thread_id: str, contact_info_json: str, conversation_summary: str = "") -> str:
    """
    Save a lead to the CRM database with automated LLM-based scoring.
    
    Args:
        thread_id: Unique conversation thread ID
        contact_info_json: JSON string containing name, email, phone
        conversation_summary: Optional summary of the conversation
        
    Returns:
        str: Lead scoring results and save confirmation
    """
    try:
        contact_info = json.loads(contact_info_json)
        scoring = crm.save_lead(
            thread_id=thread_id,
            contact_info=contact_info,
            conversation_messages=[],  
            conversation_summary=conversation_summary
        )
        
        return f"""✅ Lead saved to CRM successfully!

        **Lead Scoring Results:**
        - **Quality**: {scoring.lead_quality.upper()} 
        - **Interest Level**: {scoring.interest_level}/10
        **Analysis**: {scoring.reasoning}

        The lead has been automatically scored and saved to the database for follow-up."""
        
    except Exception as e:
        return f" Error saving lead to CRM: {str(e)}"

def view_crm_stats():
    stats = crm.get_lead_stats()
    print("\n CRM LEAD STATISTICS")
    print("=" * 30)
    print(f"Total Leads: {stats.get('total', 0)}")
    
    for quality in ['hot', 'warm', 'cold']:
        if quality in stats:
            data = stats[quality]
            print(f"{quality.upper()}: {data['count']} leads (avg interest: {data['avg_interest']}/10)")
    
    print("\nRecent HOT leads:")
    hot_leads = crm.get_leads_by_quality('hot')
    for lead in hot_leads[:3]:
        print(f"- {lead['name']} ({lead['email']}) - {lead['readiness_to_buy']}")

if __name__ == "__main__":
    print("Testing CRM Database System...")    
    view_crm_stats()
