from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    RemoveMessage,
    BaseMessage,
)
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState
from dotenv import load_dotenv
import os
import uuid
import re

# existing tools
from calculator.main import open_getsolar_calculator
from calendly.main import get_booking_link
from rag.main import rag
from mail.main import send_consultation_confirmation, extract_contact_info
from crm.main import save_lead_to_crm, CRMDatabase

load_dotenv()


class State(MessagesState):
    summary: str
    user_contact_info: Dict[str, str] = {}  # stores completed contact info
    consultation_requested: bool = False  # tracks if user wants consultation


# initialize crm database
crm_db = CRMDatabase()

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.2,
    max_retries=2,
)

# all tools available to the llm
tools = [
    rag,
    open_getsolar_calculator,
    get_booking_link,
    extract_contact_info,
    send_consultation_confirmation,
]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
you are sunny, getsolar's friendly solar assistant.

your goals:
- educate customers about solar energy
- generate qualified leads  
- guide them to consultations

workflow for consultation booking:
when a user expresses interest in scheduling a consultation, meeting, appointment, or booking:
1. use extract_contact_info tool to get their name, email, and phone
2. use get_booking_link tool to get the calendly url
3. use send_consultation_confirmation tool to send them confirmation email
4. provide them with the booking link and confirm email was sent

for other queries:
- use rag tool for solar faq and information
- use open_getsolar_calculator tool for cost estimates
- always be enthusiastic yet honest about solar benefits

remember: always use the tools provided. let the tools handle contact extraction, booking links, and email sending. do not try to manually process this information.
""".strip()


def call_model(state: State):
    # prepare messages with context
    summary = state.get("summary", "")
    if summary:
        pre = SystemMessage(content=f"conversation summary so far:\n{summary}")
        msgs = [SystemMessage(content=SYSTEM_PROMPT), pre] + state["messages"]
    else:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # get llm response (may include tool calls)
    try:
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}
    except Exception as e:
        # fallback response if llm fails
        fallback_msg = HumanMessage(
            content="i apologize, but i'm experiencing technical difficulties. please try again in a moment or contact us directly at hello@getsolar.ai"
        )
        return {"messages": [fallback_msg]}


def check_consultation_completion(state: State):
    # look for tool messages indicating successful consultation booking
    recent_messages = (
        state["messages"][-5:] if len(state["messages"]) >= 5 else state["messages"]
    )

    consultation_completed = False
    contact_extracted = False

    for msg in recent_messages:
        # check tool messages for successful completion
        if hasattr(msg, "type") and msg.type == "tool":
            if hasattr(msg, "name"):
                if (
                    msg.name == "send_consultation_confirmation"
                    and "confirmation email sent" in msg.content.lower()
                ):
                    consultation_completed = True
                elif msg.name == "extract_contact_info":
                    # validate contact extraction was successful
                    try:
                        if '"email":' in msg.content and '"name":' in msg.content:
                            contact_extracted = True
                    except:
                        pass

    # if consultation was completed, save to crm
    if consultation_completed and contact_extracted:
        return "save_crm"

    return END


def save_consultation_to_crm(state: State):
    try:
        # extract contact info from recent tool messages
        contact_info = {}
        recent_messages = state["messages"][-10:]

        for msg in recent_messages:
            if hasattr(msg, "type") and msg.type == "tool" and hasattr(msg, "name"):
                if msg.name == "extract_contact_info":
                    # parse contact info from tool result
                    import json

                    try:
                        if msg.content.startswith("{") and msg.content.endswith("}"):
                            contact_info = json.loads(msg.content)
                        else:
                            # fallback parsing for non-json format
                            content = msg.content.lower()
                            if "email" in content and "name" in content:
                                # extract basic info for crm
                                contact_info = {
                                    "name": "extracted user",
                                    "email": "user@email.com",
                                    "phone": "",
                                }
                    except:
                        contact_info = {
                            "name": "consultation user",
                            "email": "user@provided.email",
                            "phone": "",
                        }
                    break

        # save to crm if we have contact info
        if contact_info and contact_info.get("email"):
            thread_id = f"chat-{uuid.uuid4()}"
            summary = state.get("summary", "user completed solar consultation booking")

            scoring = crm_db.save_lead(
                thread_id=thread_id,
                contact_info=contact_info,
                conversation_messages=state["messages"],
                conversation_summary=summary,
            )

            print(
                f"[crm] saved {scoring.lead_quality.upper()} lead: {contact_info.get('name', 'user')} ({scoring.interest_level}/10)"
            )

        return {"messages": []}

    except Exception as e:
        print(f"[crm] error saving consultation to crm: {e}")
        return {"messages": []}

def check_consultation_completion(state):
    got_contact = any(
        getattr(m, "type", None) == "tool"
        and getattr(m, "name", "") == "extract_contact_info"
        for m in state["messages"]
    )
    got_confirm = any(
        getattr(m, "type", None) == "tool"
        and getattr(m, "name", "") == "send_consultation_confirmation"
        for m in state["messages"]
    )
    return "save_crm" if (got_contact and got_confirm) else END

def summarize_conversation(state: State):
    # filter to keep only relevant message types
    allowed_types = {"human", "system", "ai"}
    filtered: List[BaseMessage] = []

    for m in state["messages"]:
        if m.type not in allowed_types:
            continue
        if not isinstance(m.content, str):
            continue
        if getattr(m, "tool_calls", None):
            continue
        filtered.append(m)

    # create summary prompt
    summary_message = (
        "produce a concise summary of the conversation above. "
        "include the user's questions and sunny's responses. "
        "if there's an existing summary, extend it; otherwise create a new one. "
        "return 3-5 bullet points."
    )

    # generate summary without tools
    try:
        msgs = filtered + [HumanMessage(content=summary_message)]
        response = llm.invoke(msgs)

        # remove older messages to manage memory
        deletes = [RemoveMessage(id=m.id) for m in filtered[:-2]]
        return {"summary": response.content, "messages": deletes}

    except Exception as e:
        # fallback if summarization fails
        return {
            "summary": "conversation about solar energy consultation",
            "messages": [],
        }


def should_summarize(state: State):
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END


def build_workflow():
    
    # lets map the flow. we start a conversation. then if a tool is called we route to tools, and return to conversation, else if no need then we simply end the conversation
    # theres an edge from conversation to save_crm (if got contact and got an email), and another from save_crm to summarize_conversation
    
    builder = StateGraph(State)

    # add nodes
    builder.add_node("conversation", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("summarize_conversation", summarize_conversation)
    builder.add_node("save_crm", save_consultation_to_crm)

    # main conversation flow
    builder.add_edge(START, "conversation")

    # route to tools when llm makes tool calls
    builder.add_conditional_edges(
        "conversation",
        # if the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # if the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )

    # after tools, return to conversation
    builder.add_edge("tools", "conversation")
    # builder.add_conditional_edges(
    #     "tools",
    #     lambda state: "save_crm" if check_consultation_completion(state) == "save_crm"
    #                    else "conversation"
    # )

    # check if consultation was completed after conversation
    builder.add_conditional_edges(
        "conversation",
        check_consultation_completion,
    )

    
    # after crm saving, check for summarization
    builder.add_conditional_edges("save_crm", should_summarize)

    # summarization routing
    builder.add_conditional_edges(
        "conversation",
        should_summarize,
    )

    builder.add_edge("summarize_conversation", END)
    
    # compile with memory
    graph = builder.compile(checkpointer=InMemorySaver())
    return graph


def validate_system_health():
    health_status = {
        "anthropic_api": bool(os.getenv("ANTHROPIC_API_KEY")),
        "email_config": bool(os.getenv("EMAIL_USER") and os.getenv("EMAIL_PASSWORD")),
        "crm_database": True,  # crm initializes automatically
    }

    issues = [component for component, status in health_status.items() if not status]

    if issues:
        print(f"warning: system components not configured: {', '.join(issues)}")
        print("some features may not work properly")

    return len(issues) == 0


if __name__ == "__main__":
    print("sunny is ready to help with your solar journey")

    # validate system health
    system_healthy = validate_system_health()
    if not system_healthy:
        print("proceeding with limited functionality")

    print("commands: 'quit' to exit, 'health' to check system status")
    print("try asking: 'i'd like to schedule a consultation'\n")

    graph = build_workflow()
    config = {"configurable": {"thread_id": f"getsolar-{uuid.uuid4()}"}}

    while True:
        user_input = input("you: ").strip()

        if user_input.lower() in ("quit", "q", "exit"):
            print("sunny: thanks for exploring solar with getsolar. have a bright day")
            break

        if user_input.lower() == "health":
            validate_system_health()
            continue

        # process user input through workflow
        try:
            out_state = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config=config
            )

            # display ai response
            msgs = out_state.get("messages", [])
            ai_msgs = [m for m in msgs if getattr(m, "type", None) == "ai"]

            if ai_msgs:
                last_ai = ai_msgs[-1]
                print("sunny:", last_ai.content)

                # show tool calls for debugging
                for tc in getattr(last_ai, "tool_calls", []):
                    name = tc.get("name", tc.get("tool_name", "unknown"))
                    args = tc.get("args", tc.get("tool_args", {}))
                    print(f"[tool] {name} -> {args}")
            else:
                print("sunny: i'm here to help with any solar questions you have")

        except Exception as e:
            print(
                f"sunny: i apologize for the technical issue. please try again. error: {str(e)}"
            )

        print()  # blank line
