from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState
from dotenv import load_dotenv
from IPython.display import display, Image
from term_image.image import from_file
from typing import List, Dict, Any
from calculator.main import open_getsolar_calculator
from rag.main import rag
import os
import uuid
import json


load_dotenv()
class State(MessagesState):
    summary: str

STATE_PATH = "conversation_state.json"

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.2,
    max_retries=2,
    # other params...
)

tools = [rag, open_getsolar_calculator]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
You are sunny, GetSolar’s friendly solar assistant.
your goal is to educate customers about solar energy,
generate qualified leads, and guide them to next steps
(calculator, consultation, quote). always be enthusiastic yet honest.
""".strip()

def call_model(state: State):
    summary = state.get("summary","")
    if summary:
        pre = SystemMessage(content=f"summary so far:\n{summary}")
        msgs = [SYSTEM_PROMPT, pre] + state["messages"]
    else:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages":[response]}

def summarize_conversation(state: State):
    """
    description:
        produce or extend a concise bullet-point summary of the conversation.
        strip out *all* tool-related messages so the LLM only sees clean text.
    args:
        state (State): the current graph state, including history and summary
    returns:
        dict: {"summary": str, "messages": [RemoveMessage, ...]}
    """
    # keep only human/system/ai messages whose content is a string
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

    # build your single prompt
    summary_message = (
        "Please produce a concise summary of the conversation above. "
        "Include the user’s questions and Sunny’s responses. "
        "If there’s an existing summary, extend it; otherwise create a new one. "
        "Return 3–5 bullet points."
    )

    # call the LLM _without_ any tools bound
    msgs = filtered + [HumanMessage(content=summary_message)]
    response = llm.invoke(msgs)

    # trim history, keeping only the last two messages
    deletes = [RemoveMessage(id=m.id) for m in filtered[:-2]]
    return {"summary": response.content, "messages": deletes}

def should_summarize(state: State):
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END

def build_workflow():
    builder = StateGraph(State)
    # add nodes
    builder.add_node("conversation", call_model)
    builder.add_node("summarize_conversation", summarize_conversation)
    builder.add_node("tools", ToolNode(tools))
    
    # add logic
    builder.add_edge(START, "conversation")
    builder.add_conditional_edges("conversation", should_summarize)
    builder.add_conditional_edges(
        "conversation",
        # if the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # if the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("summarize_conversation", END)
    builder.add_edge("tools", "conversation")
    builder.add_edge("conversation", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    return graph

if __name__ == "__main__":
    graph = build_workflow()
    config = {"configurable": {"thread_id": f"default-{uuid.uuid4()}"}}

    print("sunny: ready to chat! (type 'quit' to exit)\n")

    while True:
        user_input = input("you: ").strip()
        if user_input.lower() in ("quit", "q", "exit"):
            print("sunny: goodbye!")
            break

        # invoke the graph with your single user message
        out_state = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        # grab the last AI message
        msgs = out_state.get("messages", [])
        ai_msgs = [m for m in msgs if getattr(m, "type", None) == "ai"]
        if ai_msgs:
            last_ai = ai_msgs[-1]
            print("sunny:", last_ai.content)

            # print any tool calls
            for tc in getattr(last_ai, "tool_calls", []):
                name = tc["tool_name"]
                args = tc.get("tool_args", {})
                print(f"[tool] {name} → {args}")
        else:
            print("sunny: (no response)")

        print()  # blank line