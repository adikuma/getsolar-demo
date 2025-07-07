import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .tools import get_enhanced_tools
from utils.config import Config

from dotenv import load_dotenv

load_dotenv()

class SWEAgent:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.tools = get_enhanced_tools()
        
        # initialize anthropic model
        self.llm = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20240620",       
            temperature=0.2,
        )

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.agent_graph = self._build_agent_graph()

    def _build_agent_graph(self) -> StateGraph:
        def agent_node(state: MessagesState) -> Dict[str, Any]:
            """main agent node that processes messages"""
            print(f"reasoning iteration {len(state['messages']) // 2 + 1}")
            
            # add system message if this is the first interaction
            messages = state["messages"]
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                system_prompt = self._get_system_prompt()
                messages = [SystemMessage(content=system_prompt)] + messages
            
            # get response from llm
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: MessagesState) -> str:
            """decide whether to use tools or end"""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        # build the graph
        workflow = StateGraph(MessagesState)
        
        # add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
        
        # compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _get_system_prompt(self) -> str:
        return """you are an advanced reasoning agent with exceptional analytical capabilities.

        your core strengths:
        - step-by-step logical reasoning and problem decomposition
        - advanced code analysis and debugging capabilities  
        - comprehensive research and data synthesis
        - creative problem solving with multiple solution paths
        - precise tool selection and orchestration

        available tools:
        - enhanced_python_repl: execute python code with full data science stack
        - web_search_enhanced: research current information with structured results
        - safe_shell_execute: run safe shell commands for system operations
        - advanced_file_analyzer: deep analysis of code files with improvement suggestions
        - project_structure_analyzer: analyze project structure and organization
        - read/write file tools: access and modify files
        - list directory tool: explore project structure

        interaction guidelines:
        - think step by step through problems
        - explain your reasoning process
        - use tools when needed to gather information or execute code
        - provide clear, actionable solutions
        - format responses with proper structure
        - cite sources when researching information

        remember: you excel at complex reasoning tasks that require multiple steps, tool coordination, and synthesis of information from various sources."""

    async def process_async(self, user_input: str, thread_id: str) -> str:
        try:
            # create config for thread continuity
            config = {"configurable": {"thread_id": thread_id}}
            
            # create input message
            input_messages = [HumanMessage(content=user_input)]
            
            # invoke the graph
            final_state = await asyncio.to_thread(
                self.agent_graph.invoke,
                {"messages": input_messages},
                config
            )
            
            # extract the final ai response
            messages = final_state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, 'content') and last_message.content:
                return last_message.content
            else:
                return "no response generated"

        except Exception as e:
            error_msg = f"reasoning error: {str(e)}"
            if self.config.debug_mode:
                import traceback
                error_msg += f"\n\ntraceback:\n{traceback.format_exc()}"
            return error_msg

    def process_sync(self, user_input: str, thread_id: str) -> str:
        return asyncio.run(self.process_async(user_input, thread_id))

    def get_conversation_history(self, thread_id: str) -> List[BaseMessage]:
        """get conversation history for a specific thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.agent_graph.get_state(config)
            return state.values.get("messages", [])
        except Exception:
            return []

    def clear_conversation(self, thread_id: str) -> None:
        """clear conversation for specific thread"""
        # note: with memorysaver, memory is cleared when the process restarts
        # for persistent storage, you'd need to implement thread-specific clearing
        pass