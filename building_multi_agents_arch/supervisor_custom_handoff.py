from dotenv import load_dotenv
import os
from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command

load_dotenv()


#llm
model = ChatOllama(
    model="qwen2.5:3b-instruct",
    temperature=0.0,
)


#tools - math
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

#search tool
tavily_api_key = os.getenv("TAVILY_API_KEY")
web_search = TavilySearchResults(
    max_results=3,
    tavily_api_key=tavily_api_key,
)


#agents
research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_agent",
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks.\n"
        "- Use the web_search tool when needed to fetch data.\n"
        "- DO NOT do any math.\n"
        "- After you're done, respond to the supervisor directly with "
        "the data you found.\n"
        "- Respond ONLY with the results of your work, no extra meta talk."
    ),
)

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply, divide],
    name="math_agent",
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks.\n"
        "- You may receive numbers or facts from the research agent.\n"
        "- Use add / multiply / divide to compute answers.\n"
        "- Respond ONLY with the final numeric result and a short explanation."
    ),
)


# Concept:
#   - Instead of letting create_supervisor auto-create `transfer_to_*` tools,
#     we create our OWN tools that:
#       * have custom names / descriptions
#       * return Command(goto=agent_name, update=..., graph=Command.PARENT)
#       * optionally modify the shared state (messages) on handoff
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """
    Create a tool that:
    - When called by the supervisor LLM,
    - Appends a 'successfully transferred' tool message to messages,
    - Returns Command telling LangGraph to go to the target agent node.
    """
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        # NEW: This tool runs inside the graph, has access to full MessagesState.
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }

        # NEW: Command here does two things:
        #  1) goto=agent_name      -> route control to that agent node
        #  2) update=...           -> add our tool_message to the message history
        #  3) graph=Command.PARENT -> jump in the parent graph (multi-agent graph)
        return Command(
            goto=agent_name,
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool

#concrete handoff tools for each worker agent
assign_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="Assign task to the research agent.",
)

assign_to_math_agent = create_handoff_tool(
    agent_name="math_agent",
    description="Assign task to the math agent.",
)

#supervisor here is a ReAct agent
supervisor_agent = create_react_agent(
    model=model,
    tools=[assign_to_research_agent, assign_to_math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- research_agent: Assign research / web lookup tasks to this agent.\n"
        "- math_agent: Assign mathematical / calculation tasks to this agent.\n\n"
        "RULES:\n"
        "- For questions like 'find GDP then compute %', FIRST send the task\n"
        "  to research_agent, then send the numeric results to math_agent.\n"
        "- Do not call agents in parallel; always one at a time.\n"
        "- Do NOT do any research or math yourself.\n"
        "- Use ONLY the handoff tools (transfer_to_*) to delegate work.\n"
    ),
    name="supervisor",
)


#build graph
supervisor_graph = (
    StateGraph(MessagesState)
    # destinations is only for visualization, not needed for logic
    .add_node("supervisor", supervisor_agent, destinations=("research_agent", "math_agent", END))
    .add_node("research_agent", research_agent)
    .add_node("math_agent", math_agent)
    # Entry: always start at supervisor
    .add_edge(START, "supervisor")
    # After each worker finishes, control returns to supervisor
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()
)


#pretty print helpers
def pretty_print_message(message, indent: bool = False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message: bool = False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")
        
        
    
#demo
def main():
    query = (
        "find US and New York state GDP in 2022. "
        "what % of US GDP was New York state?"
    )

    print(f"Query: {query}")
    print("-" * 80)

    last_chunk = None

    for chunk in supervisor_graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
        config=RunnableConfig(),
    ):
        last_chunk = chunk
        pretty_print_messages(chunk, last_message=True)

    # get final message history for supervisor node
    if last_chunk is not None and "supervisor" in last_chunk:
        final_messages = last_chunk["supervisor"]["messages"]
        print("\n=== FINAL MESSAGE HISTORY (supervisor) ===\n")
        for m in final_messages:
            m.pretty_print()


if __name__ == "__main__":
    main()