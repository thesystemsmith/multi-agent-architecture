from dotenv import load_dotenv
import os

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig

from langgraph.graph import START, END
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool

load_dotenv()

#llm
model = ChatOllama(
    model="qwen2.5:3b-instruct",
    temperature=0.0,
)


#math tools
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


#swarm handoff
# Concept:
#   - Unlike the manual Command-based handoff in supervisor_custom_handoff.py,
#     here we use create_handoff_tool from langgraph_swarm.
#   - Each agent can call these tools to hand control to the OTHER agent.
#   - There is NO central supervisor node; agents coordinate among themselves.

handoff_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description=(
        "Transfer control to the research agent for web searches and "
        "information gathering."
    ),
)

handoff_to_math_agent = create_handoff_tool(
    agent_name="math_agent",
    description=(
        "Transfer control to the math agent for numerical calculations "
        "and percentage computations."
    ),
)


# worker agents - not supervised but part of a swarm
# both are ReAct agents
research_agent = create_react_agent(
    model=model,
    tools=[web_search, handoff_to_math_agent],
    name="research_agent",
    prompt=(
        "You are a research agent specialized in web research and information gathering.\n\n"
        "INSTRUCTIONS:\n"
        "- Handle research-related tasks, web searches, and information gathering.\n"
        "- DO NOT attempt mathematical calculations yourself.\n"
        "- When you have gathered numeric data but a calculation is needed "
        "  (e.g., percentages), use handoff_to_math_agent to hand off.\n"
        "- When you finish research tasks, answer clearly with the facts you found."
    ),
)

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply, divide, handoff_to_research_agent],
    name="math_agent",
    prompt=(
        "You are a math agent specialized in numerical calculations.\n\n"
        "INSTRUCTIONS:\n"
        "- Handle mathematical calculations, such as percentages or ratios.\n"
        "- DO NOT perform web research yourself.\n"
        "- If you need missing data (e.g., GDP numbers), use handoff_to_research_agent.\n"
        "- When you finish, provide a clear numeric result and short explanation."
    ),
)


#create swarm - swarm means to large group of insects btw 😂
# default_active_agent = where we start. Here we start in math_agent
swarm_agent = create_swarm(
    agents=[research_agent, math_agent],
    default_active_agent="math_agent",
).compile()


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
def test_swarm_functionality():
    """Test swarm pattern with GDP-style query to see handoffs."""
    query = (
        "find US and New York state GDP in 2024. "
        "what % of US GDP was New York state?"
    )

    print(f"Query: {query}")
    print("-" * 80)

    try:
        for chunk in swarm_agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            },
            subgraphs=False,
            config=RunnableConfig(),
        ):
            pretty_print_messages(chunk, last_message=True)

        print("Test completed (swarm).")
        print("=" * 80)

    except Exception as e:
        print(f"Test failed with error: {e}")
        print("=" * 80)


if __name__ == "__main__":
    test_swarm_functionality()
