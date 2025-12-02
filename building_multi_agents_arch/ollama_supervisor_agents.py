from langgraph.graph import START, END
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig

from IPython.display import Image, display
from dotenv import load_dotenv
import os

load_dotenv()

# select llm
model = ChatOllama(
    model="qwen2.5:3b-instruct",
    temperature=0.0,
)


# define tools
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


tavily_api_key = os.getenv("TAVILY_API_KEY")

web_search = TavilySearchResults(
    max_results=3,
    tavily_api_key=tavily_api_key
)


#create worker agents - ReAct
research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_agent",
     prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS (MUST FOLLOW):\n"
        "- For EVERY query, you MUST call the `web_search` tool at least once.\n"
        "- Use web_search to fetch data, then summarize the results.\n"
        "- Do NOT perform math or percentage calculations.\n"
        "- After using web_search and summarizing, respond to the supervisor directly\n"
        "  with ONLY the factual data you found (numbers, facts, etc.).\n"
        "- Do NOT mention agents, tools, or transfers in your response.\n"
    ),
)

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply, divide],
    name="math_agent",
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
)


#create supervisor agent 
supervisor_graph = create_supervisor(
    model=model,
    agents=[research_agent, math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- research_agent: ONLY for information lookup and web search.\n"
        "- math_agent: ONLY for calculations.\n\n"
        "RULES (MUST FOLLOW):\n"
        "1. You MUST NOT answer the user directly until BOTH agents have been used if the question\n"
        "   involves numbers AND calculations (like percentages).\n"
        "2. For questions like GDP + percentage:\n"
        "   a) First, send the task to research_agent.\n"
        "   b) Wait for research_agent's answer.\n"
        "   c) Then send the numeric results to math_agent.\n"
        "   d) Only after math_agent responds, send ONE final answer to the user.\n"
        "3. Never write things like 'I transferred the task'; just route agents and then give the final answer.\n"
        "4. Do not do any research or math yourself. Always delegate.\n"
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
)

supervisor_agent = supervisor_graph.compile()


# If you are in a notebook, this will show the graph image
try:
    display(Image(supervisor_agent.get_graph().draw_mermaid_png()))
except Exception:
    # In plain terminal this will just be skipped
    pass


# pretty-print helpers
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
        # skip parent graph updates in the printouts
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

def test_supervisor_functionality():
    """Test the supervisor pattern with a GDP-like query to validate handoffs."""

    query = (
        "find US and New York state GDP in 2022. "
        "what % of US GDP was New York state?"
    )

    print(f"Query: {query}")
    print("-" * 80)

    try:
        # Stream updates from the supervisor agent
        for chunk in supervisor_agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            },
            subgraphs=False, #see subgraph = True
            config=RunnableConfig(),  # optional, but explicit
        ):
            pretty_print_messages(chunk, last_message=True)

        print("Test completed successfully")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")

    print("=" * 80)


if __name__ == "__main__":
    test_supervisor_functionality()