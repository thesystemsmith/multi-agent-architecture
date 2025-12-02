from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig


# state
class TicketState(TypedDict):
    user_msg: str
    category: str
    response: str
    
# tools behave like mini agents
def billing_tool(input: str) -> Command:
    return Command(
        goto=END,
        update={
            "category": "billing",
            "response": "Billing team: Your refund has been initiated."
        }
    )
    
def tech_tool(input: str) -> Command:
    return Command(
        goto=END,
        update={
            "category": "technical",
            "response": "Tech team: Please reset your password using the link provided."
        }
    )
    
def faq_tool(input: str) -> Command:
    return Command(
        goto=END,
        update={
            "category": "general",
            "response": "FAQ bot: You can find more details in our help center."
        }
    )
    

# dict of tools
TOOLS = {
    "billing": billing_tool,
    "technical": tech_tool,
    "general": faq_tool,
}


# supervisor agent
def supervisor_agent(state: TicketState):
    text = state["user_msg"].lower()

    # Decide which tool to call
    if any(w in text for w in ["invoice", "refund", "charged"]):
        selected = "billing"
    elif any(w in text for w in ["login", "password", "bug", "error"]):
        selected = "technical"
    else:
        selected = "general"

    print(f"Supervisor: routing to {selected} tool")

    # Call the subordinate agent (tool)
    tool = TOOLS[selected]

    # Tools return Command → LangGraph handles state + flow
    return tool(state["user_msg"])


# build graph
def build_supervisor_graph():
    graph = StateGraph(TicketState)

    graph.add_node("supervisor", supervisor_agent)

    # entry point
    graph.set_entry_point("supervisor")

    # supervisor will always hand off via Command
    # tools return END themselves

    return graph.compile()


# demo
def main():
    incoming = "Hi, I was charged twice and need a refund."

    initial_state: TicketState = {
        "user_msg": incoming,
        "category": "",
        "response": "",
    }

    app = build_supervisor_graph()

    print("\n=== Running Supervisor + Tool-Call Example ===\n")

    result = app.invoke(initial_state, config=RunnableConfig())

    print("\n=== FINAL OUTPUT ===")
    print("User message :", incoming)
    print("Category     :", result["category"])
    print("Response     :", result["response"])


if __name__ == "__main__":
    main()