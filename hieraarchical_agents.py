from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig


#state
class LoanState(TypedDict):
    loan_amount: int
    documents_ok: bool
    risk_score: float
    approved: bool
    log: str
    

#top level agent - boss
def boss_agent(state: LoanState) -> Command:
    log = state["log"] + " -> Boss"

    print("BossAgent: checking documents...")
    if not state["documents_ok"]:
        return Command(
            goto="verification",
            update={"log": log}
        )

    return Command(
        goto="risk",
        update={"log": log}
    )
    
#mid level agent - verification
def verification_agent(state: LoanState) -> Command:
    log = state["log"] + " -> Verification"

    print("VerificationAgent: validating documents...")
    # simulate success
    new_state = {
        "documents_ok": True,
        "log": log
    }

    return Command(
        goto="risk",
        update=new_state
    )

#low level agent - risk evaluation
def risk_agent(state: LoanState) -> Command:
    log = state["log"] + " -> Risk"

    amount = state["loan_amount"]
    print("RiskAgent: evaluating risk...")

    if amount > 100000:
        risk = 0.3
        approved = False
    else:
        risk = 0.9
        approved = True

    new_state = {
        "risk_score": risk,
        "approved": approved,
        "log": log
    }

    return Command(
        goto=END,
        update=new_state
    )
    
    
# build graph
def build_hierarchical_graph():
    graph = StateGraph(LoanState)

    graph.add_node("boss", boss_agent)
    graph.add_node("verification", verification_agent)
    graph.add_node("risk", risk_agent)

    graph.set_entry_point("boss")

    return graph.compile()


# demo
def main():
    initial_state: LoanState = {
        "loan_amount": 50000,
        "documents_ok": False,
        "risk_score": 0.0,
        "approved": False,
        "log": ""
    }

    app = build_hierarchical_graph()

    print("\n=== Running Hierarchical / Vertical Example ===\n")

    result = app.invoke(initial_state, config=RunnableConfig())

    print("\n=== FINAL OUTPUT ===")
    print("Loan amount  :", result["loan_amount"])
    print("Docs OK      :", result["documents_ok"])
    print("Risk score   :", result["risk_score"])
    print("Approved     :", result["approved"])
    print("Log          :", result["log"])


if __name__ == "__main__":
    main()