from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import random
import time

# this agent arch checks if a number is divisible by 10

# state
class LoopState(TypedDict):
    number: int
    passed: bool
    iterations: int
    max_iterations: int
    

# agents
def writer_agent(state: LoopState) -> Dict[str, Any]:
    """Generate a number (intentionally imperfect)"""
    n = random.randint(1, 100)
    print(f'Writer produced: {n}')
    return {'number': n}

def tester_agent(state: LoopState) -> Dict[str, Any]:
    """Check divisibility by 10"""
    num = state['number']
    passed = (num % 10 == 0)
    print(f'Tester: {num} is divisible by 10? {passed}')
    return {'passed': passed}


# controllers
def controller_node(state: LoopState) -> Dict[str, Any]:
    """Decide whether to loop again or end."""
    passed = state['passed']
    iter_count = state['iterations'] + 1

    print(f'Controller: iter {iter_count}, passed={passed}')

    result: Dict[str, Any] = {
        'iterations': iter_count
    }

    return result


# buld loop graph
def build_loop_graph():
    graph = StateGraph(LoopState)

    graph.add_node('writer', writer_agent)
    graph.add_node('tester', tester_agent)
    graph.add_node('controller', controller_node)

    graph.set_entry_point('writer')

    # writer → tester → controller
    graph.add_edge('writer', 'tester')
    graph.add_edge('tester', 'controller')

    # Loop logic
    # controller → writer (if NOT passed AND not exceeded)
    def should_loop(state: LoopState):
        return (not state['passed']) and (state['iterations'] < state['max_iterations'])

    graph.add_conditional_edges(
        'controller',
        should_loop,
        {
            True: 'writer',    # loop again
            False: END         # stop
        }
    )

    return graph.compile()


# demo
def main():
    initial_state: LoopState = {
        'number': 0,
        'passed': False,
        'iterations': 0,
        'max_iterations': 5
    }

    app = build_loop_graph()

    print('\n=== Running loop architecture example ===')
    final = app.invoke(initial_state, config=RunnableConfig())

    print('\n=== Final State ===')
    print(final)


if __name__ == '__main__':
    main()
    