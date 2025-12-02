from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import time

# state
class TicketState(TypedDict):
    text: str
    cleaned_text: str
    text_length: int
    urgency: str
    queue: str
    preprocess_time: float
    urgency_time: float
    triage_time: float
    
    
# agents
def preprocess_agent(state: TicketState) -> Dict[str, Any]:
    start = time.time()
    raw = state['text']
    
    cleaned = ' '.join(raw.strip().split())
    cleaned_lower = cleaned.lower()
    length = len(cleaned)
    
    return {
        'cleaned_text': cleaned_lower,
        'text_length': length,
        'preprocess_time': time.time() - start,
    }
    
def urgency_agent(state: TicketState) -> Dict[str, Any]:
    start = time.time()
    text = state['cleaned_text']

    if any(w in text for w in ['system is down', 'cannot login', 'urgent', 'immediately']):
        urgency = 'high'
    elif any(w in text for w in ['soon', 'asap', 'issue', 'problem']):
        urgency = 'medium'
    else:
        urgency = 'low'

    return {
        'urgency': urgency,
        'urgency_time': time.time() - start,
    }
    
def triage_agent(state: TicketState) -> Dict[str, Any]:
    start = time.time()
    urgency = state['urgency']

    if urgency == 'high':
        queue = 'priority_queue'
    elif urgency == 'medium':
        queue = 'standard_queue'
    else:
        queue = 'backlog'

    return {
        'queue': queue,
        'triage_time': time.time() - start,
    }


# build graph
def build_sequential_ticket_graph():
    graph = StateGraph(TicketState)
    
    graph.add_node('preprocess', preprocess_agent)
    graph.add_node('urgency', urgency_agent)
    graph.add_node('triage', triage_agent)
    
    #entry point for the graph
    graph.set_entry_point('preprocess')
    
    #edges
    graph.add_edge('preprocess', 'urgency')
    graph.add_edge('urgency', 'triage')
    graph.add_edge('triage', END)
    
    
    return graph.compile()


# demo
def main():
    text = (
        'Hi team, the system is down for all our users and we cannot login at all. '
        'Please fix this immediately, it is blocking our work.'
    )

    initial_state: TicketState = {
        'text': text,
        'cleaned_text': '',
        'text_length': 0,
        'urgency': '',
        'queue': '',
        'preprocess_time': 0.0,
        'urgency_time': 0.0,
        'triage_time': 0.0,
    }

    app = build_sequential_ticket_graph()

    print('\n=== Running sequential ticket pipeline ===')
    start = time.time()

    # NOTE: no need for parallel=True here; we WANT sequential
    config = RunnableConfig()
    result = app.invoke(initial_state, config=config)

    total = time.time() - start

    print(f'\nInput:\n{text}\n')
    print('cleaned_text :', result['cleaned_text'])
    print('text_length  :', result['text_length'])
    print('urgency      :', result['urgency'])
    print('queue        :', result['queue'])

    print('\nTimes (seconds):')
    print('preprocess   :', f'{result["preprocess_time"]:.4f}')
    print('urgency      :', f'{result["urgency_time"]:.4f}')
    print('triage       :', f'{result["triage_time"]:.4f}')
    print('wall-clock   :', f'{total:.4f}')


if __name__ == '__main__':
    main()