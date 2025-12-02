from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import time

# state
class TicketState(TypedDict):
    text: str
    is_spam: bool
    urgency: str
    category: str
    spam_time: float
    urgency_time: float
    category_time: float
    
# agents
def spam_agent(state: TicketState) -> Dict[str, Any]:
    start = time.time()
    text = state['text'].lower()

    spam_keywords = ['win money', 'free gift', 'click here', 'lottery']
    is_spam = any(k in text for k in spam_keywords)

    return {
        'is_spam': is_spam,
        'spam_time': time.time() - start,
    }
    
def urgency_agent(state: TicketState) -> Dict[str, Any]:
    start = time.time()
    text = state['text'].lower()

    if any(w in text for w in ['down', 'cannot login', 'urgent', 'immediately']):
        urgency = 'high'
    elif any(w in text for w in ['soon', 'asap', 'issue']):
        urgency = 'medium'
    else:
        urgency = 'low'

    return {
        'urgency': urgency,
        'urgency_time': time.time() - start,
    }
    
def category_agent(state: TicketState) -> Dict[str, Any]:
    start = time.time()
    text = state['text'].lower()

    if any(w in text for w in ['invoice', 'payment', 'charged', 'refund']):
        category = 'billing'
    elif any(w in text for w in ['password', 'login', '2fa', 'bug', 'error']):
        category = 'technical'
    elif any(w in text for w in ['account', 'profile', 'username']):
        category = 'account'
    else:
        category = 'other'

    return {
        'category': category,
        'category_time': time.time() - start,
    }
    
    
# join node
def join_node(state: TicketState) -> TicketState:
    return state

# build graph
def build_parallel_ticket_graph():
    graph = StateGraph(TicketState)

    graph.add_node('spam', spam_agent)
    graph.add_node('urgency', urgency_agent)
    graph.add_node('category', category_agent)

    graph.add_node('branch', lambda s: s)
    graph.add_node('join', join_node)

    graph.set_entry_point('branch')

    graph.add_edge('branch', 'spam')
    graph.add_edge('branch', 'urgency')
    graph.add_edge('branch', 'category')

    graph.add_edge('spam', 'join')
    graph.add_edge('urgency', 'join')
    graph.add_edge('category', 'join')

    graph.add_edge('join', END)

    return graph.compile()

# demo

def main():
    text = (
        'Hi team, my account was charged twice for the same invoice and I '
        'need a refund as soon as possible. This is urgent because my card '
        'is almost at the limit.'
    )
    
    initial_state: TicketState = {
        'text': text,
        'is_spam': False,
        'urgency': '',
        'category': '',
        'spam_time': 0.0,
        'urgency_time': 0.0,
        'category_time': 0.0,
    }
    
    app = build_parallel_ticket_graph()
    
    print('\n=== Running parallel ticket agents ===')
    start = time.time()
    # usually langraph runs in sequencial, the following is what makes parallel processing possible here
    config = RunnableConfig(parallel=True)
    result = app.invoke(initial_state, config=config)
    total = time.time() - start

    print(f'\nInput:\n{text}\n')
    print('is_spam   :', result['is_spam'])
    print('urgency   :', result['urgency'])
    print('category  :', result['category'])

    print('\nTimes (seconds):')
    print('spam      :', result['spam_time'])
    print('urgency   :', result['urgency_time'])
    print('category  :', result['category_time'])
    print('wall-clock:', f'{total:.4f}')

if __name__ == '__main__':
    main()