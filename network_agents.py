from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

# state
class TicketState(TypedDict):
    text: str
    category: str          # 'billing' | 'technical' | 'other'
    has_required_info: bool
    auto_resolved: bool
    escalated: bool
    history: str
    

# agents
def intake_agent(state: TicketState) -> Dict[str, Any]:
    text = state['text'].lower()
    history = state['history'] + ' -> intake'
    
    # classify very roughly
    if any(w in text for w in ['invoice', 'refund', 'charged']):
        category = 'billing'
    elif any(w in text for w in ['error', 'bug', 'crash', 'login']):
        category = 'technical'
    else:
        category = 'other'
        
    # simulate missing info
    has_required_info = 'account id' in text or 'order id' in text
    
    print(f'intake_agent: category={category}, has_required_info={has_required_info}')
    return {
        'category': category,
        'has_required_info': has_required_info,
        'history': history,
    }

def info_agent(state: TicketState) -> Dict[str, Any]:
    history = state['history'] + ' -> info'
    print('info_agent: requesting more info from user (simulated).')

    # for demo, pretend user responds with required info
    updated_text = state['text'] + ' Account ID: 12345'
    return {
        'text': updated_text,
        'has_required_info': True,
        'history': history,
    }
    
def auto_resolve_agent(state: TicketState) -> Dict[str, Any]:
    history = state['history'] + ' -> auto'
    category = state['category']

    # pretend we can auto-resolve only simple billing issues
    if category == 'billing':
        print('auto_resolve_agent: auto-resolved billing issue.')
        auto_resolved = True
    else:
        print('auto_resolve_agent: cannot auto-resolve, needs escalation.')
        auto_resolved = False

    return {
        'auto_resolved': auto_resolved,
        'history': history,
    }
    
def escalate_agent(state: TicketState) -> Dict[str, Any]:
    history = state['history'] + ' -> escalate'
    print('escalate_agent: ticket sent to human support.')

    return {
        'escalated': True,
        'history': history,
    }
    
    
# build graph
def build_network_graph():
    graph = StateGraph(TicketState)

    graph.add_node('intake', intake_agent)
    graph.add_node('info', info_agent)
    graph.add_node('auto', auto_resolve_agent)
    graph.add_node('escalate', escalate_agent)

    # entry point
    graph.set_entry_point('intake')

    # after intake, we decide where to go:
    # - missing info  -> info
    # - has info + simple issue -> auto
    # - has info + complex issue -> escalate
    def intake_next(state: TicketState) -> str:
        if not state['has_required_info']:
            return 'info'
        # simple rule: billing = try auto, others = escalate
        if state['category'] == 'billing':
            return 'auto'
        return 'escalate'

    graph.add_conditional_edges(
        'intake',
        intake_next,
        {
            'info': 'info',
            'auto': 'auto',
            'escalate': 'escalate',
        },
    )

    # after info, go back to intake (loop with updated text)
    graph.add_edge('info', 'intake')

    # after auto, either end or escalate
    def auto_next(state: TicketState) -> str:
        return 'end' if state['auto_resolved'] else 'escalate'

    graph.add_conditional_edges(
        'auto',
        auto_next,
        {
            'end': END,
            'escalate': 'escalate',
        },
    )

    # after escalate, just end
    graph.add_edge('escalate', END)

    return graph.compile()


# demo
def main():
    text = 'Hi, I was charged twice on my invoice but I did not include my account id.'

    initial_state: TicketState = {
        'text': text,
        'category': '',
        'has_required_info': False,
        'auto_resolved': False,
        'escalated': False,
        'history': '',
    }

    app = build_network_graph()

    print('\n=== Running Network / Horizontal Example ===\n')
    result = app.invoke(initial_state, config=RunnableConfig())

    print('\n=== FINAL STATE ===')
    print('category      :', result['category'])
    print('has_info      :', result['has_required_info'])
    print('auto_resolved :', result['auto_resolved'])
    print('escalated     :', result['escalated'])
    print('history       :', result['history'])


if __name__ == '__main__':
    main()