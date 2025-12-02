from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import re

# state
class CommandState(TypedDict):
    text: str      # full user input
    task: str      # 'summarize' | 'translate' | 'sentiment' | 'unknown'
    content: str   # text without command prefix
    result: str
    

# router node
def router_agent(state: CommandState) -> Dict[str, Any]:
    raw = state['text'].strip()
    lower = raw.lower()
    
    if lower.startswith('summarize:'):
        task = 'summarize'
        content = raw[len('summarize:'):].strip()
    elif lower.startswith('translate:'):
        task = 'translate'
        content = raw[len('translate:'):].strip()
    elif lower.startswith('sentiment:'):
        task = 'sentiment'
        content = raw[len('sentiment:'):].strip()
    else:
        task = 'unknown'
        content = raw
        
    print(f'Router decided task: {task}')
    return {
        'task': task,
        'content': content
    }
    

# handler agents
def summarize_agent(state: CommandState) -> Dict[str, Any]:
    text = state['content']
    sentences = re.split(r'(?<=[.!?]) +', text)
    summary = sentences[0] if sentences else text
    result = f'Summary: {summary}'
    print('Summarize agent ran')
    return {'result': result}

def translate_agent(state: CommandState) -> Dict[str, Any]:
    # Fake translation, just to keep example self-contained
    text = state['content']
    result = (
        'Traducción simulada al español: '
        + text
    )
    print('Translate agent ran')
    return {'result': result}

def sentiment_agent(state: CommandState) -> Dict[str, Any]:
    text = state['content'].lower()
    if any(w in text for w in ['love', 'great', 'awesome', 'good']):
        sentiment = 'Positive'
    elif any(w in text for w in ['hate', 'terrible', 'bad', 'awful', 'slow']):
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    result = f'Sentiment: {sentiment}'
    print('Sentiment agent ran')
    return {'result': result}

def fallback_agent(state: CommandState) -> Dict[str, Any]:
    print('Fallback agent ran')
    return {
        'result': 'Unknown command. Use one of: "summarize:", "translate:", "sentiment:".'
    }


# build graph
def build_command_router_graph():
    graph = StateGraph(CommandState)
    
    graph.add_node('router', router_agent)
    graph.add_node('summarize', summarize_agent)
    graph.add_node('translate', translate_agent)
    graph.add_node('sentiment', sentiment_agent)
    graph.add_node('fallback', fallback_agent)

    graph.set_entry_point('router')
    
    def choose_route(state: CommandState) -> str:
        return state['task']
    
    graph.add_conditional_edges(
        'router',
        choose_route,
        {
            'summarize': 'summarize',
            'translate': 'translate',
            'sentiment': 'sentiment',
            'unknown': 'fallback',
        },
    )
    
    graph.add_edge('summarize', END)
    graph.add_edge('translate', END)
    graph.add_edge('sentiment', END)
    graph.add_edge('fallback', END)

    return graph.compile()


# demo
def run_example(user_text: str):
    initial_state: CommandState = {
        'text': user_text,
        'task': '',
        'content': '',
        'result': '',
    }

    app = build_command_router_graph()
    result = app.invoke(initial_state, config=RunnableConfig())

    print('\nInput :', user_text)
    print('Task  :', result['task'])
    print('Result:', result['result'])
    print('-' * 40)


if __name__ == '__main__':
    run_example('summarize: The new park in the city is a wonderful addition. Families love it.')
    run_example('translate: The system is running smoothly today.')
    run_example('sentiment: I hate how slow this app is on my phone.')
    run_example('hello, what is this?')
    