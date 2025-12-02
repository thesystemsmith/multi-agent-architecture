from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import time
import random


# state
class SocialState(TypedDict):
    twitter_text: str
    instagram_text: str
    reddit_text: str

    twitter_sentiment: float
    instagram_sentiment: float
    reddit_sentiment: float

    report: str
    
    
# helper sentiment
def fake_sentiment_score(text: str) -> float:
    # produce pseudo-random sentiment [-1..1]
    return round(random.uniform(-1, 1), 2)


# collect agents
def collect_twitter(state: SocialState) -> Dict[str, Any]:
    time.sleep(0.5)
    text = 'Twitter buzz about product launch.'
    print('Collected Twitter data')
    return {'twitter_text': text}

def collect_instagram(state: SocialState) -> Dict[str, Any]:
    time.sleep(0.7)
    text = 'Instagram comments praising visuals.'
    print('Collected Instagram data')
    return {'instagram_text': text}

def collect_reddit(state: SocialState) -> Dict[str, Any]:
    time.sleep(0.9)
    text = 'Reddit users debating performance issues.'
    print('Collected Reddit data')
    return {'reddit_text': text}


# analyze agents
def analyze_twitter(state: SocialState) -> Dict[str, Any]:
    text = state['twitter_text']
    score = fake_sentiment_score(text)
    print(f'Analyzed Twitter sentiment: {score}')
    return {'twitter_sentiment': score}

def analyze_instagram(state: SocialState) -> Dict[str, Any]:
    text = state['instagram_text']
    score = fake_sentiment_score(text)
    print(f'Analyzed Instagram sentiment: {score}')
    return {'instagram_sentiment': score}

def analyze_reddit(state: SocialState) -> Dict[str, Any]:
    text = state['reddit_text']
    score = fake_sentiment_score(text)
    print(f'Analyzed Reddit sentiment: {score}')
    return {'reddit_sentiment': score}


# aggregate node
def aggregate_results(state: SocialState) -> SocialState:
    tw = state.get('twitter_sentiment', 0.0)
    ig = state.get('instagram_sentiment', 0.0)
    rd = state.get('reddit_sentiment', 0.0)

    overall = round((tw + ig + rd) / 3, 2)

    report = (
        f'Overall sentiment: {overall}\n'
        f'- Twitter: {tw}\n'
        f'- Instagram: {ig}\n'
        f'- Reddit: {rd}\n'
    )

    print('Aggregated results')
    return {'report': report}


# build graph
def build_aggregator_graph():
    graph = StateGraph(SocialState)

    # Collect
    graph.add_node('collect_twitter', collect_twitter)
    graph.add_node('collect_instagram', collect_instagram)
    graph.add_node('collect_reddit', collect_reddit)

    # Analyze
    graph.add_node('analyze_twitter', analyze_twitter)
    graph.add_node('analyze_instagram', analyze_instagram)
    graph.add_node('analyze_reddit', analyze_reddit)

    # Aggregate
    graph.add_node('aggregate', aggregate_results)

    # NEW: branch node to fan out
    graph.add_node('branch', lambda s: s)
    graph.set_entry_point('branch')

    # fan-out from branch to all collect nodes
    graph.add_edge('branch', 'collect_twitter')
    graph.add_edge('branch', 'collect_instagram')
    graph.add_edge('branch', 'collect_reddit')

    # each collect → its analyze
    graph.add_edge('collect_twitter', 'analyze_twitter')
    graph.add_edge('collect_instagram', 'analyze_instagram')
    graph.add_edge('collect_reddit', 'analyze_reddit')

    # all analyze nodes → aggregate
    graph.add_edge('analyze_twitter', 'aggregate')
    graph.add_edge('analyze_instagram', 'aggregate')
    graph.add_edge('analyze_reddit', 'aggregate')

    # end
    graph.add_edge('aggregate', END)

    return graph.compile()



# demo
def main():
    initial_state: SocialState = {
        'twitter_text': '',
        'instagram_text': '',
        'reddit_text': '',

        'twitter_sentiment': 0.0,
        'instagram_sentiment': 0.0,
        'reddit_sentiment': 0.0,

        'report': '',
    }

    app = build_aggregator_graph()

    print('\n=== Running Aggregator Pattern Example ===\n')

    # Enable parallel execution of branches
    config = RunnableConfig(parallel=True)

    result = app.invoke(initial_state, config=config)

    print('\n=== Final Report ===\n')
    print(result['report'])


if __name__ == '__main__':
    main()
    