from langgraph.graph import StateGraph # type: ignore[attr-defined]
from langchain_core.messages import HumanMessage

def execute_graph_invocation(
    compiled_graph: StateGraph,
    tickers: list[str],
    portfolio: dict,
    start_date: str,
    end_date: str,
    model_name: str,
    model_provider: str,
    show_reasoning: bool = False,
    initial_message: str = "Make trading decisions based on the provided data."
) -> dict:
    """Performs the core graph.invoke() call and returns the final state."""
    input_payload = {
        "messages": [HumanMessage(content=initial_message)],
        "data": {
            "tickers": tickers,
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},  # Initialize as empty
        },
        "metadata": {
            "show_reasoning": show_reasoning,
            "model_name": model_name,
            "model_provider": model_provider,
        },
    }
    return compiled_graph.invoke(input_payload)
