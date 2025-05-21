import asyncio
import json
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph # StateGraph used in run_graph type hint

from src.agents.portfolio_manager import portfolio_management_agent # Used in build_agent_workflow, but that's now in builder.py
from src.agents.risk_manager import risk_management_agent # Used in build_agent_workflow, but that's now in builder.py
# from src.main import start # Removed import
from src.utils.analysts import ANALYST_CONFIG # Used in build_agent_workflow, but that's now in builder.py
from src.graph.state import AgentState # Used in StateGraph type hint
from src.utils.parsing import parse_json_response
from src.graph.runner import execute_graph_invocation # Added import


async def run_graph_async(graph, portfolio, tickers, start_date, end_date, model_name, model_provider):
    """Async wrapper for run_graph to work with asyncio."""
    # Use run_in_executor to run the synchronous function in a separate thread
    # so it doesn't block the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: run_graph(graph, portfolio, tickers, start_date, end_date, model_name, model_provider))  # Use default executor
    return result


def run_graph(
    graph: StateGraph,
    portfolio: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    model_provider: str,
) -> dict:
    """
    Run the graph with the given portfolio, tickers,
    start date, end date, model name, and model provider.
    Metadata 'show_reasoning' defaults to False for backend use.
    """
    return execute_graph_invocation(
        compiled_graph=graph,
        tickers=tickers,
        portfolio=portfolio,
        start_date=start_date,
        end_date=end_date,
        model_name=model_name,
        model_provider=model_provider,
        show_reasoning=False # Explicitly False for backend
    )
