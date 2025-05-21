from langgraph.graph import END, StateGraph
from src.graph.state import AgentState
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.utils.analysts import ANALYST_CONFIG # Or get_analyst_nodes, ensure consistency

# Start node function (moved from src.main)
def default_start_node(state: AgentState) -> AgentState:
    """Initialize the workflow with the input state."""
    return state

def build_agent_workflow(selected_agents: list[str] | None = None) -> StateGraph:
    """
    Builds the agent StateGraph based on selected analysts.
    If selected_agents is None, all configured analysts are used.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", default_start_node)

    # Process analyst nodes
    all_analyst_keys = list(ANALYST_CONFIG.keys())
    agents_to_use = []
    if selected_agents is None:
        agents_to_use = all_analyst_keys
    else:
        agents_to_use = [agent for agent in selected_agents if agent in ANALYST_CONFIG]
        if not agents_to_use and selected_agents: # If user selected agents but none were valid
            print(f"Warning: No valid analysts selected from: {selected_agents}. Using all analysts.")
            agents_to_use = all_analyst_keys
        elif not agents_to_use: # If selected_agents was empty list
             print("Warning: No analysts selected. Proceeding with only Risk and Portfolio Managers.")


    analyst_node_details = {key: (f"{key}_agent", ANALYST_CONFIG[key]["agent_func"]) for key in ANALYST_CONFIG}

    for agent_key in agents_to_use:
        node_name, node_func = analyst_node_details[agent_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # Connect selected analysts to risk management
    # If no analysts were selected or used, this loop won't run.
    for agent_key in agents_to_use:
        node_name, _ = analyst_node_details[agent_key]
        workflow.add_edge(node_name, "risk_management_agent")
    
    # If there were no analysts, connect start_node directly to risk_management_agent
    if not agents_to_use:
        workflow.add_edge("start_node", "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow
