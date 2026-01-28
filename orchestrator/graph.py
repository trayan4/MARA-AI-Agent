"""
Main orchestrator graph using LangGraph.
Coordinates execution of all MARA agents in a workflow.
"""

from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from loguru import logger

from config import settings
from orchestrator.state import (
    MARAState,
    create_initial_state,
    update_execution_metadata,
    mark_task_complete,
    get_next_task,
    merge_agent_results,
    calculate_execution_time
)
from orchestrator.router import (
    route_next_step,
    route_after_executor,
    route_after_critic
)
from orchestrator.error_handler import (
    handle_agent_error,
    with_error_handling,
    validate_state
)

from agents.planner import get_planner
from agents.rag import get_rag_agent
from agents.vision import get_vision_agent
from agents.data import get_data_agent
from agents.web_search import get_web_search_agent
from agents.critic import get_critic_agent
from agents.report import get_report_agent


# Agent Node Functions

@with_error_handling("planner")
def planner_node(state: MARAState) -> MARAState:
    """Plan execution by decomposing query into tasks."""
    logger.info("=== PLANNER NODE ===")
    
    planner = get_planner()
    
    # Check if this is a replan
    if state.get('should_replan'):
        logger.info("Replanning after failure")
        
        # For now, skip replanning and just proceed to report
        # Replanning is complex and causes issues
        logger.warning("Replanning triggered but skipped - proceeding to report")
        state['should_replan'] = False
        state['next_step'] = 'report'
        return state
        
        state['execution_plan'] = revised_plan.to_dict()
        state['should_replan'] = False
        state['replan_count'] = state.get('replan_count', 0) + 1
    else:
        # Initial planning
        plan = planner.plan(
            query=state['query'],
            context=state.get('context')
        )
        
        state['execution_plan'] = plan.to_dict()
    
    # Set next step to executor
    state['next_step'] = 'executor'
    
    state = update_execution_metadata(state, 'planner')
    
    logger.info(f"Plan created with {len(state['execution_plan']['tasks'])} tasks")
    return state


@with_error_handling("executor")
def executor_node(state: MARAState) -> MARAState:
    """Execute the next task from the plan."""
    logger.info("=== EXECUTOR NODE ===")
    
    # Get next task
    task = get_next_task(state)
    
    if not task:
        logger.info("No more tasks to execute")
        state['next_step'] = 'critic'
        return state
    
    task_id = task['task_id']
    agent_type = task['agent_type']
    tool_name = task['tool_name']
    parameters = task['parameters']
    
    logger.info(f"Executing task {task_id}: {agent_type}.{tool_name}")
    
    try:
        # Route to appropriate agent
        if agent_type == "rag":
            result = execute_rag_task(tool_name, parameters, state)
            state['rag_results'] = result
        
        elif agent_type == "vision":
            result = execute_vision_task(tool_name, parameters, state)
            state['vision_results'] = result
        
        elif agent_type == "data":
            result = execute_data_task(tool_name, parameters, state)
            state['data_results'] = result
        
        elif agent_type == "web_search":
            result = execute_web_search_task(tool_name, parameters, state)
            state['web_search_results'] = result
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Mark task as complete
        state = mark_task_complete(state, task_id)
        state = update_execution_metadata(state, agent_type)
        
        logger.info(f"Task {task_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        state = handle_agent_error(agent_type, e, state, task_id)
    
    # Check if there are more tasks
    next_task = get_next_task(state)
    if next_task:
        state['next_step'] = 'executor'
    else:
        state['next_step'] = 'critic'
    
    return state


def execute_rag_task(tool_name: str, parameters: Dict[str, Any], state: MARAState) -> Dict[str, Any]:
    """Execute RAG agent task."""
    rag_agent = get_rag_agent()
    
    if tool_name == "search_documents":
        result = rag_agent.answer_question(
            query=parameters.get('query', state['query']),
            top_k=parameters.get('top_k', 5),
            use_hybrid=parameters.get('use_hybrid', True)
        )
        return result.to_dict()

    elif tool_name == "get_document":
        doc_id = parameters.get('doc_id')
        if not doc_id:
            return {'error': 'No doc_id provided'}
        
        doc = rag_agent.get_document(doc_id)
        if not doc:
            return {'error': f'Document {doc_id} not found'}
        
        return doc    
    
    else:
        return {'error': f'Unknown RAG tool: {tool_name}'}


def execute_vision_task(tool_name: str, parameters: Dict[str, Any], state: MARAState) -> Dict[str, Any]:
    """Execute Vision agent task."""
    vision_agent = get_vision_agent()
    
    # Get image path from parameters or context
    image_path = parameters.get('image_path')
    
    if not image_path and state.get('context', {}).get('uploaded_files'):
        # Try to use first uploaded image
        uploaded_files = state['context']['uploaded_files']
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        image_path = next(
            (f for f in uploaded_files if any(f.endswith(ext) for ext in image_extensions)),
            None
        )
    
    if not image_path:
        return {'error': 'No image path provided'}
    
    if tool_name == "analyze_image":
        result = vision_agent.analyze_image(
            image_path=image_path,
            query=parameters.get('query'),
            detail_level=parameters.get('detail_level', 'high')
        )
        return result.to_dict()
    
    elif tool_name == "extract_chart_data":
        result = vision_agent.extract_chart_data(
            image_path=image_path,
            chart_type=parameters.get('chart_type', 'auto')
        )
        return result.to_dict()
    
    elif tool_name == "ocr_extract_text":
        result = vision_agent.ocr_extract_text(
            image_path=image_path,
            language=parameters.get('language', 'eng')
        )
        return result.to_dict()
    
    else:
        return {'error': f'Unknown Vision tool: {tool_name}'}


def execute_data_task(tool_name: str, parameters: Dict[str, Any], state: MARAState) -> Dict[str, Any]:
    """Execute Data agent task."""
    data_agent = get_data_agent()
    
    # Get data path from parameters or context
    data_path = parameters.get('data_path')
    
    if not data_path and state.get('context', {}).get('uploaded_files'):
        # Try to use first uploaded data file
        uploaded_files = state['context']['uploaded_files']
        data_extensions = ['.csv', '.xlsx', '.xls', '.json']
        data_path = next(
            (f for f in uploaded_files if any(f.endswith(ext) for ext in data_extensions)),
            None
        )
    
    if not data_path and tool_name != "execute_pandas_code":
        return {'error': 'No data path provided'}
    
    if tool_name == "analyze_dataframe":
        result = data_agent.analyze_dataframe(
            data_path=data_path,
            analysis_type=parameters.get('analysis_type', 'describe'),
            columns=parameters.get('columns')
        )
        return result.to_dict()
    
    elif tool_name == "execute_pandas_code":
        result = data_agent.execute_pandas_code(
            code=parameters.get('code', ''),
            data_path=data_path
        )
        return result.to_dict()
    
    elif tool_name == "detect_anomalies":
        result = data_agent.detect_anomalies(
            data_path=data_path,
            column=parameters.get('column', ''),
            method=parameters.get('method', 'zscore'),
            threshold=parameters.get('threshold', 3.0)
        )
        return result.to_dict()
    
    elif tool_name == "calculate_trends":
        result = data_agent.calculate_trends(
            data_path=data_path,
            value_column=parameters.get('value_column', ''),
            time_column=parameters.get('time_column'),
            trend_type=parameters.get('trend_type', 'growth_rate'),
            window=parameters.get('window', 7)
        )
        return result.to_dict()
    
    elif tool_name == "group_and_aggregate":
        result = data_agent.group_and_aggregate(
            data_path=data_path,
            group_by=parameters.get('group_by', []),
            aggregations=parameters.get('aggregations', {})
        )
        return result.to_dict()
    
    else:
        return {'error': f'Unknown Data tool: {tool_name}'}


def execute_web_search_task(tool_name: str, parameters: Dict[str, Any], state: MARAState) -> Dict[str, Any]:
    """Execute Web Search agent task."""
    web_agent = get_web_search_agent()
    
    if tool_name == "search":
        result = web_agent.search(
            query=parameters.get('query', state['query']),
            num_results=parameters.get('num_results', 5),
            time_range=parameters.get('time_range')
        )
        return result.to_dict()
    
    else:
        return {'error': f'Unknown Web Search tool: {tool_name}'}


@with_error_handling("critic")
def critic_node(state: MARAState) -> MARAState:
    """Verify outputs from all agents."""
    logger.info("=== CRITIC NODE ===")
    
    critic = get_critic_agent()
    
    # Collect all agent outputs
    agent_outputs = merge_agent_results(state)
    
    if not agent_outputs:
        logger.warning("No agent outputs to verify")
        state['next_step'] = 'report'
        return state
    
    # Verify each agent's output
    verification_results = {}
    
    for agent_type, output in agent_outputs.items():
        result = critic.verify_output(
            agent_type=agent_type,
            output=output,
            context={'query': state['query']}
        )
        verification_results[agent_type] = result
    
    # Generate overall verification report
    report = critic.generate_verification_report(verification_results)
    
    state['critic_results'] = report
    
    # Check if we should trigger replanning
    overall_confidence = report['average_confidence']
    
    if overall_confidence < critic.trigger_replan_threshold and state['replan_count'] < 3:
        logger.warning(f"Low confidence ({overall_confidence:.2f}), triggering replan")
        state['should_replan'] = True
        state['next_step'] = 'planner'
    else:
        state['next_step'] = 'report'
    
    state = update_execution_metadata(state, 'critic')
    
    logger.info(f"Verification complete: confidence={overall_confidence:.2f}, passed={report['overall_passed']}")
    return state


@with_error_handling("report")
def report_node(state: MARAState) -> MARAState:
    """Generate final report."""
    logger.info("=== REPORT NODE ===")
    
    report_agent = get_report_agent()
    
    # Collect all agent outputs
    agent_outputs = merge_agent_results(state)
    
    # Add critic results to outputs
    if state.get('critic_results'):
        agent_outputs['critic'] = state['critic_results']
    
    # Calculate execution metadata
    execution_metadata = state['execution_metadata'].copy()
    execution_metadata['execution_time'] = calculate_execution_time(state)
    execution_metadata['end_time'] = datetime.now().isoformat()
    
    # Generate report
    report = report_agent.generate_report(
        query=state['query'],
        agent_outputs=agent_outputs,
        execution_metadata=execution_metadata
    )
    
    state['final_report'] = report.to_dict()
    state['next_step'] = 'end'
    
    state = update_execution_metadata(state, 'report')
    
    logger.info(f"Report generated (confidence: {report.confidence:.2f})")
    return state


# Build the graph

def create_mara_graph() -> StateGraph:
    """
    Create the MARA orchestration graph.
    
    Returns:
        Compiled StateGraph
    """
    logger.info("Building MARA orchestration graph")
    
    # Initialize graph
    graph = StateGraph(MARAState)
    
    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report", report_node)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # Add edges
    graph.add_edge("planner", "executor")
    
    # Executor can loop to itself or proceed to critic
    graph.add_conditional_edges(
        "executor",
        lambda state: "executor" if get_next_task(state) else "critic"
    )
    
    # Critic can trigger replan or proceed to report
    graph.add_conditional_edges(
        "critic",
        lambda state: "planner" if state.get('should_replan') else "report"
    )
    
    # Report always goes to END
    graph.add_edge("report", END)
    
    # Compile graph
    compiled_graph = graph.compile()
    
    logger.info("MARA graph compiled successfully")
    return compiled_graph


# Global graph instance
_mara_graph = None


def get_mara_graph():
    """Get or create the compiled MARA graph."""
    global _mara_graph
    if _mara_graph is None:
        _mara_graph = create_mara_graph()
    return _mara_graph


# Main execution function

def execute_query(
    query: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute a query through the MARA system.
    
    Args:
        query: User query
        context: Optional context (uploaded files, etc.)
    
    Returns:
        Final report or error report
    """
    logger.info(f"Executing query: '{query[:100]}...'")
    
    # Create initial state
    initial_state = create_initial_state(query, context)
    
    # Validate state
    is_valid, error = validate_state(initial_state)
    if not is_valid:
        logger.error(f"Invalid initial state: {error}")
        return {'success': False, 'error': error}
    
    # Get graph
    graph = get_mara_graph()
    
    # Execute workflow
    try:
        final_state = graph.invoke(initial_state)
        
        # Return final report
        if final_state.get('final_report'):
            return final_state['final_report']
        else:
            # Return error report if no final report
            from orchestrator.error_handler import create_error_report
            return create_error_report(final_state)
    
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'query': query
        }