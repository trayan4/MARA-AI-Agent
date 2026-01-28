"""
State management for MARA orchestrator.
Defines the shared state that flows through the agent workflow.
"""

from typing import Any, Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import MessagesState


class MARAState(TypedDict):
    """
    Shared state for MARA workflow.
    This state is passed between all agents in the graph.
    """
    
    # Input
    query: str  # User's original query
    context: Dict[str, Any]  # Additional context (uploaded files, etc.)
    
    # Planning
    execution_plan: Optional[Dict[str, Any]]  # Plan from Planner Agent
    current_task_id: Optional[int]  # Currently executing task
    completed_tasks: Annotated[List[int], operator.add]  # List of completed task IDs
    failed_tasks: Annotated[List[int], operator.add]  # List of failed task IDs
    
    # Agent Outputs
    rag_results: Optional[Dict[str, Any]]  # RAG Agent results
    vision_results: Optional[Dict[str, Any]]  # Vision Agent results
    data_results: Optional[Dict[str, Any]]  # Data Agent results
    web_search_results: Optional[Dict[str, Any]]  # Web Search results
    
    # Verification
    critic_results: Optional[Dict[str, Any]]  # Critic Agent verification
    should_replan: bool  # Flag to trigger replanning
    replan_count: int  # Number of replanning attempts
    
    # Final Output
    final_report: Optional[Dict[str, Any]]  # Final report from Report Agent
    
    # Metadata
    execution_metadata: Dict[str, Any]  # Execution tracking
    errors: Annotated[List[str], operator.add]  # Error messages
    
    # Control Flow
    next_step: Optional[str]  # Next node to execute


def create_initial_state(
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> MARAState:
    """
    Create initial state for a new query.
    
    Args:
        query: User query
        context: Optional context (files, previous results, etc.)
    
    Returns:
        Initialized MARAState
    """
    return MARAState(
        # Input
        query=query,
        context=context or {},
        
        # Planning
        execution_plan=None,
        current_task_id=None,
        completed_tasks=[],
        failed_tasks=[],
        
        # Agent Outputs
        rag_results=None,
        vision_results=None,
        data_results=None,
        web_search_results=None,
        
        # Verification
        critic_results=None,
        should_replan=False,
        replan_count=0,
        
        # Final Output
        final_report=None,
        
        # Metadata
        execution_metadata={
            'start_time': datetime.now().isoformat(),
            'agents_called': [],
            'total_tokens': 0,
            'total_cost': 0.0
        },
        errors=[],
        
        # Control Flow
        next_step='planner'
    )


def update_execution_metadata(
    state: MARAState,
    agent_name: str,
    tokens_used: int = 0,
    cost: float = 0.0
) -> MARAState:
    """
    Update execution metadata after an agent runs.
    
    Args:
        state: Current state
        agent_name: Name of agent that just ran
        tokens_used: Tokens consumed
        cost: Cost incurred
    
    Returns:
        Updated state
    """
    metadata = state['execution_metadata'].copy()
    
    # Track which agents were called
    if 'agents_called' not in metadata:
        metadata['agents_called'] = []
    metadata['agents_called'].append(agent_name)
    
    # Update token/cost tracking
    metadata['total_tokens'] = metadata.get('total_tokens', 0) + tokens_used
    metadata['total_cost'] = metadata.get('total_cost', 0.0) + cost
    
    # Update timestamp
    metadata['last_update'] = datetime.now().isoformat()
    
    state['execution_metadata'] = metadata
    return state


def mark_task_complete(state: MARAState, task_id: int) -> MARAState:
    """
    Mark a task as completed.
    
    Args:
        state: Current state
        task_id: Task ID to mark complete
    
    Returns:
        Updated state
    """
    if task_id not in state['completed_tasks']:
        state['completed_tasks'].append(task_id)
    return state


def mark_task_failed(state: MARAState, task_id: int, error: str) -> MARAState:
    """
    Mark a task as failed.
    
    Args:
        state: Current state
        task_id: Task ID that failed
        error: Error message
    
    Returns:
        Updated state
    """
    if task_id not in state['failed_tasks']:
        state['failed_tasks'].append(task_id)
    
    state['errors'].append(f"Task {task_id} failed: {error}")
    return state


def should_continue_execution(state: MARAState) -> bool:
    """
    Determine if execution should continue.
    
    Args:
        state: Current state
    
    Returns:
        True if should continue, False otherwise
    """
    # Stop if we've exceeded max iterations
    max_iterations = 10  # Could be from settings
    total_tasks = len(state['completed_tasks']) + len(state['failed_tasks'])
    
    if total_tasks >= max_iterations:
        return False
    
    # Stop if too many replans
    if state['replan_count'] >= 3:
        return False
    
    # Stop if we have a final report
    if state['final_report'] is not None:
        return False
    
    return True


def get_next_task(state: MARAState) -> Optional[Dict[str, Any]]:
    """
    Get the next task to execute from the plan.
    
    Args:
        state: Current state
    
    Returns:
        Next task or None if no tasks remain
    """
    if not state['execution_plan']:
        return None
    
    plan = state['execution_plan']
    tasks = plan.get('tasks', [])
    completed = set(state['completed_tasks'])
    failed = set(state['failed_tasks'])
    
    # Find next task whose dependencies are satisfied
    for task in tasks:
        task_id = task['task_id']
        
        # Skip if already completed or failed
        if task_id in completed or task_id in failed:
            continue
        
        # Check if dependencies are met
        dependencies = task.get('dependencies', [])
        if all(dep in completed for dep in dependencies):
            return task
    
    return None


def merge_agent_results(state: MARAState) -> Dict[str, Any]:
    """
    Merge all agent results into a single dictionary.
    
    Args:
        state: Current state
    
    Returns:
        Merged results
    """
    results = {}
    
    if state['rag_results']:
        results['rag'] = state['rag_results']
    
    if state['vision_results']:
        results['vision'] = state['vision_results']
    
    if state['data_results']:
        results['data'] = state['data_results']
    
    if state['web_search_results']:
        results['web_search'] = state['web_search_results']
    
    return results


def calculate_execution_time(state: MARAState) -> float:
    """
    Calculate total execution time in seconds.
    
    Args:
        state: Current state
    
    Returns:
        Execution time in seconds
    """
    metadata = state['execution_metadata']
    
    start_time = datetime.fromisoformat(metadata['start_time'])
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    return round(duration, 2)