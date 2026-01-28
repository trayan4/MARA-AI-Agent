"""
Router for MARA orchestrator.
Decides which agent should run next based on current state.
"""

from typing import Literal
from loguru import logger

from orchestrator.state import MARAState, get_next_task


def route_next_step(
    state: MARAState
) -> Literal["planner", "executor", "critic", "report", "end"]:
    """
    Determine which node to execute next based on current state.
    
    Args:
        state: Current workflow state
    
    Returns:
        Name of next node to execute
    """
    logger.debug(f"Routing from state: next_step={state.get('next_step')}")
    
    # If explicitly set, use that
    if state.get('next_step'):
        next_step = state['next_step']
        logger.info(f"Routing to explicit next step: {next_step}")
        return next_step
    
    # If we need to replan, go to planner
    if state.get('should_replan', False):
        logger.info("Replanning triggered, routing to planner")
        return "planner"
    
    # If we have a final report, we're done
    if state.get('final_report'):
        logger.info("Final report ready, routing to end")
        return "end"
    
    # If no plan yet, go to planner
    if not state.get('execution_plan'):
        logger.info("No execution plan, routing to planner")
        return "planner"
    
    # Check if there are more tasks to execute
    next_task = get_next_task(state)
    
    if next_task:
        logger.info(f"More tasks to execute, routing to executor for task {next_task['task_id']}")
        return "executor"
    
    # All tasks done, check if we have any results
    has_results = any([
        state.get('rag_results'),
        state.get('vision_results'),
        state.get('data_results'),
        state.get('web_search_results')
    ])
    
    if has_results and not state.get('critic_results'):
        # Results available but not verified yet
        logger.info("Results ready for verification, routing to critic")
        return "critic"
    
    if has_results and state.get('critic_results'):
        # Verified results, ready for report
        logger.info("Verified results ready, routing to report")
        return "report"
    
    # Fallback: if we have no results and no tasks, something went wrong
    logger.warning("No clear next step, routing to end")
    return "end"


def should_run_critic(state: MARAState) -> bool:
    """
    Determine if Critic Agent should run.
    
    Args:
        state: Current state
    
    Returns:
        True if critic should run
    """
    # Run critic if we have results but haven't verified yet
    has_results = any([
        state.get('rag_results'),
        state.get('vision_results'),
        state.get('data_results'),
        state.get('web_search_results')
    ])
    
    already_verified = state.get('critic_results') is not None
    
    return has_results and not already_verified


def should_generate_report(state: MARAState) -> bool:
    """
    Determine if Report Agent should run.
    
    Args:
        state: Current state
    
    Returns:
        True if report should be generated
    """
    # Generate report if:
    # 1. We have results
    # 2. Critic has verified (or critic disabled)
    # 3. No report yet
    
    has_results = any([
        state.get('rag_results'),
        state.get('vision_results'),
        state.get('data_results'),
        state.get('web_search_results')
    ])
    
    is_verified = state.get('critic_results') is not None
    no_report_yet = state.get('final_report') is None
    
    return has_results and is_verified and no_report_yet


def get_agent_for_task(task: dict) -> str:
    """
    Map task agent_type to actual agent node name.
    
    Args:
        task: Task from execution plan
    
    Returns:
        Agent node name
    """
    agent_type = task.get('agent_type', '')
    
    # Map agent types to node names
    agent_map = {
        'rag': 'rag_agent',
        'vision': 'vision_agent',
        'data': 'data_agent',
        'web_search': 'web_search_agent'
    }
    
    return agent_map.get(agent_type, 'rag_agent')  # Default to RAG


def check_max_iterations(state: MARAState, max_iterations: int = 10) -> bool:
    """
    Check if max iterations exceeded.
    
    Args:
        state: Current state
        max_iterations: Maximum allowed iterations
    
    Returns:
        True if exceeded
    """
    total_tasks = len(state['completed_tasks']) + len(state['failed_tasks'])
    return total_tasks >= max_iterations


def check_timeout(state: MARAState, timeout_seconds: int = 300) -> bool:
    """
    Check if execution has timed out.
    
    Args:
        state: Current state
        timeout_seconds: Timeout in seconds
    
    Returns:
        True if timed out
    """
    from datetime import datetime
    
    metadata = state['execution_metadata']
    start_time = datetime.fromisoformat(metadata['start_time'])
    elapsed = (datetime.now() - start_time).total_seconds()
    
    return elapsed >= timeout_seconds


def determine_parallel_tasks(state: MARAState) -> list:
    """
    Determine which tasks can be executed in parallel.
    
    Args:
        state: Current state
    
    Returns:
        List of task groups that can run in parallel
    """
    if not state.get('execution_plan'):
        return []
    
    plan = state['execution_plan']
    parallel_groups = plan.get('parallel_groups', [])
    completed = set(state['completed_tasks'])
    failed = set(state['failed_tasks'])
    
    # Find groups that are ready to execute
    ready_groups = []
    
    for group in parallel_groups:
        # Check if all tasks in group are not yet done
        group_tasks = [t for t in group if t not in completed and t not in failed]
        
        if group_tasks:
            # Check if all dependencies for this group are met
            plan_tasks = {t['task_id']: t for t in plan.get('tasks', [])}
            
            all_deps_met = True
            for task_id in group_tasks:
                task = plan_tasks.get(task_id)
                if task:
                    deps = task.get('dependencies', [])
                    if not all(d in completed for d in deps):
                        all_deps_met = False
                        break
            
            if all_deps_met:
                ready_groups.append(group_tasks)
    
    return ready_groups


def route_after_executor(state: MARAState) -> str:
    """
    Decide next step after executor runs.
    
    Args:
        state: Current state
    
    Returns:
        Next node name
    """
    # Check if there are more tasks
    next_task = get_next_task(state)
    
    if next_task:
        return "executor"
    
    # No more tasks, proceed to verification
    return "critic"


def route_after_critic(state: MARAState) -> str:
    """
    Decide next step after critic verification.
    
    Args:
        state: Current state
    
    Returns:
        Next node name
    """
    # Check if we should replan
    if state.get('should_replan', False):
        logger.warning("Critic triggered replanning")
        return "planner"
    
    # Otherwise, proceed to report generation
    return "report"