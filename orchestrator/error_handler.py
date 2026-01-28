"""
Error handling for MARA orchestrator.
Provides centralized error management, retry logic, and graceful degradation.
"""

from typing import Any, Callable, Dict, Optional
import traceback
from functools import wraps

from loguru import logger

from orchestrator.state import MARAState, mark_task_failed


class MARAError(Exception):
    """Base exception for MARA system."""
    pass


class PlanningError(MARAError):
    """Error during planning phase."""
    pass


class ExecutionError(MARAError):
    """Error during task execution."""
    pass


class VerificationError(MARAError):
    """Error during verification phase."""
    pass


class ReportGenerationError(MARAError):
    """Error during report generation."""
    pass


def handle_agent_error(
    agent_name: str,
    error: Exception,
    state: MARAState,
    task_id: Optional[int] = None
) -> MARAState:
    """
    Handle errors from individual agents.
    
    Args:
        agent_name: Name of the agent that failed
        error: The exception that occurred
        state: Current state
        task_id: Optional task ID that failed
    
    Returns:
        Updated state with error information
    """
    error_msg = f"{agent_name} failed: {str(error)}"
    logger.error(error_msg)
    logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Add to error list
    state['errors'].append(error_msg)
    
    # Mark task as failed if task_id provided
    if task_id is not None:
        state = mark_task_failed(state, task_id, str(error))
    
    # Determine if we should trigger replanning
    if isinstance(error, (PlanningError, ExecutionError)):
        # Critical errors should trigger replan if we haven't exceeded max retries
        if state['replan_count'] < 3:
            state['should_replan'] = True
            logger.info("Triggering replan due to critical error")
    
    return state


def with_error_handling(agent_name: str):
    """
    Decorator for agent functions to add error handling.
    
    Args:
        agent_name: Name of the agent
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: MARAState, *args, **kwargs) -> MARAState:
            try:
                return func(state, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {agent_name}: {e}")
                return handle_agent_error(agent_name, e, state)
        
        return wrapper
    
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 2.0
):
    """
    Decorator to retry operations on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for wait time between retries
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            wait_time = 1.0
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        logger.error(f"All {max_retries} retry attempts failed")
            
            raise last_exception
        
        return wrapper
    
    return decorator


def validate_state(state: MARAState) -> tuple[bool, Optional[str]]:
    """
    Validate state integrity.
    
    Args:
        state: State to validate
    
    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    if not state.get('query'):
        return False, "Missing required field: query"
    
    if not isinstance(state.get('completed_tasks', []), list):
        return False, "completed_tasks must be a list"
    
    if not isinstance(state.get('failed_tasks', []), list):
        return False, "failed_tasks must be a list"
    
    # Check for circular dependencies in plan
    if state.get('execution_plan'):
        plan = state['execution_plan']
        tasks = plan.get('tasks', [])
        
        for task in tasks:
            task_id = task.get('task_id')
            dependencies = task.get('dependencies', [])
            
            # Check for self-dependency
            if task_id in dependencies:
                return False, f"Task {task_id} depends on itself"
            
            # Check for invalid dependency IDs
            valid_ids = {t.get('task_id') for t in tasks}
            for dep in dependencies:
                if dep not in valid_ids:
                    return False, f"Task {task_id} has invalid dependency: {dep}"
    
    return True, None


def handle_timeout(state: MARAState) -> MARAState:
    """
    Handle execution timeout.
    
    Args:
        state: Current state
    
    Returns:
        Updated state with timeout error
    """
    error_msg = "Execution timeout exceeded"
    logger.error(error_msg)
    
    state['errors'].append(error_msg)
    
    # Force completion with partial results
    state['next_step'] = 'report'
    
    return state


def handle_max_iterations(state: MARAState) -> MARAState:
    """
    Handle max iterations exceeded.
    
    Args:
        state: Current state
    
    Returns:
        Updated state
    """
    error_msg = "Maximum iterations exceeded"
    logger.warning(error_msg)
    
    state['errors'].append(error_msg)
    
    # Force completion with partial results
    state['next_step'] = 'report'
    
    return state


def graceful_degradation(
    state: MARAState,
    failed_agent: str
) -> MARAState:
    """
    Implement graceful degradation when an agent fails.
    
    Args:
        state: Current state
        failed_agent: Name of failed agent
    
    Returns:
        Updated state with degradation strategy
    """
    logger.info(f"Implementing graceful degradation for {failed_agent}")
    
    # Strategy: Continue with available results
    # Mark the failed agent's tasks as completed (with empty results)
    # This allows the workflow to continue
    
    if state.get('execution_plan'):
        plan = state['execution_plan']
        tasks = plan.get('tasks', [])
        
        for task in tasks:
            if task.get('agent_type') == failed_agent:
                task_id = task.get('task_id')
                if task_id not in state['completed_tasks'] and task_id not in state['failed_tasks']:
                    # Mark as completed with note
                    state['completed_tasks'].append(task_id)
                    state['errors'].append(
                        f"Task {task_id} ({failed_agent}) skipped due to agent failure"
                    )
    
    return state


def create_error_report(state: MARAState) -> Dict[str, Any]:
    """
    Create comprehensive error report.
    
    Args:
        state: Current state
    
    Returns:
        Error report
    """
    from orchestrator.state import calculate_execution_time
    
    return {
        'success': False,
        'query': state.get('query', ''),
        'errors': state.get('errors', []),
        'completed_tasks': state.get('completed_tasks', []),
        'failed_tasks': state.get('failed_tasks', []),
        'replan_count': state.get('replan_count', 0),
        'execution_time': calculate_execution_time(state),
        'partial_results': {
            'rag': state.get('rag_results') is not None,
            'vision': state.get('vision_results') is not None,
            'data': state.get('data_results') is not None,
            'web_search': state.get('web_search_results') is not None
        }
    }


def log_error_details(error: Exception, context: Dict[str, Any]) -> None:
    """
    Log detailed error information.
    
    Args:
        error: The exception
        context: Additional context
    """
    logger.error("=" * 80)
    logger.error(f"ERROR: {type(error).__name__}: {str(error)}")
    logger.error(f"Context: {context}")
    logger.error("Traceback:")
    logger.error(traceback.format_exc())
    logger.error("=" * 80)


# Circuit breaker pattern for preventing cascading failures
class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    Opens after max_failures, preventing further calls until timeout.
    """
    
    def __init__(self, max_failures: int = 5, timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            max_failures: Number of failures before opening
            timeout: Seconds before attempting to close
        """
        self.max_failures = max_failures
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception if circuit is open
        """
        import time
        
        # Check if we should try to close the circuit
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                logger.info("Circuit breaker attempting to close")
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset or close circuit
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
                logger.info("Circuit breaker CLOSED")
            
            return result
        
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.max_failures:
                self.state = "open"
                logger.error(f"Circuit breaker OPENED after {self.failures} failures")
            
            raise e