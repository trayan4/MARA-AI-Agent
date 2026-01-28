"""
Safe Python code executor for data analysis.
Provides a sandboxed environment with restricted imports and operations.
"""

import ast
import sys
import io
import traceback
from typing import Any, Dict, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
import signal

import pandas as pd
import numpy as np
from loguru import logger

from config import settings


class CodeExecutionError(Exception):
    """Custom exception for code execution errors."""
    pass


class PythonExecutor:
    """
    Safe Python code executor with restrictions.
    Allows data analysis operations while blocking dangerous code.
    """
    
    # Allowed imports in safe mode
    SAFE_IMPORTS = {
        'pandas', 'pd',
        'numpy', 'np',
        'math',
        'statistics',
        'datetime',
        'json',
        're',
        'collections',
        'itertools',
        'functools',
    }
    
    # Blocked operations in safe mode
    BLOCKED_OPERATIONS = {
        'open', 'exec', 'eval', 'compile',
        '__import__', 'globals', 'locals',
        'input', 'raw_input',
        'file', 'execfile',
        'reload', '__builtins__',
    }
    
    # Blocked AST node types
    BLOCKED_AST_NODES = {
        ast.Import,  # Will be checked against whitelist
        ast.ImportFrom,
        # ast.Delete,  # Allow delete for cleanup
        # ast.Exec,  # Doesn't exist in Python 3
    }
    
    def __init__(
        self,
        timeout: Optional[int] = None,
        max_rows: Optional[int] = None,
        safe_mode: Optional[bool] = None
    ):
        """
        Initialize Python executor.
        
        Args:
            timeout: Execution timeout in seconds
            max_rows: Maximum number of rows to process
            safe_mode: Enable safety restrictions
        """
        self.timeout = timeout or settings.agents.data.timeout
        self.max_rows = max_rows or settings.agents.data.max_rows
        self.safe_mode = safe_mode if safe_mode is not None else settings.agents.data.safe_mode
        
        logger.info(
            f"Python executor initialized: timeout={self.timeout}s, "
            f"max_rows={self.max_rows}, safe_mode={self.safe_mode}"
        )
    
    def _check_imports(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code imports are safe.
        
        Args:
            code: Python code to check
        
        Returns:
            (is_safe, error_message)
        """
        if not self.safe_mode:
            return True, None
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.SAFE_IMPORTS:
                        return False, f"Unsafe import: {alias.name}"
            
            elif isinstance(node, ast.ImportFrom):
                if node.module not in self.SAFE_IMPORTS:
                    return False, f"Unsafe import from: {node.module}"
            
            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_OPERATIONS:
                        return False, f"Blocked operation: {node.func.id}"
            
            # Check for attribute access to dangerous methods
            elif isinstance(node, ast.Attribute):
                # Block __ methods except common safe ones
                if (node.attr.startswith('__') and 
                    node.attr not in {'__init__', '__str__', '__repr__', '__len__'}):
                    return False, f"Blocked attribute access: {node.attr}"
        
        return True, None
    
    def _timeout_handler(self, signum, frame):
        """Handler for execution timeout."""
        raise TimeoutError(f"Code execution exceeded {self.timeout} seconds")
    
    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        return_stdout: bool = True
    ) -> Dict[str, Any]:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            context: Variables to inject into execution context
            return_stdout: Include stdout/stderr in results
        
        Returns:
            Dictionary with execution results:
            {
                "success": bool,
                "result": Any,  # Last expression value or None
                "stdout": str,  # If return_stdout=True
                "stderr": str,  # If return_stdout=True
                "error": str,   # If success=False
                "variables": dict  # All variables after execution
            }
        """
        # Check code safety
        is_safe, error_msg = self._check_imports(code)
        if not is_safe:
            logger.error(f"Unsafe code detected: {error_msg}")
            return {
                "success": False,
                "result": None,
                "error": error_msg,
                "stdout": "",
                "stderr": ""
            }
        
        # Prepare execution context
        exec_globals = {
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
        }
        
        if context:
            exec_globals.update(context)
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Set timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(self.timeout)
            
            # Execute code with output capture
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile code
                compiled = compile(code, '<string>', 'exec')
                
                # Execute
                exec(compiled, exec_globals)
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            # Extract result (last expression value if any)
            result = exec_globals.get('result', None)
            
            # Get all variables (excluding builtins and modules)
            variables = {
                k: v for k, v in exec_globals.items()
                if not k.startswith('_') and not callable(v) and k not in {'pd', 'np', 'pandas', 'numpy'}
            }
            
            response = {
                "success": True,
                "result": result,
                "variables": variables
            }
            
            if return_stdout:
                response["stdout"] = stdout_capture.getvalue()
                response["stderr"] = stderr_capture.getvalue()
            
            logger.debug("Code executed successfully")
            return response
        
        except TimeoutError as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            logger.error(f"Code execution timeout: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "stdout": stdout_capture.getvalue() if return_stdout else "",
                "stderr": stderr_capture.getvalue() if return_stdout else ""
            }
        
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            error_trace = traceback.format_exc()
            logger.error(f"Code execution error: {e}\n{error_trace}")
            return {
                "success": False,
                "result": None,
                "error": f"{type(e).__name__}: {str(e)}",
                "stdout": stdout_capture.getvalue() if return_stdout else "",
                "stderr": stderr_capture.getvalue() if return_stdout else "",
                "traceback": error_trace
            }
    
    def execute_dataframe_operation(
        self,
        df: pd.DataFrame,
        operation: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a pandas operation on a DataFrame.
        
        Args:
            df: Input DataFrame
            operation: Operation code to execute
            **kwargs: Additional context variables
        
        Returns:
            Execution results
        """
        # Validate DataFrame size
        if len(df) > self.max_rows:
            logger.warning(f"DataFrame has {len(df)} rows, truncating to {self.max_rows}")
            df = df.head(self.max_rows)
        
        # Inject DataFrame into context
        context = {'df': df, **kwargs}
        
        # Execute operation
        return self.execute(operation, context=context)
    
    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code without executing it.
        
        Args:
            code: Python code to validate
        
        Returns:
            (is_valid, error_message)
        """
        # Check imports
        is_safe, error_msg = self._check_imports(code)
        if not is_safe:
            return False, error_msg
        
        # Check syntax
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def get_safe_builtins(self) -> Dict[str, Any]:
        """
        Get dictionary of safe builtin functions.
        
        Returns:
            Safe builtins dictionary
        """
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'hex': hex,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }
        return safe_builtins


# Convenience function
def execute_code(
    code: str,
    context: Optional[Dict[str, Any]] = None,
    safe_mode: bool = True
) -> Dict[str, Any]:
    """
    Execute Python code with default executor.
    
    Args:
        code: Python code to execute
        context: Execution context variables
        safe_mode: Enable safety restrictions
    
    Returns:
        Execution results
    """
    executor = PythonExecutor(safe_mode=safe_mode)
    return executor.execute(code, context=context)


# Example usage and tests
if __name__ == "__main__":
    executor = PythonExecutor(safe_mode=True)
    
    # Test 1: Simple calculation
    print("Test 1: Simple calculation")
    result = executor.execute("""
import pandas as pd
import numpy as np

data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
result = df.mean()
print(result)
""")
    print(f"Success: {result['success']}")
    print(f"Output: {result['stdout']}")
    print()
    
    # Test 2: Blocked operation
    print("Test 2: Blocked operation (should fail)")
    result = executor.execute("""
import os
os.system('ls')
""")
    print(f"Success: {result['success']}")
    print(f"Error: {result.get('error', 'None')}")
    print()
    
    # Test 3: DataFrame operation
    print("Test 3: DataFrame operation")
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]})
    result = executor.execute_dataframe_operation(
        df,
        """
result = df.describe()
print("Statistics:")
print(result)
"""
    )
    print(f"Success: {result['success']}")
    print(f"Output: {result['stdout']}")