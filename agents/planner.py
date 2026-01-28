"""
Planner Agent - Decomposes tasks and decides which tools/agents to use.
This is the orchestration brain of the MARA system.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from loguru import logger

from config import settings
from tools.openai_client import get_openai_client


@dataclass
class Task:
    """Represents a task in the execution plan."""
    
    task_id: int
    description: str
    agent_type: str  # 'rag', 'vision', 'data', 'web_search'
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[int]  # Task IDs that must complete first
    priority: int = 1  # Higher = more important
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'description': self.description,
            'agent_type': self.agent_type,
            'tool_name': self.tool_name,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'priority': self.priority
        }


@dataclass
class ExecutionPlan:
    """Represents a complete execution plan."""
    
    query: str
    tasks: List[Task]
    reasoning: str
    execution_order: List[int]  # Task IDs in execution order
    parallel_groups: List[List[int]]  # Tasks that can run in parallel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'tasks': [task.to_dict() for task in self.tasks],
            'reasoning': self.reasoning,
            'execution_order': self.execution_order,
            'parallel_groups': self.parallel_groups
        }


class PlannerAgent:
    """
    Plans task execution by analyzing user queries and selecting appropriate tools.
    """
    
    def __init__(self):
        """Initialize Planner Agent."""
        self.client = get_openai_client()
        self.max_retries = settings.agents.planner.max_retries
        self.timeout = settings.agents.planner.timeout
        
        # Load tool definitions
        self.tools = self._load_tool_definitions()
        
        logger.info("Planner Agent initialized")
    
    def _load_tool_definitions(self) -> Dict[str, List[Dict]]:
        """Load tool definitions from JSON files."""
        tools_dir = Path(__file__).parent.parent / "tools" / "definitions"
        tools = {}
        
        for tool_file in ["rag_tools.json", "vision_tools.json", "data_tools.json", "web_search_tools.json"]:
            file_path = tools_dir / tool_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    agent_type = tool_file.replace('_tools.json', '')
                    tools[agent_type] = data.get('tools', [])
        
        logger.debug(f"Loaded {sum(len(t) for t in tools.values())} tool definitions")
        return tools
    
    def _build_planning_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Build the planning prompt for the LLM.
        
        Args:
            query: User query
            context: Optional context (uploaded files, previous results, etc.)
        
        Returns:
            Planning prompt
        """
        # Format available tools
        tools_description = []
        for agent_type, tools_list in self.tools.items():
            tools_description.append(f"\n**{agent_type.upper()} Agent Tools:**")
            for tool in tools_list:
                params = ", ".join(tool['parameters']['required'])
                tools_description.append(
                    f"- {tool['name']}({params}): {tool['description']}"
                )
        
        tools_text = "\n".join(tools_description)
        
        # Build context section
        context_text = ""
        if context:
            context_text = "\n**Available Context:**\n"
            if context.get('uploaded_files'):
                context_text += f"- Uploaded files: {', '.join(context['uploaded_files'])}\n"
            if context.get('previous_results'):
                context_text += f"- Previous results available: {len(context['previous_results'])} items\n"
        
        prompt = f"""You are a task planning agent for MARA (Multimodal Agentic Reasoning Assistant).

Your job is to analyze the user's query and create an execution plan by selecting appropriate tools.

**User Query:**
{query}
{context_text}

**Available Tools:**
{tools_text}

**Planning Instructions:**
1. Decompose the query into atomic tasks
2. Select the appropriate agent and tool for each task
3. Identify dependencies between tasks
4. Determine which tasks can run in parallel
5. Assign priority (1-5, higher = more important)

**Output Format (JSON only, no explanation):**
{{
    "reasoning": "Brief explanation of your planning logic",
    "tasks": [
        {{
            "task_id": 1,
            "description": "What this task does",
            "agent_type": "rag|vision|data|web_search",
            "tool_name": "exact_tool_name_from_above",
            "parameters": {{"param1": "value1"}},
            "dependencies": [],
            "priority": 3
        }}
    ],
    "execution_order": [1, 2, 3],
    "parallel_groups": [[1, 2], [3]]
}}

**Important:**
- Only use tools listed above
- execution_order must list all task_ids
- parallel_groups contains tasks that can run simultaneously
- dependencies should reference task_ids

Create the execution plan now:"""
        
        return prompt
    
    def plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for the query.
        
        Args:
            query: User query
            context: Optional context information
        
        Returns:
            ExecutionPlan object
        """
        logger.info(f"Planning execution for query: {query[:100]}...")
        
        # Build planning prompt
        prompt = self._build_planning_prompt(query, context)
        
        # Get plan from LLM
        messages = [
            {
                "role": "system",
                "content": "You are an expert task planner. Respond ONLY with valid JSON, no markdown or explanation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.1,  # Low temperature for consistent planning
                max_tokens=2000
            )
            
            # Extract and parse response
            content = self.client.extract_content(response)
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            plan_data = json.loads(content.strip())
            
            # Convert to ExecutionPlan
            tasks = [
                Task(
                    task_id=t['task_id'],
                    description=t['description'],
                    agent_type=t['agent_type'],
                    tool_name=t['tool_name'],
                    parameters=t['parameters'],
                    dependencies=t.get('dependencies', []),
                    priority=t.get('priority', 1)
                )
                for t in plan_data['tasks']
            ]
            
            plan = ExecutionPlan(
                query=query,
                tasks=tasks,
                reasoning=plan_data.get('reasoning', ''),
                execution_order=plan_data.get('execution_order', []),
                parallel_groups=plan_data.get('parallel_groups', [])
            )
            
            logger.info(f"Created plan with {len(tasks)} tasks")
            logger.debug(f"Reasoning: {plan.reasoning}")
            
            return plan
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.error(f"Response content: {content}")
            
            # Fallback: create simple sequential plan
            return self._create_fallback_plan(query, context)
        
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._create_fallback_plan(query, context)
    
    def _create_fallback_plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create a simple fallback plan when planning fails.
        
        Args:
            query: User query
            context: Optional context
        
        Returns:
            Simple ExecutionPlan
        """
        logger.warning("Using fallback planning strategy")
        
        # Determine query type based on keywords
        query_lower = query.lower()
        
        tasks = []
        
        # Check for data analysis keywords
        if any(word in query_lower for word in ['analyze', 'data', 'statistics', 'trend', 'calculate']):
            tasks.append(Task(
                task_id=1,
                description="Analyze the data",
                agent_type="data",
                tool_name="analyze_dataframe",
                parameters={"analysis_type": "describe"},
                dependencies=[],
                priority=3
            ))
        
        # Check for document/RAG keywords
        if any(word in query_lower for word in ['document', 'search', 'find', 'information', 'what', 'how']):
            tasks.append(Task(
                task_id=len(tasks) + 1,
                description="Search relevant documents",
                agent_type="rag",
                tool_name="search_documents",
                parameters={"query": query, "top_k": 5},
                dependencies=[],
                priority=2
            ))
        
        # Check for image/vision keywords
        if any(word in query_lower for word in ['image', 'chart', 'graph', 'visual', 'picture', 'screenshot']):
            tasks.append(Task(
                task_id=len(tasks) + 1,
                description="Analyze visual content",
                agent_type="vision",
                tool_name="analyze_image",
                parameters={"detail_level": "high"},
                dependencies=[],
                priority=2
            ))
        
        # If no tasks identified, default to RAG
        if not tasks:
            tasks.append(Task(
                task_id=1,
                description="Search for relevant information",
                agent_type="rag",
                tool_name="search_documents",
                parameters={"query": query, "top_k": 5},
                dependencies=[],
                priority=2
            ))
        
        execution_order = [t.task_id for t in tasks]
        
        plan = ExecutionPlan(
            query=query,
            tasks=tasks,
            reasoning="Fallback plan based on query keyword analysis",
            execution_order=execution_order,
            parallel_groups=[execution_order] if len(tasks) > 1 else [[execution_order[0]]]
        )
        
        return plan
    
    def replan(
        self,
        original_plan: ExecutionPlan,
        failure_reason: str,
        completed_tasks: List[int]
    ) -> ExecutionPlan:
        """
        Create a revised plan after a task failure.
        
        Args:
            original_plan: The original execution plan
            failure_reason: Why the plan failed
            completed_tasks: List of successfully completed task IDs
        
        Returns:
            Revised ExecutionPlan
        """
        logger.info(f"Replanning due to: {failure_reason}")
        
        # Build replanning prompt
        prompt = f"""The original plan failed. Create a revised plan.

**Original Query:** {original_plan.query}

**Original Plan:** {json.dumps(original_plan.to_dict(), indent=2)}

**Failure Reason:** {failure_reason}

**Completed Tasks:** {completed_tasks}

Create a revised plan that:
1. Keeps successfully completed tasks
2. Adjusts or replaces failed tasks
3. Uses alternative approaches if needed

Respond with the same JSON format as before."""
        
        messages = [
            {"role": "system", "content": "You are an expert task planner creating recovery plans."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat_completion(messages=messages, temperature=0.2)
            content = self.client.extract_content(response)
            
            # Parse revised plan
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            plan_data = json.loads(content.strip())
            
            tasks = [
                Task(
                    task_id=t['task_id'],
                    description=t['description'],
                    agent_type=t['agent_type'],
                    tool_name=t['tool_name'],
                    parameters=t['parameters'],
                    dependencies=t.get('dependencies', []),
                    priority=t.get('priority', 1)
                )
                for t in plan_data['tasks']
            ]
            
            revised_plan = ExecutionPlan(
                query=original_plan.query,
                tasks=tasks,
                reasoning=plan_data.get('reasoning', 'Revised plan after failure'),
                execution_order=plan_data.get('execution_order', []),
                parallel_groups=plan_data.get('parallel_groups', [])
            )
            
            logger.info(f"Created revised plan with {len(tasks)} tasks")
            return revised_plan
        
        except Exception as e:
            logger.error(f"Replanning failed: {e}")
            # Return original plan with failed tasks removed
            remaining_tasks = [t for t in original_plan.tasks if t.task_id not in completed_tasks]
            return ExecutionPlan(
                query=original_plan.query,
                tasks=remaining_tasks,
                reasoning="Continuing with remaining tasks after failure",
                execution_order=[t.task_id for t in remaining_tasks],
                parallel_groups=[[t.task_id for t in remaining_tasks]]
            )


# Singleton instance
_planner: Optional[PlannerAgent] = None


def get_planner() -> PlannerAgent:
    """Get or create singleton Planner Agent instance."""
    global _planner
    if _planner is None:
        _planner = PlannerAgent()
    return _planner