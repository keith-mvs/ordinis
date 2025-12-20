"""
Agent Runner - Executes agent workflows from YAML configurations.

This module provides the core runtime for Ordinis agents, enabling:
- Loading agent configurations from YAML files
- Executing workflow steps (tool calls, resource reads, LLM analysis)
- Managing agent lifecycle and error handling
- Audit logging of all agent actions

Example:
    >>> runner = AgentRunner()
    >>> result = await runner.run_agent("strategy_optimizer")
    >>> print(result.output)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import json
import logging
import re

import yaml

logger = logging.getLogger(__name__)


class StepAction(Enum):
    """Workflow step action types."""

    READ_RESOURCE = "read_resource"
    CALL_TOOL = "call_tool"
    LLM_ANALYZE = "llm_analyze"
    LOG = "log"
    CONDITION = "condition"
    CONDITIONAL_TOOL = "conditional_tool"
    EXIT = "exit"


@dataclass
class WorkflowStep:
    """A single step in an agent workflow.

    Attributes:
        name: Step identifier
        action: Type of action to perform
        params: Action-specific parameters
        condition: Optional condition for execution
        loop: Optional loop variable for repeated execution
    """

    name: str
    action: StepAction
    params: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None
    loop: str | None = None

    @classmethod
    def from_dict(cls, step_dict: dict[str, Any]) -> WorkflowStep:
        """Create WorkflowStep from dictionary.

        Args:
            step_dict: Step definition from YAML

        Returns:
            WorkflowStep instance
        """
        action_str = step_dict.get("action", "")
        try:
            action = StepAction(action_str)
        except ValueError:
            logger.warning(f"Unknown action '{action_str}', defaulting to log")
            action = StepAction.LOG

        return cls(
            name=step_dict.get("step", "unnamed"),
            action=action,
            params={
                k: v
                for k, v in step_dict.items()
                if k not in ("step", "action", "condition", "loop")
            },
            condition=step_dict.get("condition"),
            loop=step_dict.get("loop"),
        )


@dataclass
class AgentConfig:
    """Agent configuration loaded from YAML.

    Attributes:
        name: Agent identifier
        description: Human-readable description
        schedule: Cron schedule expression
        model: LLM model to use
        timeout_seconds: Maximum execution time
        system_prompt: Agent instructions
        tools: Available MCP tools
        resources: Available MCP resources
        workflow: List of workflow steps
    """

    name: str
    description: str
    schedule: str
    model: str
    timeout_seconds: int
    system_prompt: str
    tools: list[str]
    resources: list[str]
    workflow: list[WorkflowStep]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> AgentConfig:
        """Load agent configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            AgentConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is malformed
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        workflow_steps = [
            WorkflowStep.from_dict(step) for step in data.get("workflow", [])
        ]

        return cls(
            name=data.get("name", yaml_path.stem),
            description=data.get("description", ""),
            schedule=data.get("schedule", ""),
            model=data.get("model", "nemotron-8b-v3.1"),
            timeout_seconds=data.get("timeout_seconds", 300),
            system_prompt=data.get("system_prompt", ""),
            tools=data.get("tools", []),
            resources=data.get("resources", []),
            workflow=workflow_steps,
        )


@dataclass
class AgentResult:
    """Result of agent execution.

    Attributes:
        agent_name: Name of executed agent
        success: Whether execution completed successfully
        output: Agent output (parsed JSON or raw string)
        steps_executed: Number of workflow steps executed
        duration_seconds: Total execution time
        errors: List of errors encountered
        step_results: Results from each workflow step
        timestamp: When execution completed
    """

    agent_name: str
    success: bool
    output: Any
    steps_executed: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)
    step_results: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class AgentRunner:
    """Executes agent workflows.

    The AgentRunner loads agent configurations from YAML files and executes
    their workflow steps, bridging to MCP tools and resources.

    Example:
        >>> runner = AgentRunner(agents_dir=Path("configs/agents"))
        >>> runner.register_tool("get_strategy_config", my_tool_func)
        >>> result = await runner.run_agent("strategy_optimizer")
    """

    def __init__(
        self,
        agents_dir: Path | None = None,
        mcp_client: Any | None = None,
    ) -> None:
        """Initialize the agent runner.

        Args:
            agents_dir: Directory containing agent YAML configs
            mcp_client: Optional MCP client for tool/resource access
        """
        self.agents_dir = agents_dir or Path("configs/agents")
        self.mcp_client = mcp_client

        # Registered handlers
        self._tools: dict[str, Callable] = {}
        self._resources: dict[str, Callable] = {}
        self._llm_handler: Callable | None = None

        # Execution state
        self._step_outputs: dict[str, Any] = {}
        self._running = False

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool handler.

        Args:
            name: Tool name (matches tools in agent config)
            handler: Async function to handle tool calls
        """
        self._tools[name] = handler
        logger.debug(f"Registered tool: {name}")

    def register_resource(
        self,
        uri: str,
        handler: Callable[[], Any],
    ) -> None:
        """Register a resource handler.

        Args:
            uri: Resource URI (matches resources in agent config)
            handler: Async function to fetch resource
        """
        self._resources[uri] = handler
        logger.debug(f"Registered resource: {uri}")

    def set_llm_handler(
        self,
        handler: Callable[[str, str], Any],
    ) -> None:
        """Set the LLM handler for llm_analyze steps.

        Args:
            handler: Async function(system_prompt, user_input) -> response
        """
        self._llm_handler = handler

    def list_agents(self) -> list[str]:
        """List available agent names.

        Returns:
            List of agent names from YAML files
        """
        if not self.agents_dir.exists():
            return []
        return [p.stem for p in self.agents_dir.glob("*.yaml")]

    def load_agent(self, agent_name: str) -> AgentConfig:
        """Load an agent configuration.

        Args:
            agent_name: Name of agent to load

        Returns:
            AgentConfig instance

        Raises:
            FileNotFoundError: If agent config doesn't exist
        """
        yaml_path = self.agents_dir / f"{agent_name}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Agent config not found: {yaml_path}")
        return AgentConfig.from_yaml(yaml_path)

    async def run_agent(
        self,
        agent_name: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Execute an agent's workflow.

        Args:
            agent_name: Name of agent to run
            context: Optional initial context variables

        Returns:
            AgentResult with execution details
        """
        start_time = datetime.now(UTC)
        errors: list[str] = []
        self._step_outputs = context.copy() if context else {}
        self._running = True

        try:
            config = self.load_agent(agent_name)
            logger.info(f"Starting agent: {agent_name} (model: {config.model})")

            steps_executed = 0
            final_output = None

            for step in config.workflow:
                if not self._running:
                    break

                try:
                    # Check condition if present
                    if step.condition and not self._evaluate_condition(step.condition):
                        logger.debug(f"Skipping step {step.name}: condition not met")
                        continue

                    # Execute step
                    result = await self._execute_step(step, config)
                    self._step_outputs[step.name] = {"output": result}
                    steps_executed += 1
                    final_output = result

                    logger.debug(f"Step {step.name} completed")

                except StopIteration:
                    # Exit step triggered
                    logger.debug(f"Agent exiting at step {step.name}")
                    break
                except Exception as e:
                    error_msg = f"Step {step.name} failed: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            duration = (datetime.now(UTC) - start_time).total_seconds()

            return AgentResult(
                agent_name=agent_name,
                success=len(errors) == 0,
                output=final_output,
                steps_executed=steps_executed,
                duration_seconds=duration,
                errors=errors,
                step_results=self._step_outputs,
            )

        except Exception as e:
            duration = (datetime.now(UTC) - start_time).total_seconds()
            return AgentResult(
                agent_name=agent_name,
                success=False,
                output=None,
                steps_executed=0,
                duration_seconds=duration,
                errors=[str(e)],
            )
        finally:
            self._running = False

    async def _execute_step(
        self,
        step: WorkflowStep,
        config: AgentConfig,
    ) -> Any:
        """Execute a single workflow step.

        Args:
            step: Step to execute
            config: Agent configuration

        Returns:
            Step result
        """
        if step.action == StepAction.EXIT:
            raise StopIteration()

        elif step.action == StepAction.READ_RESOURCE:
            resource_uri = step.params.get("resource", "")
            return await self._read_resource(resource_uri)

        elif step.action == StepAction.CALL_TOOL:
            tool_name = step.params.get("tool", "")
            tool_params = self._interpolate_params(step.params.get("params", {}))
            return await self._call_tool(tool_name, tool_params)

        elif step.action == StepAction.LLM_ANALYZE:
            input_text = self._interpolate_string(step.params.get("input", ""))
            return await self._llm_analyze(config.system_prompt, input_text)

        elif step.action == StepAction.LOG:
            level = step.params.get("level", "info")
            message = self._interpolate_string(step.params.get("message", ""))
            log_func = getattr(logger, level, logger.info)
            log_func(f"[{config.name}] {message}")
            return {"logged": message}

        elif step.action == StepAction.CONDITION:
            # Condition action - check and optionally execute nested steps
            on_true = step.params.get("on_true", [])
            if self._evaluate_condition(step.condition or "true"):
                for nested in on_true:
                    if isinstance(nested, dict):
                        if nested.get("action") == "exit":
                            raise StopIteration()
            return {"condition_result": True}

        elif step.action == StepAction.CONDITIONAL_TOOL:
            conditions = step.params.get("conditions", [])
            for cond in conditions:
                if self._evaluate_condition(cond.get("condition", "false")):
                    tool_name = cond.get("tool", "")
                    tool_params = self._interpolate_params(cond.get("params", {}))
                    return await self._call_tool(tool_name, tool_params)
            return {"no_condition_matched": True}

        else:
            logger.warning(f"Unknown action: {step.action}")
            return None

    async def _read_resource(self, uri: str) -> Any:
        """Read an MCP resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if uri in self._resources:
            handler = self._resources[uri]
            if asyncio.iscoroutinefunction(handler):
                return await handler()
            return handler()

        # Try MCP client if available
        if self.mcp_client:
            return await self.mcp_client.read_resource(uri)

        logger.warning(f"Resource not found: {uri}")
        return {"error": f"Resource not found: {uri}"}

    async def _call_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> Any:
        """Call an MCP tool.

        Args:
            tool_name: Tool name
            params: Tool parameters

        Returns:
            Tool result
        """
        if tool_name in self._tools:
            handler = self._tools[tool_name]
            if asyncio.iscoroutinefunction(handler):
                return await handler(**params)
            return handler(**params)

        # Try MCP client if available
        if self.mcp_client:
            return await self.mcp_client.call_tool(tool_name, params)

        logger.warning(f"Tool not found: {tool_name}")
        return {"error": f"Tool not found: {tool_name}"}

    async def _llm_analyze(
        self,
        system_prompt: str,
        user_input: str,
    ) -> Any:
        """Perform LLM analysis.

        Args:
            system_prompt: System prompt from agent config
            user_input: Interpolated user input

        Returns:
            LLM response (parsed JSON if possible)
        """
        if self._llm_handler:
            if asyncio.iscoroutinefunction(self._llm_handler):
                response = await self._llm_handler(system_prompt, user_input)
            else:
                response = self._llm_handler(system_prompt, user_input)

            # Try to parse as JSON
            if isinstance(response, str):
                try:
                    # Extract JSON from markdown code blocks
                    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))
                    return json.loads(response)
                except json.JSONDecodeError:
                    return {"raw_response": response}
            return response

        # Mock response for testing
        logger.warning("No LLM handler set, returning mock response")
        return {
            "analysis": "Mock LLM analysis",
            "adjustments": [],
            "action_taken": "none",
            "reasoning": "No LLM handler configured",
        }

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition expression.

        Supports simple template expressions like:
        - {{ step_name.output.field }}
        - {{ value | length > 0 }}

        Args:
            condition: Condition expression

        Returns:
            Boolean result
        """
        if not condition or condition.lower() == "true":
            return True
        if condition.lower() == "false":
            return False

        # Simple template variable substitution
        interpolated = self._interpolate_string(condition)

        # Handle common comparisons
        if "<" in interpolated or ">" in interpolated or "==" in interpolated:
            try:
                # Very simple eval for numeric comparisons
                # In production, use a proper expression parser
                result = eval(interpolated, {"__builtins__": {}}, {})
                return bool(result)
            except Exception:
                pass

        # Check if it's a truthy value
        try:
            return bool(interpolated) and interpolated.lower() not in ("false", "none", "0")
        except Exception:
            return False

    def _interpolate_string(self, template: str) -> str:
        """Interpolate template variables in a string.

        Replaces {{ step_name.output.field }} with actual values.

        Args:
            template: Template string

        Returns:
            Interpolated string
        """
        if not isinstance(template, str):
            return str(template)

        def replace_var(match: re.Match) -> str:
            var_path = match.group(1).strip()
            parts = var_path.split(".")

            value = self._step_outputs
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, "")
                else:
                    value = getattr(value, part, "")

            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return str(value)

        return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", replace_var, template)

    def _interpolate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Interpolate template variables in parameters.

        Args:
            params: Parameters with potential template variables

        Returns:
            Interpolated parameters
        """
        result = {}
        for key, value in params.items():
            if isinstance(value, str):
                result[key] = self._interpolate_string(value)
            elif isinstance(value, dict):
                result[key] = self._interpolate_params(value)
            else:
                result[key] = value
        return result

    def stop(self) -> None:
        """Stop the currently running agent."""
        self._running = False
