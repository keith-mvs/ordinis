"""Tests for Agent Runner module.

Tests cover:
- StepAction enum
- WorkflowStep dataclass
- AgentConfig dataclass
- AgentResult dataclass
- AgentRunner class
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

import pytest
import yaml

from ordinis.agents.runner import (
    StepAction,
    WorkflowStep,
    AgentConfig,
    AgentResult,
    AgentRunner,
)


class TestStepAction:
    """Tests for StepAction enum."""

    @pytest.mark.unit
    def test_step_action_values(self):
        """Test all step action values exist."""
        assert StepAction.READ_RESOURCE.value == "read_resource"
        assert StepAction.CALL_TOOL.value == "call_tool"
        assert StepAction.LLM_ANALYZE.value == "llm_analyze"
        assert StepAction.LOG.value == "log"
        assert StepAction.CONDITION.value == "condition"
        assert StepAction.EXIT.value == "exit"

    @pytest.mark.unit
    def test_step_action_count(self):
        """Test correct number of step actions."""
        assert len(StepAction) >= 6


class TestWorkflowStep:
    """Tests for WorkflowStep dataclass."""

    @pytest.mark.unit
    def test_create_step_basic(self):
        """Test creating a basic workflow step."""
        step = WorkflowStep(
            name="test_step",
            action=StepAction.LOG,
        )

        assert step.name == "test_step"
        assert step.action == StepAction.LOG
        assert step.params == {}
        assert step.condition is None
        assert step.loop is None

    @pytest.mark.unit
    def test_step_with_params(self):
        """Test step with parameters."""
        step = WorkflowStep(
            name="call_tool",
            action=StepAction.CALL_TOOL,
            params={"tool": "analyze", "input": "data"},
        )

        assert step.params == {"tool": "analyze", "input": "data"}

    @pytest.mark.unit
    def test_step_with_condition(self):
        """Test step with condition."""
        step = WorkflowStep(
            name="conditional_step",
            action=StepAction.CALL_TOOL,
            condition="result.success == true",
        )

        assert step.condition == "result.success == true"

    @pytest.mark.unit
    def test_step_with_loop(self):
        """Test step with loop."""
        step = WorkflowStep(
            name="loop_step",
            action=StepAction.CALL_TOOL,
            loop="items",
        )

        assert step.loop == "items"

    @pytest.mark.unit
    def test_from_dict_basic(self):
        """Test creating step from dictionary."""
        step_dict = {
            "step": "my_step",
            "action": "log",
            "message": "Hello",
        }

        step = WorkflowStep.from_dict(step_dict)

        assert step.name == "my_step"
        assert step.action == StepAction.LOG
        assert step.params == {"message": "Hello"}

    @pytest.mark.unit
    def test_from_dict_unknown_action(self):
        """Test creating step with unknown action defaults to LOG."""
        step_dict = {
            "step": "my_step",
            "action": "unknown_action",
        }

        step = WorkflowStep.from_dict(step_dict)

        assert step.action == StepAction.LOG

    @pytest.mark.unit
    def test_from_dict_with_condition(self):
        """Test creating step with condition from dict."""
        step_dict = {
            "step": "conditional",
            "action": "call_tool",
            "condition": "x > 0",
            "tool": "my_tool",
        }

        step = WorkflowStep.from_dict(step_dict)

        assert step.condition == "x > 0"
        assert step.params == {"tool": "my_tool"}


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    @pytest.mark.unit
    def test_create_config_basic(self):
        """Test creating a basic agent config."""
        config = AgentConfig(
            name="test_agent",
            description="Test agent",
            schedule="0 * * * *",
            model="test-model",
            timeout_seconds=60,
            system_prompt="You are a test agent",
            tools=["tool1", "tool2"],
            resources=["resource1"],
            workflow=[],
        )

        assert config.name == "test_agent"
        assert config.description == "Test agent"
        assert config.model == "test-model"
        assert config.timeout_seconds == 60
        assert len(config.tools) == 2

    @pytest.mark.unit
    def test_from_yaml_basic(self, tmp_path):
        """Test loading config from YAML file."""
        yaml_content = """
name: my_agent
description: My test agent
schedule: "0 0 * * *"
model: test-model
timeout_seconds: 120
system_prompt: You are an agent
tools:
  - tool1
  - tool2
resources:
  - resource1
workflow:
  - step: log_start
    action: log
    message: Starting
"""
        yaml_path = tmp_path / "agent.yaml"
        yaml_path.write_text(yaml_content)

        config = AgentConfig.from_yaml(yaml_path)

        assert config.name == "my_agent"
        assert config.description == "My test agent"
        assert config.model == "test-model"
        assert config.timeout_seconds == 120
        assert len(config.workflow) == 1
        assert config.workflow[0].action == StepAction.LOG

    @pytest.mark.unit
    def test_from_yaml_defaults(self, tmp_path):
        """Test loading config with defaults."""
        yaml_content = """
workflow: []
"""
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text(yaml_content)

        config = AgentConfig.from_yaml(yaml_path)

        assert config.name == "minimal"  # defaults to filename
        assert config.description == ""
        assert config.model == "nemotron-8b-v3.1"
        assert config.timeout_seconds == 300

    @pytest.mark.unit
    def test_from_yaml_missing_file(self, tmp_path):
        """Test loading config from missing file raises error."""
        yaml_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            AgentConfig.from_yaml(yaml_path)


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    @pytest.mark.unit
    def test_create_result_success(self):
        """Test creating a successful result."""
        result = AgentResult(
            agent_name="test_agent",
            success=True,
            output={"result": "done"},
            steps_executed=5,
            duration_seconds=10.5,
            errors=[],
            step_results={},
            timestamp=datetime.utcnow(),
        )

        assert result.agent_name == "test_agent"
        assert result.success is True
        assert result.output == {"result": "done"}
        assert result.steps_executed == 5

    @pytest.mark.unit
    def test_create_result_failure(self):
        """Test creating a failed result."""
        result = AgentResult(
            agent_name="test_agent",
            success=False,
            output=None,
            steps_executed=2,
            duration_seconds=5.0,
            errors=["Step 3 failed", "Timeout"],
            step_results={},
            timestamp=datetime.utcnow(),
        )

        assert result.success is False
        assert len(result.errors) == 2


class TestAgentRunner:
    """Tests for AgentRunner class."""

    @pytest.mark.unit
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        runner = AgentRunner()

        assert runner.agents_dir is not None
        assert runner._tools == {}
        assert runner._resources == {}

    @pytest.mark.unit
    def test_init_custom_agents_dir(self, tmp_path):
        """Test initialization with custom agents directory."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        runner = AgentRunner(agents_dir=agents_dir)

        assert runner.agents_dir == agents_dir

    @pytest.mark.unit
    def test_register_tool(self):
        """Test registering a tool handler."""
        runner = AgentRunner()

        def my_tool():
            return "result"

        runner.register_tool("my_tool", my_tool)

        assert "my_tool" in runner._tools
        assert runner._tools["my_tool"] is my_tool

    @pytest.mark.unit
    def test_register_resource(self):
        """Test registering a resource handler."""
        runner = AgentRunner()

        def my_resource():
            return "data"

        runner.register_resource("file://test", my_resource)

        assert "file://test" in runner._resources
        assert runner._resources["file://test"] is my_resource

    @pytest.mark.unit
    def test_set_llm_handler(self):
        """Test setting LLM handler."""
        runner = AgentRunner()

        async def llm_handler(system_prompt, user_input):
            return "response"

        runner.set_llm_handler(llm_handler)

        assert runner._llm_handler is llm_handler

    @pytest.mark.unit
    def test_list_agents_empty(self, tmp_path):
        """Test listing agents in empty directory."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        runner = AgentRunner(agents_dir=agents_dir)

        agents = runner.list_agents()
        assert agents == []

    @pytest.mark.unit
    def test_list_agents_with_files(self, tmp_path):
        """Test listing agent YAML files."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create agent YAML files
        (agents_dir / "agent1.yaml").write_text("name: agent1\nworkflow: []")
        (agents_dir / "agent2.yaml").write_text("name: agent2\nworkflow: []")
        (agents_dir / "not_yaml.txt").write_text("not a yaml file")

        runner = AgentRunner(agents_dir=agents_dir)

        agents = runner.list_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents

    @pytest.mark.unit
    def test_list_agents_nonexistent_dir(self, tmp_path):
        """Test listing agents when directory doesn't exist."""
        agents_dir = tmp_path / "nonexistent"

        runner = AgentRunner(agents_dir=agents_dir)

        agents = runner.list_agents()
        assert agents == []

    @pytest.mark.unit
    def test_load_agent(self, tmp_path):
        """Test loading a single agent."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        yaml_content = """
name: test_agent
description: Test
schedule: "* * * * *"
workflow: []
"""
        (agents_dir / "test_agent.yaml").write_text(yaml_content)

        runner = AgentRunner(agents_dir=agents_dir)
        config = runner.load_agent("test_agent")

        assert config is not None
        assert config.name == "test_agent"

    @pytest.mark.unit
    def test_load_agent_not_found(self, tmp_path):
        """Test loading nonexistent agent raises FileNotFoundError."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        runner = AgentRunner(agents_dir=agents_dir)

        with pytest.raises(FileNotFoundError):
            runner.load_agent("nonexistent")
