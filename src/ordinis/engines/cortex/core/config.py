"""Cortex engine configuration.

Defines configuration for the Cortex LLM orchestration engine,
including model routing, governance settings, and safety controls.
"""

from dataclasses import dataclass, field
import re
from typing import Any

from ordinis.engines.base.config import AIEngineConfig

# Model ID constants
MODEL_DEEPSEEK_R1 = "deepseek-r1"
MODEL_NEMOTRON_49B = "nemotron-super-49b-v1.5"
MODEL_NEMOTRON_8B = "nemotron-8b-v3.1"


@dataclass
class ModelConfig:
    """Configuration for a specific model task.

    Attributes:
        primary: Primary model ID.
        fallback: List of fallback model IDs (tried in order).
        temperature: Model temperature (0.0-2.0).
        max_tokens: Maximum tokens for generation.
        timeout_seconds: Timeout for this operation.
    """

    primary: str = MODEL_DEEPSEEK_R1
    fallback: list[str] = field(default_factory=list)
    temperature: float = 0.6
    max_tokens: int = 4096
    timeout_seconds: float = 60.0


@dataclass
class SafetyConfig:
    """Safety and sanitization configuration.

    Attributes:
        sanitize_inputs: Enable input sanitization.
        strip_pii: Strip PII patterns from inputs.
        strip_secrets: Strip secret patterns from inputs.
        max_prompt_length: Maximum prompt length (chars).
        max_code_length: Maximum code snippet length for analysis.
        blocked_patterns: Regex patterns to block from inputs.
    """

    sanitize_inputs: bool = True
    strip_pii: bool = True
    strip_secrets: bool = True
    max_prompt_length: int = 32000
    max_code_length: int = 16000
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?[\w@#$%^&*]+",
        r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*['\"]?[\w\-]+",
        r"(?i)bearer\s+[\w\-\.]+",
        r"`[^`]+`",  # Backtick command substitution
        r"\$\([^)]+\)",  # $(command) substitution
        r"/etc/(passwd|shadow|hosts)",  # Sensitive file paths
        r"rm\s+-rf",  # Dangerous commands
    ])


@dataclass
class CortexConfig(AIEngineConfig):
    """Configuration for Cortex LLM orchestration engine.

    Extends AIEngineConfig with Cortex-specific settings for
    model routing, safety controls, and task configurations.

    Attributes:
        rag_enabled: Enable RAG context retrieval.
        hypothesis_model: Model config for hypothesis generation.
        code_analysis_model: Model config for code analysis.
        research_model: Model config for research synthesis.
        safety: Safety and sanitization config.
        persist_outputs: Persist outputs to storage.
        output_retention_hours: Hours to retain outputs (0 = forever).
    """

    name: str = "cortex"
    rag_enabled: bool = False
    use_structured_output: bool = True  # Use JSON mode for hypothesis generation
    use_confidence_calibration: bool = False  # Use reward model for confidence scoring

    # Model configurations per task
    hypothesis_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        primary=MODEL_DEEPSEEK_R1,
        fallback=[MODEL_NEMOTRON_49B, MODEL_NEMOTRON_8B],
        temperature=0.6,
        max_tokens=4096,
    ))
    code_analysis_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        primary=MODEL_DEEPSEEK_R1,
        fallback=[MODEL_NEMOTRON_49B],
        temperature=0.2,
        max_tokens=4096,
    ))
    research_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        primary=MODEL_DEEPSEEK_R1,
        fallback=[MODEL_NEMOTRON_49B],
        temperature=0.4,
        max_tokens=8192,
    ))

    # Safety settings
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Persistence settings
    persist_outputs: bool = False
    output_retention_hours: int = 168  # 7 days

    def validate(self) -> list[str]:
        """Validate Cortex configuration."""
        errors = super().validate()

        # Validate model configs
        for task_name in ["hypothesis_model", "code_analysis_model", "research_model"]:
            model_config = getattr(self, task_name)
            if not model_config.primary:
                errors.append(f"{task_name}.primary cannot be empty")
            if model_config.temperature < 0 or model_config.temperature > 2.0:
                errors.append(f"{task_name}.temperature must be between 0 and 2.0")
            if model_config.max_tokens < 1:
                errors.append(f"{task_name}.max_tokens must be positive")

        # Validate safety patterns
        for i, pattern in enumerate(self.safety.blocked_patterns):
            try:
                re.compile(pattern)
            except re.error as e:
                errors.append(f"safety.blocked_patterns[{i}] is invalid regex: {e}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        base = super().to_dict()
        base.update({
            "rag_enabled": self.rag_enabled,
            "hypothesis_model": {
                "primary": self.hypothesis_model.primary,
                "fallback": self.hypothesis_model.fallback,
                "temperature": self.hypothesis_model.temperature,
                "max_tokens": self.hypothesis_model.max_tokens,
            },
            "code_analysis_model": {
                "primary": self.code_analysis_model.primary,
                "fallback": self.code_analysis_model.fallback,
                "temperature": self.code_analysis_model.temperature,
                "max_tokens": self.code_analysis_model.max_tokens,
            },
            "research_model": {
                "primary": self.research_model.primary,
                "fallback": self.research_model.fallback,
                "temperature": self.research_model.temperature,
                "max_tokens": self.research_model.max_tokens,
            },
            "safety": {
                "sanitize_inputs": self.safety.sanitize_inputs,
                "strip_pii": self.safety.strip_pii,
                "strip_secrets": self.safety.strip_secrets,
                "max_prompt_length": self.safety.max_prompt_length,
                "max_code_length": self.safety.max_code_length,
            },
            "persist_outputs": self.persist_outputs,
            "output_retention_hours": self.output_retention_hours,
        })
        return base
