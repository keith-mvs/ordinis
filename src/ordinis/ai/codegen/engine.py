"""
CodeGen - AI Code Generation Engine.

Leverages Codestral 25.01 via Helix for high-performance code generation.
"""

import asyncio
import re
from typing import TYPE_CHECKING, Optional

from ordinis.ai.helix import Helix
from ordinis.ai.helix.models import ChatMessage
from ordinis.core.logging import TraceContext
from ordinis.engines.base import (
    BaseEngine,
    BaseEngineConfig,
    EngineState,
    HealthLevel,
    HealthStatus,
)

if TYPE_CHECKING:
    from ordinis.ai.synapse.engine import Synapse


class CodeGenEngine(BaseEngine):
    """
    CodeGen Engine.

    Specialized engine for software development tasks using Codestral 25.01.
    """

    def __init__(self, helix: Helix, synapse: Optional["Synapse"] = None):
        """
        Initialize CodeGen engine.

        Args:
            helix: Helix LLM provider instance
            synapse: Synapse RAG engine instance (optional but recommended)
        """
        super().__init__(config=BaseEngineConfig(name="codegen"))
        self.helix = helix
        self.synapse = synapse
        # Use configured default code model instead of hardcoded "codestral"
        self._model = helix.config.default_code_model

    async def _do_initialize(self) -> None:
        """Initialize CodeGen engine."""
        self._logger.info("Initializing CodeGen engine...")
        self._health_status = HealthStatus(
            level=HealthLevel.HEALTHY,
            message="CodeGen engine initialized",
        )

    async def _do_shutdown(self) -> None:
        """Shutdown CodeGen engine."""
        self._logger.info("Shutting down CodeGen engine...")

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        # Check Helix health
        helix_health = await self.helix.health_check()
        if helix_health.level != HealthLevel.HEALTHY:
            return HealthStatus(
                level=HealthLevel.DEGRADED,
                message=f"Helix provider degraded: {helix_health.message}",
            )

        return self._health_status

    async def generate_code(
        self,
        prompt: str,
        context: str = "",
        language: str = "python",
        mode: str = "complex",  # "simple" or "complex"
    ) -> str:
        """
        Generate code based on prompt.

        Args:
            prompt: Description of code to generate
            context: Additional context (existing code, interfaces)
            language: Target programming language
            mode: Complexity mode ("simple" for fast/cheap, "complex" for deep reasoning)

        Returns:
            Generated code
        """
        if self._state != EngineState.READY:
            await self.initialize()

        # Select model based on mode
        model = (
            self.helix.config.fallback_code_model
            if mode == "simple"
            else self.helix.config.default_code_model
        )

        with TraceContext() as trace_id:
            self._logger.info(
                "Generating code",
                data={
                    "language": language,
                    "mode": mode,
                    "model": model,
                    "prompt_preview": prompt[:50],
                },
            )

            # Enhance context with Synapse if available (only for complex mode)
            if self.synapse and mode == "complex":
                try:
                    # Run retrieval in thread pool with timeout to avoid blocking/hanging
                    rag_context = await asyncio.wait_for(
                        asyncio.to_thread(self.synapse.retrieve_for_codegen, prompt, language),
                        timeout=5.0,  # 5 second timeout for RAG
                    )

                    if rag_context:
                        context = f"{context}\n\n--- RAG Context ---\n{rag_context}"
                        self._logger.info("Enhanced prompt with RAG context")
                except TimeoutError:
                    self._logger.warning("RAG retrieval timed out, proceeding without context")
                except Exception as e:
                    self._logger.warning(f"Failed to retrieve RAG context: {e}")

            system_prompt = f"""You are an expert {language} developer.
Your task is to generate high-quality, production-ready code.
- Follow best practices and idioms for {language}.
- Include type hints and docstrings.
- Handle errors and edge cases.
- Return ONLY the code, no markdown formatting or explanations unless requested.
"""

            user_prompt = f"""
Request: {prompt}

Context:
{context}
"""

            try:
                response = await self.helix.generate(
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    model=model,
                    temperature=0.1,  # Low temperature for precision
                )
                self._logger.info("Code generation successful")
                return self._sanitize_output(response.content)
            except Exception as e:
                self._logger.error(f"Code generation failed: {e}")
                raise

    async def refactor_code(
        self,
        code: str,
        instructions: str,
    ) -> str:
        """
        Refactor existing code.

        Args:
            code: Code to refactor
            instructions: Refactoring instructions

        Returns:
            Refactored code
        """
        if self._state != EngineState.READY:
            await self.initialize()

        with TraceContext() as trace_id:
            self._logger.info("Refactoring code", data={"instructions": instructions})

            system_prompt = """You are an expert code refactoring assistant.
Refactor the provided code according to the instructions.
Maintain existing functionality unless asked to change it.
Improve readability, performance, and safety.
"""

            user_prompt = f"""
Instructions: {instructions}

Code:
```
{code}
```
"""

            try:
                response = await self.helix.generate(
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    model=self._model,
                    temperature=0.1,
                )
                self._logger.info("Code refactoring successful")
                return self._sanitize_output(response.content)
            except Exception as e:
                self._logger.error(f"Code refactoring failed: {e}")
                raise

    async def generate_tests(
        self,
        code: str,
        framework: str = "pytest",
    ) -> str:
        """
        Generate unit tests for code.

        Args:
            code: Code to test
            framework: Testing framework to use

        Returns:
            Test code
        """
        if self._state != EngineState.READY:
            await self.initialize()

        with TraceContext() as trace_id:
            self._logger.info("Generating tests", data={"framework": framework})

            system_prompt = f"""You are an expert QA engineer.
Generate comprehensive unit tests using {framework}.
- Cover happy paths and edge cases.
- Mock external dependencies.
- Use descriptive test names.
"""

            user_prompt = f"""
Code to test:
```
{code}
```
"""

            try:
                response = await self.helix.generate(
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    model=self._model,
                    temperature=0.2,
                )
                self._logger.info("Test generation successful")
                return self._sanitize_output(response.content)
            except Exception as e:
                self._logger.error(f"Test generation failed: {e}")
                raise

    def _sanitize_output(self, content: str) -> str:
        """Strip model-added markdown and <think> blocks."""
        # Remove <think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove markdown code fences, handling optional language identifier
        # Matches ```python ... ``` or just ``` ... ```
        # The group(2) captures the content inside the fences
        code_block_pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(code_block_pattern, cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(1)
        else:
            # Fallback: if no closed block, try to strip just the markers if they exist
            cleaned = cleaned.replace("```", "")

        return cleaned.strip()

    async def explain_code(
        self,
        code: str,
        detail_level: str = "detailed",
    ) -> str:
        """
        Explain code functionality.

        Args:
            code: Code to explain
            detail_level: 'brief' or 'detailed'

        Returns:
            Explanation
        """
        if self._state != EngineState.READY:
            await self.initialize()

        with TraceContext() as trace_id:
            system_prompt = f"""You are an expert code instructor.
Explain the provided code.
Detail Level: {detail_level}
"""
            user_prompt = f"Code:\n```\n{code}\n```"

            try:
                response = await self.helix.generate(
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    model=self._model,
                    temperature=0.2,
                )
                return response.content
            except Exception as e:
                self._logger.error(f"Code explanation failed: {e}")
                raise

    async def review_code(
        self,
        code: str,
        focus: str = "security,performance,style",
    ) -> str:
        """
        Review code for issues.

        Args:
            code: Code to review
            focus: Comma-separated list of focus areas

        Returns:
            Review report
        """
        if self._state != EngineState.READY:
            await self.initialize()

        with TraceContext() as trace_id:
            # Get similar implementations for comparison if Synapse is available
            context = ""
            if self.synapse:
                try:
                    similar = self.synapse.find_similar_implementations(code[:200])
                    if similar:
                        context = "\nSimilar implementations found in codebase:\n" + "\n".join(
                            similar
                        )
                except Exception:
                    pass

            system_prompt = f"""You are an expert code reviewer.
Review the code focusing on: {focus}.
Identify bugs, security risks, and performance bottlenecks.
Provide actionable suggestions.
"""
            user_prompt = f"""
Code to review:
```
{code}
```

{context}
"""

            try:
                response = await self.helix.generate(
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    model=self._model,
                    temperature=0.1,
                )
                return response.content
            except Exception as e:
                self._logger.error(f"Code review failed: {e}")
                raise

    async def generate_nl2code(
        self,
        query: str,
        language: str = "python",
    ) -> str:
        """
        Generate code from natural language query (NL2SQL style but for code).

        Args:
            query: Natural language query
            language: Target language

        Returns:
            Generated code
        """
        # This is effectively an alias for generate_code but emphasizes the NL aspect
        # and might use different prompting strategies in the future.
        return await self.generate_code(query, language=language)
