"""
Helix Development Assistant CLI.

Interactive development assistant powered by NVIDIA Nemotron Super 49B
for code review, refactoring guidance, documentation generation, and
architecture consultation.

Usage:
    python scripts/helix_dev_assistant.py review <file_path>
    python scripts/helix_dev_assistant.py chat
    python scripts/helix_dev_assistant.py strategy "<description>"
    python scripts/helix_dev_assistant.py docs <file_path>
    python scripts/helix_dev_assistant.py architecture "<question>"

Examples:
    # Code review
    python scripts/helix_dev_assistant.py review src/ordinis/engines/cortex/core/engine.py

    # Interactive chat
    python scripts/helix_dev_assistant.py chat

    # Strategy design
    python scripts/helix_dev_assistant.py strategy "Iron condor with 10 delta wings"

    # Documentation generation
    python scripts/helix_dev_assistant.py docs src/ordinis/ai/helix/engine.py

    # Architecture consultation
    python scripts/helix_dev_assistant.py architecture "How should engines communicate?"
"""

import asyncio
from pathlib import Path
import sys

from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.helix.engine import Helix


class HelixDevAssistant:
    """Development assistant CLI powered by Helix."""

    def __init__(self, temperature: float = 0.1):
        """
        Initialize dev assistant.

        Args:
            temperature: Generation temperature (0.0-1.0)
                        Lower = more deterministic (good for code)
                        Higher = more creative (good for strategy)
        """
        self.helix = Helix(HelixConfig(default_temperature=temperature))
        self.temperature = temperature

    async def review_code(self, file_path: str) -> None:
        """
        Review code file and provide actionable feedback.

        Args:
            file_path: Path to file to review

        Focuses on:
        - Architecture and design patterns
        - Type safety and error handling
        - Performance and scalability
        - Production readiness
        """
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f"HELIX CODE REVIEW: {path.name}")
        print(f"{'='*70}\n")

        code = path.read_text(encoding="utf-8")

        # Limit code size for API
        if len(code) > 10000:
            print("Warning: File is large, reviewing first 10,000 characters\n")
            code = code[:10000]

        response = await self.helix.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python code reviewer for production trading systems. "
                        "Focus on: architecture, type safety, error handling, performance, "
                        "testing, and production readiness. "
                        "Be specific and actionable. Provide concrete code examples for improvements."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Review this Python file:\n\n```python\n{code}\n```",
                },
            ],
            max_tokens=2500,
        )

        print(response.content)
        print(f"\n{'='*70}")
        print(
            f"Tokens: {response.usage.total_tokens} | "
            f"Latency: {response.latency_ms:.0f}ms | "
            f"Cost: ~${response.usage.total_tokens * 0.000001:.4f}"
        )
        print(f"{'='*70}\n")

    async def interactive_chat(self) -> None:
        """
        Start interactive chat session for development questions.

        Enter 'exit', 'quit', or Ctrl+C to end session.
        """
        print(f"\n{'='*70}")
        print("HELIX INTERACTIVE CHAT")
        print(f"{'='*70}")
        print("Ask development questions, get architecture guidance, discuss code.")
        print("Type 'exit' or 'quit' to end session.\n")

        conversation_history = [
            {
                "role": "system",
                "content": (
                    "You are an expert software architect and Python developer "
                    "for the Ordinis algorithmic trading platform. "
                    "Provide concrete, actionable guidance. "
                    "Include code examples when relevant."
                ),
            }
        ]

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("\nEnding chat session.")
                    break

                conversation_history.append({"role": "user", "content": user_input})

                response = await self.helix.generate(
                    messages=conversation_history,
                    max_tokens=2000,
                )

                print(f"\nHelix: {response.content}")

                # Add assistant response to history
                conversation_history.append({"role": "assistant", "content": response.content})

                # Keep history manageable (last 10 exchanges)
                if len(conversation_history) > 21:  # system + 10 exchanges
                    conversation_history = [conversation_history[0]] + conversation_history[-20:]

            except KeyboardInterrupt:
                print("\n\nEnding chat session.")
                break
            except Exception as e:
                print(f"\nError: {e}")

    async def design_strategy(self, description: str) -> None:
        """
        Generate strategy implementation design.

        Args:
            description: Strategy description and requirements

        Provides:
        - Entry/exit logic
        - Risk management
        - Position sizing
        - Implementation code structure
        """
        print(f"\n{'='*70}")
        print("HELIX STRATEGY DESIGN")
        print(f"{'='*70}\n")

        response = await self.helix.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert quantitative trading strategist. "
                        "Design Python implementations for the Ordinis trading platform. "
                        "Include: entry/exit logic, risk management, position sizing, "
                        "and complete implementation structure. "
                        "Provide runnable code examples."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Design a trading strategy with these requirements:\n\n{description}",
                },
            ],
            max_tokens=3000,
        )

        print(response.content)
        print(f"\n{'='*70}")
        print(f"Tokens: {response.usage.total_tokens} | Latency: {response.latency_ms:.0f}ms")
        print(f"{'='*70}\n")

    async def generate_docs(self, file_path: str) -> None:
        """
        Generate comprehensive documentation for code file.

        Args:
            file_path: Path to file to document

        Generates:
        - Module overview
        - Function/class documentation
        - Usage examples
        - API reference
        """
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f"HELIX DOCUMENTATION: {path.name}")
        print(f"{'='*70}\n")

        code = path.read_text(encoding="utf-8")

        if len(code) > 10000:
            print("Warning: File is large, documenting first 10,000 characters\n")
            code = code[:10000]

        response = await self.helix.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical documentation expert. "
                        "Generate comprehensive, clear documentation for Python code. "
                        "Include: module overview, function/class descriptions, "
                        "usage examples, and API reference. "
                        "Use Markdown format."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Generate documentation for this code:\n\n```python\n{code}\n```",
                },
            ],
            max_tokens=2500,
        )

        print(response.content)
        print(f"\n{'='*70}")
        print(f"Tokens: {response.usage.total_tokens} | Latency: {response.latency_ms:.0f}ms")
        print(f"{'='*70}\n")

    async def architecture_consult(self, question: str) -> None:
        """
        Get architectural guidance and recommendations.

        Args:
            question: Architecture question or problem statement

        Provides:
        - Pattern recommendations
        - Trade-off analysis
        - Implementation guidance
        - Concrete examples
        """
        print(f"\n{'='*70}")
        print("HELIX ARCHITECTURE CONSULTATION")
        print(f"{'='*70}\n")

        response = await self.helix.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software architect for the Ordinis "
                        "algorithmic trading platform. "
                        "Provide concrete, actionable architectural guidance. "
                        "Consider: scalability, maintainability, performance, "
                        "testability, and production readiness. "
                        "Discuss trade-offs and provide code examples."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=2500,
        )

        print(response.content)
        print(f"\n{'='*70}")
        print(f"Tokens: {response.usage.total_tokens} | Latency: {response.latency_ms:.0f}ms")
        print(f"{'='*70}\n")


async def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    # Temperature settings per command
    temperature_map = {
        "review": 0.1,  # Deterministic for code review
        "chat": 0.2,  # Slightly creative for discussion
        "strategy": 0.3,  # More creative for strategy design
        "docs": 0.1,  # Deterministic for documentation
        "architecture": 0.2,  # Balanced for architecture
    }

    temperature = temperature_map.get(command, 0.1)
    assistant = HelixDevAssistant(temperature=temperature)

    try:
        if command == "review":
            if len(sys.argv) < 3:
                print("Error: File path required")
                print("Usage: python scripts/helix_dev_assistant.py review <file_path>")
                sys.exit(1)
            await assistant.review_code(sys.argv[2])

        elif command == "chat":
            await assistant.interactive_chat()

        elif command == "strategy":
            if len(sys.argv) < 3:
                print("Error: Strategy description required")
                print('Usage: python scripts/helix_dev_assistant.py strategy "<description>"')
                sys.exit(1)
            await assistant.design_strategy(sys.argv[2])

        elif command == "docs":
            if len(sys.argv) < 3:
                print("Error: File path required")
                print("Usage: python scripts/helix_dev_assistant.py docs <file_path>")
                sys.exit(1)
            await assistant.generate_docs(sys.argv[2])

        elif command == "architecture":
            if len(sys.argv) < 3:
                print("Error: Question required")
                print('Usage: python scripts/helix_dev_assistant.py architecture "<question>"')
                sys.exit(1)
            await assistant.architecture_consult(sys.argv[2])

        else:
            print(f"Error: Unknown command: {command}")
            print(__doc__)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
