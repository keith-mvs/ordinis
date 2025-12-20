"""
Helix Interactive Work Session Orchestrator.

Orchestrates extended development consultation sessions with Helix,
maintaining context and tracking metrics across multiple interactions.

Usage:
    python helix_work_session.py

Provides:
- Multi-turn conversation with context preservation
- Session metrics tracking (tokens, cost, latency)
- Conversation history export
- File content integration for code review
"""

import asyncio
from datetime import datetime
from pathlib import Path

from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.helix.engine import Helix


class WorkSession:
    """Interactive work session with Helix."""

    def __init__(self, session_name: str | None = None):
        """
        Initialize work session.

        Args:
            session_name: Optional name for this session (for tracking)
        """
        self.helix = Helix(HelixConfig(default_temperature=0.1))
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history: list[dict[str, str]] = []
        self.session_start = datetime.now()

        # Session metrics
        self.total_tokens = 0
        self.total_requests = 0
        self.total_latency_ms = 0.0

        # Initialize with system prompt
        self.conversation_history.append(
            {
                "role": "system",
                "content": (
                    "You are an expert software architect and Python developer "
                    "working on the Ordinis algorithmic trading platform. "
                    "Provide specific, actionable guidance with code examples. "
                    "Focus on production readiness, type safety, and testing."
                ),
            }
        )

        print(f"\n{'='*70}")
        print(f"HELIX WORK SESSION: {self.session_name}")
        print(f"{'='*70}\n")
        print("Commands:")
        print("  /file <path>  - Include file content in next message")
        print("  /metrics      - Show session metrics")
        print("  /export       - Export conversation history")
        print("  /exit, /quit  - End session")
        print()

    async def run(self) -> None:
        """Run interactive work session."""
        file_context: str | None = None

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.split()[0].lower()

                    if command in ["/exit", "/quit"]:
                        await self._end_session()
                        break

                    if command == "/metrics":
                        self._show_metrics()
                        continue

                    if command == "/export":
                        self._export_conversation()
                        continue

                    if command == "/file":
                        file_context = self._load_file_context(user_input)
                        continue

                    print(f"Unknown command: {command}")
                    continue

                # Build message with optional file context
                message = user_input
                if file_context:
                    message = f"{user_input}\n\n{file_context}"
                    file_context = None  # Clear after use

                # Add to history
                self.conversation_history.append({"role": "user", "content": message})

                # Get response from Helix
                response = await self.helix.generate(
                    messages=self.conversation_history,
                    max_tokens=2500,
                )

                # Update metrics
                self.total_requests += 1
                self.total_tokens += response.usage.total_tokens
                self.total_latency_ms += response.latency_ms

                # Add response to history
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                    }
                )

                # Display response
                print(f"\nHelix: {response.content}")
                print(
                    f"\n[Tokens: {response.usage.total_tokens} | "
                    f"Latency: {response.latency_ms:.0f}ms]"
                )

                # Keep history manageable (last 15 exchanges + system)
                if len(self.conversation_history) > 31:
                    self.conversation_history = [
                        self.conversation_history[0]
                    ] + self.conversation_history[-30:]

            except KeyboardInterrupt:
                print("\n")
                await self._end_session()
                break

            except Exception as e:
                print(f"\nError: {e}")

    def _load_file_context(self, command: str) -> str:
        """
        Load file content for inclusion in next message.

        Args:
            command: /file command with path

        Returns:
            Formatted file content or error message
        """
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /file <path>")
            return ""

        file_path = Path(parts[1])
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return ""

        try:
            content = file_path.read_text(encoding="utf-8")
            print(f"Loaded: {file_path} ({len(content)} chars)")
            return f"File: {file_path}\n```python\n{content}\n```"
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""

    def _show_metrics(self) -> None:
        """Display session metrics."""
        duration = datetime.now() - self.session_start
        avg_latency = self.total_latency_ms / max(self.total_requests, 1)
        estimated_cost = self.total_tokens * 0.000001

        print(f"\n{'='*70}")
        print("SESSION METRICS")
        print(f"{'='*70}")
        print(f"Session: {self.session_name}")
        print(f"Duration: {duration}")
        print(f"Requests: {self.total_requests}")
        print(f"Total Tokens: {self.total_tokens:,}")
        print(f"Avg Latency: {avg_latency:.0f}ms")
        print(f"Est. Cost: ${estimated_cost:.4f}")
        print(f"{'='*70}")

    def _export_conversation(self) -> None:
        """Export conversation history to markdown file."""
        export_path = Path(f"{self.session_name}_conversation.md")

        content = f"# Helix Work Session: {self.session_name}\n\n"
        content += f"Date: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += "---\n\n"

        for msg in self.conversation_history:
            if msg["role"] == "system":
                continue  # Skip system prompt in export

            role = "User" if msg["role"] == "user" else "Helix"
            content += f"## {role}\n\n{msg['content']}\n\n"

        export_path.write_text(content, encoding="utf-8")
        print(f"\nConversation exported to: {export_path}")

    async def _end_session(self) -> None:
        """End session and display final metrics."""
        print("\nEnding session...")
        self._show_metrics()

        # Get final Helix metrics
        helix_metrics = self.helix.get_metrics()
        print("\nHelix Engine Metrics:")
        print(f"  Total Requests: {helix_metrics['total_requests']}")
        print(f"  Total Tokens: {helix_metrics['total_tokens']:,}")
        print(f"  Providers: {helix_metrics['providers']}")


async def main() -> None:
    """Entry point."""
    import sys

    session_name = None
    if len(sys.argv) > 1:
        session_name = sys.argv[1]

    session = WorkSession(session_name)
    await session.run()


if __name__ == "__main__":
    asyncio.run(main())
