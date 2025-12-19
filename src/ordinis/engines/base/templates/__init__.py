"""Engine documentation templates.

This package contains standard documentation templates for Ordinis engines.
Templates use placeholder syntax {PLACEHOLDER} that gets replaced during
generation.

Available Templates:
    - SPEC.md: Requirements Specification
    - DESIGN.md: High/Low Level Design
    - INTERFACE.md: Interface Control Document
    - STATES.md: State Machine Specification
    - DATA.md: Data Dictionary
    - TESTS.md: Test Specification
    - RISKS.md: Risk Assessment
    - RUNBOOK.md: Operations Runbook
    - SECURITY.md: Security Specification

Standard Placeholders:
    {ENGINE_NAME} - Human-readable engine name (e.g., "Signal")
    {ENGINE_ID} - Engine identifier in UPPER_SNAKE (e.g., "SIGNAL")
    {VERSION} - Document version (e.g., "1.0.0")
    {DATE} - Generation date (e.g., "2024-01-15")
    {AUTHOR} - Document author
    {CLASS_NAME} - Engine class name (e.g., "SignalEngine")
    {MODULE_PATH} - Module path (e.g., "ordinis.engines.signalcore")
    {CONFIG_CLASS} - Config class name (e.g., "SignalEngineConfig")
    {engine} - Lowercase engine name (e.g., "signal")
"""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent

TEMPLATE_FILES = [
    "SPEC.md",
    "DESIGN.md",
    "INTERFACE.md",
    "STATES.md",
    "DATA.md",
    "TESTS.md",
    "RISKS.md",
    "RUNBOOK.md",
    "SECURITY.md",
]


def get_template(name: str) -> str:
    """Load a template by name.

    Args:
        name: Template name without extension (e.g., "SPEC").

    Returns:
        Template content as string.

    Raises:
        FileNotFoundError: If template doesn't exist.
    """
    template_path = TEMPLATES_DIR / f"{name}.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {name}")
    return template_path.read_text(encoding="utf-8")


def list_templates() -> list[str]:
    """List all available templates.

    Returns:
        List of template names.
    """
    return [f.replace(".md", "") for f in TEMPLATE_FILES]
