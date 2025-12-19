"""Documentation generator for Ordinis engines.

This module provides automated generation of documentation skeletons
from engine code, using the standard templates.

Usage:
    from ordinis.engines.base.docs_generator import DocsGenerator

    # Generate docs for an engine
    generator = DocsGenerator(SignalEngine)
    generator.generate_all("path/to/output")

    # Generate specific doc
    generator.generate_spec("path/to/SPEC.md")
"""

from dataclasses import dataclass, fields
from datetime import UTC, datetime
import inspect
from pathlib import Path
import re
from typing import Any

from ordinis.engines.base.config import BaseEngineConfig
from ordinis.engines.base.engine import BaseEngine


@dataclass
class EngineMetadata:
    """Extracted metadata from an engine class."""

    name: str
    engine_id: str
    class_name: str
    module_path: str
    config_class: str
    config_fields: list[dict[str, Any]]
    public_methods: list[dict[str, Any]]
    requirements: list[dict[str, Any]]
    docstring: str
    version: str = "1.0.0"


class DocsGenerator:
    """Generates documentation from engine classes.

    Inspects engine code and generates skeleton documentation
    using the standard templates.
    """

    TEMPLATES_DIR = Path(__file__).parent / "templates"
    DOC_TYPES = [
        "SPEC",
        "DESIGN",
        "INTERFACE",
        "STATES",
        "DATA",
        "TESTS",
        "RISKS",
        "RUNBOOK",
        "SECURITY",
    ]

    def __init__(
        self,
        engine_class: type[BaseEngine],
        version: str = "1.0.0",
        author: str = "Ordinis Team",
    ) -> None:
        """Initialize the documentation generator.

        Args:
            engine_class: The engine class to document.
            version: Document version.
            author: Document author.
        """
        self.engine_class = engine_class
        self.version = version
        self.author = author
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> EngineMetadata:
        """Extract metadata from the engine class."""
        # Get engine name
        name = getattr(self.engine_class, "__name__", "Unknown")
        engine_id = self._to_engine_id(name)

        # Get config class
        config_class = "BaseEngineConfig"
        config_fields = []

        # Try to get config from type hints
        orig_bases = getattr(self.engine_class, "__orig_bases__", [])
        for base in orig_bases:
            if hasattr(base, "__args__"):
                for arg in base.__args__:
                    if isinstance(arg, type) and issubclass(arg, BaseEngineConfig):
                        config_class = arg.__name__
                        config_fields = self._extract_dataclass_fields(arg)
                        break

        # Get public methods
        public_methods = self._extract_public_methods()

        # Get requirements if instance exists
        requirements: list[dict[str, Any]] = []

        return EngineMetadata(
            name=name.replace("Engine", ""),
            engine_id=engine_id,
            class_name=name,
            module_path=self.engine_class.__module__,
            config_class=config_class,
            config_fields=config_fields,
            public_methods=public_methods,
            requirements=requirements,
            docstring=self.engine_class.__doc__ or "",
            version=self.version,
        )

    def _to_engine_id(self, name: str) -> str:
        """Convert engine name to ID format (e.g., SignalEngine -> SIGNAL)."""
        # Remove 'Engine' suffix and convert to uppercase
        base = name.replace("Engine", "")
        # Convert camelCase to UPPER_SNAKE
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", base)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).upper()

    def _extract_dataclass_fields(self, cls: type) -> list[dict[str, Any]]:
        """Extract fields from a dataclass."""
        result = []
        try:
            for f in fields(cls):
                result.append(
                    {
                        "name": f.name,
                        "type": str(f.type),
                        "default": f.default if f.default is not f.default_factory else None,
                        "has_default": f.default is not f.default_factory,
                    }
                )
        except TypeError:
            pass
        return result

    def _extract_public_methods(self) -> list[dict[str, Any]]:
        """Extract public method signatures from the engine class."""
        methods = []
        for name, method in inspect.getmembers(self.engine_class, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue

            sig = inspect.signature(method)
            params = []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                params.append(
                    {
                        "name": pname,
                        "type": str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else "Any",
                        "default": str(param.default)
                        if param.default != inspect.Parameter.empty
                        else None,
                    }
                )

            return_type = (
                str(sig.return_annotation)
                if sig.return_annotation != inspect.Signature.empty
                else "None"
            )

            methods.append(
                {
                    "name": name,
                    "params": params,
                    "return_type": return_type,
                    "docstring": method.__doc__ or "",
                    "is_async": inspect.iscoroutinefunction(method),
                }
            )

        return methods

    def _load_template(self, doc_type: str) -> str:
        """Load a template file."""
        template_path = self.TEMPLATES_DIR / f"{doc_type}.md"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path.read_text(encoding="utf-8")

    def _substitute_placeholders(self, template: str) -> str:
        """Replace placeholders in template with actual values."""
        replacements = {
            "{ENGINE_NAME}": self.metadata.name,
            "{ENGINE_ID}": self.metadata.engine_id,
            "{VERSION}": self.metadata.version,
            "{DATE}": datetime.now(UTC).strftime("%Y-%m-%d"),
            "{AUTHOR}": self.author,
            "{CLASS_NAME}": self.metadata.class_name,
            "{MODULE_PATH}": self.metadata.module_path,
            "{CONFIG_CLASS}": self.metadata.config_class,
            "{engine}": self.metadata.name.lower(),
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def _generate_config_section(self) -> str:
        """Generate configuration documentation."""
        if not self.metadata.config_fields:
            return "No configuration fields defined."

        lines = ["| Field | Type | Default | Description |"]
        lines.append("|-------|------|---------|-------------|")

        for field in self.metadata.config_fields:
            default = field["default"] if field["has_default"] else "Required"
            lines.append(f"| `{field['name']}` | `{field['type']}` | {default} | |")

        return "\n".join(lines)

    def _generate_methods_section(self) -> str:
        """Generate methods documentation."""
        if not self.metadata.public_methods:
            return "No public methods defined."

        lines = []
        for method in self.metadata.public_methods:
            async_prefix = "async " if method["is_async"] else ""
            params_str = ", ".join(f"{p['name']}: {p['type']}" for p in method["params"])
            lines.append(
                f"#### `{async_prefix}{method['name']}({params_str}) -> {method['return_type']}`"
            )
            if method["docstring"]:
                lines.append(f"\n{method['docstring']}\n")
            else:
                lines.append("\n*No description available.*\n")

        return "\n".join(lines)

    def _generate_requirements_section(self) -> str:
        """Generate requirements table from registry."""
        if not self.metadata.requirements:
            return "| Req ID | Title | Priority | Status |\n|--------|-------|----------|--------|\n| *No requirements registered* | | | |"

        lines = ["| Req ID | Title | Priority | Status |"]
        lines.append("|--------|-------|----------|--------|")

        for req in self.metadata.requirements:
            lines.append(f"| {req['id']} | {req['title']} | {req['priority']} | {req['status']} |")

        return "\n".join(lines)

    def generate(self, doc_type: str) -> str:
        """Generate a specific documentation type.

        Args:
            doc_type: One of SPEC, DESIGN, INTERFACE, STATES, DATA,
                     TESTS, RISKS, RUNBOOK, SECURITY.

        Returns:
            Generated documentation content.
        """
        if doc_type not in self.DOC_TYPES:
            raise ValueError(f"Invalid doc type: {doc_type}. Must be one of {self.DOC_TYPES}")

        template = self._load_template(doc_type)
        content = self._substitute_placeholders(template)

        return content

    def generate_all(self, output_dir: str | Path) -> dict[str, Path]:
        """Generate all documentation types.

        Args:
            output_dir: Directory to write documentation files.

        Returns:
            Dictionary mapping doc type to output path.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = {}
        for doc_type in self.DOC_TYPES:
            content = self.generate(doc_type)
            file_path = output_path / f"{doc_type}.md"
            file_path.write_text(content, encoding="utf-8")
            generated[doc_type] = file_path

        # Generate index
        self._generate_index(output_path)
        generated["INDEX"] = output_path / "INDEX.md"

        return generated

    def _generate_index(self, output_dir: Path) -> None:
        """Generate an index file linking all docs."""
        lines = [
            f"# {self.metadata.name} Engine Documentation",
            "",
            f"> Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"> Version: {self.metadata.version}",
            "",
            "## Documents",
            "",
            "| Document | Purpose |",
            "|----------|---------|",
            "| [SPEC.md](SPEC.md) | Requirements Specification |",
            "| [DESIGN.md](DESIGN.md) | High/Low Level Design |",
            "| [INTERFACE.md](INTERFACE.md) | Interface Control Document |",
            "| [STATES.md](STATES.md) | State Machine Specification |",
            "| [DATA.md](DATA.md) | Data Dictionary |",
            "| [TESTS.md](TESTS.md) | Test Specification |",
            "| [RISKS.md](RISKS.md) | Risk Assessment |",
            "| [RUNBOOK.md](RUNBOOK.md) | Operations Runbook |",
            "| [SECURITY.md](SECURITY.md) | Security Specification |",
            "",
            "## Engine Summary",
            "",
            f"**Class:** `{self.metadata.class_name}`",
            f"**Module:** `{self.metadata.module_path}`",
            f"**Config:** `{self.metadata.config_class}`",
            "",
            "### Description",
            "",
            self.metadata.docstring or "*No description available.*",
            "",
            "### Public Methods",
            "",
            self._generate_methods_section(),
        ]

        index_path = output_dir / "INDEX.md"
        index_path.write_text("\n".join(lines), encoding="utf-8")


def generate_engine_docs(
    engine_class: type[BaseEngine],
    output_dir: str | Path,
    version: str = "1.0.0",
    author: str = "Ordinis Team",
) -> dict[str, Path]:
    """Convenience function to generate all docs for an engine.

    Args:
        engine_class: The engine class to document.
        output_dir: Directory to write documentation.
        version: Document version.
        author: Document author.

    Returns:
        Dictionary mapping doc type to output path.

    Example:
        from ordinis.engines.signalcore.core.engine import SignalEngine

        paths = generate_engine_docs(
            SignalEngine,
            "docs/engines/signalcore",
            version="1.0.0",
            author="Engineering Team"
        )
    """
    generator = DocsGenerator(engine_class, version=version, author=author)
    return generator.generate_all(output_dir)
