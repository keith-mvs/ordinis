"""
Document all Ordinis models and engines into knowledge base.
Generates markdown documentation for each component without requiring external AI.
"""

import importlib
import inspect
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def document_class(cls: type) -> str:
    """Generate markdown documentation for a class."""
    doc = f"## {cls.__name__}\n\n"

    if cls.__doc__:
        doc += f"{inspect.cleandoc(cls.__doc__)}\n\n"

    doc += "### Methods\n\n"
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("_") or name == "__init__":
            sig = inspect.signature(method)
            doc += f"#### `{name}{sig}`\n\n"
            if method.__doc__:
                doc += f"{inspect.cleandoc(method.__doc__)}\n\n"

    return doc


def document_module(module_path: str) -> str:
    """Generate markdown documentation for a module."""
    try:
        module = importlib.import_module(module_path)

        doc = f"# {module_path}\n\n"

        if module.__doc__:
            doc += f"{inspect.cleandoc(module.__doc__)}\n\n"

        # Document classes
        for name, obj in inspect.getmembers(module, predicate=inspect.isclass):
            if obj.__module__ == module_path:
                doc += document_class(obj)
                doc += "\n---\n\n"

        return doc
    except Exception as e:
        return f"# {module_path}\n\nError loading module: {e}\n"


def main():
    """Document all key modules."""
    kb_path = Path("docs/knowledge-base")
    kb_path.mkdir(parents=True, exist_ok=True)

    modules_to_document = [
        # Core engines
        ("ordinis.ai.core.signal_core", "models/signal_core.md"),
        ("ordinis.ai.helix.engine", "models/helix.md"),
        ("ordinis.engines.learning", "engines/learning_engine.md"),
        # Optimizations
        ("ordinis.optimizations.confidence_filter", "optimizations/confidence_filter.md"),
        ("ordinis.optimizations.confidence_calibrator", "optimizations/confidence_calibrator.md"),
        # Trading
        ("ordinis.trading.executor", "trading/executor.md"),
        ("ordinis.trading.risk_manager", "trading/risk_manager.md"),
        # Data
        ("ordinis.data.market_data", "data/market_data.md"),
        ("ordinis.data.feature_store", "data/feature_store.md"),
    ]

    documented = []
    failed = []

    for module_path, output_file in modules_to_document:
        print(f"Documenting {module_path}...")
        try:
            content = document_module(module_path)

            output_path = kb_path / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            documented.append((module_path, output_path))
            print(f"  ✓ Written to {output_path}")
        except Exception as e:
            failed.append((module_path, str(e)))
            print(f"  ✗ Failed: {e}")

    # Create index
    index_content = "# Ordinis Knowledge Base Index\n\n"
    index_content += "Auto-generated documentation for all Ordinis components.\n\n"
    index_content += "## Models & Engines\n\n"

    for module_path, output_path in documented:
        rel_path = output_path.relative_to(kb_path)
        index_content += f"- [{module_path}]({rel_path})\n"

    if failed:
        index_content += "\n## Failed to Document\n\n"
        for module_path, error in failed:
            index_content += f"- {module_path}: {error}\n"

    index_path = kb_path / "INDEX.md"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_content)

    print(f"\n{'='*80}")
    print("Documentation complete!")
    print(f"  Documented: {len(documented)} modules")
    print(f"  Failed: {len(failed)} modules")
    print(f"  Index: {index_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
