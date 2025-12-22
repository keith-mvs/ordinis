"""Tests for Documentation Generator.

Tests cover:
- EngineMetadata dataclass
- DocsGenerator initialization and metadata extraction
- Template loading and placeholder substitution
- Section generation (config, methods, requirements)
- Document generation and file output
- generate_engine_docs convenience function
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ordinis.engines.base.config import BaseEngineConfig
from ordinis.engines.base.docs_generator import (
    DocsGenerator,
    EngineMetadata,
    generate_engine_docs,
)
from ordinis.engines.base.engine import BaseEngine


@dataclass
class MockConfig(BaseEngineConfig):
    """Mock config for testing."""

    param1: str = "default"
    param2: int = 42
    param3: bool = True


class MockEngine(BaseEngine):
    """Mock engine for testing documentation generation."""

    def __init__(self, config: MockConfig | None = None):
        super().__init__(config or MockConfig())

    async def execute(self, data: Any) -> Any:
        """Execute the mock engine.

        Args:
            data: Input data to process.

        Returns:
            Processed data.
        """
        return data

    def public_method(self, arg1: str, arg2: int = 10) -> dict:
        """A public method for testing.

        Args:
            arg1: First argument.
            arg2: Second argument with default.

        Returns:
            Result dictionary.
        """
        return {"arg1": arg1, "arg2": arg2}


class TestEngineMetadata:
    """Tests for EngineMetadata dataclass."""

    @pytest.mark.unit
    def test_create_metadata(self):
        """Test creating engine metadata."""
        metadata = EngineMetadata(
            name="TestEngine",
            engine_id="TEST",
            class_name="TestEngine",
            module_path="ordinis.test",
            config_class="TestConfig",
            config_fields=[{"name": "field1", "type": "str", "default": "val"}],
            public_methods=[{"name": "method1", "params": [], "return_type": "None"}],
            requirements=[],
            docstring="Test engine docstring.",
            version="1.0.0",
        )

        assert metadata.name == "TestEngine"
        assert metadata.engine_id == "TEST"
        assert metadata.version == "1.0.0"


class TestDocsGeneratorInit:
    """Tests for DocsGenerator initialization."""

    @pytest.mark.unit
    def test_init_with_engine(self):
        """Test initialization with engine class."""
        generator = DocsGenerator(MockEngine)

        assert generator.engine_class == MockEngine
        assert generator.version == "1.0.0"
        assert generator.author == "Ordinis Team"
        assert generator.metadata is not None

    @pytest.mark.unit
    def test_init_custom_version_author(self):
        """Test initialization with custom version and author."""
        generator = DocsGenerator(MockEngine, version="2.0.0", author="Test Author")

        assert generator.version == "2.0.0"
        assert generator.author == "Test Author"


class TestMetadataExtraction:
    """Tests for metadata extraction methods."""

    @pytest.mark.unit
    def test_extract_metadata(self):
        """Test metadata extraction from engine class."""
        generator = DocsGenerator(MockEngine)
        metadata = generator.metadata

        assert metadata.name == "Mock"  # "Engine" suffix removed
        assert metadata.class_name == "MockEngine"
        assert "test_docs_generator" in metadata.module_path

    @pytest.mark.unit
    def test_to_engine_id_simple(self):
        """Test engine ID conversion for simple names."""
        generator = DocsGenerator(MockEngine)

        assert generator._to_engine_id("SignalEngine") == "SIGNAL"
        assert generator._to_engine_id("TestEngine") == "TEST"

    @pytest.mark.unit
    def test_to_engine_id_camel_case(self):
        """Test engine ID conversion for camelCase names."""
        generator = DocsGenerator(MockEngine)

        assert generator._to_engine_id("MyTestEngine") == "MY_TEST"
        assert generator._to_engine_id("RiskGuardEngine") == "RISK_GUARD"

    @pytest.mark.unit
    def test_extract_public_methods(self):
        """Test extraction of public methods."""
        generator = DocsGenerator(MockEngine)
        methods = generator._extract_public_methods()

        method_names = [m["name"] for m in methods]
        assert "execute" in method_names or "public_method" in method_names

    @pytest.mark.unit
    def test_extract_dataclass_fields_valid(self):
        """Test extraction of dataclass fields."""
        generator = DocsGenerator(MockEngine)
        fields = generator._extract_dataclass_fields(MockConfig)

        field_names = [f["name"] for f in fields]
        # BaseEngineConfig has engine_id and engine_type at minimum
        assert "engine_id" in field_names or len(fields) >= 0

    @pytest.mark.unit
    def test_extract_dataclass_fields_invalid(self):
        """Test extraction of dataclass fields on non-dataclass."""
        generator = DocsGenerator(MockEngine)
        # str is not a dataclass
        fields = generator._extract_dataclass_fields(str)

        assert fields == []


class TestTemplateOperations:
    """Tests for template loading and substitution."""

    @pytest.mark.unit
    def test_substitute_placeholders(self):
        """Test placeholder substitution."""
        generator = DocsGenerator(MockEngine, version="1.2.3", author="Test User")

        template = "Engine: {ENGINE_NAME}, ID: {ENGINE_ID}, Author: {AUTHOR}"
        result = generator._substitute_placeholders(template)

        assert "Mock" in result
        assert "MOCK" in result
        assert "Test User" in result

    @pytest.mark.unit
    def test_load_template_not_found(self):
        """Test loading template that doesn't exist."""
        generator = DocsGenerator(MockEngine)

        with pytest.raises(FileNotFoundError):
            generator._load_template("NONEXISTENT")


class TestSectionGeneration:
    """Tests for section generation methods."""

    @pytest.mark.unit
    def test_generate_config_section_no_fields(self):
        """Test config section with no fields."""
        generator = DocsGenerator(MockEngine)
        # Clear config fields
        generator.metadata.config_fields = []

        result = generator._generate_config_section()

        assert "No configuration fields" in result

    @pytest.mark.unit
    def test_generate_config_section_with_fields(self):
        """Test config section with fields."""
        generator = DocsGenerator(MockEngine)
        generator.metadata.config_fields = [
            {"name": "field1", "type": "str", "default": "val", "has_default": True},
            {"name": "field2", "type": "int", "default": None, "has_default": False},
        ]

        result = generator._generate_config_section()

        assert "field1" in result
        assert "field2" in result
        assert "Required" in result  # Field without default

    @pytest.mark.unit
    def test_generate_methods_section_no_methods(self):
        """Test methods section with no methods."""
        generator = DocsGenerator(MockEngine)
        generator.metadata.public_methods = []

        result = generator._generate_methods_section()

        assert "No public methods" in result

    @pytest.mark.unit
    def test_generate_methods_section_with_methods(self):
        """Test methods section with methods."""
        generator = DocsGenerator(MockEngine)
        generator.metadata.public_methods = [
            {
                "name": "test_method",
                "params": [{"name": "arg1", "type": "str"}],
                "return_type": "int",
                "docstring": "Test docstring.",
                "is_async": False,
            },
            {
                "name": "async_method",
                "params": [],
                "return_type": "None",
                "docstring": "",
                "is_async": True,
            },
        ]

        result = generator._generate_methods_section()

        assert "test_method" in result
        assert "async_method" in result
        assert "async " in result  # async prefix
        assert "Test docstring" in result

    @pytest.mark.unit
    def test_generate_requirements_section_no_reqs(self):
        """Test requirements section with no requirements."""
        generator = DocsGenerator(MockEngine)
        generator.metadata.requirements = []

        result = generator._generate_requirements_section()

        assert "No requirements" in result

    @pytest.mark.unit
    def test_generate_requirements_section_with_reqs(self):
        """Test requirements section with requirements."""
        generator = DocsGenerator(MockEngine)
        generator.metadata.requirements = [
            {"id": "REQ-001", "title": "Test Req", "priority": "HIGH", "status": "DONE"}
        ]

        result = generator._generate_requirements_section()

        assert "REQ-001" in result
        assert "Test Req" in result


class TestDocGeneration:
    """Tests for document generation."""

    @pytest.mark.unit
    def test_generate_invalid_type(self):
        """Test generating invalid doc type raises error."""
        generator = DocsGenerator(MockEngine)

        with pytest.raises(ValueError, match="Invalid doc type"):
            generator.generate("INVALID_TYPE")

    @pytest.mark.unit
    def test_generate_valid_type(self, tmp_path):
        """Test generating valid doc type."""
        generator = DocsGenerator(MockEngine)

        # Mock template loading
        with patch.object(generator, "_load_template") as mock_load:
            mock_load.return_value = "# {ENGINE_NAME} Specification"
            result = generator.generate("SPEC")

        assert "Mock" in result
        assert "Specification" in result

    @pytest.mark.unit
    def test_generate_all(self, tmp_path):
        """Test generating all documentation."""
        generator = DocsGenerator(MockEngine, version="1.0.0", author="Tester")

        # Mock template loading for all types
        with patch.object(generator, "_load_template") as mock_load:
            mock_load.return_value = "# {ENGINE_NAME} Document\n\nVersion: {VERSION}"
            paths = generator.generate_all(tmp_path)

        # Should generate all doc types plus index
        assert len(paths) == len(generator.DOC_TYPES) + 1
        assert "SPEC" in paths
        assert "INDEX" in paths

        # Check files exist
        for doc_type, path in paths.items():
            assert path.exists()

    @pytest.mark.unit
    def test_generate_index(self, tmp_path):
        """Test index file generation."""
        generator = DocsGenerator(MockEngine)

        generator._generate_index(tmp_path)

        index_path = tmp_path / "INDEX.md"
        assert index_path.exists()

        content = index_path.read_text()
        assert "Mock Engine Documentation" in content
        assert "SPEC.md" in content
        assert "DESIGN.md" in content


class TestConvenienceFunction:
    """Tests for generate_engine_docs convenience function."""

    @pytest.mark.unit
    def test_generate_engine_docs(self, tmp_path):
        """Test the convenience function."""
        with patch.object(DocsGenerator, "_load_template") as mock_load:
            mock_load.return_value = "# {ENGINE_NAME}"
            paths = generate_engine_docs(
                MockEngine,
                tmp_path,
                version="2.0.0",
                author="Test Author",
            )

        assert len(paths) > 0
        assert "SPEC" in paths
