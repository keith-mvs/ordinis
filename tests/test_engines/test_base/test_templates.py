"""Tests for Engine Documentation Templates.

Tests cover:
- TEMPLATES_DIR and TEMPLATE_FILES constants
- get_template function
- list_templates function
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ordinis.engines.base.templates import (
    TEMPLATES_DIR,
    TEMPLATE_FILES,
    get_template,
    list_templates,
)


class TestTemplatesConstants:
    """Tests for module constants."""

    @pytest.mark.unit
    def test_templates_dir_is_path(self):
        """Test TEMPLATES_DIR is a Path."""
        assert isinstance(TEMPLATES_DIR, Path)

    @pytest.mark.unit
    def test_templates_dir_exists(self):
        """Test TEMPLATES_DIR exists."""
        assert TEMPLATES_DIR.exists()

    @pytest.mark.unit
    def test_template_files_not_empty(self):
        """Test TEMPLATE_FILES is not empty."""
        assert len(TEMPLATE_FILES) > 0

    @pytest.mark.unit
    def test_template_files_are_md(self):
        """Test all template files end with .md."""
        for template in TEMPLATE_FILES:
            assert template.endswith(".md")

    @pytest.mark.unit
    def test_standard_templates_exist(self):
        """Test standard templates are in the list."""
        assert "SPEC.md" in TEMPLATE_FILES
        assert "DESIGN.md" in TEMPLATE_FILES
        assert "TESTS.md" in TEMPLATE_FILES


class TestGetTemplate:
    """Tests for get_template function."""

    @pytest.mark.unit
    def test_get_existing_template(self):
        """Test getting an existing template."""
        # Check if any template file exists before testing
        for template_name in ["SPEC", "DESIGN", "TESTS"]:
            template_path = TEMPLATES_DIR / f"{template_name}.md"
            if template_path.exists():
                content = get_template(template_name)
                assert isinstance(content, str)
                assert len(content) > 0
                return

        # Skip if no template files exist
        pytest.skip("No template files found")

    @pytest.mark.unit
    def test_get_nonexistent_template_raises(self):
        """Test getting a nonexistent template raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Template not found"):
            get_template("NONEXISTENT_TEMPLATE_12345")


class TestListTemplates:
    """Tests for list_templates function."""

    @pytest.mark.unit
    def test_list_templates_returns_list(self):
        """Test list_templates returns a list."""
        templates = list_templates()
        assert isinstance(templates, list)

    @pytest.mark.unit
    def test_list_templates_without_extension(self):
        """Test template names don't include .md extension."""
        templates = list_templates()
        for name in templates:
            assert not name.endswith(".md")

    @pytest.mark.unit
    def test_list_templates_matches_template_files(self):
        """Test list matches TEMPLATE_FILES."""
        templates = list_templates()
        expected = [f.replace(".md", "") for f in TEMPLATE_FILES]
        assert templates == expected
