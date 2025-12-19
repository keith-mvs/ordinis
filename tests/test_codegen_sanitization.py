from unittest.mock import MagicMock

import pytest

from ordinis.ai.codegen.engine import CodeGenEngine
from ordinis.ai.helix import Helix


@pytest.mark.asyncio
async def test_sanitize_output_removes_think_tags():
    # Setup
    helix_mock = MagicMock(spec=Helix)
    helix_mock.config.default_code_model = "codestral-latest"
    engine = CodeGenEngine(helix=helix_mock)

    # Test cases
    raw_output = """<think>
Here is my reasoning process.
I should import os.
</think>
import os

def hello():
    print("Hello")
"""
    expected = """import os

def hello():
    print("Hello")"""

    # Execute
    # Accessing the private method for direct testing
    sanitized = engine._sanitize_output(raw_output)

    # Verify
    assert sanitized.strip() == expected.strip()


@pytest.mark.asyncio
async def test_sanitize_output_removes_markdown_fences():
    # Setup
    helix_mock = MagicMock(spec=Helix)
    helix_mock.config.default_code_model = "codestral-latest"
    engine = CodeGenEngine(helix=helix_mock)

    # Test cases
    raw_output = """```python
def add(a, b):
    return a + b
```"""
    expected = """def add(a, b):
    return a + b"""

    # Execute
    sanitized = engine._sanitize_output(raw_output)

    # Verify
    assert sanitized.strip() == expected.strip()


@pytest.mark.asyncio
async def test_sanitize_output_handles_mixed_content():
    # Setup
    helix_mock = MagicMock(spec=Helix)
    helix_mock.config.default_code_model = "codestral-latest"
    engine = CodeGenEngine(helix=helix_mock)

    # Test cases
    raw_output = """<think>Thinking...</think>
```python
print("code")
```"""
    expected = """print("code")"""

    # Execute
    sanitized = engine._sanitize_output(raw_output)

    # Verify
    assert sanitized.strip() == expected.strip()
