"""Test configuration and fixtures."""

from pathlib import Path
import sys

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
