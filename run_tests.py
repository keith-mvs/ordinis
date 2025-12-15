"""Simple test runner script to diagnose issues."""

<<<<<<< Updated upstream
<<<<<<< Updated upstream
import subprocess
import sys
=======
import subprocess  # noqa: S404
import sys
from importlib.util import find_spec
>>>>>>> Stashed changes
=======
import subprocess  # noqa: S404
import sys
from importlib.util import find_spec
>>>>>>> Stashed changes

print("=" * 80)
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("=" * 80)

# Check pytest installation
try:
    import pytest

    print(f"✓ pytest installed: {pytest.__version__}")
except ImportError:
    print("✗ pytest NOT installed")
    print("\nInstalling pytest...")
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "pytest-cov"],
=======
=======
>>>>>>> Stashed changes
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
        ],
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
        check=False,
    )
    import pytest

    print(f"✓ pytest installed: {pytest.__version__}")

print("=" * 80)

# Try to import ordinis
<<<<<<< Updated upstream
<<<<<<< Updated upstream
try:
    import ordinis

    print("✓ ordinis package found")
except ImportError as e:
    print(f"✗ ordinis import failed: {e}")
=======
=======
>>>>>>> Stashed changes
if find_spec("ordinis") is not None:
    print("✓ ordinis package found")
else:
    print("✗ ordinis import failed: package not found")
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

print("=" * 80)

# Run pytest
print("\nRunning pytest...")
print("=" * 80)
exit_code = pytest.main(
    [
        "tests/",
        "-v",
        "--tb=short",
        "--maxfail=5",
        "-x",  # Stop on first failure
    ]
)

print("=" * 80)
print(f"Test run completed with exit code: {exit_code}")
sys.exit(exit_code)
