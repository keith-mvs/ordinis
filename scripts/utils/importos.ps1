import os
from ordinis.engines.cortex.core.engine import CortexEngine

# Pull the key from the conventional env-var name
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("NVIDIA_API_KEY not found in the environment.")

# Build the CortexEngine - NVIDIA proxy endpoint
engine = CortexEngine(
    nvidia_api_key=api_key,
    usd_code_enabled=True,
)

# Quick sanity-check (optional)
if __name__ == "__main__":
    test_code = "def foo(x): return x * 2"
    out = engine.analyze_code(test_code, "review")
    print("model_used:", out.model_used)
    print("analysis:", out.content.get("analysis"))
