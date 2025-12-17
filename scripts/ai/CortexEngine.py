# CortexEngine usage with NVIDIA Llama 3.3 Nemotron Super 49B v1.5

import os

from ordinis.engines.cortex.core.engine import CortexEngine

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("Set NVIDIA_API_KEY to your nvapi-... token")

engine = CortexEngine(
    nvidia_api_key=api_key,
    usd_code_enabled=True,
)

# Three ways to use the engine:

# 1. Use default model (nvidia/llama-3.3-nemotron-super-49b-v1.5)
code1 = "def foo(x): return x*2"
out1 = engine.analyze_code(code1, "review")
print("Default model:", out1.model_used)

# 2. Override model globally
engine.model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
out2 = engine.analyze_code(code1, "review")
print("Custom model:", out2.model_used)

# 3. Pass generation parameters per call
out3 = engine.analyze_code(code1, "review", max_tokens=4096, temperature=0.7)
print("With kwargs:", out3.model_used)
print("Analysis:", out3.content.get("analysis"))
