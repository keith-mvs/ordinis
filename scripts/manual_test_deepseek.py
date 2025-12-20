# --------------------------------------------------------------
# test_deepseek.py
# --------------------------------------------------------------
import os

from ordinis.engines.cortex.core.engine import CortexEngine

# ------------------------------------------------------------------
# 1️⃣  Pull the API key from the environment
# ------------------------------------------------------------------
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("NVIDIA_API_KEY not set – export your nvapi‑… token before running.")

# ------------------------------------------------------------------
# 2️⃣  Build the engine – **only** the arguments that exist in the
#     constructor signature are used.
# ------------------------------------------------------------------
engine = CortexEngine(
    nvidia_api_key=api_key,  # <-- correct keyword name
    usd_code_enabled=True,  # enable the USD‑Code (code‑review) model
    # embeddings_enabled=False,     # leave default
    # rag_enabled=False,           # leave default
)

# ------------------------------------------------------------------
# 3️⃣  (Optional) Explicitly set the model – the engine will read
#     CORTEX_NVIDIA_MODEL automatically, but setting it here makes the
#     flow obvious.
# ------------------------------------------------------------------
model_override = os.getenv("CORTEX_NVIDIA_MODEL")
if model_override:
    # The attribute name is `deepseek_model` in the current implementation.
    # (If your version uses a different name, inspect `engine.__dict__`.)
    engine.deepseek_model = model_override

# ------------------------------------------------------------------
# 4️⃣  Run a tiny code‑review request
# ------------------------------------------------------------------
code_to_review = "def foo(x): return x * 2"
out = engine.analyze_code(
    code_to_review,
    analysis_type="review",  # other options: "explain", "chat", etc.
)

# ------------------------------------------------------------------
# 5️⃣  Print the result
# ------------------------------------------------------------------
print("\n=== RESULT ===")
print("model_used :", out.model_used)  # should show the model you set
print("analysis   :", out.content.get("analysis"))  # dict with suggestions, scores, …
