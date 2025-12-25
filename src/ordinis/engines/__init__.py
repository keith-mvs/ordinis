"""Trading engines for the Intelligent Investor system."""

# Lazy imports: avoid importing heavy optional dependencies at package import time.
# Subpackages (e.g., signalcore, cortex) can be imported explicitly when needed.
__all__ = [
    "cortex",
    "flowroute",
    "governance",
    "learning",
    "orchestration",
    "portfolio",
    "proofbench",
    "riskguard",
    "signalcore",
]
