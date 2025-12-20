### 1.1 Probability Spaces and Random Variables

**Formal Framework**:

A probability space is a triple (Ω, ℱ, ℙ) where:
- Ω = sample space (all possible outcomes)
- ℱ = σ-algebra (collection of events)
- ℙ = probability measure

```python
# Conceptual representation
@dataclass
class ProbabilitySpace:
    sample_space: Set       # Ω
    events: SigmaAlgebra    # ℱ
    measure: Probability    # ℙ

# Random variable X: Ω → ℝ
# Maps outcomes to real numbers (e.g., stock returns)
```

**Key Distributions in Finance**:

| Distribution | Use Case | Parameters |
|--------------|----------|------------|
| Normal | Log returns (approx) | μ, σ² |
| Log-normal | Asset prices | μ, σ² |
| Student-t | Fat-tailed returns | ν (degrees of freedom) |
| Poisson | Jump events | λ (intensity) |
| Exponential | Time between events | λ |

---
