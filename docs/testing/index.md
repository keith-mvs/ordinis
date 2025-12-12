# 6. Testing Documentation

**Last Updated:** {{ git_revision_date_localized }}

---

## 6.1 Overview

Testing procedures, benchmarks, and quality assurance documentation.

## 6.2 Documents

| Document | Description |
|----------|-------------|
| [Phase 1 Testing Setup](phase1-testing-setup.md) | Initial testing infrastructure |
| [ProofBench Guide](proofbench-guide.md) | Strategy validation framework |
| [User Testing Guide](user-testing-guide.md) | End-user testing procedures |

## 6.3 Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| SignalCore | 45 | 87% |
| RiskGuard | 32 | 92% |
| FlowRoute | 28 | 78% |
| Governance | 85+ | 95% |

## 6.4 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific engine tests
pytest tests/test_engines/ -v
```
