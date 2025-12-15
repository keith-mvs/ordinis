# QUICK START: Coverage Expansion Guide

## Current Status ✅
- **Coverage**: 54.30% (2,543 passing tests)
- **Goal**: 90%
- **Gap**: 35.70%

## High-ROI Modules for Next Session

### 🔥 TOP PRIORITY: SignalCore Engine
**Current**: 15.44% | **Target**: 60% | **Gain**: +10-15%

Files to target:
- `src/ordinis/engines/signalcore/core/engine.py` (15.44%)
- `src/ordinis/engines/signalcore/core/ensemble.py` (17.97%)
- `src/ordinis/engines/signalcore/core/model.py` (34.74%)

**Why**: Already has test files, just needs coverage fixes
**How**: Copy patterns from `tests/test_ai/test_helix/conftest.py`

---

### 🎯 NEXT: Portfolio Optimizer
**Current**: 49.54% | **Target**: 80% | **Gain**: +4-5%

Files to target:
- `src/ordinis/engines/portfolio/optimizer.py` (49.54%)
- `src/ordinis/engines/portfolio/config.py` (high coverage already)

**Why**: Close to 50%, just needs edge case tests
**How**: Add tests for solver settings, scenario edge cases

---

### 🧠 IMPORTANT: RAG Module
**Current**: 25% | **Target**: 70% | **Gain**: +8-12%

Files to target:
- `src/ordinis/rag/embedders/code_embedder.py` (9.66%)
- `src/ordinis/rag/embedders/text_embedder.py` (11.61%)
- `src/ordinis/rag/pipeline/code_indexer.py` (0%)
- `src/ordinis/rag/pipeline/kb_indexer.py` (0%)

**Why**: Largest uncovered RAG component
**How**: Use existing `test_integration.py` patterns as template
**Note**: Watch for schema mismatches (RetrievalResult needs `text` field)

---

## Quick Test Creation Checklist

✅ **Before Writing Tests**
- [ ] Read the actual implementation (not assumptions)
- [ ] Check `conftest.py` for existing fixtures
- [ ] Look for similar working tests to copy patterns from
- [ ] Verify method signatures exist (not .record_failure() but .call())

✅ **While Writing Tests**
- [ ] Start with one simple test, run it, verify it passes
- [ ] Use Mock/AsyncMock from unittest.mock
- [ ] Use patch decorator for dependency injection
- [ ] Check Pydantic model requirements for schema fields

✅ **After Writing Tests**
- [ ] Run single test file: `pytest tests/test_xxx/test_yyy.py -v`
- [ ] Check for import errors first
- [ ] Run full suite: `pytest tests --ignore=tests/integration -q`
- [ ] Measure coverage: `pytest tests --cov=src/ordinis --cov-report=html`

---

## Key Files to Reference

### Test Patterns That Work ✅
- `tests/test_ai/test_helix/conftest.py` - Good fixture patterns
- `tests/test_ai/test_helix/test_engine.py` - Working Helix tests
- `tests/test_engines/test_base/test_engine.py` - BaseEngine examples
- `tests/test_rag/test_integration.py` - RAG actual usage patterns

### Implementation Files to Check First
- `src/ordinis/runtime/config.py` - Settings structure (Settings.system.environment)
- `src/ordinis/rag/vectordb/schema.py` - RetrievalResult requires `text` field
- `src/ordinis/safety/circuit_breaker.py` - Uses async (await self.call())
- `src/ordinis/ai/helix/engine.py` - Provider routing pattern

---

## Coverage Measurement Commands

```bash
# Quick run (current state)
conda run -n ordinis-env pytest tests --ignore=tests/integration -q

# With coverage report
conda run -n ordinis-env pytest tests --ignore=tests/integration --cov=src/ordinis --cov-report=html

# Check specific module
conda run -n ordinis-env pytest tests/test_signalcore/ -v --cov=src/ordinis/engines/signalcore

# Get coverage percentage
Get-Content htmlcov/index.html | Select-String "pc_cov" | Select-Object -First 1
```

---

## Common Pitfalls & Solutions

### ❌ "AttributeError: object has no attribute 'record_failure'"
**Cause**: Method doesn't exist, need to check actual implementation
**Fix**: Read the implementation and use correct method names

### ❌ "ValidationError: Field required [type=missing, input_value=...]"
**Cause**: Pydantic model missing required field
**Fix**: Check schema.py for required fields (e.g., `text` not `content`)

### ❌ "TypeError: object of type 'Mock' has no len()"
**Cause**: Mock object doesn't have proper return_value
**Fix**: Use `return_value=[]` or specific mock configuration

### ❌ "ImportError: cannot import name 'OpenAI'"
**Cause**: Mock patch path is wrong
**Fix**: Patch where it's imported, not where it's defined

---

## Success Indicators

✅ When tests are working:
- No import errors at collection
- Individual test file runs without errors
- Coverage increases on each module targeted
- Existing tests continue to pass

⚠️ When to pivot:
- More than 50% of new tests fail
- API doesn't match expectations in 3+ tests
- Diminishing returns (effort > benefit)

---

## Estimated Timeline to 90%

| Session | Focus | Start % | Target % | Effort |
|---------|-------|---------|----------|--------|
| This | Infrastructure | 14.8% | 54.30% | ✅ Done |
| Next | SignalCore + Portfolio | 54.30% | 70% | 2-3 hrs |
| +2 | RAG + Safety | 70% | 82% | 2-3 hrs |
| +3 | Final push | 82% | 90%+ | 2-3 hrs |

**Total to 90%: ~6-9 hours of focused work**

---

## Key Success Factor

✨ **Test Quality Over Quantity**

2,543 passing tests > 5,000 failing tests

Focus on:
- Tests that actually run
- Tests that match actual APIs
- Tests that add meaningful coverage
- Tests that maintainers can understand

---

## You Are Here 📍

```
Coverage Progress: ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                   54.30%                                     90%

Next Target: 70% (SignalCore + Portfolio)
Time to Target: 2-3 hours
Confidence: HIGH ✅
```

🚀 **Ready to continue? Check SignalCore test failures first!**
