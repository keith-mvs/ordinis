# Documentation Modernization Plan

**Date:** December 15, 2025
**Status:** ANALYSIS COMPLETE - Ready to Execute

---

## Current Problems Identified

### 1. **Dated Content**
- Docs reference "Phase 1 Complete" but don't reflect recent backtest validation
- Architecture docs mention "future enhancement" for governance/compliance
- Knowledge base marked "(Preview)" — is this still active?
- Planning docs are 2-4 months old
- No mention of Learning Engine, confidence calibration, or performance metrics

### 2. **Wrong/Inaccurate Information**
- Architecture section shows 12 documents but many are analysis/exploration pieces, not final specs
- "Layered System Architecture" is historical reference but not marked clearly
- Model alternatives and "NVIDIA Blueprint Integration" appear exploratory, not implemented
- Some guides reference datasets/skills that may not exist
- Navigation has 12 architecture docs but unclear which are authoritative vs. exploratory

### 3. **Inconsistent Structure**
- Mixing "Current State" (production-architecture) with "Exploratory" docs (model-alternatives)
- Section numbering inconsistent (some docs have 2.x, others are standalone)
- Cross-references are broken or inconsistent
- Archive folder is huge but not clearly separated from active docs
- guides/ has 5+ guides but no clear prioritization

### 4. **No Flow/Poor Navigation**
- Home page mentions 7 sections but doesn't create a learning journey
- "Quick Start" missing entirely — where does a new developer begin?
- No clear progression: What to read first? What's foundational vs. advanced?
- Architecture index has 12 links but no narrative explaining WHY to read each
- Knowledge base and internal docs are separate tiers but relationship is unclear
- Heavy on "what exists" (navigation menus) but light on "what to do" (workflow docs)

---

## Modernization Strategy

### Phase A: Content Consolidation (THIS WEEK)

**Goal:** Merge recent backtest + performance findings into core docs

**New Content:**
1. **Getting Started Guide** (`docs/getting-started/index.md`)
   - 5-minute overview
   - Development quick start
   - System requirements
   - First backtest tutorial

2. **Trading System Fundamentals** (`docs/fundamentals/index.md`)
   - What is Ordinis?
   - Core trading workflow
   - Signal generation → Risk management → Execution
   - Key metrics (win rate, Sharpe, profit factor)

3. **Performance & Results** (`docs/performance/index.md`)
   - Real backtest results (2019-2024)
   - Confidence filtering effectiveness
   - Calibration quality metrics
   - Benchmark comparisons
   - Links to reports in `/reports/`

4. **Architecture (CLEAN)** (`docs/architecture/index.md`)
   - **KEEP:**
     - production-architecture.md (Phase 1 baseline)
     - execution-path.md (trade flow)
     - signalcore-system.md (signals)
     - rag-system.md (knowledge base)
     - nvidia-integration.md (NVIDIA setup)
   - **MOVE TO ARCHIVE:**
     - layered-system-architecture.md (historical)
     - model-alternatives-framework.md (exploratory)
     - architecture-review-response.md (meeting notes)
     - tensortrade-alpaca-deployment.md (alternative approach)
     - simulation-engine.md (see performance/ instead)

5. **User Guides (CURATED)** (`docs/guides/`)
   - CLI usage (keep)
   - Dataset management (keep if current, else move to archive)
   - Due diligence skill (verify it exists)
   - **ADD:** Running backtests
   - **ADD:** Interpreting results
   - Remove outdated skill references

### Phase B: Navigation Restructure (WEEK 2)

**New mkdocs.yml structure:**
```
nav:
  - Home: index.md
  - Getting Started:
    - Quick Start: getting-started/index.md
    - Installation: getting-started/installation.md
    - First Backtest: getting-started/first-backtest.md
    - Development Setup: getting-started/dev-setup.md

  - Fundamentals:
    - Overview: fundamentals/index.md
    - Trading Workflow: fundamentals/workflow.md
    - Signal Generation: fundamentals/signals.md
    - Risk Management: fundamentals/risk.md
    - Execution: fundamentals/execution.md

  - Performance Results:
    - Results Summary: performance/index.md
    - Real Market Backtest (2019-2024): performance/real-market.md
    - Calibration & Confidence: performance/calibration.md
    - Benchmark Comparison: performance/benchmarks.md

  - Architecture:
    - Overview: architecture/index.md
    - Production Architecture: architecture/production-architecture.md
    - Execution Path: architecture/execution-path.md
    - SignalCore System: architecture/signalcore-system.md
    - RAG System: architecture/rag-system.md
    - NVIDIA Integration: architecture/nvidia-integration.md

  - User Guides:
    - CLI Usage: guides/cli-usage.md
    - Dataset Management: guides/dataset-management.md
    - Running Backtests: guides/running-backtests.md
    - Interpreting Results: guides/interpreting-results.md

  - Reference:
    - API Reference: reference/phase1-api-reference.md
    - Connectors: reference/connectors-quick-reference.md

  - Decisions:
    - ADRs: decisions/index.md

  - Internal:
    - Developer Docs: internal/index.md

  - Archive:
    - Historical Docs: archive/index.md
```

### Phase C: Content Quality Pass (WEEK 3)

**Dates:** Update all `**Last Updated:**` with actual edit dates
**Links:** Verify all cross-references work
**Examples:** Add code examples from actual codebase
**Accuracy:** Verify all technical claims against current code
**Voice:** Consistent technical writing tone

---

## Quick Wins (Can do TODAY)

1. **Update mkdocs.yml:**
   - Clean up Architecture section (remove 8+ exploratory docs)
   - Reorganize into Getting Started → Fundamentals → Performance → Reference
   - Move outdated docs to Archive subsection

2. **Create Performance Results Page:**
   - Pull data from `DOCUMENTATION_STATUS.md`
   - Reference actual JSON reports in `/reports/`
   - Add real numbers: 59.7% win rate, 2.56 Sharpe, 231 high-confidence trades

3. **Fix index.md:**
   - Add "Getting Started" button (primary CTA)
   - Link to performance results instead of buried "Planning/roadmap"
   - Clarify status: "Phase 1 Complete + Validation Testing Done"

4. **Create Archive Index:**
   - Move 8+ old docs to `docs/archive/`
   - Create `archive/index.md` explaining "exploratory" vs "implemented"
   - Prevents cluttering main nav

5. **Update guides/index.md:**
   - Add "Quick Start Tutorial: Your First Backtest" (high priority)
   - Add "How to Interpret Backtest Results" (critical for users)
   - Mark which guides are "current" vs "legacy"

---

## Modernization Checklist

### CONTENT
- [ ] Create `getting-started/` section with 4 docs
- [ ] Create `performance/` section with results summaries
- [ ] Move 8+ old architecture docs to `archive/`
- [ ] Create `archive/index.md` explaining what's there
- [ ] Update all cross-links to point to new structure

### NAVIGATION
- [ ] Update mkdocs.yml with new nav structure
- [ ] Remove dead links from index.md
- [ ] Add "Getting Started" as primary call-to-action
- [ ] Reorder sections: Getting Started → Fundamentals → Performance → Reference

### ACCURACY
- [ ] Update all `**Last Updated:**` dates
- [ ] Verify technical claims match current code
- [ ] Add actual code examples (from src/)
- [ ] Link backtest reports in performance/

### USER EXPERIENCE
- [ ] Add 5-minute onboarding flow in Getting Started
- [ ] Add "What should I read?" decision tree
- [ ] Add "How to interpret results?" tutorial
- [ ] Add "Troubleshooting" section to each guide

---

## Success Criteria

✅ **Documentation Flow**
- New user can understand system in 15 minutes
- Clear path: Getting Started → Run Demo → Understand Results → Read Architecture

✅ **Accuracy**
- All content reflects current system (Phase 1 + Backtest Validation)
- Links between docs are correct
- Code examples are current

✅ **Navigation**
- Max 3 clicks to find any topic
- No "where do I start?" confusion
- Clear distinction between exploratory and implemented features

✅ **Maintenance**
- Archive clearly labeled (no confusion with active docs)
- Review schedule: quarterly
- Performance results auto-updated from `/reports/`

---

## Resources Needed

**From Workspace:**
- Actual backtest results: `/reports/phase1_real_market_backtest.json` ✓
- Performance metrics: `/DOCUMENTATION_STATUS.md` ✓
- Session logs: `SESSION_*.md` files ✓
- Architecture docs: Already identified above ✓
- Code examples: `src/` directory ✓

**Writing Effort:**
- Getting Started (4 docs × 1000 words): 4h
- Performance Results (3 docs): 2h
- Navigation/mkdocs update: 1h
- Link verification: 2h
- Quality pass: 3h
- **Total: ~12h**

---

## Next Steps

1. **Approval:** Confirm this plan addresses the "dated, wrong, inconsistent, no flow" issues
2. **Quick Wins:** Do step-by-step updates to mkdocs.yml and index.md today
3. **Content Creation:** Write Getting Started + Performance + Fundamentals docs
4. **Migration:** Archive old docs, update all links
5. **Launch:** Test site build and review flow

**Ready to execute?** 🚀
