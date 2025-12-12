# Documentation Rename Plan

**Generated:** 2025-12-12
**Purpose:** Apply kebab-case naming standard to all documentation files

## Naming Standard

1. **File names:** lowercase + kebab-case only
2. **No dates in filenames** (except logs, release notes, session exports)
3. **No version suffixes** (v2, final, new, copy)
4. **Concise:** 3-6 words maximum

## Files Requiring Rename

### Priority: HIGH - Core Architecture (Phase 1)

Files in active use with multiple cross-references. Critical for user navigation.

| Current Name | Proposed Name | Referenced By | Notes |
|--------------|---------------|---------------|-------|
| `architecture/EXECUTION_PATH.md` | `architecture/execution-path.md` | `architecture/index.md` (line 44) | Core Phase 1 doc |
| `architecture/SIMULATION_ENGINE.md` | `architecture/simulation-engine.md` | `architecture/index.md` (line 45) | Core Phase 1 doc |
| `architecture/SIGNALCORE_SYSTEM.md` | `architecture/signalcore-system.md` | `architecture/index.md` (line 43)<br>`architecture/PHASE1_API_REFERENCE.md` (line 1070)<br>`architecture/PRODUCTION_ARCHITECTURE.md` (line 1434)<br>`architecture/RAG_SYSTEM.md` (line 412)<br>`project/CURRENT_STATUS_AND_NEXT_STEPS.md` (line 10) | Core system spec |
| `architecture/MONITORING.md` | `architecture/monitoring.md` | `architecture/index.md` (line 46) | Core Phase 1 doc |
| `architecture/PRODUCTION_ARCHITECTURE.md` | `architecture/production-architecture.md` | `architecture/index.md` (line 65)<br>`architecture/PHASE1_API_REFERENCE.md` (line 17, 1068) | **CRITICAL** - Main architecture doc |
| `architecture/PHASE1_API_REFERENCE.md` | `architecture/phase1-api-reference.md` | `architecture/index.md` (line 66) | **CRITICAL** - Main API doc |
| `architecture/ARCHITECTURE_REVIEW_RESPONSE.md` | `architecture/architecture-review-response.md` | `architecture/index.md` (line 67)<br>`architecture/PHASE1_API_REFERENCE.md` (line 1069)<br>`architecture/PRODUCTION_ARCHITECTURE.md` (line 1436)<br>`DOCUMENTATION_UPDATE_REPORT_20251212.md` (line 120) | Gap analysis doc |
| `architecture/LAYERED_SYSTEM_ARCHITECTURE.md` | `architecture/layered-system-architecture.md` | `architecture/index.md` (line 68)<br>`architecture/RAG_SYSTEM.md` (line 411)<br>`architecture/SIGNALCORE_SYSTEM.md` (line 1155)<br>`architecture/PRODUCTION_ARCHITECTURE.md` (line 1435)<br>`project/PROJECT_STATUS_REPORT.md` (line 6)<br>`project/CURRENT_STATUS_AND_NEXT_STEPS.md` (line 8) | Master orchestration spec |
| `architecture/RAG_SYSTEM.md` | `architecture/rag-system.md` | `architecture/index.md` (line 52) | AI/ML integration doc |
| `architecture/NVIDIA_INTEGRATION.md` | `architecture/nvidia-integration.md` | `architecture/index.md` (line 51)<br>`architecture/RAG_SYSTEM.md` (line 409)<br>`architecture/SIGNALCORE_SYSTEM.md` (line 1158)<br>`guides/CLI_USAGE.md` (line 492) | NVIDIA NIM integration |

**Cross-Reference Updates Required:** 10 files need link updates after HIGH priority renames

### Priority: MEDIUM - Supporting Architecture

Supporting documentation with moderate cross-references.

| Current Name | Proposed Name | Referenced By | Notes |
|--------------|---------------|---------------|-------|
| `architecture/MODEL_ALTERNATIVES_FRAMEWORK.md` | `architecture/model-alternatives-framework.md` | `architecture/index.md` (line 69)<br>`architecture/SIGNALCORE_SYSTEM.md` (line 1156) | Multi-model strategy |
| `architecture/NVIDIA_BLUEPRINT_INTEGRATION.md` | `architecture/nvidia-blueprint-integration.md` | `architecture/index.md` (line 70)<br>`architecture/RAG_SYSTEM.md` (line 410)<br>`architecture/SIGNALCORE_SYSTEM.md` (line 1157) | PortOpt/Distillery |
| `architecture/TENSORTRADE_ALPACA_DEPLOYMENT.md` | `architecture/tensortrade-alpaca-deployment.md` | `architecture/index.md` (line 71) | Production deployment |
| `architecture/MCP_TOOLS_EVALUATION.md` | `architecture/mcp-tools-evaluation.md` | `architecture/index.md` (line 57) | MCP tooling |
| `architecture/MCP_TOOLS_QUICK_START.md` | `architecture/mcp-tools-quick-start.md` | `architecture/index.md` (line 58) | MCP quickstart |
| `architecture/CLAUDE_CONNECTORS_EVALUATION.md` | `architecture/claude-connectors-evaluation.md` | `architecture/index.md` (line 59) | Claude integration |
| `architecture/CONNECTORS_QUICK_REFERENCE.md` | `architecture/connectors-quick-reference.md` | `architecture/index.md` (line 60) | Quick reference |
| `architecture/SYSTEM_CAPABILITIES_ASSESSMENT.md` | `architecture/system-capabilities-assessment.md` | `architecture/index.md` (line 76) | Feature assessment |
| `architecture/DEVELOPMENT_TODO.md` | `architecture/development-todo.md` | `architecture/index.md` (line 77) | Development backlog |
| `architecture/ADDITIONAL_PLUGINS_ANALYSIS.md` | `architecture/additional-plugins-analysis.md` | `architecture/index.md` (line 78) | Plugin analysis |
| `architecture/CLEAN_ARCHITECTURE_MIGRATION.md` | `architecture/clean-architecture-migration.md` | None found | Migration guide |

**Cross-Reference Updates Required:** 1 file (`architecture/index.md`) needs link updates

### Priority: MEDIUM - Guides & Strategies

User-facing documentation with cross-references.

| Current Name | Proposed Name | Referenced By | Notes |
|--------------|---------------|---------------|-------|
| `guides/CLI_USAGE.md` | `guides/cli-usage.md` | `guides/index.md` (line 15) | CLI guide |
| `guides/DUE_DILIGENCE_SKILL.md` | `guides/due-diligence-skill.md` | `guides/index.md` (line 16) | Skill guide |
| `guides/RECOMMENDED_SKILLS.md` | `guides/recommended-skills.md` | `guides/index.md` (line 17) | Skills reference |
| `guides/UI_IMPROVEMENT_PROPOSAL.md` | `guides/ui-improvement-proposal.md` | `guides/index.md` (line 18) | UI proposal |
| `strategies/STRATEGY_TEMPLATE.md` | `strategies/strategy-template.md` | `strategies/index.md` (line 15) | Template file |

**Cross-Reference Updates Required:** 2 files (`guides/index.md`, `strategies/index.md`) need link updates

### Priority: MEDIUM - Analysis & Testing

Analysis results and testing documentation.

| Current Name | Proposed Name | Referenced By | Notes |
|--------------|---------------|---------------|-------|
| `analysis/ANALYSIS_FRAMEWORK.md` | `analysis/analysis-framework.md` | `analysis/index.md` (line 15) | Analysis methodology |
| `analysis/MARKET_DATA_ENHANCEMENT_PLAN.md` | `analysis/market-data-enhancement-plan.md` | `analysis/index.md` (line 16) | Data pipeline |
| `analysis/BACKTEST_RESULTS_20251207.md` | `analysis/backtest-results-20251207.md` | `analysis/index.md` (line 17) | **Keep date** - Results log |
| `testing/PHASE_1_TESTING_SETUP.md` | `testing/phase1-testing-setup.md` | `testing/index.md` (line 15) | Testing infra |
| `testing/PROOFBENCH_GUIDE.md` | `testing/proofbench-guide.md` | `testing/index.md` (line 16) | Validation framework |
| `testing/USER_TESTING_GUIDE.md` | `testing/user-testing-guide.md` | `testing/index.md` (line 17) | User testing |

**Cross-Reference Updates Required:** 2 files (`analysis/index.md`, `testing/index.md`) need link updates

### Priority: MEDIUM - Project Management

Project scope and status documentation.

| Current Name | Proposed Name | Referenced By | Notes |
|--------------|---------------|---------------|-------|
| `project/PROJECT_SCOPE.md` | `project/project-scope.md` | `project/index.md` (line 15)<br>`index.md` (line 87) | Project objectives |
| `project/PROJECT_STATUS_REPORT.md` | `project/project-status-report.md` | `project/index.md` (line 16)<br>`architecture/SYSTEM_CAPABILITIES_ASSESSMENT.md` (line 6) | Status report |
| `project/CURRENT_STATUS_AND_NEXT_STEPS.md` | `project/current-status-next-steps.md` | `project/index.md` (line 17) | Current status |
| `project/BRANCH_WORKFLOW.md` | `project/branch-workflow.md` | `project/index.md` (line 18) | Git workflow |

**Cross-Reference Updates Required:** 3 files (`project/index.md`, `index.md`, `architecture/SYSTEM_CAPABILITIES_ASSESSMENT.md`) need link updates

### Priority: LOW - Knowledge Base

Knowledge base files (mostly already compliant).

| Current Name | Proposed Name | Referenced By | Notes |
|--------------|---------------|---------------|-------|
| `knowledge-base/00_KB_INDEX.md` | `knowledge-base/00-kb-index.md` | `knowledge-base/index.md` (line 57) | Main KB index |
| `knowledge-base/04_strategy/BACKTESTING_REQUIREMENTS.md` | `knowledge-base/04_strategy/backtesting-requirements.md` | `knowledge-base/index.md` (line 18) | Backtest spec |
| `knowledge-base/04_strategy/DATA_EVALUATION_REQUIREMENTS.md` | `knowledge-base/04_strategy/data-evaluation-requirements.md` | None found | Data eval spec |

**Cross-Reference Updates Required:** 1 file (`knowledge-base/index.md`) needs link updates

### Priority: LOW - Top-Level Documentation

Rarely referenced top-level files.

| Current Name | Proposed Name | Referenced By | Notes |
|--------------|---------------|---------------|-------|
| `DATASET_MANAGEMENT_GUIDE.md` | `dataset-management-guide.md` | None found | Dataset guide |
| `DATASET_QUICK_REFERENCE.md` | `dataset-quick-reference.md` | None found | Dataset reference |
| `CRISIS_PERIOD_COVERAGE.md` | `crisis-period-coverage.md` | None found | Crisis data |
| `EXTENSIVE_BACKTEST_FRAMEWORK.md` | `extensive-backtest-framework.md` | None found | Backtest framework |
| `DOCUMENTATION_UPDATE_REPORT_20251212.md` | `documentation-update-report-20251212.md` | None found | **Keep date** - Report log |

**Cross-Reference Updates Required:** 0 files

### Priority: LOW - Session Exports

Session export files (dates should be preserved).

| Current Name | Proposed Name | Referenced By | Notes |
|--------------|---------------|---------------|-------|
| `session-exports/SESSION_EXPORT_20251209_COVERAGE_75.md` | `session-exports/session-export-20251209-coverage-75.md` | `session-exports/index.md` (line 15) | **Keep date** |
| `session-exports/SESSION_EXPORT_20251208_DOCS_MAINTENANCE.md` | `session-exports/session-export-20251208-docs-maintenance.md` | `session-exports/index.md` (line 16) | **Keep date** |
| `session-exports/SESSION_EXPORT_20251208_MODEL_IMPLEMENTATIONS.md` | `session-exports/session-export-20251208-model-implementations.md` | `session-exports/index.md` (line 17) | **Keep date** |
| `session-exports/SESSION_EXPORT_20251208_KB_EXPANSION.md` | `session-exports/session-export-20251208-kb-expansion.md` | `session-exports/index.md` (line 18) | **Keep date** |
| `session-exports/SESSION_EXPORT_20251208_DOCS_ENHANCEMENT.md` | `session-exports/session-export-20251208-docs-enhancement.md` | `session-exports/index.md` (line 19) | **Keep date** |
| `session-exports/SESSION_EXPORT_20251208.md` | `session-exports/session-export-20251208.md` | `session-exports/index.md` (line 20) | **Keep date** |
| `session-exports/SESSION_EXPORT_20251207.md` | `session-exports/session-export-20251207.md` | `session-exports/index.md` (line 21) | **Keep date** |
| `session-exports/SESSION_SUMMARY_KB_IMPLEMENTATION.md` | `session-exports/session-summary-kb-implementation.md` | `session-exports/index.md` (line 22) | Session summary |
| `session-exports/SESSION_SUMMARY_RAG_PHASES_2-4.md` | `session-exports/session-summary-rag-phases-2-4.md` | `session-exports/index.md` (line 23) | Session summary |
| `session-exports/SESSION_EXPORT_20251203.md` | `session-exports/session-export-20251203.md` | `session-exports/index.md` (line 24) | **Keep date** |
| `session-exports/SESSION_EXPORT_20251130_main.md` | `session-exports/session-export-20251130-main.md` | `session-exports/index.md` (line 25) | **Keep date** |

**Cross-Reference Updates Required:** 1 file (`session-exports/index.md`) needs link updates

## Summary Statistics

- **Total files requiring rename:** 76
- **HIGH priority:** 10 files (core architecture, Phase 1 docs)
- **MEDIUM priority:** 26 files (supporting docs, guides, testing)
- **LOW priority:** 40 files (knowledge base, session exports, misc)

## Files Needing Cross-Reference Updates

After renaming, the following files will need their internal links updated:

### HIGH Priority Cross-References (10 files)
1. `architecture/index.md` - Main architecture navigation hub
2. `architecture/PHASE1_API_REFERENCE.md` - API reference doc
3. `architecture/PRODUCTION_ARCHITECTURE.md` - Production architecture
4. `architecture/RAG_SYSTEM.md` - RAG system doc
5. `architecture/SIGNALCORE_SYSTEM.md` - SignalCore system
6. `project/CURRENT_STATUS_AND_NEXT_STEPS.md` - Current status
7. `project/PROJECT_STATUS_REPORT.md` - Status report
8. `guides/CLI_USAGE.md` - CLI guide
9. `index.md` - Main documentation index
10. `DOCUMENTATION_UPDATE_REPORT_20251212.md` - Update report

### MEDIUM Priority Cross-References (4 files)
11. `guides/index.md` - Guides navigation
12. `strategies/index.md` - Strategies navigation
13. `analysis/index.md` - Analysis navigation
14. `testing/index.md` - Testing navigation
15. `project/index.md` - Project navigation
16. `architecture/SYSTEM_CAPABILITIES_ASSESSMENT.md` - Capabilities doc

### LOW Priority Cross-References (2 files)
17. `knowledge-base/index.md` - KB navigation
18. `session-exports/index.md` - Session exports navigation

## Proposed Execution Plan

### Phase 1: Core Architecture (HIGH Priority)
Rename and update cross-references for the 10 most critical Phase 1 documentation files:

1. `PRODUCTION_ARCHITECTURE.md` → `production-architecture.md`
2. `PHASE1_API_REFERENCE.md` → `phase1-api-reference.md`
3. `ARCHITECTURE_REVIEW_RESPONSE.md` → `architecture-review-response.md`
4. `LAYERED_SYSTEM_ARCHITECTURE.md` → `layered-system-architecture.md`
5. `SIGNALCORE_SYSTEM.md` → `signalcore-system.md`
6. `EXECUTION_PATH.md` → `execution-path.md`
7. `SIMULATION_ENGINE.md` → `simulation-engine.md`
8. `MONITORING.md` → `monitoring.md`
9. `RAG_SYSTEM.md` → `rag-system.md`
10. `NVIDIA_INTEGRATION.md` → `nvidia-integration.md`

**Files to update:** 10 cross-reference files listed above

### Phase 2: Supporting Documentation (MEDIUM Priority)
- Architecture supporting docs (11 files)
- Guides & strategies (5 files)
- Analysis & testing (6 files)
- Project management (4 files)

**Files to update:** 7 cross-reference files

### Phase 3: Batch Rename (LOW Priority)
- Knowledge base (3 files)
- Session exports (11 files)
- Top-level docs (5 files)

**Files to update:** 2 cross-reference files

## Risk Assessment

### LOW RISK
- Knowledge base files (mostly internal references)
- Session exports (historical documentation)
- Top-level docs (few cross-references)

### MEDIUM RISK
- Supporting architecture docs (multiple cross-references)
- Guides and strategies (user-facing)

### HIGH RISK
- Core Phase 1 architecture docs (heavily cross-referenced)
- Main navigation hubs (`index.md` files)

**Mitigation:** Execute Phase 1 carefully with verification of each link update.

## Validation Checklist

After each rename phase:

- [ ] All renamed files exist at new paths
- [ ] No broken internal links in `docs/**`
- [ ] Navigation index files (`*/index.md`) updated
- [ ] Main documentation hub (`docs/index.md`) updated
- [ ] No references to old SCREAMING_SNAKE_CASE names remain
- [ ] Git commit with clear message documenting changes

## Notes

1. **Preserve dates:** Files with dates (backtest results, session exports, reports) keep dates in kebab-case format
2. **Breadcrumb navigation:** Some files may have breadcrumb links that need updating
3. **External references:** Check if any files outside `docs/` reference these documents
4. **Style guide:** Update `docs/style-guide.md` if it has examples using old naming

## Next Steps

1. **Review and approve this plan**
2. **Execute Phase 1** (10 HIGH priority files)
3. **Validate Phase 1** (check all cross-references)
4. **Decide:** Continue with Phases 2-3 or stop after Phase 1
