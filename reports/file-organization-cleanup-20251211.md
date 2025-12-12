# File Organization Cleanup - December 11, 2025

Comprehensive reorganization of project files following clean naming conventions and logical directory structure.

## Summary

- **Files renamed**: 6 example scripts
- **Files moved**: 17 documents
- **Directories organized**: 4 (examples, reports, docs/strategies, docs/planning)
- **Index files created**: 2

## Changes Made

### 1. Examples Directory - Script Renaming

All example scripts renamed to follow clean hyphen-separated convention:

| Old Name | New Name |
|----------|----------|
| `backtest_covered_call.py` | `covered-call-backtest.py` |
| `backtest_married_put.py` | `married-put-backtest.py` |
| `complete_backtest_demo.py` | `complete-backtest-demo.py` |
| `covered_call_analysis.py` | `covered-call-analysis.py` |
| `covered_call_quick_demo.py` | `covered-call-demo.py` |
| `optionscore_demo.py` | `optionscore-demo.py` |

**Created**: `examples/README.md` - Index of all example scripts

### 2. Documentation - Strategy Guides

Moved strategy documentation to logical location:

| Old Location | New Location |
|--------------|--------------|
| `examples/README_MARRIED_PUT.md` | `docs/strategies/married-put-strategy.md` |

### 3. Reports Directory - Session & Status Files

Moved and renamed session files with consistent date format (YYYYMMDD):

| Old Location | New Location |
|--------------|--------------|
| `SESSION_STATUS.md` | `reports/session-status-20251210.md` |
| `SESSION_SUMMARY.md` | `reports/session-summary-20251130.md` |
| `SESSION_CONTEXT.md` | `reports/session-context-20251201.md` |
| `COMPREHENSIVE_SUITE_GUIDE.md` | `reports/comprehensive-suite-guide-20251210.md` |
| `TECHNICAL_INDICATORS_FINAL_REPORT.md` | `reports/technical-indicators-final-report-20251210.md` |
| `TECHNICAL_INDICATORS_INTEGRATION_REPORT.md` | `reports/technical-indicators-integration-report-20251210.md` |
| `ACTUAL_PROJECT_STATUS.md` | `reports/project-status-20251201.md` |
| `PROJECT_STATUS_CARD.md` | `reports/project-status-card-20251203.md` |
| `RELEASE_NOTES_v0.2.0-dev.md` | `reports/release-notes-v020-dev-20251203.md` |

Renamed existing reports for consistency:

| Old Name | New Name |
|----------|----------|
| `COMPREHENSIVE_SUITE_DIAGNOSTIC.md` | `comprehensive-suite-diagnostic-20251210.md` |
| `TEST_REPORT_20251210.md` | `test-report-20251210.md` |

**Created**: `reports/README.md` - Index of all reports and summaries

### 4. Planning Documentation

Created new directory for historical planning documents:

**Directory**: `docs/planning/`

| Old Location | New Location |
|--------------|--------------|
| `BRANCH_MERGE_PLAN.md` | `docs/planning/branch-merge-plan-20251201.md` |
| `CONSOLIDATION_COMPLETE.md` | `docs/planning/consolidation-complete-20251201.md` |
| `FIRST_BUILD_PLAN.md` | `docs/planning/first-build-plan-20251201.md` |
| `PAPER_BROKER_PLAN.md` | `docs/planning/paper-broker-plan-20251201.md` |
| `READY_TO_RUN.md` | `docs/planning/ready-to-run-20251201.md` |
| `SIMULATION_SETUP.md` | `docs/planning/simulation-setup-20251201.md` |

## Final Directory Structure

```
ordinis/
├── CHANGELOG.md                          # Project changelog (kept in root)
├── README.md                             # Main project README (kept in root)
│
├── examples/                             # Example scripts (6 files)
│   ├── README.md                         # Examples index
│   ├── complete-backtest-demo.py
│   ├── covered-call-analysis.py
│   ├── covered-call-backtest.py
│   ├── covered-call-demo.py
│   ├── married-put-backtest.py
│   └── optionscore-demo.py
│
├── reports/                              # Reports and summaries (12 files)
│   ├── README.md                         # Reports index
│   ├── comprehensive-suite-diagnostic-20251210.md
│   ├── comprehensive-suite-guide-20251210.md
│   ├── project-status-20251201.md
│   ├── project-status-card-20251203.md
│   ├── release-notes-v020-dev-20251203.md
│   ├── session-context-20251201.md
│   ├── session-status-20251210.md
│   ├── session-summary-20251130.md
│   ├── technical-indicators-final-report-20251210.md
│   ├── technical-indicators-integration-report-20251210.md
│   └── test-report-20251210.md
│
├── docs/
│   ├── strategies/                       # Strategy documentation
│   │   ├── index.md
│   │   ├── married-put-strategy.md       # NEW
│   │   └── STRATEGY_TEMPLATE.md
│   │
│   └── planning/                         # Historical planning docs (6 files)
│       ├── branch-merge-plan-20251201.md
│       ├── consolidation-complete-20251201.md
│       ├── first-build-plan-20251201.md
│       ├── paper-broker-plan-20251201.md
│       ├── ready-to-run-20251201.md
│       └── simulation-setup-20251201.md
│
└── data/
    └── backtest_results/                 # Generated backtest results
        ├── AAPL_covered_call_*.csv
        ├── AAPL_married_put_*.csv
        └── *_summary_*.txt
```

## Naming Conventions Applied

### Script Files (examples/)
- **Format**: `{strategy}-{type}.py`
- **Example**: `covered-call-backtest.py`
- **Convention**: Lowercase, hyphen-separated, descriptive

### Report Files (reports/)
- **Format**: `{type}-{date}.md` or `{description}-{date}.md`
- **Example**: `session-status-20251210.md`
- **Date Format**: YYYYMMDD
- **Convention**: Lowercase, hyphen-separated, chronologically organized

### Documentation (docs/)
- **Format**: `{topic}-{type}.md`
- **Example**: `married-put-strategy.md`
- **Convention**: Lowercase, hyphen-separated, descriptive

## Benefits

1. **Cleaner Root Directory**
   - Only essential files (README, CHANGELOG) remain in root
   - Easier navigation and reduced clutter

2. **Logical Organization**
   - Reports together in one location
   - Examples clearly separated
   - Documentation properly categorized

3. **Consistent Naming**
   - All files follow hyphen-separated lowercase convention
   - Dates in YYYYMMDD format for easy sorting
   - Descriptive names indicate content

4. **Improved Discoverability**
   - Index files (README.md) in each major directory
   - Clear categorization of file types
   - Chronological organization where appropriate

5. **Professional Structure**
   - Follows industry best practices
   - Ready for version control
   - Easy to maintain and extend

## Index Files Created

### examples/README.md
- Lists all example scripts
- Provides usage instructions
- Documents data requirements

### reports/README.md
- Categorizes all reports by type
- Explains report organization
- Provides overview of contents

## Root Directory Status

**Before**: 17 markdown files (cluttered)
**After**: 2 markdown files (clean)

Remaining files:
- `README.md` - Project documentation
- `CHANGELOG.md` - Version history

## Verification

All files successfully moved and renamed:
- ✅ 6 example scripts renamed
- ✅ 11 reports moved to reports/
- ✅ 6 planning docs moved to docs/planning/
- ✅ 1 strategy guide moved to docs/strategies/
- ✅ 2 index files created
- ✅ 2 existing reports renamed for consistency

Total files organized: **26 files**

## Next Steps

1. Update any scripts that reference old file paths
2. Add these changes to `.gitignore` if needed
3. Commit the reorganization with appropriate message
4. Update any documentation that references old paths

---

**Cleanup Date**: 2025-12-11
**Status**: Complete
**Files Affected**: 26
**New Directories**: 1 (docs/planning/)
