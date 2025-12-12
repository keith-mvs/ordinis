# Branch Consolidation Plan

**Date:** 2025-12-01
**Goal:** Merge all work to master branch, clean up

---

## Current Branch Status

### Local Branches
1. **main** (current) - Base branch
2. **research/general** - Has RAG system code
3. **user/interface** - Has branch policies, testing docs
4. **features/general-enhancements** - Has additional strategies
5. **help** - Old
6. **new** - Old
7. **add-claude-github-actions-1764364603130** - Old
8. **claude/document-repo-purpose-01UM4K4xVgxAyQPfdXDU2LSv** - Old
9. **claude/resume-session-017EStJ1N8yXYb9QaFsktU9e** - Old

### Stashes (11 total)
- stash@{0}: Simulation setup work
- stash@{1-10}: Various old stashes

---

## What Needs to be Saved

### From research/general
- **RAG system** - Complete implementation (CRITICAL)
  - src/rag/ directory
  - tests/test_rag/
  - RAG-related configs

### From user/interface
- .github/BRANCH_POLICY.md
- docs/BRANCH_WORKFLOW.md
- docs/USER_TESTING_GUIDE.md
- Modified CURRENT_STATUS_AND_NEXT_STEPS.md

### From features/general-enhancements
- Additional strategy implementations
- Enhanced tests

### Current Uncommitted Work on Main
- New documentation files:
  - ACTUAL_PROJECT_STATUS.md
  - FIRST_BUILD_PLAN.md
  - SESSION_CONTEXT.md
  - SIMULATION_SETUP.md
- scripts/ directory with test_data_fetch.py
- .claude/settings.local.json updates

---

## Merge Strategy

### Step 1: Commit Current Work on Main
Commit all new docs and scripts created today.

### Step 2: Merge research/general → main
Get RAG system onto main branch.

### Step 3: Merge user/interface → main
Get branch policies and testing guides.

### Step 4: Merge features/general-enhancements → main
Get additional strategies.

### Step 5: Apply Stashes (if needed)
Check stash@{0} for simulation setup work.

### Step 6: Rename main → master
Switch default branch name.

### Step 7: Delete Old Branches
Remove all branches except master.

### Step 8: Clean Remotes
Push master, remove old remote branches.

---

## Execution Steps

```bash
# 1. Commit current work
git add ACTUAL_PROJECT_STATUS.md FIRST_BUILD_PLAN.md SESSION_CONTEXT.md SIMULATION_SETUP.md scripts/
git commit -m "Add project status docs and data fetch scripts"

# 2. Merge research/general (RAG system)
git merge research/general --no-ff -m "Merge RAG system from research/general"

# 3. Merge user/interface (docs)
git merge user/interface --no-ff -m "Merge branch policies and testing docs"

# 4. Merge features/general-enhancements (strategies)
git merge features/general-enhancements --no-ff -m "Merge additional strategies"

# 5. Rename to master
git branch -m main master
git push origin -u master
git push origin --delete main

# 6. Delete local branches
git branch -D research/general user/interface features/general-enhancements help new
git branch -D add-claude-github-actions-1764364603130
git branch -D claude/document-repo-purpose-01UM4K4xVgxAyQPfdXDU2LSv
git branch -D claude/resume-session-017EStJ1N8yXYb9QaFsktU9e

# 7. Clean stashes (after confirming nothing critical)
git stash clear
```

---

**Status:** Ready to execute
