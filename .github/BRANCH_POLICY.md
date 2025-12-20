# Branch Protection Policy

## Overview

This document defines the branch protection rules and merge requirements for the Intelligent Investor project.

## Protected Branches

### `main` Branch Protection

**Status:** Fully Protected

**Required Checks Before Merge:**
1. All automated tests pass
2. Code coverage meets minimum threshold (50%)
3. Pre-commit hooks pass
4. User authorization obtained
5. Testing checklist completed
6. No merge conflicts
7. Documentation updated

**Direct Push:** ❌ Prohibited
**Force Push:** ❌ Prohibited

### `user/interface` Branch Protection

**Status:** Partially Protected

**Required Checks:**
1. Most tests pass (minor failures documented)
2. No critical errors introduced
3. Code follows project standards

**Direct Push:** ✅ Allowed for authorized developers
**Force Push:** ⚠️ Discouraged

## Merge Requirements

### Merging to `main` (Production)

**Pre-Merge Requirements:**

#### 1. Automated Tests
```powershell
pytest  # All critical tests must pass
pytest --cov  # Coverage ≥50%
```

#### 2. Code Quality Checks
```powershell
pre-commit run --all-files  # All checks pass
ruff check .  # No violations
mypy src/  # No type errors
```

#### 3. Documentation
- [ ] Code changes documented
- [ ] README updated (if applicable)
- [ ] Architecture docs updated (if applicable)

#### 4. User Authorization
- [ ] User explicitly approves merge

#### 5. Testing Checklist
- [ ] Complete `.github/TESTING_CHECKLIST.md`
- [ ] All items verified

### Merging to `user/interface` (Testing)

**Pre-Merge Requirements:**

#### 1. Basic Tests
```powershell
pytest  # >90% pass rate, failures documented
```

#### 2. Code Standards
```powershell
ruff check .
ruff format .
```

#### 3. No Critical Issues
- [ ] No syntax errors
- [ ] No import errors
- [ ] Core functionality intact

## Merge Process

### Standard Merge to `main`

```powershell
# 1. Prepare
git checkout user/interface
git pull origin user/interface
pytest --cov
pre-commit run --all-files

# 2. Get user authorization

# 3. Execute merge
git checkout main
git pull origin main
git merge --no-ff user/interface
pytest --cov
git push origin main
```

## Rollback Policy

### When to Rollback

- Critical bug introduced
- Tests failing in production
- Performance degradation
- User requests rollback

### Rollback Process

```powershell
git log --oneline
git checkout -b rollback/issue-description main
git revert <commit-hash>
pytest --cov
# Get user approval
git checkout main
git merge --no-ff rollback/issue-description
git push origin main
```

## Related Documents

- Branch Workflow: `docs/BRANCH_WORKFLOW.md`
- Testing Checklist: `.github/TESTING_CHECKLIST.md`
- User Testing Guide: `docs/USER_TESTING_GUIDE.md`

---

**Version:** 1.0
**Last Updated:** 2025-11-30
