    # Ordinis Task Docs Standard
    **Status:** Active
    **Owner:** Repo Admin
    **Applies to:** All task-specific documentation and supporting assets

    ---

    ## 1) Purpose
    Standardize where task docs live, how they’re named, and how they’re cleaned up so:
    - Docs don’t sprawl into random directories
    - Filenames don’t drift (e.g., `Methodology.md` vs `methods.md` vs `methodology_final.md`)
    - Task scratch notes can exist locally without ever being committed

    ---

    ## 2) Scope
    This standard governs:
    - Task-specific docs (WIP, per task)
    - Task doc structure and filenames
    - Markdown naming conventions for task docs
    - Local-only notes behavior
    - Enforcement via tooling (pre-commit + CI)

    This standard does **not** govern:
    - Code output artifacts (handled by `artifacts/` rules)
    - Curated permanent documentation (separate doc standards apply)

    ---

    ## 3) Canonical Storage Locations
    ### 3.1 Task-specific docs (WIP)
    All task docs MUST live under:



docs/tasks/
    ### 3.2 Permanent curated docs (non-task)
    Permanent docs MUST live under the appropriate area:
    - `docs/architecture/`
    - `docs/reference/`
    - `docs/guides/`
    - `docs/decisions/` (ADRs)

    ### 3.3 Generated outputs
    Generated outputs MUST live under:
    - `artifacts/` (never committed)

    ---

    ## 4) Task Folder Naming Convention
    Each task MUST have exactly one folder:



docs/tasks/t####-/
    ### 4.1 Task ID format
    Task ID MUST match:
    - `t####` (zero-padded, 4 digits)

    Regex:
    - `^t\d{4}$`

    Examples:
    - ✅ `t0001`
    - ✅ `t0042`
    - ❌ `t42`
    - ❌ `T0042`

    ### 4.2 Slug format
    Slug MUST be lowercase kebab-case:

    Regex:
    - `^[a-z0-9]+(?:-[a-z0-9]+)*$`

    Examples:
    - ✅ `event-bus-contract`
    - ✅ `data-provenance`
    - ❌ `event_bus_contract`
    - ❌ `Event-Bus-Contract`

    ### 4.3 Full task folder regex
    Folder name MUST match:
    - `^t\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*$`

    ---

    ## 5) Required Files in Each Task Folder
    Each task folder MUST contain the following files (exact names):



index.md
methodology.md
results.md
next-steps.md
    ### Purpose of each
    - `index.md` — task summary, status, links, key outcomes
    - `methodology.md` — approach, assumptions, procedure, validation
    - `results.md` — outputs, evidence, findings, metrics
    - `next-steps.md` — action list, backlog, open questions

    ---

    ## 6) Optional Local Notes (Allowed, Never Committed)
    ### 6.1 Optional file
    Each task folder MAY contain:



notes.md
    ### 6.2 Non-negotiable rule
    `notes.md` MUST NOT be committed to git under any circumstances.

    ### 6.3 Ignore rule
    Repo MUST include this `.gitignore` entry:

    ```gitignore
    # Local task notes (never commit)
    docs/tasks/**/notes.md

### 6.4 Enforcement rule

Repo MUST fail pre-commit/CI if `notes.md` becomes tracked.

Required check logic:

* If `git ls-files "docs/tasks/**/notes.md"` returns anything → FAIL with remediation message:

  * `git rm --cached <path>` then commit removal

* * *

7) Task Assets Convention

-------------------------

Task-specific assets MUST live under the task folder:
    docs/tasks/t####-<slug>/assets/
      diagrams/
      images/

Rules:

* Task assets MUST NOT be stored in global `docs/_assets/` unless promoted to permanent docs.

* If a diagram is meant to become permanent, it MUST be promoted with the doc (see Section 10).

* * *

8) Markdown Naming Rules (Task Docs)

------------------------------------

### 8.1 Fixed task filenames

The required task docs are fixed filenames (no variants allowed):

* `index.md`

* `methodology.md`

* `results.md`

* `next-steps.md`

* `notes.md` (optional, local-only)

### 8.2 Any additional markdown

Additional markdown files inside a task folder are discouraged.
If needed, they MUST be kebab-case and placed under a subfolder (e.g., `assets/` or `appendix/`).

Kebab-case filename regex:

* `^[a-z0-9]+(?:-[a-z0-9]+)*\.md$`

* * *

9) Tooling Requirements (Standardization + Enforcement)

-------------------------------------------------------

### 9.1 Scaffolder (creation must be standardized)

Repo MUST provide a scaffolder script:

* `tools/new_task_doc.py`

Requirements:

* Inputs: `--id t####` and `--slug kebab-case`

* Creates:

  * folder: `docs/tasks/t####-<slug>/`

  * required files: `index.md`, `methodology.md`, `results.md`, `next-steps.md`

  * assets folders: `assets/diagrams/`, `assets/images/`

* Must not overwrite existing content unless explicitly intended.

### 9.2 Convention checker (hard enforcement)

Repo MUST provide:

* `tools/check_docs_conventions.py`

It MUST enforce:

* Task folders match naming regex

* Required files exist (Section 5)

* Root-level markdown files are limited to required + optional (`notes.md`)

* All markdown filenames are kebab-case where applicable

* `notes.md` is not tracked (Section 6.4)

### 9.3 Local enforcement (pre-commit)

Repo SHOULD include `pre-commit` config to run the checker locally before commits.

### 9.4 CI enforcement (merge gate)

Repo MUST run the checker in CI (GitHub Actions) so noncompliant docs cannot merge.

* * *

10) Task Completion Cleanup Policy

----------------------------------

When a task is completed, it MUST be handled in one of these ways:

### Option A — Promote to permanent docs

* Consolidate the final spec/knowledge into:

  * `docs/architecture/`, `docs/reference/`, or `docs/guides/`

* Create/update an ADR in `docs/decisions/` if decisions were made.

* Keep task folder but mark status in `index.md` as:

  * `Status: complete` (and link to promoted docs)

### Option B — Close and archive

* Mark `index.md` as `Status: complete`

* Keep only canonical content

* Remove any redundant/duplicated drafts

Rules:

* No duplicate “final specs” across multiple locations.

* Git history is the versioning mechanism; filenames must not include `final`, `v2`, `new`, `copy`.

* * *

11) Acceptance Criteria

-----------------------

This standard is considered successfully implemented when:

1. All new task docs are created via `tools/new_task_doc.py` (or equivalent CLI wrapper)

2. CI fails on:

   * invalid task folder names

   * missing required task files

   * tracked `notes.md`

3. `docs/tasks/` contains only folders named `t####-slug/`

4. There are no committed `notes.md` files anywhere in the repository

* * *

12) Quick Reference

-------------------

**Create a task doc folder**

* `python tools/new_task_doc.py --id t0042 --slug event-bus-contract`

**Valid example path**

* `docs/tasks/t0042-event-bus-contract/methodology.md`

**Local notes**

* `docs/tasks/t0042-event-bus-contract/notes.md` (allowed, must remain untracked)



    If you want, I can also give you a second Markdown doc that defines the **permanent docs** standard (architecture/reference/guides) + a diagram standard (Mermaid + assets + naming) so the refinement phase is clean and consistent too.
    ::contentReference[oaicite:0]{index=0}
