Markdown file naming standard
-----------------------------

Pick **one convention and enforce it everywhere**. This will keep links stable and the docs tree predictable.

### File name rules

* **lowercase + kebab-case only**
  ✅ `architecture-overview.md`
  ❌ `Architecture Overview.md`, `architecture_overview.md`

* **no dates in filenames** (unless it’s a log or release notes)
  Put dates in the doc body or front matter.

* **no “final”, “v2”, “new”, “copy”**
  Those belong in git history, not filenames.

* **keep it short but specific (3–6 words max)**
  ✅ `event-bus-contract.md`
  ❌ `detailed-event-bus-contract-and-event-taxonomy-spec.md`

* **one topic per file** (if it needs multiple topics, split it)

### Directory + index conventions

* Each docs folder gets an **index** page:

  * `docs/<area>/index.md` = the landing page for that area

* Prefer stable top-level areas:

  * `docs/architecture/`

  * `docs/guides/`

  * `docs/reference/`

  * `docs/operations/`

  * `docs/decisions/` (ADRs)

  * `docs/rfc/` (proposals, optional)

### Ordering convention (only where needed)

When you need ordered reading (guides), use **two‑digit prefixes**:

* `docs/guides/01-installation.md`

* `docs/guides/02-configuration.md`

* `docs/guides/03-first-backtest.md`

Do **not** use numeric prefixes in reference/architecture docs unless there’s a real sequence.

### Attachments naming

* Put non-markdown assets in a sibling folder:

  * `docs/_assets/`

  * `docs/_assets/diagrams/`

  * `docs/_assets/images/`

* Asset filenames mirror the doc:

  * `event-bus-contract.md`

  * `docs/_assets/diagrams/event-bus-contract.sequence.mmd`

  * `docs/_assets/images/event-bus-contract.sequence.svg`

* * *

Docs refinement standard: visualizations + clean diagrams
---------------------------------------------------------

### Diagram standard (GitHub-friendly)

Use **Mermaid** by default (renders in GitHub + most doc stacks):

* `flowchart` for high-level flows

* `sequenceDiagram` for message order

* `stateDiagram-v2` for OMS/order lifecycle

* `classDiagram` for contracts (ports/adapters)

**Rule:** diagrams live as **source** + optional exported image:

* Source: `docs/_assets/diagrams/<name>.<type>.mmd`

* Export: `docs/_assets/images/<name>.<type>.svg` (SVG preferred)

Example naming:

* `trading-flow.backtest.sequence.mmd`

* `execution.oms.state.mmd`

* `architecture.c4.container.mmd`

### “Clean diagram” quality gates (non-negotiable)

* No crossing lines if you can avoid it

* Every arrow has a verb label (`publishes`, `subscribes`, `calls`, `persists`)

* Include a legend when symbols repeat

* Separate diagrams by concern:

  * C4 Context (system + external services)

  * C4 Container (major modules: SignalCore, RiskGuard, ProofBench, etc.)

  * Component diagrams per engine

  * Sequence diagrams per workflow (backtest, paper)

  * State machine for orders/fills

### C4 model recommendation

Adopt **C4** as the “architecture backbone”:

* `architecture-context.md`

* `architecture-container.md`

* `architecture-components-signalcore.md`

* `architecture-components-proofbench.md`
  …each with a diagram + a short “contracts & invariants” section.

* * *

“Docs cleanup after task completion” (same discipline as code)
--------------------------------------------------------------

* All WIP docs live in:

  * `docs/rfc/` (proposal stage)

* Once accepted:

  * Move to the final home (`docs/architecture/`, `docs/reference/`, etc.)

  * Add an ADR entry in `docs/decisions/` explaining _why_

* Delete stale duplicates. If you need history, git already has it.

* * *

Next step: I’ll set you up with a docs style guide + diagram templates
----------------------------------------------------------------------

I can draft a **`docs/style-guide.md`** that codifies:

* naming rules above

* header structure (Purpose / Scope / Contracts / Examples)

* link conventions

* diagram templates (Mermaid snippets for C4/sequence/state)

* a “doc done” checklist

One question before I lock this in: **Are you building the docs site with MkDocs (Material) or staying GitHub-rendered Markdown only?**
