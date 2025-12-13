Documentation Migration Plan forOrdinis
=======================================

Current Documentation Audit & Classification
--------------------------------------------

The Ordinis repository’s docs/ directory contains a mix of domainknowledge articles, system design docs, user guides, project notes, and logs.Below is a breakdown of all existing Markdown files by topic and purpose, withnotes on their domain relevance and structural conformity:

* **Knowledge Base Domain Articles (Trading Concepts):** Under docs/knowledge-base/ are ~79 files grouped into numbered folders (01_foundations through 07_references). These cover trading domain knowledge – e.g. market foundations, signal generation methods, risk management, strategy design, execution, options, and reference materials[_[1]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L28-L46)[_[2]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L34-L43). Many are comprehensive, but some are placeholders (e.g. empty events/ subsections in _Signals_, or stubs in _Foundations_)[_[3]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L32-L40)[_[4]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L94-L102). File naming here is mostly already in lowercase/kebab-case (except a few like 00_KB_INDEX.md)[_[5]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L102-L105). These articles currently lack YAML frontmatter metadata (instead, they often begin with a title and a **Purpose/Created** note in the content)[_[6]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L3-L10), so they only partially conform to the intended structure.
* **Architecture & Engine Design Docs (Internal System):** The docs/architecture/ folder contains ~18 technical docs describing Ordinis’s system architecture: e.g. SIMULATION_ENGINE.md (backtest engine), SIGNALCORE_SYSTEM.md (signal engine), RAG_SYSTEM.md (retrieval/LLM integration), EXECUTION_PATH.md (workflow through engines), PRODUCTION_ARCHITECTURE.md (overall system design), etc. These are highly detailed internal design documents (engines, integration, tooling) not currently organized under the knowledge base. They use inconsistent file naming (ScreamingSnakeCase) and lack standardized metadata. For example, **SignalCore**, **RiskGuard**, **Cortex** (orchestrator) and **FlowRoute** (execution engine) are discussed in these docs[_[7]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L9-L13)[_[8]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/04_strategy/nvidia_integration.md#L42-L50), but there is no dedicated “Cortex Engine” or “FlowRoute” markdown file – their details are embedded in broader architecture docs. These files need restructuring into an “Engines” section of the knowledge base.
* **Strategy Examples (Technical Analysis/Strategies):** The docs/strategies/ folder contains a few strategy-specific writeups (bollinger-bands.md, macd.md, married-put-strategy.md) and a STRATEGY_TEMPLATE.md. These cover trading strategies and indicators (Bollinger Bands, MACD, an options strategy) which overlap with content in the knowledge base _Signals_ (technical analysis) and _Options_ domains. In fact, the knowledge base already has entries for MACD and Bollinger Bands[_[9]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L84-L93)[_[10]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L100-L108), and the Options domain is slated to include strategies like married puts[_[11]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L32-L40). Thus, these separate strategy files are redundant. They are not integrated into the knowledge base structure and use inconsistent naming (some lowercase, one uppercase template). They should be merged into the domain knowledge hierarchy or archived if duplicative.
* **User Guides & Skill/Prompt Docs:** In docs/guides/ are a handful of user-facing guides: CLI_USAGE.md (CLI usage instructions), DUE_DILIGENCE_SKILL.md (an LLM skill how-to), RECOMMENDED_SKILLS.md (list of Claude skills), and UI_IMPROVEMENT_PROPOSAL.md (internal UI proposal). These are related to interacting with the system (CLI or Claude AI skills) rather than core trading knowledge. They are not currently in the knowledge base structure. They also violate naming standards (MixedCase and underscores)[_[12]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L156-L165). The skill-related docs are largely superseded by the new knowledge base content – e.g. _Due Diligence Skill_ content has been integrated into the Strategy domain as a due diligence framework[_[13]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L73-L76). These guides need to be relocated to an appropriate section (likely a new “Prompts & Skills” or “User Guide” section) or archived if obsolete.
* **Analysis & Testing Reports:** Under docs/analysis/ are analytical plans/results (e.g. ANALYSIS_FRAMEWORK.md, MARKET_DATA_ENHANCEMENT_PLAN.md, and a dated backtest result log). Similarly, docs/testing/ holds testing-related docs (PHASE_1_TESTING_SETUP.md, PROOFBENCH_GUIDE.md, USER_TESTING_GUIDE.md). These are project-specific or technical how-tos, not general knowledge. For instance, _ProofBench Guide_ is a user manual for the backtesting engine, which should be retained (as engine documentation), whereas the Phase 1 test setup and user testing guides are one-off internal documents. None of these are in the knowledge base structure, and their file names use improper casing[_[14]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L168-L176). These should be split – essential technical guides moved under an Engines or Sources category (e.g. ProofBench guide under Engines), and outdated plans or logs archived.
* **Project Planning & Status:** The docs/planning/ folder contains dated plan memos (all suffixed 20251201.md) for branch merges, build plans, etc., and docs/project/ contains project scope and status docs (e.g. PROJECT_SCOPE.md, PROJECT_STATUS_REPORT.md, CURRENT_STATUS_AND_NEXT_STEPS.md, etc). These are internal project management documents, time-sensitive and now largely outdated (e.g. the status reports correspond to Phase 1 completion). They have no place in the finalized knowledge base and have no references in user docs[_[15]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L114-L121). They should be removed from the main docs (archived outside the knowledge base).
* **Session Export Logs:** docs/session-exports/ holds numerous SESSION_EXPORT_<date>.md files – transcripts or logs of development sessions. These are essentially historical records (with dates in filenames) and are not useful to end-users. They are rarely referenced (few or no cross-links)[_[16]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L257-L265) and can be safely archived or deleted. They do not conform to structural standards beyond being in a separate folder.
* **Miscellaneous Top-Level Docs:** A few standalone .md files exist at the root of docs/ – e.g. DATASET_MANAGEMENT_GUIDE.md, DATASET_QUICK_REFERENCE.md (covering data pipeline details), CRISIS_PERIOD_COVERAGE.md (data coverage analysis), EXTENSIVE_BACKTEST_FRAMEWORK.md (initial backtest framework design), and DOCUMENTATION_UPDATE_REPORT_20251212.md (a log of documentation updates). These have no clear home in the new structure and currently have no inbound references[_[17]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L114-L119), indicating low value to current documentation. They will either be incorporated into the new structure (if still useful) or archived. For example, dataset guides could be moved under a “Sources” section, whereas the crisis data analysis and the outdated backtest framework design can likely be dropped.

**Structural Conformity:** Overall, the currentdocs are fragmented. The new knowledge base architecture (with **Domains**, **Engines**, **Prompts**, **Sources**, **Inbox** categories) will centralize allcontent. Many existing files need renaming to fit the slug style (lowercase,kebab-case) – currently ~66 files are still in CamelCase_or_SCREAMING_SNAKE.md form[_[14]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L168-L176)[_[18]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L178-L186). Additionally, virtually none of the Markdown files have YAMLfrontmatter for metadata (title, tags, etc.), which will be needed for the newsite. The following sections provide a detailed migration plan to realigneverything under docs/knowledge-base/ with the canonicalstructure.
Mapping of Files to New Structure
---------------------------------

The table below maps each existing documentation file to its proposednew location under the docs/knowledge-base/ hierarchy. Files aregrouped by the target top-level category (Domain, Engines, Prompts, Sources, orInbox). File name changes to kebab-case are indicated where applicable:

### **Domain Knowledge(Trading Topics)** – move all existing trading domain .md files into knowledge-base/domains/ subdirectoriescorresponding to their topic:

| **Source Path**                                           | **→ New Path**                                                                                                | **Notes**                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| docs/knowledge-base/01_foundations/**...**                | docs/knowledge-base/domains/foundations/**...**                                                               | _(Rename folder:_ 01_foundations → foundations)*                                                                                                                                                                                                                                                                            |
| • 01_foundations/README.md                                | → domains/foundations/README.md                                                                               | Foundations index (market basics)                                                                                                                                                                                                                                                                                           |
| • 01_foundations/publications/harris_trading_exchanges.md | → domains/foundations/publications/harris_trading_exchanges.md                                                | Keep under foundations domain                                                                                                                                                                                                                                                                                               |
| • _(Advanced math placeholders)_                          | → domains/foundations/advanced_mathematics/_..._                                                              | _(See “Missing Content” below)_                                                                                                                                                                                                                                                                                             |
| docs/knowledge-base/02_signals/**...**                    | docs/knowledge-base/domains/signals/**...**                                                                   | _(Rename folder:_ 02_signals → signals)*                                                                                                                                                                                                                                                                                    |
| • .../technical/ _(all files)_                            | → domains/signals/technical/ _(same filenames)_                                                               | Technical analysis indicators (retain existing files like rsi.md, etc.)                                                                                                                                                                                                                                                     |
| • .../fundamental/README.md _(placeholder)_               | → domains/signals/fundamental/README.md                                                                       | Fundamental analysis index (to expand)                                                                                                                                                                                                                                                                                      |
| • _(New fundamental files)_                               | → domains/signals/fundamental/financial_statements.md<br>→ .../valuation_analysis.md, yield_analysis.md, etc. | Integrate skills content (see below)                                                                                                                                                                                                                                                                                        |
| • .../events/_(placeholders)_                             | → domains/signals/events/_(retain structure)_                                                                 | Corporate/Macro event strategy stubs (to fill in)                                                                                                                                                                                                                                                                           |
| • .../quantitative/_(files)_                              | → domains/signals/quantitative/_(files)_                                                                      | Quant strategies (already organized)                                                                                                                                                                                                                                                                                        |
| docs/knowledge-base/03_risk/**...**                       | docs/knowledge-base/domains/risk/**...**                                                                      | _(Rename folder:_ 03_risk → risk)*                                                                                                                                                                                                                                                                                          |
| • 03_risk/README.md                                       | → domains/risk/README.md                                                                                      | Risk management overview (keep)                                                                                                                                                                                                                                                                                             |
| • 03_risk/advanced_risk_methods.md                        | → domains/risk/advanced_risk_methods.md                                                                       | Retain (advanced topics)                                                                                                                                                                                                                                                                                                    |
| • _(New risk files)_                                      | → domains/risk/interest_rate_risk.md<br>→ domains/risk/credit_risk_analysis.md                                | Add fixed-income risk content[_[19]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L64-L67)                                                                                                                                                       |
| docs/knowledge-base/04_strategy/**...**                   | docs/knowledge-base/domains/strategy/**...**                                                                  | _(Rename folder:_ 04_strategy → strategy)*                                                                                                                                                                                                                                                                                  |
| • 04_strategy/strategy_formulation_framework.md           | → domains/strategy/strategy_formulation_framework.md                                                          | Keep primary strategy framework doc                                                                                                                                                                                                                                                                                         |
| • 04_strategy/backtesting-requirements.md                 | → domains/strategy/backtesting-requirements.md                                                                | (kebab-case rename applied)[_[5]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L102-L105)                                                                                                                                                                   |
| • 04_strategy/data-evaluation-requirements.md             | → domains/strategy/data-evaluation-requirements.md                                                            | (kebab-case applied)[_[5]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L102-L105)                                                                                                                                                                          |
| • 04_strategy/nvidia_integration.md                       | → domains/strategy/nvidia_integration.md                                                                      | NVIDIA AI integration in strategy (retain)                                                                                                                                                                                                                                                                                  |
| • _(New strategy file)_                                   | → domains/strategy/due_diligence_framework.md                                                                 | Add due diligence methodology (from skill)[_[13]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L73-L76)                                                                                                                                          |
| docs/knowledge-base/05_execution/**...**                  | docs/knowledge-base/domains/execution/**...**                                                                 | _(Rename folder:_ 05_execution → execution)*                                                                                                                                                                                                                                                                                |
| • 05_execution/README.md                                  | → domains/execution/README.md                                                                                 | Execution systems overview (keep)                                                                                                                                                                                                                                                                                           |
| • 05_execution/data_pipelines.md                          | → domains/execution/data_pipelines.md                                                                         | Retain (data pipeline patterns)                                                                                                                                                                                                                                                                                             |
| • 05_execution/deployment_patterns.md                     | → domains/execution/deployment_patterns.md                                                                    | Retain (deployment architectures)                                                                                                                                                                                                                                                                                           |
| • 05_execution/governance_engines.md                      | → domains/execution/governance_engines.md                                                                     | Retain (governance mechanisms)                                                                                                                                                                                                                                                                                              |
| • 05_execution/monitoring.md                              | → domains/execution/monitoring.md                                                                             | Retain (system monitoring) – **remove duplicate** architecture/MONITORING.md[_[20]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L2-L5)                                                                                          |
| • _(Possible new doc)_                                    | → domains/execution/flowroute_engine.md                                                                       | _Add execution engine (FlowRoute) overview – see Engines below_                                                                                                                                                                                                                                                             |
| docs/knowledge-base/06_options/**...**                    | docs/knowledge-base/domains/options/**...**                                                                   | _(Rename folder:_ 06_options → options)*                                                                                                                                                                                                                                                                                    |
| • 06_options/README.md                                    | → domains/options/README.md                                                                                   | Options domain overview (retain)                                                                                                                                                                                                                                                                                            |
| • 06_options/publications/hull_options_futures.md         | → domains/options/publications/hull_options_futures.md                                                        | Keep reference (textbook excerpt)                                                                                                                                                                                                                                                                                           |
| • _(New subfolder)_ 06_options/strategy_implementations/  | → domains/options/strategy_implementations/                                                                   | **Add** options strategy implementations section                                                                                                                                                                                                                                                                            |
| • _(New files in above)_                                  | → e.g. .../iron_condors.md, vertical_spreads.md, protective_strategies.md, etc.                               | Integrate 13 options strategies from skills[_[21]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L29-L37)[_[22]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L38-L41) |
| • _(New advanced doc)_                                    | → domains/options/advanced/oas_analysis.md                                                                    | Add advanced topic: option-adjusted spread[_[23]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L66-L69)                                                                                                                                          |
| • _(New greeks doc)_                                      | → domains/options/greeks_library.md                                                                           | Add “Greeks and pricing models” library (from skill)[_[24]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L39-L41)                                                                                                                                |
| docs/knowledge-base/07_references/**...**                 | docs/knowledge-base/domains/references/**...**                                                                | _(Rename folder:_ 07_references → references)*                                                                                                                                                                                                                                                                              |
| • 07_references/README.md                                 | → domains/references/README.md                                                                                | Bibliography index (retain)                                                                                                                                                                                                                                                                                                 |
| • 07_references/index.json                                | → domains/references/index.json                                                                               | (Keep search index JSON)                                                                                                                                                                                                                                                                                                    |
| • _(Potential new refs)_                                  | → domains/references/textbooks/fixed_income.md                                                                | Add fixed-income notes (e.g. bond pricing skill output)[_[25]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L56-L61)                                                                                                              |

_Rationale:_ All domain-related content is consolidated under knowledge-base/domains/, eliminating thenumeric prefixes in folder names. This preserves the logical structure (assummarized in the KB index[_[2]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L34-L43)[_[26]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L44-L47))while aligning with the new schema. Existing knowledge base articles remain intheir domain, just relocated (and renamed if needed). For example, BACKTESTING_REQUIREMENTS.md becomes backtesting-requirements.md in **strategy**[_[5]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L102-L105).Redundant files from docs/strategies/ are merged here rather than moved outright: e.g. the content of strategies/bollinger-bands.md is alreadycovered in domains/signals/technical/overlays/bollinger_bands.md, so the standalone file will be archived. Similarly, strategies/macd.md duplicates the MACD docin _Signals_, and married-put-strategy.md is superseded bythe new protective strategies doc in _Options_. The only strategies filenot yet represented is STRATEGY_TEMPLATE.md, which will bearchived (or moved to **Inbox**) as it’s an internal template.

### **Engine &Architecture Docs** – create a new knowledge-base/engines/ section and moveall internal system documentation here. Each major engine or component gets itsown markdown, consolidating content from the docs/architecture/ folder:

| **Source Path**                                                                         | **→ New Path**                                                                  | **Notes**                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| docs/architecture/PRODUCTION_ARCHITECTURE.md                                            | → knowledge-base/engines/system-architecture.md                                 | **Rename:** production-architecture.md (core system overview)[_[27]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L23-L28). Serves as top-level architecture doc in Engines section.                                                                                                                                        |
| docs/architecture/LAYERED_SYSTEM_ARCHITECTURE.md                                        | → _merge into_ engines/system-architecture.md or engines/cortex-orchestrator.md | This doc overlaps with general architecture (layered design)[_[28]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L26-L29). Integrate key content into the main System Architecture write-up.                                                                                                                                |
| docs/architecture/PHASE1_API_REFERENCE.md                                               | → **Archive/Inbox** (deprecated)                                                | Phase1 API spec – outdated (internal use). Not included in new KB (could live in Inbox for dev reference).                                                                                                                                                                                                                                                                                  |
| docs/architecture/ARCHITECTURE_REVIEW_RESPONSE.md                                       | → **Archive/Inbox**                                                             | One-off Q&A response, not needed in KB (historical).                                                                                                                                                                                                                                                                                                                                        |
| docs/architecture/EXECUTION_PATH.md                                                     | → knowledge-base/engines/execution-path.md                                      | Overview of end-to-end execution flow. Renamed execution-path.md (already done in Phase1)[_[29]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L46-L52). Consider integrating with _FlowRoute_ engine docs or keep as high-level sequence description.                                                 |
| docs/architecture/SIMULATION_ENGINE.md                                                  | → knowledge-base/engines/proofbench.md (Backtest Engine)                        | **Rename:** simulation-engine.md → proofbench-engine.md for clarity. Documents the simulation/backtesting engine (ProofBench).                                                                                                                                                                                                                                                              |
| docs/architecture/SIGNALCORE_SYSTEM.md                                                  | → knowledge-base/engines/signalcore-engine.md                                   | **Rename:** signalcore-system.md → signalcore-engine.md. Documentation for SignalCore engine (signal generation)[_[29]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L46-L52).                                                                                                                        |
| docs/architecture/RAG_SYSTEM.md                                                         | → knowledge-base/engines/rag-engine.md                                          | **Rename:** rag-system.md → rag-engine.md. LLM Retrieval-Augmentation component (embedding DB, etc.).                                                                                                                                                                                                                                                                                       |
| docs/architecture/NVIDIA_INTEGRATION.md                                                 | → knowledge-base/engines/nvidia-integration.md                                  | Already renamed[_[30]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L48-L52). This detailed NVIDIA AI integration doc will reside under Engines (covers AI models usage across engines). Note: overlaps with Strategy’s higher-level NVIDIA doc – ensure cross-references or merge where appropriate. |
| docs/architecture/NVIDIA_BLUEPRINT_INTEGRATION.md                                       | → **Archive** or engines/nvidia-blueprint.md (TBD)                              | If still relevant (an experimental integration), it can be kept as an Engines sub-article. Otherwise, archive as outdated.                                                                                                                                                                                                                                                                  |
| docs/architecture/SYSTEM_CAPABILITIES_ASSESSMENT.md                                     | → **Archive** (Project archive)                                                 | Internal capabilities gap analysis (superseded by project status). No user-facing value.                                                                                                                                                                                                                                                                                                    |
| docs/architecture/MODEL_ALTERNATIVES_FRAMEWORK.md                                       | → **Archive** (or engines/model-alternatives.md)                                | Internal evaluation of alternative ML models. If still useful, place under Engines; otherwise archive.                                                                                                                                                                                                                                                                                      |
| docs/architecture/CLAUDE_CONNECTORS_EVALUATION.md                                       | → **Archive**                                                                   | Evaluation of Claude connectors (historical research). Not needed in formal docs.                                                                                                                                                                                                                                                                                                           |
| docs/architecture/CONNECTORS_QUICK_REFERENCE.md                                         | → knowledge-base/sources/connectors-quick-reference.md                          | _Reclassify under Sources._ (See Sources below)                                                                                                                                                                                                                                                                                                                                             |
| docs/architecture/MCP_TOOLS_EVALUATION.md<br>docs/architecture/MCP_TOOLS_QUICK_START.md | → **Archive** (or engines/mcp-tools.md)                                         | Likely no longer needed (specific tools analysis/setup). Archive unless those tools remain in use (then document under Engines).                                                                                                                                                                                                                                                            |
| docs/architecture/ADDITIONAL_PLUGINS_ANALYSIS.md                                        | → **Archive**                                                                   | Outdated plugin analysis (not part of core system).                                                                                                                                                                                                                                                                                                                                         |
| docs/architecture/TENSORTRADE_ALPACA_DEPLOYMENT.md                                      | → knowledge-base/sources/tensortrade-alpaca.md                                  | _Reclassify under Sources_ (Alpaca broker integration guide).                                                                                                                                                                                                                                                                                                                               |
| docs/architecture/DEVELOPMENT_TODO.md                                                   | → **Archive**                                                                   | Old development task list (superseded by issue tracker).                                                                                                                                                                                                                                                                                                                                    |

Additionally,we will **create new engine docs** in this section for components thatcurrently lack dedicated files:

* **Cortex Orchestrator:** _Add_ knowledge-base/engines/cortex-engine.md – Document the **Cortex** engine (strategy orchestration layer). Cortex’s role and capabilities are mentioned in other docs (e.g. NVIDIA integration, execution path)[_[31]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L9-L18), but it needs a focused description of its design and how it coordinates SignalCore, RiskGuard, etc.
* **RiskGuard Engine:** _Add_ knowledge-base/engines/riskguard-engine.md – A dedicated doc for the **RiskGuard** risk management engine. Currently, RiskGuard is only briefly referenced in broader docs (e.g. part of execution flow or AI integration)[_[7]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L9-L13). This new file will detail its rule-based risk checks, integration with LLM (if any), and usage.
* **FlowRoute Execution Engine:** _Add_ knowledge-base/engines/flowroute-engine.md – Document the **FlowRoute** component (order execution module). FlowRoute appears in architecture diagrams[_[8]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/04_strategy/nvidia_integration.md#L42-L50) but has no write-up yet. This doc will describe how trade orders are routed to brokers (including the paper-trading broker implementation from the planning docs).

_Rationale:_ By moving these into an **Engines** section, we isolate theOrdinis-specific technical documentation. Each engine’s markdown will includearchitecture diagrams or code snippets as needed. Duplicative content betweenold architecture docs will be consolidated. For example, the former _ProductionArchitecture_, _Layered Architecture_, and _Execution Path_ docscollectively describe how engines interact – this will be distilled into **SystemArchitecture** and **Cortex/FlowRoute** docs, with appropriatecross-references. Obsolete or one-off files (reviews, to-dos, etc.) will not bebrought into the knowledge base (marked for archive).

### **Prompt &Skill Documentation** – establish knowledge-base/prompts/ for content relatedto LLM prompts, Claude skills, and user interaction:

| **Source Path**                                    | **→ New Path**                                 | **Notes**                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| docs/guides/CLI_USAGE.md                           | → knowledge-base/prompts/cli-usage.md          | CLI usage guide (kebab-case rename)[_[32]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L60-L64). Could also live in a general “User Guide” section, but we include it under prompts as it’s user-facing.                                                                                                                                                      |
| docs/guides/RECOMMENDED_SKILLS.md                  | → knowledge-base/prompts/recommended-skills.md | Rename from RECOMMENDED_SKILLS.md[_[32]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L60-L64). However, verify relevance: if all recommended skills are now integrated into the KB, this may be repurposed as an index of available AI capabilities or archived.                                                                                              |
| docs/guides/DUE_DILIGENCE_SKILL.md                 | → **Archive** (integrated into KB)             | The content was integrated into _Strategy domain_ as due_diligence_framework.md[_[13]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L73-L76). No standalone doc needed.                                                                                                                                                                             |
| docs/guides/UI_IMPROVEMENT_PROPOSAL.md             | → **Archive**                                  | Internal proposal, not part of KB.                                                                                                                                                                                                                                                                                                                                                                                             |
| docs/knowledge-base/SKILLS_INDEX.md                | → knowledge-base/prompts/skills-index.md       | This cross-reference index of Claude skills to KB sections (already kebab-case named) will be kept for maintainers[_[33]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L11-L19). It can live in the Prompts section as an internal index (status: completed integration). Mark it as an internal reference (or move to Inbox if not for end users). |
| docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md | → **Archive**                                  | Detailed plan for skills integration (execution complete[_[34]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L13-L21)). No longer needed in docs after migration (keep in archive for record).                                                                                                                                                      |
| _Add:_ knowledge-base/prompts/skills-guide.md      | (new file)                                     | Create a **“Claude Skills & Prompts Guide”** for users/developers. This would explain how LLM skills are structured in Ordinis, how to enable/disable or update them, and how the system’s prompt-engine works. It can draw on content from the skills evaluation and integration docs.                                                                                                                                        |

_Rationale:_ The **Prompts** section will house documentation on using andmanaging the AI assistant features of Ordinis. After migration, much of theactual _content_ of skills (due diligence, technical analysis, etc.) livesin the domain articles, but we still need guides for how the skill systemfunctions. The CLI usage doc fits here as it’s about interacting with the AIvia command line. The SKILLS_INDEX.md is a valuable referencemapping skills to knowledge base content and will be retained (possibly flaggedas internal-only if not relevant to end users). The detailed integrationstrategy document is no longer needed once the content is in place (its purposewas planning), so it will be archived. In summary, **Prompts** will containany user-facing instructions for AI/skills, while the domain sections containthe integrated knowledge those skills provided.

### **Sources &Data Documentation** – gather all docs related to datasources, external connectors, and datasets under knowledge-base/sources/:

| **Source Path**                                    | **→ New Path**                                         | **Notes**                                                                                                                                                                                                                                                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| docs/DATASET_MANAGEMENT_GUIDE.md                   | → knowledge-base/sources/dataset-management-guide.md   | Rename to kebab-case. Guide to managing datasets (data import, update processes). Relevant to data pipeline users – keep under Sources.                                                                                                                                                                                    |
| docs/DATASET_QUICK_REFERENCE.md                    | → knowledge-base/sources/dataset-quick-reference.md    | Quick reference of datasets/fields. Retain if useful for developers; otherwise archive if redundant with other docs.                                                                                                                                                                                                       |
| docs/architecture/CONNECTORS_QUICK_REFERENCE.md    | → knowledge-base/sources/connectors-quick-reference.md | Rename from CONNECTORS_QUICK_REFERENCE.md[_[35]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L142-L150). Consolidate with any similar content. This provides an at-a-glance list of external API connectors and integration points. |
| docs/architecture/TENSORTRADE_ALPACA_DEPLOYMENT.md | → knowledge-base/sources/alpaca-deployment.md          | Guide for deploying via Alpaca broker using TensorTrade. Keep under Sources as an external integration tutorial. (Rename to concise kebab-case).                                                                                                                                                                           |
| docs/CRISIS_PERIOD_COVERAGE.md                     | → **Archive** (or sources/crisis-data-coverage.md)     | Likely archive: one-off analysis of data coverage during crisis periods. If considered useful, could be reformatted as a reference case study under Sources. Otherwise drop.                                                                                                                                               |
| docs/EXTENSIVE_BACKTEST_FRAMEWORK.md               | → **Archive**                                          | Outdated design for backtest framework. Superseded by actual implementation (ProofBench) – not needed in new docs.                                                                                                                                                                                                         |
| docs/architecture/CLAUDE_CONNECTORS_EVALUATION.md  | → **Archive** (if not repurposed elsewhere)            | (Listed under Engines above for archive). Not needed here.                                                                                                                                                                                                                                                                 |
| _Add:_ knowledge-base/sources/data-sources.md      | (new overview doc)                                     | Consider adding a short overview of **Ordinis data sources**. This can describe the various data feeds, APIs, and connectors Ordinis uses (market data, news, etc.), tying together the details from dataset guides and connector references.                                                                              |

_Rationale:_ A **Sources** section centralizes all documentation about externaldata and integration points. By moving dataset guides here, we ensure allcontent about managing or connecting data (which is part of system executionbut not trading theory) is separate from the strategy knowledge. The datasetmanagement and quick reference docs are retained to help users manage theirdata – they have no cross-references currently[_[17]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L114-L119) (meaning they weren’t in navigation), but placing them in Sources gives them aproper home. The connectors reference from the architecture folder belongs hereas well, since it lists external integration points. The Alpaca deployment docis an example of a broker integration how-to; it fits the Sources theme(external brokerage as a “source” of execution). If any of these prove outdatedor low-value (e.g. if the Alpaca workflow is no longer supported), they can bearchived later, but initially we migrate them for completeness. The crisisperiod coverage analysis is very niche – unless needed, it will be archivedrather than moved, to avoid clutter.

### **Inbox (Archive& Unsorted)** – move all redundant, outdated, orlow-value .md files to knowledge-base/inbox/ (an unpublishedholding area):

Thefollowing files will **not** be part of the main knowledge base navigation.They will be either deleted or placed in inbox/ for reference, with the intent toremove them in the future. These include one-time reports, completed plans, andduplicate content:

* **Project Planning Docs:** docs/planning/*.md (all the *-20251201.md plans) and docs/project/PROJECT_STATUS_REPORT.md, CURRENT_STATUS_AND_NEXT_STEPS.md, etc. – These were relevant during development Phase 1 but are now completed. For example, _consolidation-complete-20251201.md_ indicates the task was finished Dec 2025, so the plan can be archived. These will be moved to knowledge-base/inbox/planning/ (or simply removed). **Rationale:** They have served their purpose and are not useful to end-users or future maintainers beyond historical curiosity.
* **Session Exports:** all files under docs/session-exports/ – These raw session transcripts will be removed from the docs. They might be zipped and stored outside the docs tree if needed for audit, but otherwise deleted. They have zero integration with the documentation (as noted, they are “historical, few dependencies”[_[16]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L257-L265)).
* **Internal Proposals & Reviews:** UI_IMPROVEMENT_PROPOSAL.md, ARCHITECTURE_REVIEW_RESPONSE.md – Archive these in inbox/ or remove. They represent transient discussions and are not part of a polished knowledge base.
* **To-Do Lists and Status Logs:** DEVELOPMENT_TODO.md, DOCUMENTATION_UPDATE_REPORT_20251212.md – Migrate to inbox/ for record or delete. The documentation update report (dated 2025-12-12) is essentially a changelog of this very migration effort, which doesn’t need to live in the final docs site. The dev to-do is long outdated (tracked in issue tracker now).
* **Redundant Strategy Docs:** docs/strategies/bollinger-bands.md, macd.md, married-put-strategy.md – Since these have been superseded by domain articles (or will be after skill integration), we will remove them from the main docs. We can keep them temporarily in inbox/strategies/ to verify all content was carried over, then delete. The STRATEGY_TEMPLATE.md will also go to inbox/ (it’s not user-facing).
* **Miscellaneous Outdated Content:** CRISIS_PERIOD_COVERAGE.md and EXTENSIVE_BACKTEST_FRAMEWORK.md – These will be archived. Neither is referenced in any index[_[17]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L114-L119) and their information is either obsolete or absorbed elsewhere (the backtest framework design is implemented in code now).

Allarchived files in **Inbox** will be omitted from navigation and search. Wewill retain them short-term for historical reference. Ultimately, afterverifying nothing critical was lost in the migration, these can be deleted ormoved to a separate archive/ outside the docs/knowledge-base tree.
Renaming & Restructuring Recommendations
----------------------------------------

To align with documentation standards and the new structure, we mustuniformly rename files and update cross-references:

* **Apply Kebab-Case Filenames:** All files will use lowercase, hyphen-separated names (no spaces, no CamelCase). This was partially completed for 10 core architecture docs[_[36]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L41-L49). We must finish renaming ~66 remaining files across guides, analysis, testing, project, and low-priority areas[_[14]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L168-L176)[_[18]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L178-L186). For example, DUE_DILIGENCE_SKILL.md → due-diligence-skill.md, PROJECT_STATUS_REPORT.md → project-status-report.md, PHASE_1_TESTING_SETUP.md → phase1-testing-setup.md, etc. Dates in filenames will be kept only where they convey unique context (session logs, dated reports)[_[37]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L74-L77)[_[38]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L126-L134). All others will drop dates. We will also remove version suffixes like “_v2” or “_final” if any (none observed in current set). This renaming will improve consistency and navigation[_[39]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L8-L12). After renaming, **all internal links must be updated** to the new filenames. (A scripted search-and-replace will be used, as automation scripts were prepared for this purpose[_[40]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L54-L63)[_[41]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L76-L85).)
* **Rename Directories for Clarity:** As part of moving into the new /knowledge-base/ subfolders, we will remove numeric prefixes from directory names (as shown in mapping). The ordering of domain sections will be preserved via index pages or weighting in the documentation tool, rather than filesystem naming. For instance, 02_signals/ becomes signals/. The top-level knowledge-base/index.md will be adjusted to point to the new paths (it currently lists the numbered folders[_[2]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L34-L43), which we will update to the new domain folder names). Likewise, any references in text to “02_signals” etc., will be revised.
* **Merge or Split Content as Needed:** Where duplicate or overlapping documents exist, perform content refactoring:

·         Merge **Strategy docs** intodomain articles (as discussed, incorporate any unique info from standalonestrategy files into the Signals/Options domain docs).

·         The two NVIDIA integration docs:consider combining them or clearly differentiating scope. The one underStrategy (now Domains) gives a high-level integration overview, while the oneunder Engines is very detailed with code examples[_[42]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L46-L55)[_[43]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L116-L124). We should cross-link them and perhaps move some deep technical configfrom the domain article into the Engines article to avoid confusion.

·         Ensure the **Execution Path** narrative (workflow through Cortex → SignalCore → RiskGuard → ProofBench →FlowRoute) is integrated either into the System Architecture doc or as its ownEngines doc as mapped. This prevents fragmentation (so a reader can either readthe overall flow in one place or follow links to each engine’s page fordetails).

·         The **Production vs LayeredArchitecture** docs will be consolidated. We likely only need one “SystemArchitecture” page; any unique diagrams or analysis from the other can beincluded there or in an appendix.

* If any content in archived docs is still relevant (e.g. **Crisis data coverage** might have useful findings), consider extracting a summary into an appropriate reference or footnote in a domain doc (for example, a note in Risk management about data reliability in crises). Otherwise, let it go.
* **Navigation Updates:** With files relocated and renamed, all index pages (docs/index.md, project/index.md, guides/index.md, etc.) should be reviewed:

·         The main docs/index.md (if one exists as an entrypoint) will likely now mostly point to knowledge-base/index.md (KB home) andpossibly a “User Guide” if any remains. We will remove references to obsoletesections (project, etc.) from it. For example, if docs/index.md currently links to ProjectScope or similar, those links will be dropped.

·         The knowledge-base/index.md itself will beupdated to reflect the new folder names (domains, etc.) and to add any newsections like Engines or Prompts if we want them listed. Currently it outlinesonly the seven domain sections[_[44]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L17-L25). We may append an “**Engine Documentation**” section to that indexso that internal system docs are discoverable (unless we keep them separatefrom the public KB – but the instruction is to put everything under KB, sopresumably they should be accessible). Another approach is to create a separatetop-level index for Engines and have the main index link to it.

·         Create a small index for knowledge-base/engines/ listing all enginedocs, and similarly for prompts/ and sources/ if those have multiple entries, sothat navigation is clear.

·         Cross-references within documentsmust be revised after moves. E.g. any link pointing to ../architecture/EXECUTION_PATH.md shouldpoint to the new ../engines/execution-path.md. According tothe rename compliance report, about 9 files needed reference updates afterPhase 1 renames[_[45]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L59-L67); after this full reorg, a comprehensive link check will be required.We will use automated tools to grep for old filenames and paths and update themen masse (the provided update_doc_references.py script will behelpful, extended for path changes).
YAMLFrontmatter Metadata
------------------------

We will add or correct YAML frontmatter in each Markdown file to ensureconsistency and enable advanced documentation features (like automaticindexing, tagging, and status labels). Currently, many docs includepseudo-metadata in the content (for example, **Purpose/Created/Status** lines at the top)[_[6]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L3-L10)[_[46]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L2-L9). These will be converted into YAML format at the very top of eachfile, enclosed by --- delimiters.

The following guidelines apply per file (with variations by type ofdoc):

* **Title:** Every file gets a title: field matching its official name. This should be a human-friendly title for navigation (we might use the first # H1 content as the title). E.g. in signalcore-engine.md, add title: "SignalCore Engine Overview". This ensures the sidebar or page title is properly formatted (instead of relying on the filename or H1).
* **Category/Domain:** Use a frontmatter field to classify the doc’s domain. For all files under domains/, we add domain: <name> (e.g. domain: signals). For engine docs, we can use domain: engines (or a more specific key like engine: SignalCore). Since the new structure explicitly groups them, this metadata serves double duty for search filters. For example, in a technical article like **Regime Classification** under Signals, frontmatter might include domain: signals and tags like tags: [machine learning, regime detection]. In an engine doc like **ProofBench Engine**, we might put domain: engines and also an engine: ProofBench identifier.
* **Status:** Many docs are in-progress or planned. We will introduce a status: field (with values like draft, placeholder, complete, archived). For instance, all placeholder README files (empty stubs) can be marked status: draft or status: outline. In the YAML, e.g. status: draft for an incomplete page. Completed, polished pages can be status: complete (or omitted if we consider complete as default). The _Skills Index_ or planning-related docs (if kept) might be marked status: internal to denote not for broad consumption.
* **Version/Date:** Where applicable, include a created: and/or last_updated: timestamp in ISO or a version number. Some docs have a **Created** date listed in text[_[47]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L5-L8) – we will move that into YAML as created: 2024-12-12, etc. For living documents that track changes (like the architecture compliance report or status reports), also add version: 1.0.0 or similar if relevant. However, for static knowledge articles, version can often be omitted or tied to software release version if appropriate. We will add a version for formal reports (e.g. the compliance report might keep version: 1.0.0 as it had)[_[48]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L320-L323).
* **Authors (optional):** If we need to document authorship or approvers, a field can be added (some internal reports have an **Author: Technical Writer Agent** line[_[48]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L320-L323) – that could become author: "Technical Writer Bot" in YAML for record, though for most KB articles the author might not be needed to display).
* **Tags:** Add tags: [] to key pages to facilitate search. For example, the technical analysis pages could have tags like tags: [momentum, indicator, technical analysis], risk pages might include tags: [risk management, VaR], engine pages might have tags: [architecture, engine, AI], etc. We will generate a set of relevant keywords for each article to populate this field. This was not present before, so it’s entirely new metadata to improve the knowledge base usability.
* **Summary/Description:** Optionally, include a brief summary: in frontmatter for each page to show hover-text or search result snippets. We can use the existing first paragraph or purpose lines for this. For instance, the **Knowledge Base index** first line _"Foundational trading knowledge for automated strategy development."_ can become summary: "Foundational trading knowledge for automated strategy development." in YAML.

**Files requiring YAML addition:** Virtually allfiles in the new structure will get frontmatter inserted. Particularly: - All **domainarticles** (currently none have YAML; we will add to all ~80 of them). - All **enginedocs** (new and moved ones) will get a standardized frontmatter (title,engine name, etc.). - Main indexes like knowledge-base/index.md will get a title: "Ordinis Knowledge Base" and perhaps be marked with toc: false if we don’t want a TOC on that page. - The **session export** logsand archived docs won’t need frontmatter since they are not published (or if wedo keep them accessible in some way, we might label them status: archived clearly in YAML). - Theinternal reports (if any moved to Inbox) can have a note in YAML like status: archived and hide: true (depending on site generatorfeatures) to keep them out of nav.

For example, here’s how we will adjust one file’s header:

<!-- Before(excerpt from SKILLS_INDEX.md): -->

# Skills IntegrationIndex

**Purpose**: Mastercross-reference between Claude skills and Knowledge Base sections **Last Updated**:2025-12-12 **Version**: 1.2

---

<!-- After (withYAML frontmatter): -->

---

title: "SkillsIntegration Index"
domain: prompts
status: internal
last_updated:"2025-12-12"
version:"1.2"
tags: [Claude,skills, integration]

---

# Skills IntegrationIndex

Mastercross-reference between Claude skills and Knowledge Base sections.

All metadata like **Created**, **Last Updated**, **Purpose** will thus be captured in YAML (and removed from the body text). This makes theinformation more machine-readable and consistent. We will ensure the **title** in YAML matches the H1 heading for each document (to avoid confusion – or wemay remove the redundant H1 from body if the site generator uses the YAMLtitle).

Finally, we’ll run the docs site generator (e.g. MkDocs or Sphinx asconfigured) in a test mode to verify that: - All renamed paths resolve (nobroken links). - Frontmatter is parsed correctly (no rendering of raw YAML onpages). - The new structure (domains, engines, prompts, sources) appearscorrectly in the navigation and all placeholder pages are accounted for.
Identified Redundancies & Content Gaps
--------------------------------------

In the process of mapping, we flagged several items for removal orarchival (see **Inbox/Archive** above). Here we summarize _what_ isbeing discarded and _why_:

* **Outdated Planning/Status Docs:** These served internal project management functions and do not contribute to the knowledge base. For example, the Branch Merge Plan and Status Reports from 2025 were relevant to development coordination, but a reader of the docs now does not need to see these old plans. We retain them off-site for historical purposes. _Impact:_ No loss to knowledge base content, reduces noise.
* **Session Logs:** Purely raw transcripts. They were never meant to be polished documentation. Removing them greatly declutters the docs (11+ files). _Impact:_ None on knowledge base quality; these were not referenced[_[49]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L259-L265).
* **Duplicate Strategy Content:** The standalone strategy files (Bollinger, MACD, etc.) are duplicates of content now in the Signals domain. We verified that the knowledge base technical analysis section covers these indicators comprehensively[_[50]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L85-L93)[_[51]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L101-L108). Thus, the separate files can be dropped to avoid confusion. _Impact:_ Users will consult the unified Signals technical docs instead.
* **Old Backtest Framework Doc:** This described the intended design of a backtester, but Ordinis now has a working ProofBench engine. The important parts of that design either made it into the ProofBench implementation (documented in the ProofBench guide) or are no longer relevant. We choose not to carry forward a design spec that doesn’t match the final state. _Impact:_ None on users; ProofBench’s actual usage is documented in its guide.
* **Internal Evaluations (Claude connectors, MCP tools, etc.):** These documents were essentially research notes during development – e.g. evaluating different connector approaches, plugins, or tools. The outcomes of those evaluations (which approach was chosen) are reflected in the final architecture, so documenting the decision process is not necessary in the KB. Removing them streamlines the Engines section to only actual implemented components. _Impact:_ Possible loss of some rationale context, but if needed, a short note can be added to relevant engine docs (“Nvidia connector was chosen over alternative X as evaluated in dev”). Otherwise, safe to drop.

The **Content Gaps** that need filling (new docs to scaffold) werealso identified:

* **Foundations Domain:** We noted that _Advanced Mathematics_ content is missing. The KB plan calls for ~10 new files on topics like game theory, information theory, control theory, etc.[_[52]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L136-L145)[_[53]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L150-L158). These need to be written. Additionally, _Market Microstructure_ (exchange mechanics, order types) is touched on in the Foundations overview but lacks dedicated deep-dive articles. We should add a few files in Foundations covering Microstructure in detail (order book dynamics, etc., as listed in the index’s key concepts[_[54]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L56-L65)). These were not present in the docs inventory – we’ll create them to complete the section.
* **Signals Domain:** The **Events** subfolders (corporate_actions, earnings_events, macro_events) are currently just placeholders[_[3]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L32-L40). We must create content for each type of event-driven strategy (covering typical approaches to trading earnings releases, merger arbitrage, economic announcements, etc.). The **Fundamental** subfolder needs expansion – per the Claude skills integration, we expect new pages like _financial_statements.md_, _valuation_analysis.md_, _yield_analysis.md_, _fixed_income_analysis.md_ to appear[_[55]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L55-L62). The skills integration index confirms that _benchmarking_ and _financial-analysis_ skills were integrated into a fundamental/skill_integration.md (perhaps as a combined page)[_[56]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L51-L59), and _bond-benchmarking_ into _fixed_income_analysis.md_, etc. We need to ensure those files exist and, if not, scaffold them based on the skill content. Also, the **Volume** section is just a placeholder README currently[_[57]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L130-L138) – we should add at least one page on volume indicators/metrics beyond what’s briefly listed (OBV, VWAP, etc.). The **Sentiment** section (news_sentiment, social_media) similarly has empty stubs[_[58]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L61-L64); content about sentiment analysis techniques should be written. These gaps were anticipated in the expansion plan, so we must now fill them.
* **Risk Domain:** The skill integration added _interest_rate_risk.md_ and _credit_risk_analysis.md_ (likely combined as fixed_income_risk.md per Skills Index)[_[59]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L62-L69). We should verify if we have a general _Risk Management 101_ page (covering basic concepts like VaR, drawdown, etc.) – currently the advanced methods page exists, but a basic intro might be missing. If missing, create a “Risk Management Basics” page to complement _Advanced Risk Methods_.
* **Strategy Domain:** Thanks to skill integration, _due_diligence_framework.md_ is added[_[13]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L73-L76). Aside from that, the Strategy section looks fairly complete with formulation, backtesting requirements, etc. One possible gap: maybe a page on **Portfolio Management** or **Position Sizing** – though some of that is in Risk domain, if not covered it could be worth adding under Strategy (e.g. portfolio optimization techniques beyond mean-variance, unless covered under Signals quantitative). We’ll check overlap; if needed, add a stub for _Portfolio Optimization_ or _Strategy Tuning_.
* **Execution Domain:** The documentation here is mostly about system implementation (which we are also covering in Engines). One thing to scaffold is the **FlowRoute** engine doc as mentioned – essentially, _Execution (FlowRoute) engine_ documentation. Additionally, if the planning docs like “paper-broker-plan” were executed, perhaps a short page about setting up a paper trading broker (FlowRoute in paper mode) could be added under Execution or Sources. We should confirm if _FlowRoute_ is internal name only or if there’s a user-facing concept to document (it might just be internal, so an engine doc suffices). We will also ensure the _governance_engines.md_ content (which is quite specific) covers everything needed; if not, identify if any “compliance” or “governance” engine in the system lacks documentation.
* **Options Domain:** This saw a huge expansion from integrated skills: an entire strategy implementations subsection (iron condors, spreads, straddles, etc.) was created. We need to ensure all those files are present. The Skills Index suggests some were consolidated (e.g. multiple skills into one file like _volatility_strategies.md_)[_[60]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L31-L39). We should verify if any option strategy is missing (the index covered 13 common ones, which is pretty thorough). Moreover, the plan mentioned adding _pricing_models.md_ (Black-Scholes, etc.) in addition to the _greeks_library.md_. The current integration indicates options-strategies skill content went into _greeks_library.md_ (maybe it includes pricing models too). If not, we should add a dedicated _options/pricing_models.md_ for theoretical pricing frameworks. Also, an _advanced_ folder exists now with OAS analysis; we should see if any other advanced option topics (like volatility surfaces or exotic options) are planned – none explicitly, but we can leave room.
* **Engines:** As noted, _Cortex_, _RiskGuard_, _FlowRoute_ need new docs. We will scaffold those with at least a basic template (purpose, key functions, maybe a diagram). The existing engine docs like _SignalCore System_ and _Simulation Engine_ should be reviewed to ensure they’re up to date with the current code (for example, if Phase 2 changes happened since they were written). If any engine (like an _Analytics engine_ or _Data engine_) is missing documentation, add it. It appears all major ones are covered now (Cortex orchestrates, SignalCore signals, RiskGuard risk, ProofBench simulates, FlowRoute executes, plus RAG/LLM integration as a support engine). No further engine modules are known, so these cover it.
* **Prompts:** We identified the need for a high-level _Skills/Prompt Usage Guide_, since we removed some guides. We will create a scaffold for one, describing how .claude/skills/ works in Ordinis (some of that info might come from the archived _RECOMMENDED_SKILLS.md_ or references to Claude OS usage). This ensures the knowledge base isn’t missing an explanation of how the AI assistance layer functions from a user perspective.
* **References:** The references section might need scaffolding if we want to include summaries of important papers or textbooks. The manifest shows only one PDF link currently. If the comprehensive plan intended more academic references (like adding Hull’s Option Futures summary, etc.), we should add placeholder pages or at least list citations. For instance, a _references/fixed_income.md_ was mentioned for bond pricing[_[25]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L56-L61) – if not already present, create it (perhaps summarizing key points from a fixed income textbook or linking to external reference).

In summary, after migrating existing files, we expect to create roughly15–20 new markdown files to address the above gaps (many already drafted viaskill integration). This includes ~10 advanced math pages in Foundations[_[61]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L136-L144), ~5–6 fundamental analysis and fixed-income pages in Signals/Risk[_[62]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L62-L70), ~3–4 event strategy pages, a few option strategy and pricing pages,and 3 engine docs. These scaffolds can initially be minimal (just a header, abrief outline, and a “(To be written)” note with status draft), so that the structure is in placeand contributors can fill them in subsequent iterations.

* * *

**Next Steps:** With this migration plan, theexecution will proceed in phases: 1. **File Operations:** Perform filemoves/renames as per the mapping table. Use git moves to preserve history wherepossible. Update all links and references across the docs accordingly(leveraging the planned scripts[_[41]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L76-L85)). 2. **Metadata Injection:** Add YAML frontmatter to all filespost-move. We can script this for consistency (for example, insert a templateinto each file based on its category). 3. **Asset Relocation:** Extract anyembedded diagrams (if any were stored in the docs or created during thisprocess) to the assets/ directory. For instance, the ASCIIart diagrams in NVIDIA docs could be replaced with actual images under assets/diagrams/engines/nvidia_integration.png for better readability in the future. Ensure references in markdownare updated to point to the new asset paths. 4. **Validation:** Rebuild thedocumentation site locally (e.g. run MkDocs) to check that all pages render,links work, images load, and nav reflects the new structure. Pay specialattention to the main index and cross-section links. 5. **Clean-up:** Removeor archive the files slated for deletion (ensuring they are no longerreferenced). We might keep the docs/ folder of old structure in a branchor backup tag if we ever need to refer back, but the main branch will only havethe new docs/knowledge-base/... content (plus anydecided inbox/ archives). 6. **FutureContributions:** Mark the placeholders and draft pages with a clear status soit’s visible that they need content. Possibly open issues or tasks for eachmissing topic (advanced math topics, etc.) to track completion.

By following this plan, we will achieve a well-organized docs/knowledge-base that is logicallydivided into Domains (trading knowledge) and Engines/Systems (internal docs),plus supporting sections for Prompts (AI integration) and Sources (dataintegrations). This will make the Ordinis documentation far more accessible andmaintainable going forward, while preserving all valuable content from thecurrent repository.

* * *

[_[1]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L28-L46) [_[3]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L32-L40) [_[4]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L94-L102) [_[20]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L2-L5) [_[52]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L136-L145) [_[53]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L150-L158) [_[58]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L61-L64) [_[61]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md#L136-L144) COMPREHENSIVE_MIGRATION_PLAN.md

[ordinis/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md at af5c281405cea8dbda5fbf2b94cafc4e55e655be · keith-mvs/ordinis · GitHub](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/COMPREHENSIVE_MIGRATION_PLAN.md)

[_[2]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L34-L43) [_[9]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L84-L93) [_[10]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L100-L108) [_[26]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L44-L47) [_[44]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L17-L25) [_[50]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L85-L93) [_[51]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L101-L108) [_[54]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L56-L65) [_[57]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md#L130-L138) index.md

[ordinis/docs/knowledge-base/index.md at af5c281405cea8dbda5fbf2b94cafc4e55e655be · keith-mvs/ordinis · GitHub](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/index.md)

[_[5]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L102-L105) [_[15]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L114-L121) [_[17]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L114-L119) [_[27]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L23-L28) [_[28]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L26-L29) [_[32]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L60-L64) [_[37]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L74-L77) [_[38]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L126-L134) [_[39]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md#L8-L12) docs-rename-plan.md

https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-rename-plan.md

[_[6]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L3-L10) [_[11]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L32-L40) [_[25]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L56-L61) [_[47]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L5-L8) [_[55]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md#L55-L62) SKILLS_INTEGRATION_STRATEGY.md

[ordinis/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md at af5c281405cea8dbda5fbf2b94cafc4e55e655be · keith-mvs/ordinis · GitHub](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INTEGRATION_STRATEGY.md)

[_[7]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L9-L13) [_[31]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L9-L18) [_[42]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L46-L55) [_[43]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md#L116-L124) nvidia-integration.md

[ordinis/docs/architecture/nvidia-integration.md at af5c281405cea8dbda5fbf2b94cafc4e55e655be · keith-mvs/ordinis · GitHub](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/architecture/nvidia-integration.md)

[_[8]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/04_strategy/nvidia_integration.md#L42-L50) nvidia_integration.md

[ordinis/docs/knowledge-base/04_strategy/nvidia_integration.md at af5c281405cea8dbda5fbf2b94cafc4e55e655be · keith-mvs/ordinis · GitHub](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/04_strategy/nvidia_integration.md)

[_[12]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L156-L165) [_[14]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L168-L176) [_[16]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L257-L265) [_[18]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L178-L186) [_[29]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L46-L52) [_[30]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L48-L52) [_[35]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L142-L150) [_[36]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L41-L49) [_[40]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L54-L63) [_[41]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L76-L85) [_[45]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L59-L67) [_[48]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L320-L323) [_[49]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md#L259-L265) docs-naming-standard-compliance-report.md

[ordinis/docs/docs-naming-standard-compliance-report.md at af5c281405cea8dbda5fbf2b94cafc4e55e655be · keith-mvs/ordinis · GitHub](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/docs-naming-standard-compliance-report.md)

[_[13]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L73-L76) [_[19]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L64-L67) [_[21]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L29-L37) [_[22]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L38-L41) [_[23]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L66-L69) [_[24]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L39-L41) [_[33]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L11-L19) [_[34]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L13-L21) [_[46]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L2-L9) [_[56]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L51-L59) [_[59]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L62-L69) [_[60]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L31-L39) [_[62]_](https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md#L62-L70) SKILLS_INDEX.md

https://github.com/keith-mvs/ordinis/blob/af5c281405cea8dbda5fbf2b94cafc4e55e655be/docs/knowledge-base/SKILLS_INDEX.md

* * *

1) First principle (this is the anchor)

---------------------------------------

Your KB should be **domain‑centric, not document‑centric**.

Every domain answers the same questions:

* _What is this field?_

* _What problems does it solve?_

* _What skills are required?_

* _What are the canonical sources?_

* _How does it map to engines, prompts, and systems?_

If a page doesn’t answer one of those, it doesn’t belong in the KB.

* * *

2) Top‑level KB layout (single source of truth)

-----------------------------------------------

Put this under:
    docs/knowledge-base/

### Canonical structure

    docs/knowledge-base/
      index.md                 # how to use the KB
      taxonomy.md              # controlled vocabulary + domain map
      glossary.md              # shared definitions
      governance.md            # quality rules, citation rules

      domains/
        trading/
        systems/
        mathematics/
        ml-ai/
        operations/

      engines/
      prompts/
      sources/
      research-inbox/

This separation is critical:

* **domains/** = _what you know_

* **engines/** = _how the platform reasons_

* **prompts/** = _how models are driven_

* **sources/** = _where truth comes from_

* **research-inbox/** = _where raw material lands before curation_

* * *

3) Domain structure (this is the heart)

---------------------------------------

Each domain uses **the same internal layout**, no exceptions.

Example:
    docs/knowledge-base/domains/trading/market-microstructure/
      index.md
      problems.md
      skills.md
      publications.md
      engine-mapping.md
      prompts.md
      examples.md
      assets/

### What each file means

* `index.md`
  Definition, scope, why it matters

* `problems.md`
  What this domain actually solves in practice

* `skills.md`
  Competencies required (analyst, quant, system, ops)

* `publications.md`
  **Curated, institutional‑grade sources only**
  (books, standards, papers — no blogs)

* `engine-mapping.md`
  Which engines consume this knowledge and how

* `prompts.md`
  Domain‑specific prompt templates

* `examples.md`
  Applied usage (models, workflows, failure modes)

This enforces consistency and makes the KB machine‑parsable.

* * *

4) Map to your actual platform domains

--------------------------------------

Under `domains/`, I would mirror your stated scope:
    domains/trading/
      market-microstructure/
      technical-analysis/
      volume-order-flow/
      macro-fundamental/
      sentiment-news/
      options-derivatives/
      risk-management/
      strategy-design/
      system-architecture/

    domains/mathematics/
      probability/
      stochastic-processes/
      optimization/
      time-series/
      linear-algebra/

    domains/systems/
      system-architecture/
      automation/
      data-pipelines/
      windows-performance/
      hardware-optimization/

    domains/ml-ai/
      model-evaluation/
      feature-engineering/
      llm-rag/
      ml-ops/

    domains/operations/
      compliance/
      auditability/
      monitoring/
      failure-analysis/

This aligns **exactly** with your engine and platform goals.

* * *

5) Engine knowledge (separate from domain theory)

-------------------------------------------------

Engines are **not domains**. They are **consumers of domains**.
    docs/knowledge-base/engines/
      microstructure-engine/
        index.md
        io-contracts.md
        core-functions.md
        integration-hooks.md
        failure-modes.md

      hardware-optimization-engine/
      risk-engine/
      execution-engine/

This avoids mixing “what the world is” with “how the system behaves”.

* * *

6) Prompt library (first‑class, not scattered)

----------------------------------------------

Centralize prompts so they don’t rot.
    docs/knowledge-base/prompts/
      deep-research.md
      publication-ingestion.md
      skill-extraction.md
      domain-skillset.md
      engine-integration.md

      hardware/
        diagnostics.md
        cuda-tuning.md

Each prompt:

* purpose

* inputs

* outputs

* failure cases

* extension hooks (Claude, future models)

* * *

7) Sources = evidence locker (no prose here)

--------------------------------------------

    docs/knowledge-base/sources/
      books/
      papers/
      standards/
      datasets/

Rules:

* PDFs live here

* Each source has a **metadata sidecar** (`.yaml` or front‑matter doc)

* Domain docs _reference_ sources, they don’t duplicate them

This keeps your KB defensible and auditable.

* * *

8) Research inbox (prevents KB pollution)

-----------------------------------------

Raw ideas never go straight into domains.
    docs/knowledge-base/research-inbox/
      unreviewed/
      in-review/
      rejected/

Promotion rule:

* Inbox → Domain doc only after curation

* Otherwise it stays quarantined

This single rule prevents long‑term decay.

* * *

9) Mandatory metadata (for RAG + governance)

--------------------------------------------

Every KB page gets YAML front‑matter:
    id: kb-trading-market-microstructure
    type: domain
    status: canonical
    version: 1.0
    owner: platform
    sources:
      - book-harris-microstructure
    related:
      - engine-microstructure

This enables:

* traceability

* RAG filtering

* version control

* audit readiness

* * *

10) Golden rules (non‑negotiable)

---------------------------------

1. One domain = one folder

2. Same file names across all domains

3. Sources are cited or the page doesn’t exist

4. Inbox before promotion

5. Engines never define theory

6. No “misc”, “notes”, or “scratch” in KB

7. Git is history — filenames never carry versions

* * *

### One clarifying question (to tune the next step)

Is this knowledge base intended to serve:
**A)** internal platform development only
**B)** internal + user‑facing (eventually public)

Answer **A or B** and I’ll refine the governance + exposure rules accordingly.
