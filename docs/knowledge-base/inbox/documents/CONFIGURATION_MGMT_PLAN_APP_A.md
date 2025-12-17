Organizing an Evolving Automated Trading System Project
An automated trading system project can grow chaotic as it evolves. In this case, the project's structure has been changing continually, leading to many files scattered at the top level and inconsistent organization. A recent comprehensive refactoring transformed the codebase from a cluttered state into a more professional layout[1], but further refinement is needed. Below we detail the issues identified and the structured improvements – including file reorganization, renaming for consistency, and implementing change control through documentation – to solidify the project architecture.
Issues with the Current Structure
Top-Level Clutter: Originally, numerous files and artifacts resided in the repository root (e.g. cache folders, build outputs, coverage reports), making it hard to navigate. This clutter has been addressed by removing over 186 unnecessary files and directories (like caches, build artifacts, duplicate files)[2][3]. As a result, the root now contains only essential configuration and documentation files (e.g. pyproject.toml, README, requirements, config files)[4]. Reducing top-level noise improves clarity and discoverability of key files[3].
Inconsistent Naming and Legacy Artifacts: The project was initially named "Intelligent Investor," but not all references and file names reflected the new name. This was corrected by renaming the project to "ordinis" universally (e.g. updating the package name in pyproject.toml)[5]. File naming conventions were standardized and duplicate/outdated files (such as versioned script copies) were removed[6]. Consistent naming eliminates confusion and ensures the project identity is clear in all modules.
Unstructured Scripts and Data Files: Previously, many standalone scripts lived in one folder, and data outputs were intermixed, contributing to a disorganized top level. This was solved by categorizing scripts into subdirectories by purpose (data processing, backtesting, trading, etc.)[7]. For example, rather than 40+ scripts in one place, they are now grouped under scripts/ in logical categories like data/, backtesting/, trading/, demo/, etc., each with a clear focus[7]. Similarly, data files and logs have been organized into domain-specific folders. A structured data/ directory was introduced with subfolders for raw datasets, processed data, historical price data, synthetic data, backtest results, etc.[8]. A new logs/ directory contains separate logs for data fetches, backtests, and live trading runs[9]. This hierarchy keeps the repository tidy and makes it obvious where each type of file belongs.
Lack of Change Control and Documentation: The project lacked formal change tracking and development guidelines, which are vital as the system grows. To address this, new documentation and processes were put in place. A CHANGELOG was added to record notable changes over time[10], and a comprehensive CONTRIBUTING.md guide (180+ lines) was created[11]. The contributing guide introduces development standards such as coding style, testing practices, and commit message conventions[12]. Adopting a standard commit format (e.g. a header line with a change type/scope and a body detailing the change, plus any footers for issue references) helps enforce consistency in how changes are recorded. These conventions, along with a new pre-commit hook configuration (.pre-commit-config.yaml at the root), establish change control discipline so that each code change is well-documented and reviewed in a controlled manner. Additionally, high-level documentation like ARCHITECTURE.md (over 350 lines) was written to outline system design, layers, and future roadmap[13], giving developers a clear reference for the project's structure and guiding principles.
Refactoring and Reorganization Improvements
The recent refactoring implemented a series of structural improvements to build a maintainable foundation:
•    Cleanup of Junk and Build Files: All irrelevant or auto-generated files were purged. This included test cache directories (.pytest_cache/, .mypy_cache/), coverage reports, build outputs (like a site/ directory from docs), and even a stray Windows NUL device file[14][15]. By updating .gitignore and removing these artifacts, the repository shed over half a million lines of noise, reducing its size by ~40%[16][3]. This makes the repo leaner and faster to clone, and it ensures developers don't accidentally commit temporary files.
•    Structured Directories for Code and Scripts: The source code under src/ is now organized into clear sub-packages representing different layers or domains of the trading system. For instance, modules are separated into core/ (core logic), engines/ (backtesting and trading engines), strategies/ (trading strategies implementations), data/ (data handling utilities), plugins/ (market data adapters), monitoring/ (metrics and risk monitors), and so on[17]. This layer-based structure reflects the architecture (data layer, engine layer, strategy layer, etc.) described in the design documents[13], and it makes the code more navigable by functionality.
Meanwhile, the scripts/ directory was reorganized into 8 subfolders by function[7]. For example, all data management scripts (for fetching or preparing datasets) reside in scripts/data/, backtesting runners and analyzers are in scripts/backtesting/, live/paper trading executors in scripts/trading/, and so forth[7]. This categorization replaces a flat collection of 40+ scripts with a logical grouping, improving discoverability. Each category can even have its own README with usage instructions (indeed a scripts/README.md was added documenting all 42 scripts and how to use them[18]).
•    Data and Outputs Organization: The data/ folder was expanded to cover various data types. It now has dedicated subfolders for raw data (original dumps), processed data (cleaned or feature-engineered datasets), historical time series (historical/ and a cache for them), synthetic data for testing, macro-economic indicators (macro/), and places to store backtest results and even ChromaDB vector store for the RAG system[8]. Each of these is documented in a data/README.md to clarify usage. Similarly, logs are kept in logs/ with separate files for different processes (fetch, backtest, trading)[9], and reports (such as analysis outputs or status reports) are organized under a reports/ directory by type (backtest reports, performance analyses, status updates)[19]. By segregating data, logs, and reports, the working files produced by the system do not clutter the code directories, and they can be easily managed or ignored in version control when appropriate.
•    Consistent Naming and Project Renaming: As part of refactoring, the project package name was updated from the legacy "intelligent_investor" to the current "ordinis" everywhere – including folder names and metadata in pyproject.toml[20]. This unified name avoids confusion (code imports now use ordinis.*). Additionally, naming conventions for files and directories were reviewed: some scripts that had version suffixes or ambiguous names were renamed for clarity (e.g., old duplicates like backtest_new_indicators.py had one version removed and the remaining one placed in the proper folder)[21][22]. The refactor explicitly removed duplicate or deprecated files and enforced a standard naming scheme[6]. This ensures that each filename clearly reflects its purpose, and it eliminates redundant files that could diverge over time.
•    Introducing Documentation and Guides: A major enhancement was adding top-level documentation to guide contributors. The CONTRIBUTING.md file now provides a comprehensive developer guide (setup, code style, testing, workflow, etc.), including specific commit message guidelines and a pull request process[12]. Having a commit convention means every commit message starts with a consistent header (e.g. "feat:", "fix:", etc., plus scope), and can include an optional footer (e.g. "Closes #issue") to improve traceability. This helps with change control, because one can generate meaningful release notes or find changes by type. Alongside that, ARCHITECTURE.md describes the system's design and layering in detail[13], which helps current and new team members understand how the pieces fit together. A specialized scripts/README.md was also created to document usage of each utility script[18], and the existing data directory README was updated to remain comprehensive[23]. These documents greatly improve the project's transparency and maintainability, serving as an internal knowledge base.
By implementing all of the above, the project structure is now much more professional and maintainable. The repository follows standard layouts and best practices, with a clear separation of concerns and thorough documentation of its components[24].
Figure: Refined Project Structure (Simplified) – The outline below reflects the updated layout after refactoring, highlighting the organized directories and key files:
{project-root}/
├── src/
│   ├── core/               # Core logic and base classes
│   ├── engines/            # Trading & backtesting engines (ProofBench, etc.)
│   ├── strategies/         # Strategy implementations (technical, options, etc.)
│   ├── data/               # Data layer modules (loaders, validators)
│   ├── plugins/            # External plugins (market data APIs, etc.)
│   ├── monitoring/         # Monitoring and risk management components
│   ├── analysis/           # Analytical tools (if separate from engines)
│   ├── dashboard/          # Dashboard or visualization back-end
│   ├── rag/                # RAG system integration (knowledge base querying)
│   └── visualization/      # Plotting and UI visualization code
├── scripts/
│   ├── data/               # Scripts for data fetching and management[25]
│   ├── backtesting/        # Scripts to run or analyze backtests[26]
│   ├── trading/            # Scripts for live/paper trading execution[27]
│   ├── demo/               # Demo scenarios and example runs[28]
│   ├── analysis/           # Analysis and reporting tools[29]
│   ├── skills/             # AI assistant skill management utilities[30]
│   ├── docs/               # Documentation generation tools[31]
│   ├── rag/                # Scripts for RAG (Retrieval-Augmented Generation) system[32]
│   └── utils/              # Miscellaneous utility scripts[33]
├── data/
│   ├── metadata/           # Dataset catalogs and metadata files[34]
│   ├── raw/                # Original raw data (unprocessed)[8]
│   ├── processed/          # Processed/cleaned data sets[35]
│   ├── historical/         # Historical market data (OHLCV etc.)[8]
│   ├── historical_cache/   # Cached historical data for quick access[8]
│   ├── synthetic/          # Synthetic data for testing strategies[8]
│   ├── macro/              # Macroeconomic indicators data[36]
│   ├── backtest_results/   # Outputs from backtest runs[37]
│   ├── chromadb/           # Vector store for knowledge base (RAG)[38]
│   └── README.md           # Documentation for data directory
├── logs/
│   ├── fetch/              # Logs from data fetching processes[39]
│   ├── backtest/           # Logs from backtesting runs[39]
│   └── trading/            # Logs from live/paper trading runs[40]
├── reports/
│   ├── backtest/           # Generated reports for backtest analyses[41]
│   ├── performance/        # Performance analysis reports[42]
│   └── status/             # Project status and session reports[43]
├── docs/
│   ├── architecture/       # System architecture documentation
│   ├── guides/             # User and developer guides
│   ├── knowledge-base/     # Domain knowledge and research docs
│   ├── references/         # External references (with PDFs)[44]
│   ├── project/            # Project planning and status docs[45]
│   └── ... (additional documentation categories)
├── tests/                  # Test suite (unit and integration tests)
├── .claude/                # (AI assistant context files, archived sessions, etc.)
├── .env.example            # Template environment variables
├── .gitignore              # Git ignore rules (updated for new structure)[46]
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── ARCHITECTURE.md         # ✨ System architecture overview document[47]
├── CONTRIBUTING.md         # ✨ Contribution guidelines (standards, workflow)[10]
├── CHANGELOG.md            # ✨ Changelog for tracking changes[10]
├── pyproject.toml          # Project metadata (renamed to 'ordinis', etc.)[20]
├── README.md               # Top-level project README
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development/testing dependencies
└── mkdocs.yml              # Documentation site configuration
(The tree above illustrates the refined layout, incorporating all reorganizations and new files. Starred ✨ entries denote newly added items after refactoring.)
Benefits of the Refined Structure
Adopting all of the above improvements yields multiple benefits:
•    Improved Discoverability & Organization: Related files are grouped logically (e.g. scripts by purpose, data by type), and each directory often contains a README explaining its contents. This modular structure makes it faster for developers to find what they need and understand the codebase at a glance[3]. New team members can easily navigate to relevant sections (for instance, all trading execution code is under scripts/trading/ and src/engines/, strategy definitions under src/strategies/, etc.), rather than trawling through a monolithic folder.
•    Reduced Clutter & Technical Debt: The removal of redundant files and artifacts declutters the repository significantly[48]. Over 535,000 lines of generated or obsolete code were deleted[16], streamlining the project. A cleaner repo not only saves disk space and speeds up operations, but also reduces the risk of confusion from outdated code. Developers can be confident that what remains is the authoritative code and documentation. The project root is now clean, containing mainly configuration and documentation, which is typical of a well-structured Python project[4].
•    Consistent Naming & Professionalism: By renaming the package and standardizing file names, the project presents a unified identity. Consistent naming prevents errors (e.g. mismatched import paths) and reflects an attention to detail. The use of a single primary branch (master consolidated from main)[49] and inclusion of version numbers (project version set to 0.2.0-dev in pyproject)[20] indicate that the project is managed in line with professional version control practices. Overall, the structure now aligns with industry best practices for Python project layout[24].
•    Better Documentation & Onboarding: The addition of rich documentation (Architecture guide, Contributing guide, script usage docs) greatly enhances transparency[50]. Team members have a clear reference for how the system is designed and how to contribute. The commit message guidelines and documented workflow in CONTRIBUTING.md instill discipline in how changes are made and reviewed[12], which is crucial as the project scales. New contributors can read these docs to get up to speed, and the project maintainer can point to these resources to ensure consistency. The presence of a changelog means important changes are recorded in one place for each release, aiding in both internal tracking and external communication of progress.
•    Easier Maintenance & Future Growth: With a logical structure in place, adding new features or modules becomes easier. For example, if a new trading strategy is developed, one knows to put its code under src/strategies/ and perhaps add corresponding tests under tests/. If a new script is needed (say for data analysis), it should fit into one of the script categories or prompt creating a new category if truly unique. This prevents the "many files at top-level" problem from recurring. The project can grow within the established framework. Moreover, suggestions have been made to further future-proof the project, such as adding an .editorconfig for consistent code style across editors, using Git LFS for large files, setting up CI/CD pipelines, and even considering a monorepo approach if the scope broadens to multiple related projects[51][52]. These recommendations ensure that as complexity increases, the structure remains manageable and robust.
In summary, the project has been thoroughly refactored and reorganized to incorporate all requested improvements – cleaning up the repository, grouping files logically, renaming for consistency, and instituting change-control through documentation and conventions. The result is a more maintainable, scalable foundation for the automated trading system, one that can accommodate ongoing changes without devolving into chaos. By continuously refining the architecture in this manner, the team can focus on developing new features and strategies rather than wrestling with project structure, thereby accelerating development in the long run[53].
Sources:
•    Ordinis Project Refactoring Summary (Dec 12, 2025) – Project reorganization, file cleanup, and new structure overview[1][6].
•    Ordinis Repository Documentation – Directory layouts for scripts, data, logs, and reports after refactor[7][8].
•    Ordinis CONTRIBUTING and Configuration – Project rename in pyproject, commit conventions, and added documentation[5][11].
•    Ordinis Refactor Benefits – Outcomes of cleanup (reduced files/lines) and best-practice structure achieved[3][24].

________________________________________

[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16] [17] [18] [19] [20] [21] [22] [23] [24] [25] [26] [27] [28] [29] [30] [31] [32] [33] [34] [35] [36] [37] [38] [39] [40] [41] [42] [43] [44] [45] [46] [47] [48] [49] [50] [51] [52] [53] session-export-20251212-refactoring.md
https://github.com/keith-mvs/ordinis/blob/c67c93b34c9a87bff87114ca0c7f3a41623aa2ad/docs/archive/session-exports/session-export-20251212-refactoring.md
