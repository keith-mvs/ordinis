# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-dev] - 2024-12-08

### Added
- **Governance Engines** - Complete governance framework implementation
  - `AuditEngine` - Immutable audit trails with SHA-256 hash chaining
  - `PPIEngine` - Personal data detection and masking
  - `EthicsEngine` - OECD AI Principles (2024) implementation
  - `GovernanceEngine` - Policy enforcement orchestration
  - `BrokerComplianceEngine` - Alpaca/IB terms of service compliance
- **Documentation System** - MkDocs Material with plugins
  - PDF export capability
  - Section numbering
  - Git revision dates
  - Mermaid diagram support
- **Knowledge Base Updates**
  - Governance engines documentation
  - Advanced risk methods
  - Strategy formulation framework
  - NVIDIA integration guide

### Changed
- Updated KB index with implementation status markers
- Enhanced governance engines documentation with usage examples

### Technical
- 85+ governance engine tests
- OECD AI Principles enumeration with all 5 pillars
- Broker compliance for PDT rules, rate limits, data usage

## [0.1.0] - 2024-11-30

### Added
- **Core Trading Infrastructure**
  - `SignalCore` - Signal generation engine
  - `RiskGuard` - Risk management with kill switches
  - `FlowRoute` - Order routing and execution
- **Knowledge Base** - 90+ markdown files organized by trading workflow
- **Paper Trading** - Alpaca Markets integration
- **AI Integration**
  - NVIDIA NIM model integration
  - RAG system for knowledge retrieval
  - Cortex analysis engine
- **Strategies**
  - SMA Crossover strategy
  - Momentum strategy template

### Technical
- Python 3.11+ codebase
- pytest testing framework
- Streamlit dashboard
- Configuration via YAML

---

## Version Naming Convention

- **Major (X.0.0)**: Breaking API changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, documentation updates
- **-dev**: Development version, not for production

## Links

- [Project Documentation](docs/index.md)
- [Knowledge Base](docs/knowledge-base/index.md)
- [Architecture](docs/architecture/index.md)
