# Ordinis - Project Scope

## Mission Statement

Design and build an AI-driven automated trading system capable of:
1. Generating trade signals across multiple instrument classes
2. Executing trades autonomously via broker APIs
3. Operating under strict, configurable risk and compliance constraints

---

## Scope Parameters

### Instruments (All Included)
| Instrument | Included | Notes |
|------------|----------|-------|
| US Equities | Yes | Primary focus |
| Options | Yes | Equity options, defined-risk strategies |
| Futures | Yes | Index, commodity futures |
| Forex | Yes | Major pairs |
| Crypto | Yes | Major cryptocurrencies |

### Trading Styles (Selectable/Configurable)
The system will support multiple trading styles, selectable per strategy:

| Style | Holding Period | Use Cases |
|-------|----------------|-----------|
| Intraday | Minutes to hours | Momentum, mean reversion, scalping |
| Swing | Days to weeks | Trend following, breakout/pullback |
| Position | Weeks to months | Macro themes, value-based |
| Mixed/Adaptive | Variable | Regime-dependent switching |

### Risk Tolerance (Adjustable)
Risk parameters will be fully configurable:

| Parameter | Range | Default |
|-----------|-------|---------|
| Risk per trade | 0.25% - 5% | 1% |
| Max daily loss | 1% - 10% | 3% |
| Max drawdown | 5% - 30% | 15% |
| Max concurrent positions | 1 - 20 | 5 |
| Max sector concentration | 10% - 50% | 25% |

### Capital Considerations
- **Starting capital**: <$25,000 USD
- **PDT Rule Impact**: Pattern Day Trader rule limits day trades to 3 per 5 rolling business days for margin accounts under $25K
- **Mitigation strategies**:
  - Cash account (no PDT, but T+2 settlement)
  - Swing/position trading focus until capital threshold met
  - Multiple broker accounts (legal but adds complexity)
  - Options strategies (some exemptions apply)

### Market Focus
- **Primary**: US markets (NYSE, NASDAQ, CBOE, CME)
- **Secondary**: International markets (open for future expansion)
- **Trading hours**: Regular session primary; extended hours optional

### Broker Integration
- **Preferred**: TD Ameritrade / Charles Schwab
- **Note**: TD Ameritrade merged with Schwab (2023). API transition in progress.
- **Alternatives to evaluate**:
  - Interactive Brokers (robust API, global markets)
  - Alpaca (commission-free, modern API)
  - Tradier (options-focused, good API)

### Knowledge Base Philosophy
- **Primary sources**: Academic/scholarly publications, peer-reviewed research
- **Secondary sources**: Credible financial publications, regulatory documents
- **Excluded**: Unverified social media, promotional content, anonymous blogs

---

## Key Constraints & Considerations

### Regulatory
- SEC/FINRA rules for US equities and options
- CFTC rules for futures
- PDT rule for accounts <$25K
- Wash sale rules for tax purposes

### Technical
- API rate limits and reliability
- Data latency considerations
- Order execution quality
- System uptime requirements

### Risk Management (Non-Negotiable)
- Kill switch capability
- Daily loss limits that halt trading
- Position size limits
- Sanity checks on all orders

---

## Open Questions for Refinement

1. **Data providers**: Preferences for market data sources?
2. **Backtesting period**: How much historical data for validation?
3. **Programming language**: Python preferred for ML/quant work?
4. **Deployment**: Cloud (AWS/GCP) or local?
5. **Options complexity**: Simple (long calls/puts, spreads) or advanced (iron condors, butterflies, etc.)?

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2024-01-XX | Initial scope definition |
