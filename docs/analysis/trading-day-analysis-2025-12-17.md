# Trading Day Analysis - December 17, 2025

## Summary

| Metric | Count |
|--------|-------|
| **Signals Generated** | ~6,000+ |
| **Successful Executions** | 0 |
| **Failed Executions** | 6,239+ |

---

## Error Breakdown

| Error Type | Count | Root Cause |
|------------|-------|------------|
| `insufficient buying power` | **6,123** | Account fully invested (21 positions consuming all margin) |
| `pending_new OrderStatus` | 68 | Alpaca API returning status not in OrderStatus enum |
| `calculated quantity is 0` | 48 | Position sizing bug (buying_power was 0) |
| WebSocket auth errors | 32 | Initial Polygon auth flow issue (fixed) |
| Reconnection attempts | 23 | Before auth fix was deployed |

---

## Top Signal Generators (RSI Oversold)

| Symbol | Signals | Notes |
|--------|---------|-------|
| TSLA | 141 | High volatility day |
| DXCM | 122 | Medical device selloff |
| DKNG | 120 | Gambling sector weak |
| HOOD | 117 | Fintech under pressure |
| OPEN | 112 | Real estate tech beaten |
| PLTR | 111 | Tech selloff |
| IWM | 110 | Small caps weak |
| SOFI | 109 | Fintech weakness |
| COIN | 109 | Crypto correlation |
| JBLU | 108 | Airlines oversold |

---

## Key Issues Identified

1. **Account State**: Started with $100k but accumulated 21 positions worth $195k (2x margin), leaving $0 buying power

2. **Bug Timeline**:
   - 07:42 - WebSocket auth failures (fixed by proper auth flow)
   - 07:45 - Connected successfully
   - 08:09 - First signals generated
   - 08:09+ - `pending_new` OrderStatus errors (enum mismatch)
   - Later - `insufficient buying power` (account fully invested)

3. **Position Accumulation**: Orders DID execute but the script wasn't tracking them properly, leading to over-investment

---

## Final Account State

| Metric | Value |
|--------|-------|
| Equity | $95,647.92 |
| Buying Power | $0 |
| Cash | -$99,844.04 |
| Open Positions | 21 |
| Total Position Value | $195,490.77 |
| Unrealized P&L | -$4,352 (approx) |

### Open Positions at End of Day

| Symbol | Shares | Entry Price | Market Value | P&L % |
|--------|--------|-------------|--------------|-------|
| AFRM | 240 | $73.96 | $17,412 | -1.91% |
| ALGN | 36 | $164.32 | $5,781 | -2.28% |
| AMD | 70 | $206.18 | $13,867 | -3.92% |
| BAC | 432 | $55.29 | $23,583 | -1.26% |
| CELH | 443 | $40.47 | $18,252 | +1.80% |
| CMC | 42 | $71.33 | $2,940 | -1.86% |
| CRWD | 24 | $483.40 | $11,321 | -2.42% |
| DASH | 13 | $225.23 | $2,890 | -1.30% |
| DDOG | 105 | $138.68 | $14,361 | -1.38% |
| ENPH | 184 | $32.54 | $5,892 | -1.60% |
| EXPE | 20 | $285.38 | $5,656 | -0.90% |
| FSLR | 22 | $260.56 | $5,617 | -2.01% |
| GNRC | 152 | $154.68 | $22,078 | -6.09% |
| MDB | 28 | $424.55 | $11,592 | -2.48% |
| PINS | 114 | $26.15 | $2,959 | -0.73% |
| QQQ | 4 | $610.05 | $2,408 | -1.31% |
| RBLX | 68 | $87.68 | $5,843 | -1.99% |
| RUN | 684 | $17.46 | $11,731 | -1.79% |
| TSLA | 12 | $490.21 | $5,621 | -4.45% |
| TTWO | 12 | $245.49 | $2,887 | -2.01% |
| VEEV | 13 | $220.38 | $2,801 | -2.23% |

---

## Recommendations

### Immediate Fixes Required

1. **Reset paper account** to $100k via Alpaca dashboard
2. **Fix OrderStatus enum** to include `pending_new` status from Alpaca API
3. **Sync positions on startup** - Query Alpaca for existing positions before trading
4. **Add position limit enforcement** - Verify against Alpaca account, not just internal tracking

### Strategy Tuning

5. **Reduce signal sensitivity** - RSI < 35 triggered too many signals; consider RSI < 30
6. **Add confirmation filters** - Require multiple conditions (e.g., RSI + volume spike)
7. **Implement sector exposure limits** - Avoid concentration in single sector

### Infrastructure Improvements

8. **Add position reconciliation loop** - Periodic sync with broker state
9. **Implement circuit breakers** - Stop trading if error rate exceeds threshold
10. **Add pre-trade buying power check** - Query Alpaca before every order

---

## Lessons Learned

1. **Always sync with broker state** - Internal tracking diverged from actual positions
2. **Test with smaller position sizes first** - 3% per position * many signals = over-investment
3. **Handle all API response states** - `pending_new` was unexpected but valid
4. **Monitor buying power, not just equity** - Margin usage matters

---

*Generated: 2025-12-17*
*Strategy: ATR-Optimized RSI Mean Reversion*
*Data Feed: Massive/Polygon WebSocket*
*Broker: Alpaca Paper Trading*
