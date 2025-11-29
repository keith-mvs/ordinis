# Claude Connectors - Quick Reference

**Last Updated:** 2025-01-28

---

## TL;DR Recommendation

**❌ DO NOT implement any connectors in Phase 1**

**Rationale:**
- Current data sources (Polygon, IEX) are sufficient for core strategies
- Focus engineering effort on SignalCore, RiskGuard, ProofBench
- Add connectors only when specific alpha sources are proven

**Exception:** Consider **Scholar Gateway** in Phase 2 for Knowledge Base quality (low cost, passive benefit)

---

## Quick Assessment Matrix

| Connector | Use Case | Priority | Cost/Month | When to Add |
|-----------|----------|----------|------------|-------------|
| 🟡 **Daloopa** | Fundamental data | Medium-High | $500-2K | IF building fundamental strategies |
| 🔴 **Crypto.com** | Crypto data | Low | $0-100 | IF adding crypto to universe (unlikely) |
| 🟡 **Scholar Gateway** | Academic papers | Medium | $0-50 | Phase 2, for KB quality |
| 🟡 **MT Newswires** | Real-time news | Medium | $1K-5K | IF event-driven is core strategy |
| 🔴 **Moody's Analytics** | Credit risk | Low | $10K-50K+ | Likely never (too expensive, wrong scope) |
| 🟡 **S&P Aiera** | Transcripts + AI | Medium | $2K-10K | IF sentiment is core alpha source |

---

## Priority Tiers

### 🔴 Tier 1: DO NOT ADD (Yet)
**All evaluated connectors are Tier 1 "Do Not Add" for now**

Why? System needs to prove core strategies first before expanding data sources.

### 🟡 Tier 2: EVALUATE Later (Phase 2-3)

**Add IF specific conditions met:**

1. **Scholar Gateway** - Low-hanging fruit
   - **Condition:** KB maintenance in Phase 2
   - **Cost:** ~$0-50/month
   - **Effort:** ~12 hours
   - **Value:** Improves KB credibility

2. **Daloopa** - Fundamental alpha
   - **Condition:** Building fundamental/value strategies
   - **Cost:** ~$500-2,000/month
   - **Effort:** ~15 hours
   - **Value:** HIGH if fundamentals are alpha source

3. **MT Newswires** - Event-driven alpha
   - **Condition:** Event-driven trading is core strategy
   - **Cost:** ~$1,000-5,000/month
   - **Effort:** ~18 hours
   - **Value:** HIGH if news is alpha source

4. **S&P Aiera** - Sentiment alpha
   - **Condition:** Sentiment from transcripts is alpha source
   - **Cost:** ~$2,000-10,000/month
   - **Effort:** ~21 hours
   - **Value:** HIGH if sentiment works, overlaps with MT Newswires

### 🔴 Tier 3: SKIP (Likely Never)

1. **Crypto.com**
   - **Reason:** Out of scope (equities/options focus)
   - **Reconsider:** Only if crypto explicitly added

2. **Moody's Analytics**
   - **Reason:** Too expensive, not relevant for equities
   - **Reconsider:** Only if managing $10M+ portfolios or adding fixed income

---

## Decision Framework

**Before adding ANY connector, ask:**

1. ✅ **Is there proven alpha?** Have we tested strategies that need this data?
2. ✅ **Have we maxed out current data?** Can Polygon/IEX provide this?
3. ✅ **Is cost justified?** Will this data generate >10x its cost in alpha?
4. ✅ **Do we have capacity?** Can we integrate and maintain this?
5. ✅ **Is there a free alternative?** Can we test with free data first?

**If ANY answer is NO → DEFER the connector**

---

## Quick Cost Analysis

### Total Cost if All Added

| Connector | Monthly Cost |
|-----------|-------------|
| Daloopa | $1,000 (mid-range) |
| Crypto.com | $0 (free tier) |
| Scholar Gateway | $25 (estimated) |
| MT Newswires | $3,000 (mid-range) |
| Moody's Analytics | $25,000 (mid-range) |
| S&P Aiera | $5,000 (mid-range) |
| **TOTAL** | **$34,025/month** |

**Annual:** ~$408,000/year

**Recommended Spend:** $0 now, <$2,000/month in Phase 2-3 (Scholar + one data source)

---

## Free Alternatives

**Use BEFORE paying for connectors:**

| Need | Free Alternative | Limitation |
|------|------------------|------------|
| Fundamentals | EDGAR SEC, Yahoo Finance, FMP | Manual work, less structured |
| Crypto | CoinGecko, Binance API | No institutional SLA |
| Academic | Google Scholar, arXiv, SSRN | Manual search |
| News | NewsAPI, Yahoo Finance News | Delayed, lower quality |
| Transcripts | Company IR sites, Seeking Alpha | Manual, no AI analysis |
| Credit | Moody's website (delayed) | Not real-time |

**Strategy:** Start with free, upgrade to paid when proven valuable

---

## Integration Effort

**Total effort if all added:** ~101 hours (2.5 weeks full-time)

**Recommended:** Add one at a time, only when needed

---

## Recommended Sequence

**IF eventually adding connectors (Phase 2+):**

1. **Scholar Gateway** (Phase 2) - Low cost, KB quality
2. **Daloopa** OR **MT Newswires** (Phase 2-3) - Pick one based on strategy type
3. **S&P Aiera** (Phase 3+) - Only if Daloopa/MT Newswires prove valuable
4. **Crypto.com** (Unlikely) - Only if scope expands
5. **Moody's** (Never) - Unless scope dramatically changes

**Do NOT add multiple expensive connectors simultaneously**

---

## Red Flags

**DO NOT add a connector if:**

- ❌ No proven strategy needs this data
- ❌ Cost >1% of expected monthly returns
- ❌ Free alternative untested
- ❌ Engineering team at capacity
- ❌ Similar connector already integrated

---

## Key Takeaways

1. **Current data is sufficient** - Polygon + IEX cover technical/volume strategies
2. **Defer all connectors to Phase 2+** - Build strategies first, add data later
3. **Start with free alternatives** - Prove alpha before paying
4. **Scholar Gateway exception** - Low cost, passive KB improvement
5. **Avoid Moody's** - Too expensive, wrong scope for equities/options
6. **One at a time** - Don't add multiple expensive sources simultaneously

---

## Next Steps

1. ✅ **Focus on core system** (SignalCore, RiskGuard, ProofBench)
2. ✅ **Prove technical strategies** (MA Crossover, breakouts, etc.)
3. 🔲 **Re-evaluate in Phase 2** (Q2 2025)
4. 🔲 **Request trials** when ready (Daloopa, MT Newswires, Aiera)
5. 🔲 **Make data-driven decision** (measure alpha, calculate ROI)

---

## Contact for More Info

**Full Analysis:** `docs/architecture/CLAUDE_CONNECTORS_EVALUATION.md`
**Questions:** Reach out to Architecture Team
**Trials:** Contact connectors directly when Phase 2 begins

---

**Document Version:** v1.0.0
**Owner:** Architecture Team
**Next Review:** Q2 2025
