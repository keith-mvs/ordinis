# Cross-Linking Scheme for Skill Package References

**Version:** 1.0
**Last Updated:** 2025-12-12
**Purpose:** Establish consistent cross-linking methodology across all skill package reference files

---

## Linking Philosophy

1. **Progressive Disclosure**: SKILL.md links to references/, references link to deeper topics
2. **Bidirectional Links**: Related files link to each other
3. **Hierarchical Navigation**: Clear parent/child relationships
4. **Relative Paths**: Use relative paths for portability
5. **Descriptive Link Text**: Make link purpose clear from text

---

## Link Types and Patterns

### 1. Parent Skill Links

From any reference file back to the main SKILL.md:

```markdown
**Parent Skill**: [Strategy Name](../SKILL.md)
```

### 2. Sibling Reference Links

Between references in the same skill:

```markdown
See also:
- [Position Management](position-management.md)
- [Greeks Analysis](greeks-analysis.md)
- [Examples](examples.md)
```

### 3. Master Skill Links

From individual strategy to master options-strategies skill:

```markdown
**Master Reference**: [Options Strategies Overview](../../options-strategies/SKILL.md)

**Shared Resources**:
- [Greeks Fundamentals](../../options-strategies/references/greeks.md)
- [Volatility Analysis](../../options-strategies/references/volatility.md)
- [Strategy Comparison](../../options-strategies/references/strategies.md)
```

### 4. Template Links

From any skill to the template:

```markdown
**Template Reference**: [Skill Package Template](../../_template/SKILL.md)
```

### 5. Related Strategy Links

Between similar strategies:

```markdown
**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md) - Bullish equivalent
- [Bear Put Spread](../../bear-put-spread/SKILL.md) - Bearish debit spread
- [Iron Condor](../../iron-condor/SKILL.md) - Combination strategy
```

### 6. Cross-Domain Links

Between different domains (options ↔ bonds ↔ financial):

```markdown
**Related Skills**:
- [Financial Analysis](../../financial-analysis/SKILL.md) - Fundamental analysis
- [Technical Analysis](../../technical-analysis/SKILL.md) - Chart patterns
```

---

## Standard Cross-Link Sections

### In Strategy Mechanics Files

```markdown
## See Also

**Within This Skill**:
- [Strike Selection](strike-selection.md) - Choosing optimal strikes
- [Position Management](position-management.md) - Managing the trade
- [Greeks Analysis](greeks-analysis.md) - Understanding risk metrics

**Master Resources**:
- [Options Greeks](../../options-strategies/references/greeks.md)
- [Volatility Concepts](../../options-strategies/references/volatility.md)

**Related Strategies**:
- [Strategy Name](../../strategy-name/SKILL.md) - Brief description
```

### In Strike Selection Files

```markdown
## See Also

**Within This Skill**:
- [Strategy Mechanics](strategy-mechanics.md) - How the strategy works
- [Examples](examples.md) - Real-world strike selection

**Master Resources**:
- [Greeks Fundamentals](../../options-strategies/references/greeks.md)
```

### In Position Management Files

```markdown
## See Also

**Within This Skill**:
- [Strategy Mechanics](strategy-mechanics.md) - Position structure
- [Greeks Analysis](greeks-analysis.md) - Risk evolution

**Master Resources**:
- [Volatility Analysis](../../options-strategies/references/volatility.md)
```

---

## Link Placement Guidelines

### 1. At File Start (After Title)

```markdown
# Strike Selection Guide

**Parent**: [Bear Put Spread](../SKILL.md) | **Related**: [Position Management](position-management.md)

---

[Content starts here]
```

### 2. In Context (Inline)

```markdown
When selecting strikes, consider the delta framework (see [Greeks Analysis](greeks-analysis.md#delta-framework)).
```

### 3. At File End (See Also Section)

```markdown
---

## See Also

[Standard cross-link section as above]
```

---

## Cross-Linking Matrix

### Options Strategies Cross-Links

| Strategy | Links To |
|----------|----------|
| bear-put-spread | bull-call-spread (opposite), long-put (simpler), iron-condor (uses this) |
| bull-call-spread | bear-put-spread (opposite), long-call (simpler), iron-condor (uses this) |
| iron-butterfly | iron-condor (similar), long-straddle (opposite) |
| iron-condor | iron-butterfly (similar), bull-call-spread, bear-put-spread |
| long-call-butterfly | iron-butterfly (similar), long-call (simpler) |
| long-straddle | long-strangle (similar), iron-butterfly (opposite), short-straddle |
| long-strangle | long-straddle (similar), iron-condor (opposite) |
| married-put | protective-collar (similar), long-put (component) |
| protective-collar | married-put (similar), covered-call (uses this) |
| covered-call | protective-collar (uses this), naked-call |

### Bond Analysis Cross-Links

| Skill | Links To |
|-------|----------|
| bond-pricing | yield-measures, duration-convexity |
| bond-benchmarking | bond-pricing, credit-risk |
| credit-risk | bond-pricing, bond-benchmarking |
| duration-convexity | bond-pricing, yield-measures |
| option-adjusted-spread | bond-pricing, duration-convexity |
| yield-measures | bond-pricing, duration-convexity |

### Financial Analysis Cross-Links

| Skill | Links To |
|-------|----------|
| financial-analysis | benchmarking, due-diligence |
| benchmarking | financial-analysis |
| due-diligence | financial-analysis, benchmarking |

---

## Implementation Standards

### 1. Required Links (Minimum)

Every reference file MUST have:
- Link back to parent SKILL.md
- "See Also" section at end with at least 2 related links

### 2. Recommended Links

Every reference file SHOULD have:
- 3-5 inline contextual links to related content
- Links to master skill resources (for domain-specific skills)
- Links to 1-2 related strategies/skills

### 3. Optional Links

Consider adding:
- Links to external resources (documentation, papers)
- Links to template for developers
- Historical/version links

### 4. Link Text Best Practices

**Good**:
```markdown
- [Position Management](position-management.md) - When and how to adjust
- [Greeks Analysis](greeks-analysis.md#theta-decay) - Understanding time decay
```

**Bad**:
```markdown
- [Click here](position-management.md)
- [More info](greeks-analysis.md)
```

---

## Validation Checklist

- [ ] All reference files have parent SKILL.md link
- [ ] All reference files have "See Also" section
- [ ] Inline links use descriptive text
- [ ] Relative paths are correct
- [ ] No broken links (404s)
- [ ] Bidirectional links exist (A→B and B→A)
- [ ] Links to master resources where applicable
- [ ] Links between related strategies exist

---

## Automated Validation

Use script: `scripts/validate_cross_links.py`

```bash
python scripts/validate_cross_links.py
```

This checks:
- Link syntax correctness
- Path validity
- Bidirectional consistency
- Coverage (% of files with proper cross-links)

---

## Maintenance

1. **When adding new reference**: Add it to "See Also" sections of related files
2. **When adding new skill**: Update related skills' cross-links
3. **Monthly audit**: Run validation script and fix issues

---

**Example Implementation**: See `bear-put-spread/references/` for complete example
