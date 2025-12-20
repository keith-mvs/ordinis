# KB Search

Search the Knowledge Base publications and concepts library.

## Usage

```bash
/kb-search <query> [--domains D1,D2] [--type TYPE] [--audience LEVEL] [--max-results N]
```

## Parameters

- `query` (required): Natural language search query or concept name
- `--domains`: Filter by domain numbers (1-9), comma-separated
- `--type`: Filter by publication type (textbook, academic_paper, practitioner_book, industry_guide)
- `--audience`: Filter by target audience (beginner, intermediate, advanced, institutional)
- `--max-results`: Maximum number of results to return (default: 10, max: 50)

## Instructions

When this command is invoked, perform the following search workflow:

### 1. Load Publications Index

Read the master publications index:
```
docs/knowledge-base/publications/index.json
```

### 2. Parse Search Parameters

Extract from user command:
- Main query string
- Optional domain filter
- Optional type filter
- Optional audience filter
- Max results limit

### 3. Keyword Matching

For each publication in the index, calculate relevance score based on:

**High Priority Matches (Score: 10 points each):**
- Query appears in `title` (case-insensitive)
- Query appears in publication `id`
- Query exact match in `key_concepts` array

**Medium Priority Matches (Score: 5 points each):**
- Query appears in `summary` field
- Query partial match in `key_concepts` (substring)
- Author name matches query

**Low Priority Matches (Score: 2 points each):**
- Query appears in `tags` array
- Related domain match

### 4. Apply Filters

Filter results by:
- **Domains:** If `--domains` specified, only include publications where ANY domain in `domains` array matches
- **Type:** If `--type` specified, match `source_type` field exactly
- **Audience:** If `--audience` specified, match `target_audience` field

### 5. Rank Results

Sort by:
1. Relevance score (descending)
2. Publication year (descending, newer first)
3. Practical/theoretical score (if practical query, prefer higher scores)

### 6. Format Output

For each result in top N (max_results):

```markdown
## [RANK]. **[TITLE]** ([AUTHORS], [YEAR])

**Domains:** [D1, D2, ...] | **Type:** [source_type] | **Audience:** [target_audience]

**Relevance:** [score]/10 | **Practical:** [practical_vs_theoretical * 100]%

**Key Concepts:** concept1, concept2, concept3...

**Summary:** [summary text]

**Matched On:** [what triggered this result - e.g., "title match: 'risk'"]

**Access:** [primary access method and URL if available]

**Document:** [Link to markdown file if exists, e.g., docs/knowledge-base/.../publication.md]

---
```

### 7. Search Summary Header

Before results, display:

```markdown
# Knowledge Base Search Results

**Query:** "[query]"
**Filters:** [list active filters or "None"]
**Total Matches:** [count]
**Showing:** Top [N] results

---
```

### 8. No Results Handling

If no matches found:

```markdown
# Knowledge Base Search Results

**Query:** "[query]"
**Filters:** [filters]

No publications found matching your search criteria.

**Suggestions:**
- Try broader search terms
- Remove filters (domains, type, audience)
- Check spelling
- Search for author names or concepts instead

**Available Domains:**
1. Market Microstructure
2. Technical Analysis
3. Volume & Liquidity
4. Fundamental & Macro
5. News & Sentiment
6. Options & Derivatives
7. Risk Management
8. Strategy & Backtesting
9. System Architecture

**Available Publication Types:**
- textbook
- academic_paper
- practitioner_book
- industry_guide
- research_report
```

### 9. Concept Search (If query looks like concept ID)

If query matches pattern `concept_\d{2}_[a-z_]+`, attempt to locate concept file:

```
docs/knowledge-base/0[domain]_[domain_name]/concepts/[concept_name].md
```

If found, display concept instead of publications.

### 10. Related Concepts (Advanced)

At the end of results, suggest related searches:

```markdown
## Related Searches

Based on this query, you might also be interested in:
- [Suggested query 1] (from related concepts)
- [Suggested query 2] (from same domain)
- [Suggested query 3] (from publication tags)
```

## Examples

### Example 1: Basic Keyword Search

```bash
/kb-search "overfitting in backtests"
```

**Expected Output:**
```markdown
# Knowledge Base Search Results

**Query:** "overfitting in backtests"
**Filters:** None
**Total Matches:** 3
**Showing:** Top 3 results

---

## 1. **Advances in Financial Machine Learning** (Marcos López de Prado, 2018)

**Domains:** [7, 8] | **Type:** practitioner_book | **Audience:** advanced

**Relevance:** 10/10 | **Practical:** 70%

**Key Concepts:** machine_learning, feature_engineering, backtesting, cross_validation, portfolio_construction

**Summary:** Modern machine learning techniques for quantitative finance. Covers feature engineering, meta-labeling, backtesting, and portfolio construction. Addresses overfitting, data leakage, and statistical pitfalls.

**Matched On:** Key concept "overfitting", summary match "backtesting"

**Access:** Hardcopy, ePub - ISBN 978-1119482086

**Document:** [View Full Notes](docs/knowledge-base/07_risk_management/publications/lopez_de_prado_advances.md)

---

## 2. **Evidence-Based Technical Analysis** (David Aronson, 2006)

**Domains:** [2, 8] | **Type:** practitioner_book | **Audience:** advanced

**Relevance:** 7/10 | **Practical:** 50%

**Key Concepts:** technical_analysis, statistical_testing, data_mining_bias, overfitting, hypothesis_testing

**Summary:** Critical examination of technical analysis through the lens of statistical rigor and scientific method. Addresses data mining bias, overfitting, and proper hypothesis testing.

**Matched On:** Key concept "overfitting", key concept "hypothesis_testing"

**Access:** Hardcopy - ISBN 978-0470008744

**Document:** Not yet documented

---

...

## Related Searches

Based on this query, you might also be interested in:
- "cross validation in finance" (related to preventing overfitting)
- "data mining bias" (common cause of false backtests)
- "sharpe ratio deflation" (adjusting for multiple testing)
```

### Example 2: Filtered Search

```bash
/kb-search "options strategies" --domains 6 --audience intermediate
```

**Expected Output:**
```markdown
# Knowledge Base Search Results

**Query:** "options strategies"
**Filters:** Domains: [6], Audience: intermediate
**Total Matches:** 1
**Showing:** Top 1 result

---

## 1. **Options, Futures, and Other Derivatives** (John C. Hull, 2021)

**Domains:** [6] | **Type:** textbook | **Audience:** intermediate

**Relevance:** 10/10 | **Practical:** 50%

**Key Concepts:** options, derivatives, black_scholes, greeks, hedging, risk_neutral_valuation

**Summary:** Industry-standard textbook on derivatives pricing and risk management. Covers Black-Scholes, binomial models, Greeks, hedging strategies, and exotic options.

**Matched On:** Title match "options", key concept "options", summary match "strategies"

**Access:** Hardcopy, ePub - ISBN 978-0136939979

**Document:** [View Full Notes](docs/knowledge-base/06_options_derivatives/publications/hull_options_futures.md)

---
```

### Example 3: Author Search

```bash
/kb-search "Larry Harris"
```

**Expected Output:**
```markdown
# Knowledge Base Search Results

**Query:** "Larry Harris"
**Filters:** None
**Total Matches:** 2
**Showing:** Top 2 results

---

## 1. **Trading and Exchanges** (Larry Harris, 2003)

**Matched On:** Author name exact match

[... full result ...]

## 2. **Market Microstructure Survey** (Bruno Biais, Larry Glosten, Chester Spatt, 2005)

**Matched On:** Co-author name match

[... full result ...]
```

## Implementation Notes

### Search Algorithm

For initial implementation (no embeddings yet):

1. **Token-based matching** - Split query into tokens, match against all text fields
2. **Weighted scoring** - Different weights for title, concepts, summary
3. **Fuzzy matching** - Allow minor spelling variations (optional enhancement)
4. **Phrase matching** - Exact phrase gets higher score than individual words

### Future Enhancements

Once embedding pipeline is established:

1. **Semantic search** - Use vector embeddings for similarity
2. **Hybrid search** - Combine keyword + semantic scores
3. **Concept graph navigation** - Traverse related concepts
4. **Citation network** - Find publications that reference each other

### Error Handling

- **File not found:** If index.json missing, display helpful error with setup instructions
- **Invalid domain:** Show valid domain numbers if invalid filter provided
- **Malformed query:** Provide example queries

## Related Skills

- `/explain-concept` - Get detailed explanation of a specific concept
- `/recommend-publications` - Get curated reading recommendations based on goals

---

**Skill Version:** v1.0.0
**Created:** 2025-01-28
**Status:** stable
**Engine Integration:** KnowledgeEngine (read-only)
**Permissions:** access_kb
