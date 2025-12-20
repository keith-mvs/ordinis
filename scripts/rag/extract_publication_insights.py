#!/usr/bin/env python
"""
Extract Strategic Insights from Research Publications.

Uses Synapse RAG engine to query indexed publications and synthesize
actionable recommendations for strategy, position sizing, and portfolio management.
"""

import asyncio
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loguru import logger


async def extract_insights():
    """Extract insights from publications using Synapse."""

    # Initialize engines
    logger.info("Initializing Helix and Synapse engines...")

    from ordinis.ai.helix import Helix
    from ordinis.ai.helix.config import HelixConfig
    from ordinis.ai.synapse import Synapse
    from ordinis.ai.synapse.config import SynapseConfig
    from ordinis.ai.synapse.models import RetrievalContext, SearchScope

    helix_config = HelixConfig()
    helix = Helix(config=helix_config)
    await helix.initialize()

    synapse_config = SynapseConfig()
    synapse = Synapse(helix=helix, config=synapse_config)
    await synapse.initialize()

    logger.info("Engines initialized. Querying publications...")

    # Define research queries for each domain
    queries = {
        "strategy_logic": [
            "momentum strategy improvements alpha generation",
            "mean reversion trading optimal entry exit signals",
            "technical indicator combinations profitable trading",
            "machine learning trading strategy feature selection",
            "market regime detection adaptive strategies",
        ],
        "position_sizing": [
            "optimal position sizing Kelly criterion risk adjusted",
            "volatility-based position sizing ATR methods",
            "portfolio allocation risk parity equal risk contribution",
            "dynamic position sizing market conditions",
            "bet sizing under uncertainty optimal f",
        ],
        "portfolio_optimization": [
            "mean variance optimization improvements robust",
            "factor investing portfolio construction",
            "risk budgeting portfolio optimization",
            "black litterman model views integration",
            "hierarchical risk parity clustering",
        ],
        "risk_management": [
            "drawdown control risk management trading",
            "value at risk expected shortfall tail risk",
            "correlation regime changes portfolio risk",
            "stop loss optimization trailing stops",
            "backtest overfitting detection prevention",
        ],
        "execution": [
            "optimal execution algorithms VWAP TWAP",
            "transaction costs market impact models",
            "order flow analysis execution timing",
            "slippage estimation trade execution",
        ],
    }

    insights = {}

    for domain, domain_queries in queries.items():
        logger.info(f"Extracting insights for: {domain}")
        domain_snippets = []

        for query in domain_queries:
            result = synapse.search_publications(
                query=query,
                top_k=3,
            )

            for snippet in result.snippets:
                if snippet.score > 0.75:  # High relevance only
                    domain_snippets.append(
                        {
                            "query": query,
                            "source": snippet.source,
                            "score": snippet.score,
                            "text": snippet.text[:500],  # Truncate for display
                        }
                    )

        insights[domain] = domain_snippets
        logger.info(f"  Found {len(domain_snippets)} relevant snippets")

    # Generate synthesis report
    print("\n" + "=" * 80)
    print("PUBLICATION INSIGHTS REPORT")
    print("=" * 80)

    total_snippets = sum(len(v) for v in insights.values())
    print(f"\nTotal relevant findings: {total_snippets} snippets from publications\n")

    for domain, snippets in insights.items():
        print(f"\n{'=' * 80}")
        print(f"## {domain.upper().replace('_', ' ')}")
        print("=" * 80)

        if not snippets:
            print("  No high-relevance findings.")
            continue

        # Group by source
        sources = {}
        for s in snippets:
            src = s["source"]
            if src not in sources:
                sources[src] = []
            sources[src].append(s)

        for source, src_snippets in sources.items():
            print(f"\n### Source: {source}")
            print(f"    Relevance: {max(s['score'] for s in src_snippets):.2f}")
            print(f"    Matching queries: {len(src_snippets)}")

            # Show best snippet
            best = max(src_snippets, key=lambda x: x["score"])
            print(f"\n    Key excerpt:")
            print(f"    {best['text'][:300]}...")

    # Now use Helix to synthesize recommendations
    print("\n" + "=" * 80)
    print("SYNTHESIZED RECOMMENDATIONS")
    print("=" * 80)

    # Build context from all insights
    context_parts = []
    for domain, snippets in insights.items():
        if snippets:
            context_parts.append(f"\n## {domain.upper()}")
            for s in snippets[:3]:  # Top 3 per domain
                context_parts.append(f"Source: {s['source']}\n{s['text']}\n")

    context_str = "\n".join(context_parts)

    synthesis_prompt = f"""
Based on the following research publication excerpts, provide actionable recommendations
for improving an algorithmic trading system. Focus on practical implementation guidance.

{context_str}

Provide recommendations in these categories:
1. STRATEGY LOGIC: Specific improvements to signal generation and entry/exit rules
2. POSITION SIZING: Methods to optimize bet sizes based on research findings
3. PORTFOLIO OPTIMIZATION: Techniques for better asset allocation
4. RISK MANAGEMENT: Controls and limits based on academic findings
5. EXECUTION: Order execution improvements

For each recommendation, cite the relevant research source.
Be specific and actionable - include formulas or parameters where applicable.
"""

    try:
        from ordinis.ai.helix.models import ChatMessage

        response = await helix.generate(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a quantitative finance expert synthesizing research findings into actionable trading system improvements.",
                ),
                ChatMessage(role="user", content=synthesis_prompt),
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        print(response.content)

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        print("\n[Manual synthesis from findings]")

        # Fallback: print key findings summary
        print("\nKey findings from publications:")
        for domain, snippets in insights.items():
            if snippets:
                print(f"\n{domain.upper()}:")
                unique_sources = {s["source"] for s in snippets}
                for src in list(unique_sources)[:3]:
                    print(f"  - {src}")

    # Save report
    report_path = PROJECT_ROOT / "artifacts" / "reports" / "publication_insights_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Publication Insights Report\n\n")
        f.write(f"**Generated:** {__import__('datetime').datetime.now().isoformat()}\n\n")
        f.write("## Summary\n\n")
        f.write(f"Total relevant findings: {total_snippets} snippets\n\n")

        for domain, snippets in insights.items():
            f.write(f"## {domain.replace('_', ' ').title()}\n\n")
            if snippets:
                sources = {s["source"] for s in snippets}
                f.write(f"Sources: {len(sources)} publications\n\n")
                for s in snippets[:5]:
                    f.write(f"### {s['source']} (score: {s['score']:.2f})\n")
                    f.write(f"Query: {s['query']}\n\n")
                    f.write(f"```\n{s['text'][:400]}...\n```\n\n")
            else:
                f.write("No high-relevance findings.\n\n")

    print(f"\n\nReport saved to: {report_path}")

    await helix.shutdown()
    await synapse.shutdown()


if __name__ == "__main__":
    asyncio.run(extract_insights())
