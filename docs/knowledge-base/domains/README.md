<think>
Okay, I need to create a comprehensive README.md for the 'domains' folder in the docs/knowledge-base directory. Let me start by understanding the user's request. They want three main sections: an overview of the 'domains' domain, brief descriptions of each subdirectory/file based on their names, and a structure to help developers navigate.

First, the overview. The 'domains' folder likely organizes different areas of expertise within the quantitative trading system. Each subdirectory represents a key domain. I should explain that this is where specialized knowledge is stored, covering various aspects from foundational concepts to specific areas like options or risk management.

Next, describing each subdirectory. Let's list them: execution, foundations, options, references, risk, signals, strategy. For each, I need to infer what they might contain based on the name. Execution probably deals with trade execution strategies, algorithms, and systems. Foundations would be the basic concepts and theories. Options would cover options trading specifics, like models and pricing. References might include external resources or standards. Risk would handle risk management techniques. Signals would be about generating trading signals. Strategy would cover different trading strategies and their implementation.

Then, the structure. The user wants a navigation guide. Maybe list each subdirectory with a brief summary and perhaps the order in which a developer should explore them. Starting with foundations, then moving to specific domains like signals, strategy, execution, risk, options, and references as needed.

I should make sure the README is clear and helpful. Use headings, bullet points, and maybe a table of contents. But the user didn't mention a TOC, so maybe just sections with headers. Also, avoid markdown fences as per the instruction. Need to ensure that the output is only the markdown content without any extra text. Let me check the example response to see the structure. They used # for main headings, ## for subsections, and bullet points. I'll follow that.

Wait, the user said "Output ONLY the markdown content. Do not include ```markdown``` fences." So I need to write the markdown without any code blocks. Just plain text with markdown syntax.

Let me start drafting:

# Domains Knowledge Base

## Overview
Explain that this is the core knowledge areas for the trading system, covering various domains. Each subdirectory is a specialized area.

## Subdirectory Descriptions
List each subdirectory with a brief description. Make sure to infer based on the name. For example, 'execution' would be about executing trades, algorithms, etc.

## Navigation Structure
Suggest a path for developers: start with foundations, then signals and strategy, then execution, risk, options, and references as needed.

Check for any possible mistakes. For example, 'references' might be documentation references or external links. 'signals' could be about generating trade signals. 'strategy' would be the different trading strategies implemented.

I should also make sure the language is professional but clear, as it's for a technical audience. Avoid jargon unless necessary, but since it's for a quantitative trading system, some terms are expected.

Putting it all together now. Make sure each section flows logically. Use bullet points for the subdirectories. Maybe bold the subdirectory names for emphasis. Alright, let's write the markdown.
</think>

# Domains Knowledge Base

## Overview
The `domains` directory serves as the core repository of specialized knowledge areas within the quantitative trading system. It organizes foundational concepts, domain-specific implementations, and strategic frameworks into discrete, navigable sections. This structure enables developers to contextualize system components, understand domain interactions, and reference technical details critical to system design, implementation, and maintenance.

## Subdirectory Descriptions
Each subdirectory corresponds to a distinct domain of expertise, inferred from its name and role in quantitative trading systems:

- **`execution`**
  Contains documentation on trade execution mechanisms, including algorithms (e.g., VWAP, TWAP), latency optimization, order routing, and integration with exchanges or brokers.

- **`foundations`**
  Covers core theoretical and technical prerequisites, such as probability theory, statistical models, time-series analysis, and mathematical frameworks underpinning quantitative trading.

- **`options`**
  Focuses on options trading specifics, including pricing models (Black-Scholes, binomial), Greeks, volatility surfaces, arbitrage strategies, and derivatives-related risk management.

- **`references`**
  Houses external references, standards, and compliance documentation, such as regulatory requirements, industry whitepapers, and third-party API specifications.

- **`risk`**
  Details risk management methodologies, including value-at-risk (VaR), stress testing, portfolio diversification, position sizing, and real-time risk monitoring systems.

- **`signals`**
  Explains signal generation frameworks, including technical indicators (RSI, MACD), machine learning models, sentiment analysis, and backtesting of signal efficacy.

- **`strategy`**
  Documents trading strategies (e.g., mean reversion, momentum, market-making) and their implementation, including logic, parameters, and performance metrics.

## Navigation Structure
Developers should navigate this section in the following order to build contextual understanding:

1. **Start with `foundations`** to grasp the mathematical and statistical underpinnings of the system.
2. **Proceed to `signals` and `strategy`** to understand how trading decisions are generated and executed.
3. **Explore `execution`** to learn about the mechanics of trade implementation and infrastructure.
4. **Review `risk`** to integrate risk constraints and safeguards into development work.
5. **Refer to `options`** for domain-specific details if working with derivatives.
6. **Use `references`** for compliance, external integrations, or deeper dives into cited methodologies.

This progression ensures alignment with the system’s design philosophy: from theory to implementation, with risk and execution as cross-cutting concerns.
