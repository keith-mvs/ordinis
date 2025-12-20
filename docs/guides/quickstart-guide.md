# Quick Start Guide for Skills

Get up and running with your options strategy skill in 5 minutes.

---

## 1. Install

### Claude Code - Project Level (Recommended)
```bash
cd your-project
mkdir -p .claude/skills
cp -r [strategy-name] .claude/skills/
pip install -r .claude/skills/[strategy-name]/requirements.txt
```

### Claude Code - Global
```bash
mkdir -p ~/.claude/skills
cp -r [strategy-name] ~/.claude/skills/
pip install -r ~/.claude/skills/[strategy-name]/requirements.txt
```

### Claude.ai - Upload Skill
1. Navigate to Settings → Skills
2. Click "Upload Skill"
3. Select the skill directory
4. Confirm installation

---

## 2. Verify Installation

Start Claude and ask:
```
What skills are available?
```

You should see your strategy skill listed.

---

## 3. Basic Usage Examples

### Example 1: Simple Position Analysis
```
Analyze a [strategy-name] position on [TICKER] at [PRICE]
with [strategy-specific parameters]
```

**Claude will automatically**:
- Load the skill
- Calculate position metrics
- Provide breakeven, max profit, max loss
- Generate recommendations

### Example 2: Strike Comparison
```
Compare different strike selections for [strategy-name] on [TICKER]
```

**Claude will**:
- Analyze multiple strike combinations
- Compare risk/reward profiles
- Provide recommendations based on your criteria

### Example 3: Position Sizing
```
How many contracts should I use for a [strategy-name] on [TICKER]
with $[AMOUNT] available capital?
```

**Claude will**:
- Calculate optimal position size
- Consider risk parameters
- Provide contract recommendations

---

## 4. Common Commands

### Calculate Metrics
```python
from scripts.[strategy]_calculator import [StrategyName]

position = [StrategyName](
    underlying_symbol="SPY",
    underlying_price=450,
    # [strategy-specific parameters]
)

print(f"Max Profit: ${position.max_profit:.2f}")
print(f"Max Loss: ${position.max_loss:.2f}")
print(f"Breakeven: ${position.breakeven_price:.2f}")
```

### Generate Visualization
```python
from scripts.visualizations import plot_payoff_diagram

plot_payoff_diagram(position, save_path="payoff.png")
```

---

## 5. Real-World Workflow

**Step 1**: Identify opportunity
```
"Find [strategy-name] opportunities on [TICKER]"
```

**Step 2**: Analyze position
```
"Analyze this [strategy-name]: [parameters]"
```

**Step 3**: Compare alternatives
```
"Compare this with [alternative strikes/expirations]"
```

**Step 4**: Check risk controls
```
"Validate this position against my risk limits"
```

**Step 5**: Get execution guidance
```
"What's the best way to execute this position?"
```

---

## 6. Tips for Best Results

### Be Specific
❌ "Analyze options"
✅ "Analyze a bull-put-spread on TGT at $152.50, sell 145 put, buy 140 put, 60 DTE"

### Provide Complete Information
Include:
- Ticker symbol
- Current stock price
- Strike prices (all legs)
- Expiration date or DTE
- Current IV (if known)
- Your capital allocation

### Use Natural Language
Claude understands conversational requests:
```
"I'm bullish on AAPL but want downside protection.
Show me a married put with $10,000 capital."
```

---

## 7. Troubleshooting

### Skill not loading?

**Check installation**:
```bash
ls -la ~/.claude/skills/[strategy-name]/
```

**Verify SKILL.md exists**:
```bash
cat ~/.claude/skills/[strategy-name]/SKILL.md
```

### Import errors?

**Install dependencies**:
```bash
pip install -r ~/.claude/skills/[strategy-name]/requirements.txt
```

**Check Python version** (3.11+ recommended):
```bash
python --version
```

### Skill triggers incorrectly?

Check the `description` field in SKILL.md frontmatter. It should include:
- What the skill does
- When to use it (specific triggers)
- Key terms that should invoke it

---

## Next Steps

- See [installation-guide.md](installation-guide.md) for advanced setup
- See [developer-guide.md](developer-guide.md) for customization
- See strategy-specific references for detailed implementation

---

**This guide provides the essentials**. For comprehensive documentation, refer to the specific reference files in the skill package.
