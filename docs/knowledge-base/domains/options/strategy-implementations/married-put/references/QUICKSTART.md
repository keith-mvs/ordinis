# Married-Put Strategy - Quick Start Guide

Get up and running with the married-put strategy skill in 5 minutes.

## 1. Install (Choose One)

### Claude Code - Project Level (Recommended)
```bash
cd your-project
mkdir -p .claude/skills
cp -r married-put-strategy .claude/skills/
pip install -r .claude/skills/married-put-strategy/requirements.txt
```

### Claude Code - Personal
```bash
cp -r married-put-strategy ~/.claude/skills/
pip install -r ~/.claude/skills/married-put-strategy/requirements.txt
```

### Claude.ai
```bash
zip -r married-put-strategy.zip married-put-strategy/
# Then upload via Settings → Features in Claude.ai
```

## 2. Verify Installation

Start Claude and ask:
```
What skills are available?
```

You should see **married-put-strategy** in the list.

## 3. Basic Usage Examples

### Example 1: Simple Position Analysis
```
I own 100 shares of a stock trading at $45.
Analyze a married-put with a $43 strike and $2.10 premium.
```

**Claude will automatically**:
- Calculate breakeven price
- Determine maximum loss
- Show protection percentage
- Display total cost

### Example 2: Compare Strike Prices
```
Stock at $50, 100 shares. Compare these puts:
- $45 strike, $0.85 premium
- $48 strike, $1.75 premium
- $50 strike, $2.90 premium
- $52 strike, $4.30 premium

Which offers best value?
```

**Claude will**:
- Create comparison table
- Calculate cost vs protection
- Recommend optimal strike
- Show payoff diagrams (if visualizations enabled)

### Example 3: Expiration Analysis
```
Stock: $52.75, Strike: $50
Compare 30-day ($1.85), 60-day ($2.95), and 90-day ($3.85) puts.
What's the most cost-effective expiration?
```

**Claude will**:
- Calculate monthly costs
- Show annual costs for rolling strategy
- Recommend best expiration cycle
- Explain trade-offs

### Example 4: Position Sizing
```
I have $25,000 capital. Stock is $48.25, put premium is $2.15.
How many shares should I buy with protection?
```

**Claude will**:
- Determine optimal shares (rounded to contracts)
- Calculate total cost
- Show remaining capital
- Recommend position size

### Example 5: Load Example Scenarios
```
Load and analyze the sample positions from examples/sample_positions.csv.
Show me the SNAP position analysis.
```

**Claude will**:
- Read the CSV data
- Analyze the high-volatility small-cap scenario
- Calculate all metrics
- Explain the trade-offs

## 4. Common Commands

### Calculate Metrics
```python
from scripts.married_put_calculator import MarriedPut

position = MarriedPut(
    stock_price=45.00,
    shares=100,
    put_strike=43.00,
    put_premium=2.10
)

print(f"Breakeven: ${position.breakeven_price:.2f}")
print(f"Max Loss: ${position.max_loss:.2f}")
```

### Generate Payoff Diagram
```python
from scripts.visualizations import plot_married_put_payoff

plot_married_put_payoff(
    stock_price=45.00,
    put_strike=43.00,
    put_premium=2.10,
    save_path="diagram.png"
)
```

### Compare Strikes
```python
from scripts.strike_comparison import compare_strike_prices

results = compare_strike_prices(
    stock_price=50.00,
    shares=100,
    strike_prices=[45, 48, 50, 52],
    premiums=[0.85, 1.75, 2.90, 4.30]
)

print(results.calculate_metrics())
```

## 5. Real-World Workflow

**Step 1**: Identify your position
```
Stock: AAPL at $175.50
Shares: 100
Capital: $18,000
```

**Step 2**: Research strikes (ask Claude)
```
AAPL at $175.50. Show me OTM, ATM, and ITM put options
for 60-day expiration. What's the trade-off?
```

**Step 3**: Calculate metrics (ask Claude)
```
Using the $170 strike at $6.25 premium, calculate:
- Breakeven point
- Maximum loss
- Protection percentage
- Total cost
```

**Step 4**: Visualize payoff (ask Claude)
```
Create a payoff diagram for this position showing
profit/loss from $150 to $200 stock prices.
```

**Step 5**: Monitor and roll (future)
```
My put has 25 days left. Should I roll to a new 60-day?
Current put worth $3.20, new 60-day $170 put costs $5.80.
```

## 6. Tips for Best Results

### Be Specific
❌ "Analyze options"
✓ "Analyze married-put: Stock $50, Strike $48, Premium $2.50"

### Provide Complete Information
Include:
- Stock price
- Number of shares (or capital available)
- Put strike price
- Put premium
- Days to expiration (optional but helpful)

### Ask for Comparisons
"Compare these three strikes..." gets better analysis than single queries.

### Request Visualizations
"Show me a payoff diagram" or "Create a chart comparing these options"

### Reference Examples
"Use the TDOC example from sample_positions.csv as a starting point"

## 7. File Organization

```
married-put-strategy/
├── SKILL.md              ← Main documentation (Claude reads this)
├── reference.md          ← Advanced Greeks, formulas
├── requirements.txt      ← Python dependencies
├── INSTALLATION.md       ← Full installation guide
├── QUICKSTART.md         ← This file
├── scripts/              ← Implementation code
│   ├── married_put_calculator.py
│   ├── strike_comparison.py
│   ├── expiration_analysis.py
│   ├── position_sizer.py
│   └── visualizations.py
└── examples/             ← Sample data
    ├── sample_positions.csv
    └── README.md
```

## 8. Next Steps

### For Basic Usage
- Read: `SKILL.md` (Sections: Quick Start, Strike Comparison, P/L Analysis)
- Try: Examples 1-3 above
- Run: `python scripts/married_put_calculator.py`

### For Advanced Usage
- Read: `reference.md` (Greeks, Volatility, Tax Implications)
- Study: `examples/README.md` and sample_positions.csv
- Experiment: Create your own position scenarios

### For Integration
- API: See `INSTALLATION.md` - Anthropic API section
- Automation: Write scripts using the calculator classes
- Portfolio: Use position_sizer.py for multiple holdings

## 9. Troubleshooting

**Skill not loading?**
```bash
# Check location
ls .claude/skills/married-put-strategy/SKILL.md

# Restart Claude Code
claude
```

**Import errors?**
```bash
pip install --upgrade numpy pandas matplotlib scipy
```

**Claude not using skill?**
- Be more specific in your query
- Mention "married-put" explicitly
- Reference strike prices and premiums

## 10. Support

- **Documentation**: See `reference.md` for detailed formulas
- **Examples**: Check `examples/README.md` for 5 scenarios
- **Installation**: Full guide in `INSTALLATION.md`
- **Updates**: Check ordinis-1 project for latest version

---

**Start Simple**: Try Example 1 above, then gradually explore more features.

**Questions?** Ask Claude: "Explain married-put strategy" or "How do I use the position sizing tool?"

Happy Trading! 📈
