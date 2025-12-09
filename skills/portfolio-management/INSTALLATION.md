# Installing the Portfolio Management Skill

## Quick Installation

The portfolio-management skill has been packaged and is ready to install.

### Step 1: Download the Skill Package

The skill package file is available in your Claude session outputs:
- **File**: `portfolio-management.skill`  
- **Size**: ~115KB (compressed)
- **Contains**: 5 Python scripts + 3 reference markdown files

### Step 2: Extract to Project

The `.skill` file is actually a ZIP archive. To extract it to your project:

#### Option A: Using File Explorer
1. Locate `portfolio-management.skill` in your downloads/outputs
2. Rename the file from `.skill` to `.zip`
3. Right-click → Extract All
4. Copy the extracted `portfolio-management` folder to:
   ```
   C:\Users\kjfle\.projects\intelligent-investor\skills\portfolio-management\
   ```

#### Option B: Using PowerShell
```powershell
# Navigate to where you downloaded the skill file
cd C:\Users\kjfle\Downloads

# Extract the skill (it's a zip file)
Expand-Archive -Path portfolio-management.skill -DestinationPath "C:\Users\kjfle\.projects\intelligent-investor\skills\portfolio-management" -Force

# Verify extraction
Get-ChildItem "C:\Users\kjfle\.projects\intelligent-investor\skills\portfolio-management" -Recurse
```

#### Option C: Using Python
```python
import zipfile
import os

skill_file = "path/to/portfolio-management.skill"
dest_dir = r"C:\Users\kjfle\.projects\intelligent-investor\skills\portfolio-management"

with zipfile.ZipFile(skill_file, 'r') as zip_ref:
    zip_ref.extractall(dest_dir)

print(f"Extracted to {dest_dir}")
```

### Step 3: Verify Installation

After extraction, your directory should look like this:

```
C:\Users\kjfle\.projects\intelligent-investor\skills\portfolio-management\
├── SKILL.md                    # Main documentation (19KB)
├── README.md                   # This installation guide
├── scripts\
│   ├── broker_connector.py     # 16KB - Broker API integrations
│   ├── market_data.py          # 14KB - Market data providers
│   ├── portfolio_analytics.py  # 14KB - Performance & risk analytics
│   ├── tax_tracker.py          # 14KB - Tax lot accounting
│   └── dividend_tracker.py     # 14KB - Dividend tracking
└── references\
    ├── formulas.md             # 5KB - Mathematical formulas
    ├── benchmarks.md           # 8KB - Market indices
    └── sectors.md              # 10KB - GICS sector classification
```

### Step 4: Install Dependencies

```bash
# Navigate to your project
cd C:\Users\kjfle\.projects\intelligent-investor

# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Install required packages
pip install pandas numpy alpaca-trade-api ib_insync yfinance alpha_vantage polygon-api-client

# Optional: Install visualization packages
pip install matplotlib plotly openpyxl
```

### Step 5: Configure API Keys

Create or update your `.env` file in the project root:

```bash
# Broker APIs
BROKER_API_KEY=your_alpaca_key_here
BROKER_API_SECRET=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# TD Ameritrade (if using)
TDA_REFRESH_TOKEN=your_tda_refresh_token
TDA_ACCOUNT_ID=your_tda_account_id

# Market Data APIs
ALPHA_VANTAGE_API_KEY=your_alphavantage_key
POLYGON_API_KEY=your_polygon_key
```

### Step 6: Test the Installation

Create a test script to verify everything works:

```python
# test_portfolio_skill.py
import sys
sys.path.append('skills/portfolio-management')

from scripts.broker_connector import create_connector
from scripts.market_data import create_provider
from scripts.portfolio_analytics import PortfolioAnalytics

print("✓ All imports successful!")

# Test market data (no API key needed for Yahoo Finance)
provider = create_provider('yahoo')
print("✓ Market data provider created")

# Test analytics
analytics = PortfolioAnalytics()
print("✓ Portfolio analytics initialized")

print("\nPortfolio Management Skill installed successfully!")
```

Run the test:
```bash
python test_portfolio_skill.py
```

## What's Next?

1. **Read the Documentation**: Open `SKILL.md` for complete API reference
2. **Review Examples**: See Quick Start Workflows in SKILL.md
3. **Check References**: Explore `references/` for formulas and benchmarks
4. **Start Building**: Integrate the scripts into your intelligent-investor workflows

## Common Issues

### Issue: Import errors
**Solution**: Make sure you've installed all dependencies and the skill is in the correct path

### Issue: API connection failures
**Solution**: Verify your API keys are set in .env and you're using the correct base URLs

### Issue: Missing dependencies
**Solution**: Run `pip install -r requirements.txt` if you create one, or install packages individually

## Project Integration

The skill is now part of your intelligent-investor project:

```python
# In your project code
from skills.portfolio_management.scripts.broker_connector import create_connector
from skills.portfolio_management.scripts.portfolio_analytics import PortfolioAnalytics

# Use in your analysis workflows
connector = create_connector('alpaca')
analytics = PortfolioAnalytics()
```

---

**Note**: The skill package is also available for import directly into Claude via Settings > Skills if you prefer to use it that way instead of or in addition to the project installation.
