 # Married-Put Strategy Skill Package
## Installation and Packaging Guide

## Directory Structure

```
married-put-strategy/
│
├── SKILL.md                           # Main skill documentation (REQUIRED)
├── reference.md                       # Advanced reference material
├── requirements.txt                   # Python dependencies
│
├── scripts/                           # Python implementation scripts
│   ├── married_put_calculator.py     # Core calculations engine
│   ├── strike_comparison.py          # Strike price analysis
│   ├── expiration_analysis.py        # Expiration cycle comparison
│   ├── position_sizer.py             # Position sizing logic
│   └── visualizations.py             # Plotting and charts
│
└── examples/                          # Sample data and examples
    ├── sample_positions.csv           # 5 real-world scenarios
    └── README.md                      # Examples documentation
```

## Installation Instructions

### For Claude Code (Desktop/CLI)

#### Option 1: Project-Level Skill (Recommended for teams)

1. Navigate to your project directory:
   ```bash
   cd /path/to/your/project
   ```

2. Create skills directory if it doesn't exist:
   ```bash
   mkdir -p .claude/skills
   ```

3. Copy the skill package:
   ```bash
   cp -r married-put-strategy .claude/skills/
   ```

4. Verify installation:
   ```bash
   ls .claude/skills/married-put-strategy/SKILL.md
   ```

5. Start Claude Code in your project:
   ```bash
   cd /path/to/your/project
   claude
   ```

The skill will be automatically available to all team members who clone the repository.

#### Option 2: Personal Skill (For individual use)

1. Copy to your personal skills directory:
   ```bash
   cp -r married-put-strategy ~/.claude/skills/
   ```

2. Verify installation:
   ```bash
   ls ~/.claude/skills/married-put-strategy/SKILL.md
   ```

3. Start Claude Code from anywhere:
   ```bash
   claude
   ```

The skill will be available in all your projects.

### For Claude.ai (Web Interface)

1. **Create a ZIP file** of the skill package:
   ```bash
   cd married-put-strategy
   zip -r ../married-put-strategy.zip .
   cd ..
   ```

2. **Upload to Claude.ai**:
   - Go to https://claude.ai
   - Navigate to Settings → Features
   - Click "Upload Custom Skill"
   - Select `married-put-strategy.zip`
   - Confirm upload

3. **Verify Installation**:
   - Start a new conversation
   - Ask: "What skills are available?"
   - Confirm "married-put-strategy" appears in the list

**Note**: Custom skills in Claude.ai are per-user, not shared organization-wide.

### For Anthropic API

1. **Create a ZIP file** (if not already created):
   ```bash
   zip -r married-put-strategy.zip married-put-strategy/
   ```

2. **Upload via API**:
   ```python
   import anthropic

   client = anthropic.Anthropic(api_key="your-api-key")

   # Upload skill
   with open("married-put-strategy.zip", "rb") as f:
       skill = client.skills.create(
           name="married-put-strategy",
           file=f
       )

   print(f"Skill uploaded: {skill.id}")
   ```

3. **Use in API calls**:
   ```python
   response = client.messages.create(
       model="claude-sonnet-4-5-20250929",
       max_tokens=4096,
       tools=[{
           "type": "code_execution_20250125",
           "container": {
               "skills": ["married-put-strategy"]
           }
       }],
       messages=[{
           "role": "user",
           "content": "Analyze a married-put position..."
       }]
   )
   ```

## Installing Python Dependencies

After installing the skill, install required Python packages:

### Using pip (Standard)

```bash
pip install -r married-put-strategy/requirements.txt
```

### Using pip with virtual environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r married-put-strategy/requirements.txt
```

### Using conda

```bash
conda create -n options-trading python=3.8
conda activate options-trading
pip install -r married-put-strategy/requirements.txt
```

## Verifying Installation

### Test the skill in Claude Code:

```
Ask Claude:

"I own 100 shares of a stock trading at $50.
Analyze a married-put with a $48 strike and $2.50 premium."
```

Claude should automatically use the married-put-strategy skill.

### Test Python scripts directly:

```bash
cd married-put-strategy/scripts
python married_put_calculator.py
```

You should see example calculations printed to console.

## Packaging for Distribution

### Create ZIP for Claude.ai or API:

```bash
cd married-put-strategy
zip -r ../married-put-strategy.zip . -x "*.pyc" -x "__pycache__/*" -x ".DS_Store"
```

This creates a clean ZIP file excluding Python cache files.

### Create shareable package for GitHub:

```bash
# If distributing via git
cd married-put-strategy
git init
git add .
git commit -m "Initial commit: Married-put strategy skill"
```

Team members can then clone and use:
```bash
git clone <repository-url>
cp -r married-put-strategy ~/.claude/skills/
```

## Updating the Skill

### For Claude Code:

Simply edit files in place:
```bash
# Edit SKILL.md or any script
code ~/.claude/skills/married-put-strategy/SKILL.md

# Restart Claude Code to load changes
```

### For Claude.ai:

1. Make changes to local files
2. Create new ZIP file
3. Re-upload via Settings → Features
4. Previous version will be replaced

### For API:

```python
# Delete old version
client.skills.delete(skill_id="old-skill-id")

# Upload new version
with open("married-put-strategy.zip", "rb") as f:
    new_skill = client.skills.create(
        name="married-put-strategy",
        file=f
    )
```

## Troubleshooting

### Skill not appearing in Claude Code

**Check installation location**:
```bash
# Project skill
ls .claude/skills/married-put-strategy/SKILL.md

# Personal skill
ls ~/.claude/skills/married-put-strategy/SKILL.md
```

**Verify YAML frontmatter**:
```bash
head -10 married-put-strategy/SKILL.md
```

Should see:
```yaml
---
name: married-put-strategy
description: Analyzes and implements...
---
```

**Restart Claude Code**:
```bash
# Exit Claude Code (Ctrl+C or type 'exit')
# Restart
claude
```

### Python import errors

**Check dependencies**:
```bash
pip list | grep numpy
pip list | grep pandas
pip list | grep matplotlib
```

**Reinstall if needed**:
```bash
pip install --upgrade -r married-put-strategy/requirements.txt
```

### Claude not using skill automatically

**Make request more specific**:
Instead of: "Calculate options"
Try: "Analyze married-put with stock at $50, strike $48, premium $2.50"

**Verify skill description matches use case**:
The description includes keywords: "married-put", "protective put", "downside insurance", "options strategies"

## Support and Updates

### Check for updates:
Visit the ordinis-1 project repository for latest version and updates.

### Report issues:
Include:
- Claude Code version
- Python version (`python --version`)
- Error messages
- Steps to reproduce

### Community:
Share your strategies and improvements via pull requests or issues on GitHub.

## License

This skill package is part of the ordinis-1 project.
See project LICENSE file for terms and conditions.

## Version History

- **v1.0.0** (2025-12-10): Initial release
  - Core married-put calculations
  - Strike and expiration comparison
  - Position sizing tools
  - 5 example scenarios
  - Complete documentation

---

**Need Help?**

1. Check the examples: `examples/README.md`
2. Read the reference: `reference.md`
3. Test scripts directly: `python scripts/married_put_calculator.py`
4. Ask Claude Code: "Show me how to use the married-put skill"
