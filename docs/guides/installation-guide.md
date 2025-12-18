# Installation Guide for Options Strategy Skills

Comprehensive installation instructions for different Claude environments.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Claude Code Installation](#claude-code-installation)
- [Claude.ai Installation](#claudeai-installation)
- [Anthropic API Integration](#anthropic-api-integration)
- [Environment Setup](#environment-setup)
- [Verification](#verification)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **OS**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Disk Space**: 500MB for skill + dependencies

### Required Knowledge
- Basic command-line usage
- Python package management (pip)
- Git (optional, for cloning repositories)

### Python Packages
All required packages are listed in `requirements.txt`:
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0

---

## Claude Code Installation

### Method 1: Project-Level Installation (Recommended)

Install skill for a specific project:

```bash
# Navigate to your project
cd /path/to/your/project

# Create skills directory
mkdir -p .claude/skills

# Copy skill package
cp -r [strategy-name] .claude/skills/

# Install dependencies
cd .claude/skills/[strategy-name]
pip install -r requirements.txt
```

**Advantages**:
- Isolated per project
- No conflicts between projects
- Easy to version control

### Method 2: Global Installation

Install skill for all Claude Code sessions:

```bash
# Create global skills directory
mkdir -p ~/.claude/skills

# Copy skill package
cp -r [strategy-name] ~/.claude/skills/

# Install dependencies
cd ~/.claude/skills/[strategy-name]
pip install -r requirements.txt
```

**Advantages**:
- Available in all projects
- Install once, use everywhere

### Method 3: Virtual Environment (Advanced)

Use project-specific virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Create skills directory
mkdir -p .claude/skills
cp -r [strategy-name] .claude/skills/

# Install dependencies in venv
pip install -r .claude/skills/[strategy-name]/requirements.txt
```

**Advantages**:
- Complete isolation
- Reproducible environments
- Version control for dependencies

---

## Claude.ai Installation

### Upload Skill Package

1. **Prepare Skill Package**:
   ```bash
   # Navigate to skill directory
   cd [strategy-name]

   # Verify SKILL.md exists
   ls -la SKILL.md
   ```

2. **Access Skills Settings**:
   - Log in to Claude.ai
   - Navigate to Settings → Skills
   - Click "Upload New Skill"

3. **Upload Files**:
   - Select the entire skill directory
   - Or upload individual files (SKILL.md, scripts/, references/)
   - Confirm upload

4. **Verify Installation**:
   - Ask Claude: "What skills do I have?"
   - Look for your strategy skill in the list

### Skill Updates

To update an existing skill:

1. Navigate to Settings → Skills
2. Find the skill in your list
3. Click "Update"
4. Upload new version
5. Confirm replacement

---

## Anthropic API Integration

### Using Skills Programmatically

#### Python SDK Example

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

# Load skill file
with open('[strategy-name]/SKILL.md', 'r') as f:
    skill_content = f.read()

# Create message with skill context
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    system=[
        {
            "type": "text",
            "text": skill_content,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{
        "role": "user",
        "content": "Analyze a bull-call-spread on SPY..."
    }]
)
```

#### TypeScript SDK Example

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { readFileSync } from 'fs';

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// Load skill
const skillContent = readFileSync('[strategy-name]/SKILL.md', 'utf-8');

const message = await client.messages.create({
  model: 'claude-sonnet-4-5-20250929',
  max_tokens: 4096,
  system: [
    {
      type: 'text',
      text: skillContent,
      cache_control: { type: 'ephemeral' }
    }
  ],
  messages: [{
    role: 'user',
    content: 'Analyze a bull-call-spread on SPY...'
  }]
});
```

### Skill Caching

Use prompt caching to reduce costs when using skills repeatedly:

```python
# First call: Skill content is cached
# Subsequent calls: Skill read from cache (cheaper)

system_content = [
    {
        "type": "text",
        "text": skill_content,
        "cache_control": {"type": "ephemeral"}  # Cache for 5 minutes
    }
]
```

---

## Environment Setup

### Environment Variables

Create `.env` file for configuration:

```bash
# API Keys (if needed)
ANTHROPIC_API_KEY=your-api-key-here
ALPACA_API_KEY=your-alpaca-key
ALPACA_SECRET_KEY=your-alpaca-secret

# Python Configuration
PYTHONPATH=/path/to/skills

# Skill-Specific Settings
DEFAULT_IV=0.20
DEFAULT_RISK_FREE_RATE=0.05
TRANSACTION_COST=0.65
```

### Path Configuration

Add skills directory to Python path:

**Linux/macOS** (`.bashrc` or `.zshrc`):
```bash
export PYTHONPATH="${PYTHONPATH}:${HOME}/.claude/skills"
```

**Windows** (PowerShell profile):
```powershell
$env:PYTHONPATH += ";$HOME\.claude\skills"
```

### IDE Integration

#### VS Code

Create `.vscode/settings.json`:
```json
{
  "python.analysis.extraPaths": [
    "${workspaceFolder}/.claude/skills"
  ],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true
}
```

#### PyCharm

1. File → Settings → Project Structure
2. Add Content Root: `.claude/skills`
3. Mark as Sources

---

## Verification

### Test Installation

```bash
# Test skill directory exists
ls -la ~/.claude/skills/[strategy-name]

# Test SKILL.md is valid
head -20 ~/.claude/skills/[strategy-name]/SKILL.md

# Test Python imports
python -c "from scripts.[strategy]_calculator import [StrategyName]; print('OK')"

# Test dependencies
python -c "import numpy, pandas, matplotlib, scipy; print('All packages installed')"
```

### Test with Claude

Start Claude Code and test:

```
Query: "What skills are available?"
Expected: Your strategy skill should be listed

Query: "Explain the [strategy-name] skill"
Expected: Claude should describe the skill's capabilities

Query: "Analyze a sample [strategy-name] position"
Expected: Claude should use the skill to perform analysis
```

### Validation Script

Create `test_installation.py`:

```python
#!/usr/bin/env python3
"""Test skill installation"""

import sys
from pathlib import Path

def test_installation():
    """Verify skill is correctly installed."""

    # Test 1: SKILL.md exists
    skill_path = Path.home() / '.claude' / 'skills' / '[strategy-name]'
    skill_md = skill_path / 'SKILL.md'

    if not skill_md.exists():
        print("❌ SKILL.md not found")
        return False
    print("✅ SKILL.md found")

    # Test 2: Scripts directory exists
    scripts_dir = skill_path / 'scripts'
    if not scripts_dir.exists():
        print("❌ scripts/ directory not found")
        return False
    print("✅ scripts/ directory found")

    # Test 3: Import main calculator
    try:
        sys.path.insert(0, str(skill_path))
        from scripts.[strategy]_calculator import [StrategyName]
        print("✅ Calculator module imports successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 4: Dependencies
    try:
        import numpy, pandas, matplotlib, scipy
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

    print("\n✅ Installation successful!")
    return True

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
```

Run validation:
```bash
python test_installation.py
```

---

## Configuration Options

### Skill Configuration

Edit skill behavior by modifying `SKILL.md` frontmatter:

```yaml
---
name: [strategy-name]
description: [Customize when skill triggers]
---
```

### Script Configuration

Some skills include configuration files:

**config.json**:
```json
{
  "default_volatility": 0.20,
  "default_risk_free_rate": 0.05,
  "transaction_cost_per_contract": 0.65,
  "max_position_percentage": 0.25
}
```

### Customization

Create local overrides:

**local_config.json** (not version controlled):
```json
{
  "transaction_cost_per_contract": 0.50,
  "preferred_dte_ranges": [30, 45, 60]
}
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Skill Not Loading

**Symptoms**: Claude doesn't recognize the skill

**Solutions**:
```bash
# Verify SKILL.md location
ls ~/.claude/skills/[strategy-name]/SKILL.md

# Check YAML frontmatter
head -10 ~/.claude/skills/[strategy-name]/SKILL.md

# Restart Claude Code
```

#### Issue 2: Import Errors

**Symptoms**: `ModuleNotFoundError` when using scripts

**Solutions**:
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.11+

# Verify PYTHONPATH
echo $PYTHONPATH
```

#### Issue 3: Permission Errors

**Symptoms**: Cannot read/write skill files

**Solutions**:
```bash
# Fix permissions (Linux/macOS)
chmod -R 755 ~/.claude/skills/[strategy-name]

# Fix ownership
chown -R $USER ~/.claude/skills/[strategy-name]
```

#### Issue 4: Dependencies Conflict

**Symptoms**: Package version conflicts

**Solutions**:
```bash
# Use virtual environment
python -m venv skill-venv
source skill-venv/bin/activate
pip install -r requirements.txt

# Or use pip constraints
pip install -r requirements.txt --constraint constraints.txt
```

#### Issue 5: Script Execution Fails

**Symptoms**: Scripts run but produce errors

**Solutions**:
```bash
# Enable debugging
python -m pdb scripts/[strategy]_calculator.py

# Check for syntax errors
python -m py_compile scripts/[strategy]_calculator.py

# Verify data files exist
ls -la assets/
```

### Getting Help

If issues persist:

1. **Check Documentation**: Review `references/` files for specific guidance
2. **Test Imports**: Run `python -c "import [module]"` for each dependency
3. **Verify File Structure**: Ensure all required files are present
4. **Check Logs**: Look for error messages in terminal output
5. **Reinstall**: Remove and reinstall the skill from scratch

### Platform-Specific Notes

#### Windows
- Use PowerShell (not CMD) for installation
- Path separators: Use forward slashes in code, even on Windows
- Line endings: Git may convert CRLF/LF (configure `core.autocrlf`)

#### macOS
- May need to install Xcode Command Line Tools
- Use `python3` instead of `python` if both Python 2/3 installed

#### Linux
- May need to install build tools: `sudo apt-get install python3-dev build-essential`
- Some distributions require separate pip install: `sudo apt-get install python3-pip`

---

## Advanced Topics

### Skill Updates and Versioning

Track skill versions using Git:

```bash
cd ~/.claude/skills/[strategy-name]
git init
git add .
git commit -m "Initial skill installation"
git tag v1.0.0
```

Update skill:
```bash
# Pull updates
git pull origin main

# Reinstall dependencies if requirements changed
pip install -r requirements.txt --upgrade
```

### Multiple Skill Versions

Run different versions side-by-side:

```bash
~/.claude/skills/
├── [strategy-name]-v1/
├── [strategy-name]-v2/
└── [strategy-name]-dev/
```

### Performance Optimization

For faster skill loading:

1. **Pre-compile Python modules**:
   ```bash
   python -m compileall scripts/
   ```

2. **Use prompt caching** (API only)

3. **Minimize SKILL.md size**: Use progressive disclosure with references/

---

**Installation complete!** See [quickstart-guide.md](quickstart-guide.md) for usage examples.
