# Skill Distribution Guide

How to distribute skills via personal, project, and plugin methods.

---

## Overview

Claude Code supports three distribution methods for Skills:

1. **Personal Skills** (~/.claude/skills/) - Individual user, all projects
2. **Project Skills** (.claude/skills/) - Team sharing via git
3. **Plugin Skills** (Claude Code plugins) - Marketplace distribution

Each method serves different use cases and has unique installation procedures.

---

## Personal Skills

### Overview

**Location**: `~/.claude/skills/`

**Scope**: Available to YOU across ALL projects

**Use cases**:
- Individual workflows and preferences
- Experimental Skills you're developing
- Personal productivity tools
- Skills not ready for team sharing

### Installation

**Create personal Skill**:
```bash
# Create directory
mkdir -p ~/.claude/skills/my-personal-skill

# Create SKILL.md
code ~/.claude/skills/my-personal-skill/SKILL.md

# Add supporting files
mkdir -p ~/.claude/skills/my-personal-skill/{scripts,references,assets}
```

**Install existing Skill**:
```bash
# Copy Skill directory to personal Skills folder
cp -r /path/to/strategy-skill ~/.claude/skills/

# Install dependencies
cd ~/.claude/skills/strategy-skill
pip install -r requirements.txt
```

### Advantages

✅ **Private**: Not shared with team
✅ **Global**: Available in all projects
✅ **Fast**: Instant availability
✅ **Flexible**: Experiment without affecting team

### Disadvantages

❌ **Not Shared**: Team doesn't benefit
❌ **Manual Updates**: Must update manually
❌ **No Version Control**: Not tracked in git

### Best Practices

**Use personal Skills for**:
- Testing new Skills before team deployment
- Personal keyboard shortcuts or workflows
- Individual analysis preferences
- Skills specific to your role

**Don't use personal Skills for**:
- Team workflows
- Project-specific requirements
- Skills that would benefit others

---

## Project Skills

### Overview

**Location**: `.claude/skills/` (within project directory)

**Scope**: Available to TEAM for THIS project

**Use cases**:
- Team workflows and conventions
- Project-specific expertise
- Shared utilities and scripts
- Standardized analysis methods

### Installation

**Create project Skill**:
```bash
# Navigate to project root
cd /path/to/your/project

# Create project Skills directory
mkdir -p .claude/skills/team-skill

# Create SKILL.md
code .claude/skills/team-skill/SKILL.md

# Add supporting files
mkdir -p .claude/skills/team-skill/{scripts,references,assets}
```

**Share with team via git**:
```bash
# Add to version control
git add .claude/skills/

# Commit
git commit -m "Add team Skill for options analysis

- Implements bull-put-spread analyzer
- Includes position sizing and risk assessment
- Full documentation and examples"

# Push to remote
git push origin main
```

**Team members install automatically**:
```bash
# Team member pulls latest changes
git pull

# Skills are immediately available (no manual install needed)
claude  # Start Claude Code - Skill is loaded
```

**Install dependencies** (team members):
```bash
# Navigate to project
cd /path/to/project

# Install Skill dependencies
pip install -r .claude/skills/team-skill/requirements.txt
```

### Advantages

✅ **Team Sharing**: Automatic distribution
✅ **Version Control**: Git tracks changes
✅ **Automatic Updates**: git pull gets latest
✅ **Project-Specific**: Tailored to project needs

### Disadvantages

❌ **Project-Limited**: Only available in this project
❌ **Requires Git**: Team must use version control
❌ **Public** (if repo is public): Anyone can see

### Best Practices

**Use project Skills for**:
- Team-wide workflows
- Project conventions (commit messages, code style)
- Domain-specific analysis (for this project)
- Shared utilities everyone uses

**Git workflow**:

```bash
# 1. Create feature branch
git checkout -b add-bull-put-skill

# 2. Develop Skill
mkdir -p .claude/skills/bull-put-spread
# ... create files

# 3. Test Skill
claude  # Test that Skill triggers correctly

# 4. Commit changes
git add .claude/skills/bull-put-spread/
git commit -m "Add bull-put-spread analysis Skill"

# 5. Push and create PR
git push origin add-bull-put-skill
# Create pull request on GitHub/GitLab

# 6. After merge, team pulls changes
git checkout main
git pull  # Team members get Skill automatically
```

**Updating project Skills**:

```bash
# Make changes
code .claude/skills/team-skill/SKILL.md

# Commit update
git add .claude/skills/team-skill/
git commit -m "Update team-skill: Add new examples"
git push

# Team members update
git pull  # Gets latest Skill version
```

---

## Plugin Skills

### Overview

**Location**: Bundled with Claude Code plugins

**Scope**: Available to ANYONE who installs the plugin

**Use cases**:
- Widely-shared Skills
- Third-party integrations
- Marketplace distribution
- Open-source contributions

### Creating Plugin with Skills

**Directory structure**:
```
my-plugin/
├── plugin.json
└── skills/
    ├── skill-1/
    │   └── SKILL.md
    └── skill-2/
        └── SKILL.md
```

**plugin.json**:
```json
{
  "name": "options-trading-plugin",
  "version": "1.0.0",
  "description": "Options trading analysis Skills",
  "skills": [
    "skills/bull-call-spread",
    "skills/bull-put-spread",
    "skills/married-put"
  ]
}
```

**See**: [Claude Code plugins documentation](https://code.claude.com/docs/en/plugins) for complete plugin creation guide.

### Installation (Users)

**From marketplace**:
1. Navigate to Claude Code plugins marketplace
2. Search for plugin
3. Click "Install"
4. Skills available immediately

**From git repository**:
```bash
# Clone plugin repository
git clone https://github.com/user/plugin-name.git

# Install plugin (method depends on Claude Code version)
# See Claude Code docs for current installation method
```

### Advantages

✅ **Wide Distribution**: Anyone can install
✅ **Marketplace**: Discoverable by users
✅ **Versioning**: Semantic versioning support
✅ **Updates**: Users get updates automatically

### Disadvantages

❌ **More Complex**: Requires plugin infrastructure
❌ **Public**: Always public (if in marketplace)
❌ **Approval**: May require marketplace approval

### Best Practices

**Use plugins for**:
- Skills useful to broad audience
- Open-source contributions
- Integrations with external services
- General-purpose utilities

**Don't use plugins for**:
- Company-specific workflows (use project Skills)
- Experimental/unfinished Skills
- Private/proprietary analysis methods

---

## Comparison Matrix

| Feature | Personal | Project | Plugin |
|---------|----------|---------|--------|
| **Scope** | User, all projects | Team, this project | Anyone, all projects |
| **Installation** | Manual copy | Git clone/pull | Plugin install |
| **Updates** | Manual | Git pull | Automatic |
| **Sharing** | No | Team only | Public |
| **Version Control** | No | Yes (git) | Yes (git + plugin) |
| **Use Cases** | Individual, experimental | Team workflows | Wide distribution |
| **Privacy** | Private | Team-visible | Public |
| **Effort** | Low | Medium | High |

---

## Decision Tree

### Should this be a Personal Skill?

**YES** if:
- Only you will use it
- Experimental or work-in-progress
- Personal preferences or shortcuts
- Testing before team deployment

**NO** if:
- Team would benefit
- Project-specific requirements
- Ready for production use

→ If NO, continue to Project Skill decision

---

### Should this be a Project Skill?

**YES** if:
- Team workflows or conventions
- Project-specific domain knowledge
- Shared utilities for this project
- Version control is important

**NO** if:
- Useful beyond this project
- Generic/general-purpose
- Want marketplace distribution

→ If NO, continue to Plugin decision

---

### Should this be a Plugin?

**YES** if:
- Useful to broad audience
- General-purpose utility
- Want marketplace distribution
- Open-source contribution

**NO** if:
- Company-proprietary
- Project-specific
- Not ready for public release

→ If NO, reconsider Personal or Project

---

## Hybrid Approach

You can use MULTIPLE distribution methods for the same Skill:

**Example workflow**:

1. **Start as Personal Skill** (development)
   ```bash
   # Create in personal Skills
   mkdir ~/.claude/skills/new-skill
   # Develop and test
   ```

2. **Promote to Project Skill** (team sharing)
   ```bash
   # Copy to project
   cp -r ~/.claude/skills/new-skill .claude/skills/
   git add .claude/skills/new-skill/
   git commit -m "Add new-skill for team use"
   git push
   ```

3. **Release as Plugin** (public distribution)
   ```bash
   # Create plugin structure
   mkdir my-plugin/skills/
   cp -r .claude/skills/new-skill my-plugin/skills/
   # Create plugin.json
   # Publish to marketplace
   ```

This allows progressive rollout: test personally → share with team → release publicly.

---

## Migration Between Distribution Methods

### Personal → Project

```bash
# Copy from personal to project
cp -r ~/.claude/skills/my-skill .claude/skills/

# Add to git
cd .claude/skills/my-skill
git add .
git commit -m "Promote my-skill to project Skill"
git push
```

### Project → Plugin

```bash
# Create plugin structure
mkdir -p my-plugin/skills/
cp -r .claude/skills/project-skill my-plugin/skills/

# Create plugin.json
cat > my-plugin/plugin.json << 'EOF'
{
  "name": "my-plugin",
  "version": "1.0.0",
  "skills": ["skills/project-skill"]
}
EOF

# Publish (see Claude Code plugin docs)
```

### Plugin → Project

```bash
# Install plugin first
# (follow plugin installation instructions)

# Copy to project
cp -r ~/.claude/plugins/plugin-name/skills/skill-name .claude/skills/

# Customize for project
code .claude/skills/skill-name/SKILL.md

# Commit
git add .claude/skills/skill-name/
git commit -m "Import skill-name from plugin for customization"
```

---

## Real-World Examples

### Example 1: Options Trading Skills

**Scenario**: Creating options strategy analysis Skills

**Distribution**:
- **Personal**: Experimental bull-put-spread analyzer (testing formulas)
- **Project** (Ordinis): Married-put, bull-call-spread (team uses)
- **Plugin**: Generic Black-Scholes calculator (useful to everyone)

**Workflow**:
```bash
# 1. Develop personally
mkdir ~/.claude/skills/bull-put-experiment
# Test and refine

# 2. Promote to project
cp -r ~/.claude/skills/bull-put-experiment ~/Workspace/ordinis/.claude/skills/bull-put-spread
cd ~/Workspace/ordinis
git add .claude/skills/bull-put-spread/
git commit -m "Add bull-put-spread Skill"
git push

# 3. Extract generic components for plugin
mkdir options-plugin/skills/black-scholes
# Copy reusable Black-Scholes code
# Publish plugin
```

### Example 2: Team Conventions

**Scenario**: Standardizing commit messages for team

**Distribution**: **Project Skill only**

**Reasoning**:
- Project-specific conventions
- Team needs consistency
- Not useful outside this project

**Implementation**:
```bash
cd ~/project
mkdir -p .claude/skills/commit-helper
# Create SKILL.md with project commit standards
git add .claude/skills/commit-helper/
git commit -m "Add commit-helper Skill for team standards"
git push
```

### Example 3: General Utility

**Scenario**: PDF form filling tool

**Distribution**: **Plugin** (public utility)

**Reasoning**:
- Useful to many people
- Not project-specific
- Open-source contribution

**Implementation**:
```bash
mkdir pdf-plugin/skills/pdf-form-filler
# Create comprehensive SKILL.md
# Add scripts
# Publish to Claude Code marketplace
```

---

## Best Practices Summary

### Personal Skills
- ✅ Use for experimentation
- ✅ Test before team deployment
- ✅ Keep private workflows private
- ❌ Don't duplicate team Skills
- ❌ Don't forget to promote when ready

### Project Skills
- ✅ Commit to git immediately
- ✅ Document team-specific context
- ✅ Keep dependencies in requirements.txt
- ❌ Don't include personal preferences
- ❌ Don't hardcode local paths

### Plugin Skills
- ✅ Make generic and reusable
- ✅ Document thoroughly
- ✅ Version semantically
- ❌ Don't include proprietary logic
- ❌ Don't hardcode project-specific details

---

## Troubleshooting

### Skill not appearing

**Personal Skill**:
```bash
# Check location
ls ~/.claude/skills/my-skill/SKILL.md

# Restart Claude Code
claude
```

**Project Skill**:
```bash
# Verify in project directory
ls .claude/skills/team-skill/SKILL.md

# Pull latest
git pull

# Restart Claude Code
```

**Plugin Skill**:
- Check plugin is installed
- Restart Claude Code
- Re-install plugin if needed

### Duplicate Skills

If same Skill exists in multiple locations:

**Priority order** (Claude Code loads in this order):
1. Project Skills (.claude/skills/)
2. Plugin Skills (~/.claude/plugins/*/skills/)
3. Personal Skills (~/.claude/skills/)

**Resolution**: Remove duplicates from lower-priority locations

### Update not reflecting

```bash
# For project Skills
git pull  # Get latest
claude --reload  # Reload Skills

# For personal Skills
# Make changes
claude --reload  # Reload Skills

# For plugin Skills
# Update plugin
# Restart Claude Code
```

---

## Summary

**Choose your distribution method based on**:
- **Scope**: Who should have access?
- **Maturity**: How stable is the Skill?
- **Purpose**: Project-specific or general-purpose?

**General guidance**:
- Start **Personal** (develop and test)
- Promote to **Project** (team benefits)
- Release as **Plugin** (public utility)

Use the distribution method that matches your Skill's intended audience and maturity level.

---

**This guide helps you choose the right distribution method for your Skills and manage them effectively across personal, team, and public contexts.**
