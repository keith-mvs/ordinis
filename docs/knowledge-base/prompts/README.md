<think>
Okay, let's tackle this. The user wants a comprehensive README.md for the 'docs/knowledge-base/prompts' folder. They mentioned the files inside: cli-usage.md, recommended-skills.md, skills-index.md, user-testing-guide.md.

First, I need to create an overview of the 'prompts' domain. Since it's part of a quantitative trading system documentation, the prompts probably relate to how users interact with the system, maybe through command-line interfaces or prompts for input. It might cover best practices, skills needed, testing procedures.

Next, each file's description. Cli-usage.md likely explains how to use the command-line interface, including commands and options. Recommended-skills.md would list the skills a user should have, like programming, statistics, etc. Skills-index.md might be an index or directory of all skills-related documents. User-testing-guide.md probably details how to test the system, maybe steps for QA or user acceptance testing.

Structure-wise, the README should help developers navigate. Maybe start with an overview, then list each file with a brief description, and then a structure section explaining how these files fit into the knowledge base. Also, include navigation tips like links to other sections if needed.

Wait, the user said to infer from the names. So I shouldn't assume too much beyond the names. For example, skills-index.md could be an index of skills, but maybe it's a list of skills required or available. User-testing-guide.md might be about testing the prompts themselves or the system's response to prompts.

Also, the output should be markdown without the fences. So just the content starting with # headings, etc.

I need to make sure the README is clear and helps a developer understand what's in the folder and how to use the documents. Maybe include a section on how these files relate to each other. For example, recommended-skills.md might link to skills-index.md, or cli-usage.md would be used alongside user-testing-guide.md.

Double-check the requirements: overview, file descriptions, structure for navigation. Don't add extra sections unless necessary. Keep it concise but comprehensive.
</think>

# README: Prompts Knowledge Base

## Overview
The `prompts` domain within the knowledge base focuses on guiding users through interactive elements of the quantitative trading system. This includes command-line interfaces (CLI), skill requirements for effective system usage, and structured workflows for testing and validation. The documentation here ensures users can efficiently engage with the system, troubleshoot interactions, and validate their workflows.

## File Descriptions
- **`cli-usage.md`**
  Details command-line interface operations, including syntax, parameters, and examples for executing trades, querying data, or configuring system settings via CLI.

- **`recommended-skills.md`**
  Outlines the technical and domain-specific skills (e.g., Python programming, statistical analysis, market dynamics) required to effectively use the trading system and interpret prompts.

- **`skills-index.md`**
  A catalog of all skill-related documentation, linking to detailed guides, tutorials, or external resources for skill development relevant to the system.

- **`user-testing-guide.md`**
  Provides step-by-step instructions for testing user interactions with the system, including prompt validation, error handling, and feedback loops during development or deployment.

## Navigation Structure
This section is organized to support both **quick reference** and **deep learning** workflows:

1. **Start with `recommended-skills.md`** to ensure you have the prerequisite knowledge for interacting with the system.
2. **Use `cli-usage.md`** for hands-on operations, referencing it during CLI-based tasks.
3. **Consult `skills-index.md`** to explore additional learning resources or troubleshoot skill gaps.
4. **Follow `user-testing-guide.md`** when validating system behavior, especially during development or QA phases.

For developers, this folder serves as a critical bridge between system design and user experience, ensuring prompts and interactions are intuitive, well-documented, and aligned with user capabilities. Links to related documentation (e.g., API references, strategy implementation guides) are included where applicable.
