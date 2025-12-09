"""
MkDocs Macros module for Ordinis Documentation.

Provides template variables and functions for dynamic documentation generation.
"""

from datetime import datetime


def define_env(env):
    """
    Define custom macros and variables for MkDocs.

    This is called by mkdocs-macros-plugin during build.
    """

    # Project metadata
    env.variables['project_name'] = 'Ordinis Trading System'
    env.variables['project_version'] = '0.2.0-dev'
    env.variables['project_author'] = 'Ordinis Development Team'

    # Version history
    env.variables['version_history'] = [
        {'version': '0.2.0-dev', 'date': '2024-12-08', 'changes': 'Governance engines, OECD principles, broker compliance'},
        {'version': '0.1.0', 'date': '2024-11-30', 'changes': 'Initial release with core trading infrastructure'},
    ]

    # Component status
    env.variables['components'] = {
        'SignalCore': {'status': 'active', 'version': '0.2.0'},
        'RiskGuard': {'status': 'active', 'version': '0.2.0'},
        'FlowRoute': {'status': 'active', 'version': '0.2.0'},
        'Cortex': {'status': 'development', 'version': '0.1.0'},
        'Governance': {'status': 'active', 'version': '0.2.0'},
    }

    @env.macro
    def now():
        """Return current datetime."""
        return datetime.now()

    @env.macro
    def version_badge(version=None):
        """Generate HTML version badge."""
        ver = version or env.variables['project_version']
        return f'<span class="version-badge">v{ver}</span>'

    @env.macro
    def status_badge(status):
        """Generate status badge with appropriate styling."""
        css_class = f'status-{status.lower().replace(" ", "-")}'
        return f'<span class="{css_class}">{status}</span>'

    @env.macro
    def component_table():
        """Generate component status table."""
        rows = []
        for name, info in env.variables['components'].items():
            status_html = status_badge(info['status'])
            rows.append(f"| {name} | {status_html} | {info['version']} |")
        return "\n".join([
            "| Component | Status | Version |",
            "|-----------|--------|---------|",
        ] + rows)

    @env.macro
    def last_updated():
        """Return last updated timestamp."""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
