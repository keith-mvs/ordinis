"""
Enable running Cortex CLI via: python -m ordinis.engines.cortex

Usage:
    python -m ordinis.engines.cortex --file src/mycode.py --type review
    python -m ordinis.engines.cortex --dir src/ --type security --output report.json
"""

from ordinis.engines.cortex.cli import main

if __name__ == "__main__":
    main()
