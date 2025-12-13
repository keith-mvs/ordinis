"""
Alerting module for Ordinis live trading.

Provides multi-channel alerting for:
- Kill switch activation
- Risk threshold breaches
- Order rejections
- System health issues

Channels:
- Desktop notifications (Windows toast + audio)
- Email (optional SMTP)
"""

from adapters.alerting.manager import Alert, AlertManager, AlertSeverity

__all__ = [
    "Alert",
    "AlertManager",
    "AlertSeverity",
]
