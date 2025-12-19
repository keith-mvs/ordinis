"""Alert channels for different notification methods."""

from ordinis.adapters.alerting.channels.desktop import DesktopNotifier
from ordinis.adapters.alerting.channels.email import EmailNotifier, EmailRecipient, SMTPConfig

__all__ = [
    "DesktopNotifier",
    "EmailNotifier",
    "EmailRecipient",
    "SMTPConfig",
]
