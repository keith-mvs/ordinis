"""Alert channels for different notification methods."""

from adapters.alerting.channels.desktop import DesktopNotifier
from adapters.alerting.channels.email import EmailNotifier, EmailRecipient, SMTPConfig

__all__ = [
    "DesktopNotifier",
    "EmailNotifier",
    "EmailRecipient",
    "SMTPConfig",
]
