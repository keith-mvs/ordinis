"""Alert channels for different notification methods."""

from alerting.channels.desktop import DesktopNotifier
from alerting.channels.email import EmailNotifier, EmailRecipient, SMTPConfig

__all__ = [
    "DesktopNotifier",
    "EmailNotifier",
    "EmailRecipient",
    "SMTPConfig",
]
