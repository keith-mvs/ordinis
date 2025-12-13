"""
Email notification channel for alerts.

Provides:
- SMTP email delivery
- HTML and plain text formats
- Rate limiting per recipient
- Template support

Optional - requires SMTP server configuration.
"""

from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import smtplib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.alerting.manager import Alert

logger = logging.getLogger(__name__)


@dataclass
class SMTPConfig:
    """SMTP server configuration."""

    host: str
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    from_address: str = ""
    from_name: str = "Ordinis Trading"
    timeout_seconds: int = 30


@dataclass
class EmailRecipient:
    """Email recipient with rate limiting."""

    address: str
    name: str = ""
    min_severity: str = "warning"
    last_sent: datetime | None = None
    rate_limit_seconds: float = 300.0


class EmailNotifier:
    """
    Email alert sender.

    Supports multiple recipients with per-recipient rate limiting
    and severity filtering.
    """

    def __init__(
        self,
        config: SMTPConfig,
        recipients: list[EmailRecipient] | None = None,
        default_rate_limit: float = 300.0,
    ):
        """
        Initialize email notifier.

        Args:
            config: SMTP server configuration
            recipients: List of recipients
            default_rate_limit: Default seconds between emails per recipient
        """
        self.config = config
        self._recipients: dict[str, EmailRecipient] = {}
        self._default_rate_limit = default_rate_limit
        self._send_count = 0
        self._last_error: str | None = None

        if recipients:
            for r in recipients:
                self.add_recipient(r)

    def add_recipient(
        self,
        recipient: EmailRecipient | None = None,
        address: str = "",
        name: str = "",
        min_severity: str = "warning",
    ) -> None:
        """
        Add email recipient.

        Args:
            recipient: EmailRecipient object
            address: Email address (if not using recipient object)
            name: Display name
            min_severity: Minimum severity to receive
        """
        if recipient:
            self._recipients[recipient.address] = recipient
        elif address:
            self._recipients[address] = EmailRecipient(
                address=address,
                name=name,
                min_severity=min_severity,
                rate_limit_seconds=self._default_rate_limit,
            )

    def remove_recipient(self, address: str) -> None:
        """Remove recipient by email address."""
        self._recipients.pop(address, None)

    def send(self, alert: "Alert") -> bool:
        """
        Send alert via email.

        Args:
            alert: Alert to send

        Returns:
            True if sent to at least one recipient
        """
        if not self._recipients:
            logger.debug("No email recipients configured")
            return False

        # Filter recipients by severity and rate limit
        eligible = self._get_eligible_recipients(alert)
        if not eligible:
            logger.debug("No eligible recipients for alert")
            return False

        # Build email
        subject, html_body, text_body = self._format_alert(alert)

        # Send to each recipient
        sent_count = 0
        for recipient in eligible:
            try:
                if self._send_email(recipient.address, subject, html_body, text_body):
                    recipient.last_sent = datetime.utcnow()
                    sent_count += 1
            except Exception as e:
                logger.exception(f"Failed to send email to {recipient.address}: {e}")
                self._last_error = str(e)

        self._send_count += sent_count
        return sent_count > 0

    def _get_eligible_recipients(self, alert: "Alert") -> list[EmailRecipient]:
        """Get recipients eligible for this alert."""
        severity_order = ["info", "warning", "critical", "emergency"]
        alert_severity_index = severity_order.index(alert.severity.value)

        eligible = []
        now = datetime.utcnow()

        for recipient in self._recipients.values():
            # Check severity
            recipient_severity_index = severity_order.index(recipient.min_severity)
            if alert_severity_index < recipient_severity_index:
                continue

            # Check rate limit
            if recipient.last_sent:
                elapsed = (now - recipient.last_sent).total_seconds()
                if elapsed < recipient.rate_limit_seconds:
                    continue

            eligible.append(recipient)

        return eligible

    def _format_alert(self, alert: "Alert") -> tuple[str, str, str]:
        """
        Format alert for email.

        Returns:
            Tuple of (subject, html_body, text_body)
        """
        severity_prefix = {
            "info": "[i]",
            "warning": "[!]",
            "critical": "[X]",
            "emergency": "[!!!]",
        }
        prefix = severity_prefix.get(alert.severity.value, "[*]")

        subject = f"{prefix} [{alert.severity.value.upper()}] {alert.title}"

        # Plain text body
        text_body = f"""
{alert.title}
{"=" * len(alert.title)}

Severity: {alert.severity.value.upper()}
Type: {alert.alert_type.value}
Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}

{alert.message}
"""

        if alert.metadata:
            text_body += "\nDetails:\n"
            for key, value in alert.metadata.items():
                text_body += f"  {key}: {value}\n"

        text_body += "\n---\nOrdinis Trading Alert System"

        # HTML body
        severity_colors = {
            "info": "#17a2b8",
            "warning": "#ffc107",
            "critical": "#dc3545",
            "emergency": "#6f42c1",
        }
        color = severity_colors.get(alert.severity.value, "#6c757d")

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .alert-box {{ border-left: 4px solid {color}; padding: 16px; margin: 16px 0; background: #f8f9fa; }}
        .severity {{ color: {color}; font-weight: bold; text-transform: uppercase; }}
        .meta {{ color: #6c757d; font-size: 0.9em; }}
        .details {{ background: #e9ecef; padding: 12px; margin-top: 12px; border-radius: 4px; }}
        .footer {{ color: #6c757d; font-size: 0.8em; margin-top: 24px; border-top: 1px solid #dee2e6; padding-top: 12px; }}
    </style>
</head>
<body>
    <div class="alert-box">
        <h2 style="margin-top: 0;">{alert.title}</h2>
        <p class="meta">
            <span class="severity">{alert.severity.value}</span> |
            {alert.alert_type.value} |
            {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
        </p>
        <p>{alert.message}</p>
"""

        if alert.metadata:
            html_body += '        <div class="details"><strong>Details:</strong><ul>'
            for key, value in alert.metadata.items():
                html_body += f"<li><strong>{key}:</strong> {value}</li>"
            html_body += "</ul></div>"

        html_body += """
    </div>
    <div class="footer">
        Ordinis Trading Alert System
    </div>
</body>
</html>
"""

        return subject, html_body, text_body

    def _send_email(
        self,
        to_address: str,
        subject: str,
        html_body: str,
        text_body: str,
    ) -> bool:
        """
        Send email via SMTP.

        Args:
            to_address: Recipient address
            subject: Email subject
            html_body: HTML content
            text_body: Plain text content

        Returns:
            True if sent successfully
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = (
            f"{self.config.from_name} <{self.config.from_address}>"
            if self.config.from_name
            else self.config.from_address
        )
        msg["To"] = to_address

        # Attach both plain and HTML versions
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # Connect and send
        if self.config.use_ssl:
            server = smtplib.SMTP_SSL(
                self.config.host,
                self.config.port,
                timeout=self.config.timeout_seconds,
            )
        else:
            server = smtplib.SMTP(
                self.config.host,
                self.config.port,
                timeout=self.config.timeout_seconds,
            )

        try:
            if self.config.use_tls and not self.config.use_ssl:
                server.starttls()

            if self.config.username and self.config.password:
                server.login(self.config.username, self.config.password)

            server.sendmail(
                self.config.from_address,
                to_address,
                msg.as_string(),
            )

            logger.info(f"Alert email sent to {to_address}")
            return True

        finally:
            server.quit()

    def test(self, to_address: str | None = None) -> bool:
        """
        Send test email.

        Args:
            to_address: Specific address to test, or first recipient

        Returns:
            True if test successful
        """
        if not to_address:
            if not self._recipients:
                logger.error("No recipients configured for test")
                return False
            to_address = next(iter(self._recipients.keys()))

        try:
            return self._send_email(
                to_address=to_address,
                subject="Ordinis Test Email",
                html_body="<p>Email notifications are working!</p>",
                text_body="Email notifications are working!",
            )
        except Exception as e:
            logger.exception(f"Email test failed: {e}")
            self._last_error = str(e)
            return False

    def get_status(self) -> dict:
        """Get notifier status."""
        return {
            "configured": bool(self.config.host),
            "recipient_count": len(self._recipients),
            "recipients": list(self._recipients.keys()),
            "send_count": self._send_count,
            "last_error": self._last_error,
        }
