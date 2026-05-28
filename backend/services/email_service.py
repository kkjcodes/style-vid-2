"""
Minimal SMTP email sender for password-reset links.

If SMTP_HOST is not configured, the reset link is logged at WARNING level
instead of sent — useful for local dev where no mail server is available.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from backend.core.config import get_settings

log = logging.getLogger("email_service")


def send_reset_email(to_email: str, reset_token: str) -> None:
    s = get_settings()
    reset_url = f"{s.app_url}/?reset={reset_token}"

    if not s.smtp_host:
        log.warning(
            f"SMTP not configured — password reset link sent to {to_email} (check logs at app startup)"
        )
        log.debug(f"Reset URL (dev only): {reset_url}")
        return

    body_html = f"""
    <p>You requested a password reset for your StyleVid account.</p>
    <p><a href="{reset_url}">Click here to reset your password</a></p>
    <p>This link expires in <strong>1 hour</strong> and can only be used once.</p>
    <p>If you didn't request this, ignore this email — your password won't change.</p>
    """
    body_text = (
        f"Reset your StyleVid password:\n{reset_url}\n\n"
        "This link expires in 1 hour. If you didn't request this, ignore this email."
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "StyleVid — reset your password"
    msg["From"] = s.smtp_from or s.smtp_user
    msg["To"] = to_email
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as srv:
            srv.ehlo()
            srv.starttls(context=context)
            srv.login(s.smtp_user, s.smtp_password)
            srv.sendmail(msg["From"], to_email, msg.as_string())
        log.info(f"Reset email sent to {to_email}")
    except Exception as exc:
        log.error(f"Failed to send reset email to {to_email}: {exc}")
        raise RuntimeError("Could not send reset email. Please try again later.") from exc
