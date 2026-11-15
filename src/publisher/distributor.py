"""
Autonomous 24/7 E-Paper News Agent - Telegram & Email Multi-Channel Distributor
Delivers the compiled E-Paper PDF and executive summary digest to Telegram Bot API
and optional Gmail SMTP.
"""

from __future__ import annotations

import email
import logging
import mimetypes
import os
import smtplib
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("epaper.publisher.distributor")


class Distributor:
    """Manages automated delivery of the daily E-Paper edition."""

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        gmail_user: Optional[str] = None,
        gmail_password: Optional[str] = None,
        gmail_recipients: Optional[List[str]] = None,
    ):
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()

        self.gmail_user = gmail_user or os.getenv("GMAIL_USER", "").strip()
        self.gmail_password = gmail_password or os.getenv("GMAIL_APP_PASSWORD", "").strip()

        recipients_env = os.getenv("GMAIL_RECIPIENTS", "")
        if gmail_recipients:
            self.gmail_recipients = gmail_recipients
        elif recipients_env:
            self.gmail_recipients = [r.strip() for r in recipients_env.split(",") if r.strip()]
        elif self.gmail_user:
            self.gmail_recipients = [self.gmail_user]
        else:
            self.gmail_recipients = []

    def format_telegram_caption(
        self,
        edition_date: str,
        total_stories: int,
        total_sources: int,
        stories: List[Dict[str, Any]],
        edition_cycle: Optional[str] = None,
    ) -> str:
        """Create a clean, formatted Telegram summary of top front-page headlines."""
        if not edition_cycle:
            utc_now = datetime.now(timezone.utc)
            edition_cycle = "Morning Broadsheet" if utc_now.hour < 7 else "Evening Wrap"

        top_leads = stories[:5]
        headlines_text = ""
        for i, story in enumerate(top_leads, start=1):
            tag = story.get("state_or_region") or story.get("category", "")
            headlines_text += f"{i}. <b>[{tag}]</b> {story.get('headline', '')}\n"

        caption = (
            f"🗞 <b>THE DAILY CHRONICLE</b>\n"
            f"📅 <i>{edition_date} • 24/7 {edition_cycle}</i>\n"
            f"📊 <b>{total_stories} Verified Stories</b> | {total_sources} Outlets\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<b>★ TOP HEADLINES:</b>\n"
            f"{headlines_text}\n"
            f"📎 <i>Full broadsheet newspaper edition attached below.</i>"
        )
        return caption

    def send_to_telegram(
        self,
        pdf_path: Optional[str],
        md_path: Optional[str],
        edition_date: str,
        total_stories: int,
        total_sources: int,
        stories: List[Dict[str, Any]],
        edition_cycle: Optional[str] = None,
    ) -> bool:
        """Sends document and summary caption to Telegram chat."""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.info(
                "Telegram credentials (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) not configured. "
                "Skipping Telegram delivery."
            )
            return False

        caption = self.format_telegram_caption(
            edition_date, total_stories, total_sources, stories, edition_cycle=edition_cycle
        )

        base_url = f"https://api.telegram.org/bot{self.telegram_token}"

        # 1. Determine file to upload (prefer PDF, fallback to Markdown)
        target_file = pdf_path if (pdf_path and os.path.exists(pdf_path)) else md_path
        if not target_file or not os.path.exists(target_file):
            logger.warning("No generated file found to deliver via Telegram.")
            # Fall back to sending text message digest
            try:
                resp = requests.post(
                    f"{base_url}/sendMessage",
                    data={
                        "chat_id": self.telegram_chat_id,
                        "text": caption,
                        "parse_mode": "HTML",
                    },
                    timeout=20,
                )
                return resp.status_code == 200
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")
                return False

        logger.info(f"Delivering E-Paper edition to Telegram chat {self.telegram_chat_id}...")

        try:
            with open(target_file, "rb") as f:
                # Telegram caption length limit is 1024 chars
                trimmed_caption = caption[:1020]
                resp = requests.post(
                    f"{base_url}/sendDocument",
                    data={
                        "chat_id": self.telegram_chat_id,
                        "caption": trimmed_caption,
                        "parse_mode": "HTML",
                    },
                    files={"document": f},
                    timeout=60,
                )
                if resp.status_code == 200:
                    logger.info("E-Paper successfully dispatched to Telegram!")
                    return True
                else:
                    logger.error(f"Telegram upload failed HTTP {resp.status_code}: {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Error dispatching to Telegram: {e}")
            return False

    def send_to_email(
        self,
        pdf_path: Optional[str],
        html_path: Optional[str],
        edition_date: str,
        total_stories: int,
        edition_cycle: Optional[str] = None,
    ) -> bool:
        """Sends edition via Gmail SMTP (optional channel)."""
        if not self.gmail_user or not self.gmail_password or not self.gmail_recipients:
            logger.debug("Gmail SMTP credentials not configured. Skipping email delivery.")
            return False

        if not edition_cycle:
            utc_now = datetime.now(timezone.utc)
            edition_cycle = "Morning Edition" if utc_now.hour < 7 else "Evening Edition"

        try:
            subject = f"The Daily Chronicle — {edition_date} ({edition_cycle} • {total_stories} Stories)"
            msg = MIMEMultipart("mixed")
            msg["From"] = f"The Daily Chronicle <{self.gmail_user}>"
            msg["To"] = ", ".join(self.gmail_recipients)
            msg["Subject"] = subject

            # Email body - Styled strictly in Times New Roman with warm off-white newsprint paper
            body_html = (
                f'<div style="font-family: \'Times New Roman\', Times, Georgia, serif !important; color: #111111; line-height: 1.45; max-width: 680px; margin: 0 auto; padding: 15px; background-color: #f7f4ec;">'
                f'<div style="border-top: 3px solid #111111; border-bottom: 2px double #111111; text-align: center; padding: 8px 0; margin-bottom: 15px;">'
                f'<h1 style="font-family: \'Times New Roman\', Times, Georgia, serif !important; font-size: 26pt; font-weight: 900; text-transform: uppercase; margin: 0; letter-spacing: -0.5px;">The Daily Chronicle</h1>'
                f'<div style="font-family: \'Times New Roman\', Times, Georgia, serif !important; font-size: 8pt; text-transform: uppercase; letter-spacing: 2px; font-style: italic; color: #444444; margin-top: 2px;">A National &amp; Bengal Journal of Record • Autonomous Edition</div>'
                f'<div style="border-top: 1px solid #111111; border-bottom: 1px solid #111111; padding: 3px; margin-top: 6px; font-size: 8pt; font-weight: bold; text-transform: uppercase; background-color: #eee9dc;">'
                f'{edition_date} • {edition_cycle.upper()} • {total_stories} Verified Stories • Attached Broadsheet PDF'
                f'</div>'
                f'</div>'
                f'<p style="font-family: \'Times New Roman\', Times, Georgia, serif !important; font-size: 10pt; line-height: 1.5;">'
                f'Attached is your daily broadsheet newspaper edition compiled in traditional broadsheet format with multi-column layouts and Times New Roman typography.'
                f'</p>'
                f'<div style="margin: 15px 0; padding: 10px; border-left: 3px solid #111111; background-color: #eee9dc;">'
                f'<strong style="font-family: \'Times New Roman\', Times, Georgia, serif !important; font-size: 9pt; text-transform: uppercase;">Executive Delivery Notice:</strong>'
                f'<p style="font-family: \'Times New Roman\', Times, Georgia, serif !important; font-size: 9pt; margin-top: 4px;">The full newspaper edition PDF is attached below for reading and printing.</p>'
                f'</div>'
                f'<hr style="border: 0; border-top: 0.5px solid #cccccc; margin: 20px 0;" />'
                f'<div style="font-family: \'Times New Roman\', Times, Georgia, serif !important; font-size: 7.5pt; color: #666666; font-style: italic; text-align: center;">'
                f'The Daily Chronicle • Autonomous Intelligence Journal'
                f'</div>'
                f'</div>'
            )
            if html_path and os.path.exists(html_path):
                try:
                    with open(html_path, "r", encoding="utf-8") as f:
                        raw_html = f.read()
                        # Wrap rendered HTML in a Times New Roman outer wrapper
                        body_html = (
                            f'<div style="font-family: \'Times New Roman\', Times, Georgia, serif !important; color: #111111;">'
                            f'{raw_html}'
                            f'</div>'
                        )
                except Exception:
                    pass

            msg.attach(MIMEText(body_html, "html"))

            # Attachment
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    part = MIMEBase("application", "pdf")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f'attachment; filename="{os.path.basename(pdf_path)}"',
                    )
                    msg.attach(part)

            # Send via Gmail SSL
            logger.info(f"Connecting to Gmail SMTP to send edition to {self.gmail_recipients}...")
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=25) as server:
                server.login(self.gmail_user, self.gmail_password)
                server.send_message(msg, to_addrs=self.gmail_recipients)
            logger.info("Email delivered successfully via Gmail SMTP!")
            return True
        except Exception as e:
            logger.error(f"Failed to deliver email via Gmail SMTP: {e}")
            return False

    def distribute_all(
        self,
        publication_result: Dict[str, Any],
        stories: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Dispatches to all configured channels."""
        edition_date = publication_result.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        cycle = publication_result.get("cycle")
        cycle_label = f"{cycle} Edition" if cycle else None
        pdf_path = publication_result.get("pdf_path")
        md_path = publication_result.get("md_path")
        html_path = publication_result.get("html_path")
        total_stories = publication_result.get("total_stories", len(stories))
        total_sources = publication_result.get("total_sources", 1)

        tg_ok = self.send_to_telegram(
            pdf_path, md_path, edition_date, total_stories, total_sources, stories, edition_cycle=cycle_label
        )
        email_ok = self.send_to_email(
            pdf_path, html_path, edition_date, total_stories, edition_cycle=cycle_label
        )

        return {"telegram": tg_ok, "email": email_ok}
