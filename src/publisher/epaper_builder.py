"""
Autonomous 24/7 E-Paper News Agent - Broadsheet PDF & Markdown Publisher
Renders multi-column broadsheet newspaper HTML via Jinja2 and compiles crisp paginated PDF
via WeasyPrint, generating paired Markdown and HTML editions archived in editions/YYYY/MM/.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger("epaper.publisher.builder")

# Lazy import WeasyPrint to handle environments where C-libraries might be missing
try:
    import weasyprint
    HAS_WEASYPRINT = True
except (ImportError, OSError):
    HAS_WEASYPRINT = False


def find_browser_executable() -> Optional[str]:
    """Finds available Chrome, Chromium, or Edge binary for headless PDF printing."""
    # 1. Check PATH
    for name in ["chrome", "google-chrome", "chromium", "chromium-browser", "msedge"]:
        path = shutil.which(name)
        if path:
            return path

    # 2. Check standard Windows installation directories
    candidate_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        candidate_paths.extend([
            os.path.join(local_app_data, r"Google\Chrome\Application\chrome.exe"),
            os.path.join(local_app_data, r"Microsoft\Edge\Application\msedge.exe"),
        ])

    for p in candidate_paths:
        if os.path.exists(p):
            return p

    # 3. Check standard Linux/macOS paths
    unix_paths = [
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for p in unix_paths:
        if os.path.exists(p):
            return p

    return None


class EPaperBuilder:
    """Builds broadsheet HTML, compiled PDF, and Markdown archive files."""

    def __init__(
        self,
        templates_dir: str = "templates",
        editions_dir: str = "editions",
    ):
        self.templates_dir = templates_dir
        self.editions_dir = editions_dir
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def prepare_edition_paths(self, date_str: str) -> Tuple[str, str, str, str]:
        """
        Creates directory structure editions/YYYY/MM/ and returns file paths
        for PDF, Markdown, and HTML.
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year_str = dt.strftime("%Y")
        month_str = dt.strftime("%m")

        target_dir = os.path.join(self.editions_dir, year_str, month_str)
        os.makedirs(target_dir, exist_ok=True)

        base_name = f"Daily_Chronicle_{date_str}"
        pdf_path = os.path.join(target_dir, f"{base_name}.pdf")
        md_path = os.path.join(target_dir, f"{base_name}.md")
        html_path = os.path.join(target_dir, f"{base_name}.html")

        return target_dir, pdf_path, md_path, html_path

    def group_stories_by_section(
        self, stories: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Groups stories into balanced Front Page Leads and desk sections.
        Ensures ZERO stories are excluded or dropped, and every desk is represented.
        """
        sections_data: Dict[str, List[Dict[str, Any]]] = {
            "kolkata_bengal": [],
            "indian_states": [],
            "national": [],
            "global": [],
            "trends": [],
        }

        for story in stories:
            sec = story.get("section", "national")
            if sec in sections_data:
                sections_data[sec].append(story)
            else:
                # Fallback to national if unrecognized section key
                sections_data["national"].append(story)

        # Build balanced Front Page Leads: pick the top story from each desk
        front_leads: List[Dict[str, Any]] = []
        used_ids = set()
        desk_order = ["national", "kolkata_bengal", "indian_states", "global", "trends"]

        for desk in desk_order:
            for s in sections_data.get(desk, []):
                if s["cluster_id"] not in used_ids:
                    front_leads.append(s)
                    used_ids.add(s["cluster_id"])
                    break

        # Fill up to 6 front leads from remaining highest-scoring stories
        for s in stories:
            if len(front_leads) >= 6:
                break
            if s["cluster_id"] not in used_ids:
                front_leads.append(s)
                used_ids.add(s["cluster_id"])

        return front_leads, sections_data

    def render_html(
        self,
        stories: List[Dict[str, Any]],
        edition_date: str,
        total_sources_count: int,
    ) -> str:
        """Renders HTML broadsheet newspaper using Jinja2 and templates/epaper.css."""
        css_file = os.path.join(self.templates_dir, "epaper.css")
        css_content = ""
        if os.path.exists(css_file):
            with open(css_file, "r", encoding="utf-8") as f:
                css_content = f.read()

        front_leads, sections_data = self.group_stories_by_section(stories)

        dt = datetime.strptime(edition_date, "%Y-%m-%d")
        formatted_date = dt.strftime("%A, %B %d, %Y")

        template = self.jinja_env.get_template("epaper.html")
        rendered = template.render(
            edition_date=formatted_date,
            edition_date_iso=edition_date,
            total_stories_count=len(stories),
            total_sources_count=total_sources_count,
            front_leads=front_leads,
            sections_data=sections_data,
            css_content=css_content,
        )
        return rendered

    def compile_pdf(self, html_path: str, html_content: str, pdf_path: str) -> bool:
        """
        Compiles rendered HTML to high-resolution broadsheet PDF.
        Supports both Headless Browser (Chrome/Edge/Chromium) and WeasyPrint engines.
        """
        # Engine 1: Chrome / Chromium / Edge Headless CLI
        browser_exe = find_browser_executable()
        if browser_exe:
            try:
                logger.info(f"Compiling newspaper PDF with Headless Browser ({os.path.basename(browser_exe)})...")
                abs_html = os.path.abspath(html_path)
                abs_pdf = os.path.abspath(pdf_path)

                cmd = [
                    browser_exe,
                    "--headless",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={abs_pdf}",
                    abs_html,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
                if os.path.exists(abs_pdf) and os.path.getsize(abs_pdf) > 1000:
                    logger.info(f"PDF successfully generated via Browser ({os.path.getsize(abs_pdf):,} bytes).")
                    return True
                else:
                    logger.warning(f"Browser PDF failed (code {res.returncode}): {res.stderr}")
            except Exception as e:
                logger.warning(f"Browser PDF generation error: {e}")

        # Engine 2: WeasyPrint Engine
        if HAS_WEASYPRINT:
            try:
                logger.info("Attempting WeasyPrint PDF compilation...")
                wp_html = weasyprint.HTML(string=html_content, base_url=self.templates_dir)
                wp_html.write_pdf(target=pdf_path, presentational_hints=True)
                if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
                    logger.info(f"PDF successfully generated via WeasyPrint ({os.path.getsize(pdf_path):,} bytes).")
                    return True
            except Exception as e:
                logger.warning(f"WeasyPrint compilation error: {e}")

        logger.error("No compatible PDF engine succeeded in compiling the PDF.")
        return False

    def build_markdown_report(
        self,
        stories: List[Dict[str, Any]],
        edition_date: str,
        total_sources: int,
    ) -> str:
        """Generates comprehensive, searchable Markdown edition archiving 100% of stories."""
        dt = datetime.strptime(edition_date, "%Y-%m-%d")
        formatted_date = dt.strftime("%A, %B %d, %Y")
        front_leads, sections_data = self.group_stories_by_section(stories)

        lines = [
            f"# THE DAILY CHRONICLE — INDIA & BENGAL EDITION",
            f"**Autonomous 24/7 Intelligence Journal** | *{formatted_date}*",
            f"",
            f"> **Edition Stats:** {len(stories)} Verified Stories | {total_sources} Outlets & Media Sources | 100% Retention",
            f"",
            f"---",
            f"",
            f"## ★ FRONT PAGE LEAD HEADLINES",
            f"",
        ]

        for idx, story in enumerate(front_leads, start=1):
            sources_str = ", ".join(story.get("sources", [])[:4])
            lines.extend([
                f"### {idx}. {story.get('headline', '')}",
                f"- **Tag:** `[{story.get('urgency_tag', 'VERIFIED')}]` `[{story.get('state_or_region') or story.get('category')}]`",
                f"- **Summary:** {story.get('summary', '')}",
                f"- **Key Takeaway:** {story.get('key_takeaway', '')}",
                f"- **Sources ({story.get('articles_count', 1)}):** {sources_str}",
            ])
            if story.get("urls"):
                lines.append(f"- **Read Original:** [Source Link]({story['urls'][0]})")
            lines.append("")

        section_titles = [
            ("kolkata_bengal", "❖ KOLKATA & WEST BENGAL DESK"),
            ("indian_states", "❖ INDIAN STATES & UNION TERRITORIES DIGEST"),
            ("national", "❖ NATIONAL AFFAIRS, GOVERNANCE & ECONOMY"),
            ("global", "❖ WORLD & INTERNATIONAL AFFAIRS DESK"),
            ("trends", "❖ VIRAL TRENDS & SOCIAL SENTIMENT RADAR"),
        ]

        for sec_key, title in section_titles:
            sec_stories = sections_data.get(sec_key, [])
            if not sec_stories:
                continue

            lines.extend([
                f"---",
                f"",
                f"## {title} ({len(sec_stories)} Stories)",
                f"",
            ])

            for story in sec_stories:
                tag = story.get("state_tag") or story.get("state_or_region") or sec_key.upper()
                sources_str = ", ".join(story.get("sources", [])[:3])
                lines.extend([
                    f"#### [{tag}] {story.get('headline', '')}",
                    f"*{story.get('summary', '')}*",
                    f"",
                    f"> **Takeaway:** {story.get('key_takeaway', '')}",
                    f"",
                    f"- *Outlets:* {sources_str} | [Link]({story.get('urls', ['#'])[0]})",
                    f"",
                ])

        lines.extend([
            f"---",
            f"*Generated autonomously by Ppodong News Intelligence System on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*",
        ])

        return "\n".join(lines)

    def publish_edition(
        self,
        stories: List[Dict[str, Any]],
        edition_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Coordinates full publishing process:
        Renders HTML, writes HTML file, compiles WeasyPrint PDF, and writes Markdown report.
        """
        if not edition_date:
            edition_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Collect unique source count
        all_sources = set()
        for s in stories:
            for src in s.get("sources", []):
                all_sources.add(src)
        total_sources = len(all_sources)

        target_dir, pdf_path, md_path, html_path = self.prepare_edition_paths(edition_date)

        # 1. Render and write HTML
        rendered_html = self.render_html(stories, edition_date, total_sources)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        logger.info(f"Saved HTML edition to {html_path}")

        # 2. Compile PDF
        pdf_success = self.compile_pdf(html_path, rendered_html, pdf_path)

        # 3. Render and write Markdown archive
        md_content = self.build_markdown_report(stories, edition_date, total_sources)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Saved Markdown archive to {md_path}")

        return {
            "date": edition_date,
            "target_dir": target_dir,
            "html_path": html_path,
            "pdf_path": pdf_path if (pdf_success and os.path.exists(pdf_path)) else None,
            "md_path": md_path,
            "total_stories": len(stories),
            "total_sources": total_sources,
        }
