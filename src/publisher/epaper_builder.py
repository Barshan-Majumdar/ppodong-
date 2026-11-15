"""
Autonomous 24/7 E-Paper News Agent - Broadsheet PDF & Markdown Publisher
Renders multi-column broadsheet newspaper HTML via Jinja2 and compiles crisp paginated PDF
via WeasyPrint, generating paired Markdown and HTML editions archived in editions/YYYY/MM/.
"""

from __future__ import annotations

import logging
import os
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
        Groups stories into Front Page Leads and exact desk sections.
        Ensures ZERO stories are excluded or dropped.
        """
        # Pick top 6-8 highest scoring stories for Front Page Lead
        front_leads = stories[: min(6, len(stories))]

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

    def compile_pdf(self, html_content: str, pdf_path: str) -> bool:
        """Compiles rendered HTML string to PDF using WeasyPrint."""
        if not HAS_WEASYPRINT:
            logger.warning(
                "WeasyPrint or required C-libraries (Pango/Cairo) not found in runtime. "
                "Skipping direct WeasyPrint PDF compilation (HTML and Markdown preserved)."
            )
            return False

        try:
            logger.info(f"Compiling newspaper PDF with WeasyPrint: {pdf_path}...")
            # WeasyPrint font & media configuration
            wp_html = weasyprint.HTML(string=html_content, base_url=self.templates_dir)
            wp_html.write_pdf(target=pdf_path, presentational_hints=True)
            logger.info(f"PDF successfully generated ({os.path.getsize(pdf_path)} bytes).")
            return True
        except Exception as e:
            logger.error(f"Error during WeasyPrint compilation: {e}")
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
            f"*Generated autonomously by Antigravity News Intelligence System on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*",
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
        pdf_success = self.compile_pdf(rendered_html, pdf_path)

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
