"""PDF export: every report in the app downloads as a branded A4 PDF.

document_pdf(title, sections) lays out headings, paragraphs, bullet points and
small tables on as many pages as needed. markdown_pdf(title, text) turns the
app's Markdown reports (headings, bullets, tables) into the same layout.
Only matplotlib is used, so no extra package is needed.
"""

from __future__ import annotations

import io
import re
import textwrap
from datetime import date
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

# EngagAI brand
GREEN, DARK_GREEN, BUTTER, BURGUNDY = "#2F6B3F", "#1F4A2E", "#F3D77A", "#8C2A3A"
TEXT, MUTED, CREAM = "#1F2A22", "#6E6A5E", "#FFFDF5"

PAGE_W, PAGE_H = 8.27, 11.69          # A4, inches
MARGIN = 0.75
LINE = 0.205                          # inches per 10-pt line
WRAP = 98                             # characters per line at 10 pt


def _clean(text: Any) -> str:
    """Plain text: no Markdown emphasis, no Streamlit icons or badges."""
    text = str(text)
    text = re.sub(r":material/[a-z_]+:", "", text)
    text = re.sub(r":[a-z]+-badge\[([^\]]*)\]", r"\1", text)
    text = text.replace("**", "").replace("\\$", "$").replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


class _Writer:

    def __init__(self, pdf: PdfPages, title: str, subtitle: str):
        self.pdf, self.title, self.subtitle = pdf, title, subtitle
        self.page, self.fig, self.y = 0, None, 0.0
        self._new_page()

    def _new_page(self):
        if self.fig is not None:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page += 1
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.fig.patch.set_facecolor("white")
        band = 1.25 if self.page == 1 else 0.55
        self.fig.add_artist(plt.Rectangle((0, 1 - band / PAGE_H), 1, band / PAGE_H,
                                          transform=self.fig.transFigure,
                                          color=DARK_GREEN, zorder=0))
        self.fig.add_artist(plt.Rectangle((0, 1 - band / PAGE_H - 0.05 / PAGE_H), 1,
                                          0.05 / PAGE_H, transform=self.fig.transFigure,
                                          color=BUTTER, zorder=0))
        self._text(MARGIN, 0.38, "Engag", 11, CREAM, weight="bold")
        self._text(MARGIN + 0.53, 0.38, "AI", 11, BUTTER, weight="bold")
        if self.page == 1:
            self._text(MARGIN, 0.82, self.title, 20, CREAM, weight="bold")
            if self.subtitle:
                self._text(MARGIN, 1.08, self.subtitle, 9.5, "#CFE0C9")
        self.y = band + 0.45

    def _footer(self):
        self._text(MARGIN, PAGE_H - 0.45,
                   "Associations in your data, not guarantees · EngagAI", 7.5, MUTED)
        self._text(PAGE_W - MARGIN, PAGE_H - 0.45, f"Page {self.page}", 7.5, MUTED, ha="right")

    def _text(self, x, y, s, size, color, weight="normal", ha="left", family=None):
        self.fig.text(x / PAGE_W, 1 - y / PAGE_H, s, fontsize=size, color=color,
                      fontweight=weight, ha=ha, va="top", parse_math=False,
                      family=family or "sans-serif")

    def _room(self, height):
        if self.y + height > PAGE_H - 0.8:
            self._new_page()

    def heading(self, text, color=GREEN):
        self._room(LINE * 3)
        self.y += LINE * 0.4
        self._text(MARGIN, self.y, _clean(text), 13, color, weight="bold")
        self.y += LINE * 1.6

    def paragraph(self, text, size=10, color=TEXT, weight="normal", indent=0.0, bullet=None):
        width = int(WRAP * 10 / size) - int(indent * 12)
        lines = textwrap.wrap(_clean(text), width) or [""]
        for i, line in enumerate(lines):
            self._room(LINE)
            if bullet and i == 0:
                self._text(MARGIN + indent - 0.18, self.y, bullet, size, GREEN, weight="bold")
            self._text(MARGIN + indent, self.y, line, size, color, weight=weight)
            self.y += LINE * size / 10
        self.y += LINE * 0.35

    def bullets(self, items, numbered=False):
        for i, item in enumerate(items, 1):
            self.paragraph(item, indent=0.28, bullet=f"{i}." if numbered else "•")

    def table(self, frame: pd.DataFrame, max_rows: int = 25):
        """Each row as a short block: the first column bold, the rest as 'name: value'."""
        if frame is None or frame.empty:
            return
        columns = list(frame.columns)
        for _, row in frame.head(max_rows).iterrows():
            self._room(LINE * 2)
            self.paragraph(f"{columns[0]}: {row[columns[0]]}", weight="bold", indent=0.28,
                           bullet="•")
            self.paragraph(" · ".join(f"{c}: {row[c]}" for c in columns[1:]),
                           size=9, color=MUTED, indent=0.28)

    def chart(self, draw, height: float = 2.4, title: str = ""):
        """A matplotlib chart across the page: draw(ax) fills the axes."""
        self._room(height + (LINE * 1.4 if title else 0) + 0.2)
        if title:
            self._text(MARGIN, self.y, _clean(title), 10, TEXT, weight="bold")
            self.y += LINE * 1.4
        left, width = (MARGIN + 1.9) / PAGE_W, (PAGE_W - 2 * MARGIN - 2.4) / PAGE_W
        bottom = 1 - (self.y + height - 0.35) / PAGE_H
        ax = self.fig.add_axes([left, bottom, width, (height - 0.45) / PAGE_H])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#BBBBBB")
        ax.spines["bottom"].set_color("#BBBBBB")
        ax.tick_params(labelsize=8, colors=TEXT)
        draw(ax)
        self.y += height + 0.15

    def close(self):
        self._footer()
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def document_pdf(title: str, sections: list[dict[str, Any]], subtitle: str = "") -> bytes:
    """sections: [{"heading": str, "lead": str, "paragraphs": [...], "bullets": [...],
    "numbered": bool, "charts": [{"draw": fn(ax), "height": in, "title": str}],
    "table": DataFrame}] -> PDF bytes."""
    subtitle = subtitle or f"Prepared {date.today():%B %d, %Y}"
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        w = _Writer(pdf, title, subtitle)
        for section in sections:
            if section.get("heading"):
                w.heading(section["heading"], section.get("color", GREEN))
            if section.get("lead"):
                w.paragraph(section["lead"], size=11, weight="bold")
            for p in section.get("paragraphs", []):
                w.paragraph(p)
            if section.get("bullets"):
                w.bullets(section["bullets"], section.get("numbered", False))
            for chart in section.get("charts", []):
                w.chart(chart["draw"], chart.get("height", 2.4), chart.get("title", ""))
            if section.get("table") is not None:
                w.table(section["table"])
        w.close()
    return buffer.getvalue()


def markdown_pdf(title: str, markdown: str, subtitle: str = "") -> bytes:
    """The app's Markdown reports (#, -, 1., | tables, paragraphs) as a PDF."""
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] = {"paragraphs": [], "bullets": []}
    header: list[str] | None = None

    def flush():
        nonlocal current
        if current.get("heading") or current["paragraphs"] or current["bullets"]:
            sections.append(current)
        current = {"paragraphs": [], "bullets": []}

    first = True
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            header = None
            continue
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            if first and line.startswith("# "):     # the title is already on the page
                first = False
                continue
            flush()
            current["heading"] = text
            first = False
            continue
        first = False
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if header is None:
                header = cells
                continue
            current["bullets"].append(" · ".join(
                f"{h}: {v}" if h else v for h, v in zip(header, cells)))
            continue
        bullet = re.match(r"^([-*]|\d+\.)\s+(.*)", line)
        if bullet:
            current["bullets"].append(bullet.group(2))
        else:
            if current["bullets"]:               # keep the reading order
                flush()
            current["paragraphs"].append(line)
    flush()
    return document_pdf(title, sections, subtitle)


# PROCESS – CHART HELPERS FOR PDF PAGES

def pdf_text(value: str) -> str:
    """Text safe for matplotlib labels ('$' is not read as math)."""
    return str(value).replace("$", r"\$")


def barh(labels, values, colors, value_labels=None, ranges=None, xlabel=""):
    """A ranked horizontal bar chart, largest at the top, for document_pdf charts."""
    def draw(ax):
        n = len(values)
        ys = list(range(n))[::-1]
        ax.barh(ys, values, color=colors, height=0.6)
        ax.set_yticks(ys, [pdf_text(l) for l in labels])
        if ranges:
            for y, (lo, hi) in zip(ys, ranges):
                if lo is not None and hi is not None and lo == lo and hi == hi:
                    ax.plot([lo, hi], [y, y], color="#C9A227", linewidth=3,
                            solid_capstyle="round")
        top = max([v for v in values] + [hi for _, hi in (ranges or []) if hi == hi and hi]
                  + [0])
        ax.set_xlim(0, top * 1.25 if top > 0 else 1)
        ends = [max(v, hi) if hi is not None and hi == hi else v
                for v, (_, hi) in zip(values, ranges or [(None, None)] * n)]
        for y, v, text in zip(ys, ends, value_labels or [""] * n):
            if text:
                ax.text(v + top * 0.02, y, text, va="center", fontsize=8, color=TEXT,
                        fontweight="bold", parse_math=False)
        ax.set_xlabel(pdf_text(xlabel), fontsize=8, color=MUTED)
        ax.xaxis.set_major_formatter(lambda x, _: pdf_text(_short(x)))
    return draw


def _short(x: float) -> str:
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= size:
            return f"{x / size:.1f}".rstrip("0").rstrip(".") + unit
    return f"{x:,.0f}"
