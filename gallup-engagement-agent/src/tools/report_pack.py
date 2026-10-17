"""Report pack: a branded deck (PowerPoint) and PDF for leaders to take away.

Every number comes from results the app already verified (CEO brief, Action
story, factor values, audit). No AI writes it. One content model, two renderers:

  build_pack(df, analysis, ...)  -> list of slides (plain dicts)
  to_pptx(slides, branding)      -> .pptx bytes (editable text and tables)
  to_pdf(slides, branding)       -> .pdf bytes (same layout, one slide per page)

Slides: title · bottom line · what drives results · what it is worth · what is
likely to happen · where to start · decisions · how sure we are · about.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

import io
import re
import textwrap
from dataclasses import dataclass
from datetime import date
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import pandas as pd                                    # noqa: E402

from .action_story import build_action_story           # noqa: E402
from .ceo_brief import build_ceo_brief                 # noqa: E402
from .financial import Assumptions, money              # noqa: E402
from .insights import analysis_frame                   # noqa: E402
from .priorities import PRIORITY_COLORS, PRIORITY_ORDER, factor_value  # noqa: E402


# PROCESS 2 – BRANDING

@dataclass
class Branding:
    company: str = "Your organization"
    color: str = "#1E7B34"                 # accent: title bar, numbers, chart bars
    logo: bytes | None = None              # PNG or JPG
    prepared_by: str = "People Analytics"


FONT = "Arial"
SECTIONS = ["Bottom line", "What drives results", "What it is worth",
            "What is likely to happen", "Where to start", "Decisions",
            "How sure we are"]
DISCLAIMER = ("Built from your survey and HR data with fixed statistical methods. Results are "
              "associations, not proof of cause; money figures use the cost assumptions shown. "
              "Gallup and Q12 are trademarks of Gallup; this report is not endorsed by Gallup.")


def _plain(text: str) -> str:
    """Strip markdown bold and escapes for slides."""
    return re.sub(r"\*\*|\\\$", lambda m: "$" if m.group(0) == "\\$" else "", str(text))


# PROCESS 3 – CONTENT (ONE LIST OF SLIDES)

def build_pack(df: pd.DataFrame, analysis: dict | None, assumptions: Assumptions | None = None,
               traps: dict | None = None, change_analysis: dict | None = None,
               audit: dict | None = None, sections: list[str] | None = None) -> list[dict]:
    a = assumptions or Assumptions(headcount=len(analysis_frame(df)))
    wanted = set(sections or SECTIONS)
    brief = build_ceo_brief(df, analysis, a, traps)
    story = build_action_story(df, analysis, change_analysis, a, traps)
    factors = factor_value(df, analysis, a, traps)

    slides: list[dict] = [{"kind": "title", "title": "Engagement and business results",
                           "subtitle": f"{brief['survey'].capitalize()} · {date.today():%B %Y}"}]
    if "Bottom line" in wanted:
        slides.append({"kind": "content", "title": "The bottom line", "lead": brief["bottom_line"],
                       "metrics": [(s["label"], s["value"], s["change"]) for s in brief["scorecard"]],
                       "bullets": []})
    if "What drives results" in wanted and brief["what_matters"]:
        slides.append({"kind": "content", "title": "What drives results",
                       "lead": "The strongest links that passed every statistical check.",
                       "bullets": brief["what_matters"]})
    if "What it is worth" in wanted and brief["money"]:
        chart = factor_chart_png(factors, a.currency) if not factors.empty else None
        slides.append({"kind": "content", "title": "What it is worth",
                       "lead": story["chapters"]["money"]["lead"],
                       "bullets": brief["money"], "chart": chart})
    if "What is likely to happen" in wanted and story["chapters"]["predict"]["available"]:
        chapter = story["chapters"]["predict"]
        slides.append({"kind": "content", "title": "What is likely to happen",
                       "lead": chapter["lead"], "bullets": chapter["points"][:4]})
    if "Where to start" in wanted:
        start = story["start_here"]
        if not start.empty:
            table = start[["team_id", "department", "why", "start_with"]].head(8).rename(columns={
                "team_id": "Team", "department": "Department", "why": "Why it is here",
                "start_with": "First step"})
            slides.append({"kind": "table", "title": "Where to start",
                           "lead": "Teams that show up on several lists: high risk, biggest "
                                   "opportunity or highest extra cost.",
                           "table": table, "bullets": brief["where_to_act"][:2]})
    if "Decisions" in wanted:
        slides.append({"kind": "content", "title": "Decisions for you",
                       "lead": "Three decisions this report supports.",
                       "bullets": brief["decisions"], "numbered": True})
    if "How sure we are" in wanted:
        bullets = list(brief["confidence"])
        if audit is not None:
            status, summary, detail = audit["verdict"]
            bullets.insert(0, f"Statistical audit: {summary}. {detail}")
        slides.append({"kind": "content", "title": "How sure we are",
                       "lead": "Every number was computed by fixed, checked methods.",
                       "bullets": bullets})
    slides.append({"kind": "content", "title": "About this report", "lead": DISCLAIMER,
                   "bullets": [f"Cost assumptions: average salary {money(a.avg_salary, a.currency)}, "
                               f"replacing a leaver costs {a.replacement_cost_pct:.0%} of salary.",
                               "Only groups of 5 or more people are shown; no individual scores."]})
    for slide in slides:
        for key in ("title", "subtitle", "lead"):
            if key in slide:
                slide[key] = _plain(slide[key])
        slide["bullets"] = [_plain(b) for b in slide.get("bullets", [])]
    return slides


# PROCESS 4 – CHART (SHARED BY BOTH FORMATS)

def factor_chart_png(factors: pd.DataFrame, currency: str = "$") -> bytes:
    data = factors[factors["yearly_value"] > 0].head(6).iloc[::-1]
    colors = dict(zip(PRIORITY_ORDER, PRIORITY_COLORS))
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=200)
    bars = ax.barh(data["label"], data["yearly_value"],
                   color=[colors[p] for p in data["priority"]])
    for bar, value in zip(bars, data["yearly_value"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, "  " + money(value, currency),
                va="center", fontsize=9)
    ax.set_xlim(0, data["yearly_value"].max() * 1.3)
    ax.set_xlabel(f"Estimated yearly value ({currency})", fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v, currency)))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)
    used = [p for p in PRIORITY_ORDER if p in set(data["priority"])]
    ax.legend(handles=[matplotlib.patches.Patch(color=colors[p], label=p) for p in used],
              fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _col_widths(table: pd.DataFrame) -> list[float]:
    """Column shares by the longest text in each column (sum to 1)."""
    longest = [max([len(str(c)), 12] + [len(str(v)) + 3 for v in table[c]])  # floor + padding
               for c in table.columns]
    total = sum(longest)
    return [n / total for n in longest]


# PROCESS 5 – POWERPOINT

def to_pptx(slides: list[dict], branding: Branding) -> bytes:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt

    accent = RGBColor.from_string(branding.color.lstrip("#").upper())
    dark, grey = RGBColor(0x22, 0x22, 0x22), RGBColor(0x66, 0x66, 0x66)
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    blank = deck.slide_layouts[6]

    def text(slide, x, y, w, h, value, size, color=dark, bold=False):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.word_wrap = True
        frame.paragraphs[0].text = value
        for run in frame.paragraphs[0].runs:
            run.font.size, run.font.bold, run.font.color.rgb = Pt(size), bold, color
            run.font.name = FONT
        return frame

    def chrome(slide, number):
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, deck.slide_width, Inches(0.12))
        bar.fill.solid(); bar.fill.fore_color.rgb = accent; bar.line.fill.background()
        if branding.logo:
            slide.shapes.add_picture(io.BytesIO(branding.logo), Inches(11.6), Inches(0.3),
                                     height=Inches(0.6))
        text(slide, 0.6, 7.0, 9, 0.3, f"{branding.company} · Engagement and business results",
             9, grey)
        text(slide, 12.2, 7.0, 0.6, 0.3, str(number), 9, grey)

    for number, s in enumerate(slides, 1):
        slide = deck.slides.add_slide(blank)
        if s["kind"] == "title":
            band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.4), deck.slide_width,
                                          Inches(2.6))
            band.fill.solid(); band.fill.fore_color.rgb = accent; band.line.fill.background()
            white = RGBColor(0xFF, 0xFF, 0xFF)
            text(slide, 0.8, 2.75, 11.5, 0.8, branding.company, 36, white, True)
            text(slide, 0.8, 3.6, 11.5, 0.6, s["title"], 24, white)
            text(slide, 0.8, 4.25, 11.5, 0.5, s["subtitle"], 16, white)
            text(slide, 0.8, 6.6, 11.5, 0.4, f"Prepared by {branding.prepared_by}", 12, grey)
            if branding.logo:
                slide.shapes.add_picture(io.BytesIO(branding.logo), Inches(0.8), Inches(0.8),
                                         height=Inches(1.0))
            continue

        chrome(slide, number)
        text(slide, 0.6, 0.35, 10.8, 0.7, s["title"], 30, dark, True)
        text(slide, 0.6, 1.1, 12.0, 0.9, s["lead"], 16, accent, True)
        top = 2.15

        if s.get("metrics"):
            width = 12.1 / len(s["metrics"])
            for i, (label, value, note) in enumerate(s["metrics"]):
                x = 0.6 + i * width
                card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(top),
                                              Inches(width - 0.15), Inches(1.5))
                card.fill.solid(); card.fill.fore_color.rgb = RGBColor(0xF4, 0xF6, 0xF4)
                card.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
                text(slide, x + 0.15, top + 0.1, width - 0.4, 0.3, label, 11, grey)
                text(slide, x + 0.15, top + 0.4, width - 0.4, 0.6, value, 26, dark, True)
                text(slide, x + 0.15, top + 1.05, width - 0.4, 0.3, note, 10, grey)
            top += 1.8

        if s.get("table") is not None:
            table = s["table"]
            shape = slide.shapes.add_table(len(table) + 1, len(table.columns), Inches(0.6),
                                           Inches(top), Inches(12.1), Inches(0.4 * (len(table) + 1)))
            grid = shape.table
            for c, share in enumerate(_col_widths(table)):
                grid.columns[c].width = int(Inches(12.1) * share)
            for c, name in enumerate(table.columns):
                grid.cell(0, c).text = str(name)
            for r, row in enumerate(table.itertuples(index=False), 1):
                for c, value in enumerate(row):
                    grid.cell(r, c).text = str(value)
            for cell in (grid.cell(r, c) for r in range(len(table) + 1) for c in range(len(table.columns))):
                for p in cell.text_frame.paragraphs:
                    for run in p.runs:
                        run.font.size, run.font.name = Pt(11), FONT
            top += 0.4 * (len(table) + 1) + 0.3

        bullet_width = 6.6 if s.get("chart") else 12.1
        if s.get("bullets"):
            frame = text(slide, 0.6, top, bullet_width, 7.0 - top, "", 14)
            for i, item in enumerate(s["bullets"]):
                p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
                p.text = (f"{i + 1}. " if s.get("numbered") else "•  ") + item
                p.space_after = Pt(8)
                for run in p.runs:
                    run.font.size, run.font.color.rgb, run.font.name = Pt(14), dark, FONT
        if s.get("chart"):
            slide.shapes.add_picture(io.BytesIO(s["chart"]), Inches(7.4), Inches(top),
                                     width=Inches(5.4))

    buffer = io.BytesIO()
    deck.save(buffer)
    return buffer.getvalue()


# PROCESS 6 – PDF (SAME LAYOUT, DRAWN WITH MATPLOTLIB)

def _figures(slides: list[dict], branding: Branding):
    """Yield one matplotlib figure per slide (used for the PDF and the previews)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    W, H = 13.333, 7.5
    accent, dark, grey = branding.color, "#222222", "#666666"
    logo = plt.imread(io.BytesIO(branding.logo)) if branding.logo else None

    def put(ax, x, y, value, size, color=dark, bold=False, width_in=12.0):
        chars = max(20, int(width_in * 120 / size))       # rough characters per line
        wrapped = "\n".join(textwrap.fill(line, chars) for line in str(value).split("\n"))
        ax.text(x / W, 1 - y / H, wrapped, fontsize=size, color=color, va="top",
                fontweight="bold" if bold else "normal", transform=ax.transAxes, wrap=False,
                parse_math=False)                       # "$2.5M ... $22M" is money, not math
        return wrapped.count("\n") + 1

    for number, s in enumerate(slides, 1):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        if s["kind"] == "title":
            ax.add_patch(Rectangle((0, 1 - 5.0 / H), 1, 2.6 / H, color=accent, transform=ax.transAxes))
            put(ax, 0.8, 2.75, branding.company, 36, "white", True)
            put(ax, 0.8, 3.65, s["title"], 24, "white")
            put(ax, 0.8, 4.3, s["subtitle"], 16, "white")
            put(ax, 0.8, 6.6, f"Prepared by {branding.prepared_by}", 12, grey)
            if logo is not None:
                fig.add_axes([0.8 / W, 1 - 1.9 / H, 2.2 / W, 1.0 / H]).imshow(logo)
                fig.axes[-1].axis("off")
            yield fig
            continue

        ax.add_patch(Rectangle((0, 1 - 0.12 / H), 1, 0.12 / H, color=accent, transform=ax.transAxes))
        if logo is not None:
            fig.add_axes([11.4 / W, 1 - 0.95 / H, 1.6 / W, 0.6 / H]).imshow(logo)
            fig.axes[-1].axis("off")
        put(ax, 0.6, 0.4, s["title"], 30, dark, True)
        lines = put(ax, 0.6, 1.15, s["lead"], 16, accent, True)
        top = max(2.15, 1.15 + 0.32 * lines + 0.25)
        put(ax, 0.6, 7.05, f"{branding.company} · Engagement and business results", 9, grey)
        put(ax, 12.2, 7.05, str(number), 9, grey)

        if s.get("metrics"):
            width = 12.1 / len(s["metrics"])
            for i, (label, value, note) in enumerate(s["metrics"]):
                x = 0.6 + i * width
                ax.add_patch(FancyBboxPatch((x / W, 1 - (top + 1.5) / H), (width - 0.15) / W,
                                            1.5 / H, boxstyle="round,pad=0.002",
                                            facecolor="#F4F6F4", edgecolor="#DDDDDD",
                                            transform=ax.transAxes))
                put(ax, x + 0.15, top + 0.12, label, 11, grey, width_in=width)
                put(ax, x + 0.15, top + 0.45, value, 24, dark, True, width_in=width)
                put(ax, x + 0.15, top + 1.1, note, 10, grey, width_in=width)
            top += 1.8

        if s.get("table") is not None:
            table = s["table"]
            cell = ax.table(cellText=table.astype(str).to_numpy(), colLabels=list(table.columns),
                            cellLoc="left", colLoc="left", colWidths=_col_widths(table),
                            bbox=[0.6 / W, 1 - (top + 0.38 * (len(table) + 1)) / H,
                                  12.1 / W, 0.38 * (len(table) + 1) / H])
            cell.auto_set_font_size(False); cell.set_fontsize(10)
            for (r, _), c in cell.get_celld().items():
                c.set_edgecolor("#DDDDDD")
                if r == 0:
                    c.set_facecolor(accent); c.get_text().set_color("white")
            top += 0.38 * (len(table) + 1) + 0.35

        bullet_width = 6.6 if s.get("chart") else 12.1
        y = top
        for i, item in enumerate(s.get("bullets", [])):
            marker = f"{i + 1}. " if s.get("numbered") else "•  "
            lines = put(ax, 0.6, y, marker + item, 14, dark, width_in=bullet_width)
            y += 0.3 * lines + 0.18
        if s.get("chart"):
            chart = plt.imread(io.BytesIO(s["chart"]))
            height = 5.4 * chart.shape[0] / chart.shape[1]
            fig.add_axes([7.4 / W, 1 - (top + height) / H, 5.4 / W, height / H]).imshow(chart)
            fig.axes[-1].axis("off")
        yield fig


def to_pdf(slides: list[dict], branding: Branding) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for fig in _figures(slides, branding):
            pdf.savefig(fig)
            plt.close(fig)
    return buffer.getvalue()


def slide_previews(slides: list[dict], branding: Branding, dpi: int = 60) -> list[bytes]:
    """Small PNG images of each slide, for previewing before download."""
    images = []
    for fig in _figures(slides, branding):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
        images.append(buffer.getvalue())
    return images
