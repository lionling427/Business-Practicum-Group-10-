# APP LAYOUT – the one file to edit when you move, merge, rename or hide pages.
#
# The sidebar is built from NAVIGATION below: seven pages for CEOs and CFOs.
# Each entry is one sidebar page. An entry with one view runs that page file as
# it is; an entry with several views shows a switcher at the top and runs the
# chosen page file underneath. The page files in app_pages/ keep working on
# their own, so teammates can keep editing them and merges stay simple.
#
# "stack": True shows every view one under the other instead of a switcher.
# A view can hide sections of its page file: ("Name", "app_pages/x.py",
# {"hide": ["x.section"]}). The section names are listed in HIDE below and are
# checked in the page files with src.layout.hidden("x.section").
#
# To try a different layout: reorder entries, move a view to another entry,
# rename a title, or delete a line. Nothing else needs to change.

from pathlib import Path

import streamlit as st

APP_NAME = "EngagAI"
APP_TAGLINE = "Engagement intelligence for leaders"
_ASSETS = Path(__file__).resolve().parents[1] / "assets"
LOGO = str(_ASSETS / "engagai_logo.png")
LOGO_ICON = str(_ASSETS / "engagai_icon.png")

NAVIGATION = {
    "For CEO & CFO": [
        {
            "title": "Executive summary",
            "icon": ":material/summarize:",
            "default": True,
            "description": "The bottom line, what drives results, what it is worth and "
                           "what to decide; then where engagement stands.",
            "stack": True,
            "views": [
                ("Bottom line and decisions", "app_pages/ceo_brief.py", {"hide": ["page.header"]}),
                ("Engagement at a glance", "app_pages/overview.py",
                 {"hide": ["page.header", "overview.tiles"]}),
            ],
        },
        {
            "title": "Strategy: where to act",
            "icon": ":material/target:",
            "description": "The story in one sentence, what drives results, what it is worth, "
                           "where to act and which teams to start with.",
            "stack": True,
            "views": [
                ("Action story", "app_pages/action_story.py", {"hide": ["page.header"]}),
                ("Evidence", "app_pages/outcomes.py",
                 {"hide": ["page.header", "outcomes.all_but_evidence"]}),
            ],
        },
        {
            "title": "Financial impact",
            "icon": ":material/payments:",
            "description": "What turnover and absence cost today, what the recommended "
                           "program is worth, and what-if scenarios on the same numbers.",
            "views": [
                ("Cost and value", "app_pages/financials.py"),
                ("Scenarios", "app_pages/scenarios.py"),
            ],
        },
        {
            "title": "Predictions & risk",
            "icon": ":material/online_prediction:",
            "description": "Where engagement is heading, how it changed for the same "
                           "people, and where turnover risk is concentrated.",
            "stack": True,
            "views": [
                ("Engagement forecast", "app_pages/predictions.py",
                 {"hide": ["page.header", "predictions.group_risk", "predictions.monthly_risk"]}),
                ("Change for the same people", "app_pages/change.py",
                 {"hide": ["page.header", "change.teams"]}),
                ("Turnover risk by group", "app_pages/predictions.py",
                 {"hide": ["page.header", "predictions.forecast"]}),
            ],
        },
    ],
    "For HR and managers": [
        {
            "title": "Team drill-down",
            "icon": ":material/groups:",
            "description": "Which teams need support first, and a plan for each manager.",
            "views": [
                ("Watchlist and coaching", "app_pages/teams.py"),
                ("Team plans", "app_pages/recommendations.py"),
                ("Question scores", "app_pages/engagement.py"),
                ("Teams that moved", "app_pages/change.py",
                 {"hide": ["change.overall", "change.linked"]}),
                ("Strengths", "app_pages/strengths.py"),
            ],
        },
    ],
    "Evidence and tools": [
        {
            "title": "Trust",
            "icon": ":material/verified:",
            "description": "Can the data and the numbers be trusted?",
            "views": [
                ("Data quality", "app_pages/data_quality.py"),
                ("Statistical audit", "app_pages/audit.py"),
                ("Analytical checks", "app_pages/outcomes.py", {"hide": ["outcomes.drivers"]}),
                ("Prediction tests", "app_pages/teams.py", {"hide": ["teams.only_tests"]}),
            ],
        },
        {
            "title": "Tools",
            "icon": ":material/build:",
            "description": "Ask your own question, or take a deck to the board.",
            "views": [
                ("Ask AI", "app_pages/agent.py"),
                ("Board deck", "app_pages/report_pack.py"),
            ],
        },
    ],
}

# Sections hidden everywhere in this layout. Set to False to show again.
#   predictions.monthly_risk          monthly model on separate person-month or demo data
#   overview.key_findings             "Key findings" at the bottom of the overview
#   change.linked                     "Is change in engagement linked to results?"
# Sections a view can hide (see "hide" above):
#   page.header                       the page file's own title and description
#   outcomes.all_but_evidence         everything except the two evidence expanders
#   predictions.forecast / .group_risk  the engagement forecast / turnover risk by group
#   change.overall / .teams / .linked  overall change and chart / teams that moved /
#                                       is change linked to results
#   overview.tiles                    the headline tiles (the CEO brief shows its own)
#   outcomes.drivers                  tiles and "which questions matter"
#   outcomes.checks                   analytical traps and sorting effect
#   recommendations.where_to_start    tiles and the "where to start" chart
#   recommendations.team_plans        the team table and each team's plan
HIDE = {
    "predictions.monthly_risk": False,
    "overview.key_findings": True,
    "change.linked": True,
}

VIEW_HIDE_KEY = "_view_hide"


def hidden(section):
    """True when a page section is switched off, everywhere or in the current view."""
    if HIDE.get(section, False):
        return True
    return section in st.session_state.get(VIEW_HIDE_KEY, ())


# WHERE A PAGE FILE IS SHOWN – so text and tests never hard-code page names.

def slug(text):
    """URL path of a sidebar page: 'Team drill-down' -> 'team-drill-down'."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def locate(path, view=None):
    """Where a page file appears: the page title, its id for st.switch_page /
    AppTest.switch_page, and the view to pick (None on single or stacked pages).
    If the file appears more than once, the view that hides the least wins."""
    best, wanted = None, view
    for entries in NAVIGATION.values():
        for entry in entries:
            views = entry["views"]
            switcher = len(views) > 1 and not entry.get("stack")
            for view in views:
                if view[1] != path or (wanted and view[0] != wanted):
                    continue
                hide = view[2].get("hide", []) if len(view) > 2 else []
                score = len([h for h in hide if h != "page.header"])
                if best is None or score < best[0]:
                    best = (score, {
                        "title": entry["title"],
                        "page": entry["title"] if entry.get("default") else slug(entry["title"]),
                        "view": view[0] if switcher else None,
                        "view_key": f"view_{slug(entry['title'])}" if switcher else None,
                    })
    return best[1] if best else None


def where(path, bold=True, view=None):
    """'**Team drill-down → Team plans**': how to find a page file in the sidebar."""
    spot = locate(path, view)
    if spot is None:
        return ""
    label = spot["title"] + (f" → {spot['view']}" if spot["view"] else "")
    return f"**{label}**" if bold else label
