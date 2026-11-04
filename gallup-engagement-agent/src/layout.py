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
            "description": "Where engagement stands now and how it is changing.",
            "views": [
                ("Engagement at a glance", "app_pages/overview.py", {"hide": ["page.header"]}),
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
                 {"hide": ["page.header", "predictions.group_risk"]}),
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
#   outcomes.drivers                  tiles and "which questions matter"
#   outcomes.checks                   analytical traps and sorting effect
#   recommendations.where_to_start    tiles and the "where to start" chart
#   recommendations.team_plans        the team table and each team's plan
HIDE = {
    "predictions.monthly_risk": True,
    "overview.key_findings": True,
    "change.linked": True,
}

VIEW_HIDE_KEY = "_view_hide"


def hidden(section):
    """True when a page section is switched off, everywhere or in the current view."""
    if HIDE.get(section, False):
        return True
    return section in st.session_state.get(VIEW_HIDE_KEY, ())
