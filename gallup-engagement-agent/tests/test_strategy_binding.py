"""Offline checks of the Strategy presentation against controlled current results."""

import ast
import html
import re
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.tools.action_story import MAX_START_TEAMS, where_to_start
from src.tools.data_ingestion import upload_fingerprint
from src.tools.financial import Assumptions, money


PAGE = Path(__file__).resolve().parents[1] / "app_pages" / "action_story.py"


class FakeStreamlit:
    def __init__(self):
        self.html_parts = []
        self.markdown_parts = []

    def container(self, **_):
        return nullcontext()

    def html(self, value):
        self.html_parts.append(value)

    def markdown(self, value):
        self.markdown_parts.append(value)


def render_summary(factor, count, value=2_400_000):
    """Execute only the actual top-of-page presentation block with controlled inputs."""
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "top_factor"
                     for target in node.targets))
    section = []
    for node in tree.body[start:]:
        section.append(node)
        if len([part for part in section if isinstance(part, ast.With)]) == 2:
            break
    code = compile(ast.Module(body=section, type_ignores=[]), str(PAGE), "exec")
    st = FakeStreamlit()
    frame = pd.DataFrame([{"label": factor, "yearly_value": value}]) if factor else pd.DataFrame()
    teams = pd.DataFrame({"why": ["High turnover risk · Highest extra cost"] * count})
    story = {"start_here": teams,
             "headline": "If nothing changes, about 120 people are likely to leave next year.",
             "chapters": {"predict": {"points": []}}}
    ns = {"factors": frame, "story": story, "c": "$", "st": st,
          "html": html, "re": re, "money": money, "MAX_START_TEAMS": MAX_START_TEAMS}
    exec(code, ns)
    return " ".join(st.html_parts), " ".join(st.markdown_parts)


def test_same_and_different_current_results():
    old, old_explanation = render_summary("Recognition", 10)
    new, new_explanation = render_summary("Learn and grow", 3, 500_000)
    assert "Recognition" in old and "10 teams" in old and "Recognition" in old_explanation
    assert "Learn and grow" in new and "3 teams" in new and "$500K" in new
    assert "Learn and grow" in new_explanation and "3 teams" in new_explanation
    assert "Recognition" not in new and "Recognition" not in new_explanation
    assert "Review the 10 teams" not in new and "the 10 teams shown" not in new_explanation


def test_one_team_and_missing_factor():
    one, _ = render_summary("Learn and grow", 1)
    missing, explanation = render_summary(None, 0)
    assert "1 team with" in one
    assert "No verified factor" in missing
    assert "No cross-signal teams" in missing
    assert "No verified company-wide factor" in explanation


def test_more_qualifying_teams_are_capped():
    ids = {f"T{i}" for i in range(MAX_START_TEAMS + 4)}
    facts = {"high_teams": {"turnover": ids, "absence": ids}}
    shown = where_to_start(facts, None, pd.DataFrame(), Assumptions())
    assert len(shown) == MAX_START_TEAMS
    output, explanation = render_summary("Recognition", len(shown))
    assert f"{MAX_START_TEAMS} teams" in output
    assert f"up to {MAX_START_TEAMS} teams" in explanation
    assert "others may qualify" in explanation


def test_equal_name_and_size_with_different_contents_have_distinct_identity():
    first = BytesIO(b"factor=Recognition")
    second = BytesIO(b"factor=OtherFactor")
    assert len(first.getvalue()) == len(second.getvalue())
    assert upload_fingerprint(first) != upload_fingerprint(second)
    assert upload_fingerprint(first) == upload_fingerprint(first)
