"""Gallup action tools: link each recommendation to Gallup action planning and
manager learning resources, and turn it into an action plan.

Resources live in config/gallup_resources.json (editable, checked links):
  elements          one Gallup article per Q12 element ("The Fourth Element of
                    Great Managing" for Q04 Recognition, ...)
  action_planning   Gallup Access action planning help + survey-results guidance
  manager_learning  Gallup manager courses and guides
  gallup_access_url optional: the organization's own Gallup Access sign-in link

action_plan_rows() writes plans with the fields Gallup's action planning uses
(goal, next steps, owner, due date, success indicator, follow-up), one row per
step, ready to copy into Gallup Access or any tracker. It does not upload
anything: there is no public Gallup Access API.

This app is independent and not endorsed by Gallup; links point to Gallup's
public pages.

    python -m src.tools.action_tools --check     # check every link still works
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .insights import ACTION_LIBRARY, ITEM_LABELS


# PROCESS 2 – CONFIGURATION

RESOURCE_FILE = Path(__file__).resolve().parents[2] / "config" / "gallup_resources.json"
DISCLAIMER = ("External links to Gallup's public resources. This app is independent and "
              "not endorsed by Gallup.")
# (when, days from the plan start) – matches the Recommendations timeline.
PLAN_STEPS = [("Weeks 1–2", 14), ("Weeks 3–6", 42), ("Week 8", 56)]
FOLLOW_UP_DAYS = 90
PLAN_COLUMNS = ["team_id", "department", "q12_item", "element", "goal", "step", "next_step",
                "owner", "due_date", "success_indicator", "follow_up", "learn_more",
                "action_planning_help"]


@lru_cache(maxsize=4)
def load_resources(path: str | None = None) -> dict[str, Any]:
    """The resource file, read once."""
    return json.loads(Path(path or RESOURCE_FILE).read_text(encoding="utf-8"))


# PROCESS 3 – RESOURCES FOR A RECOMMENDATION

def element_resource(item: str) -> dict[str, str] | None:
    entry = load_resources()["elements"].get(item)
    if not entry:
        return None
    return {**entry, "kind": "Element guide",
            "why": f"Gallup on {ITEM_LABELS.get(item, item)}: why it matters and how managers move it."}


def action_planning_links() -> list[dict[str, str]]:
    res = load_resources()
    links = [{**r, "kind": "Action planning"} for r in res["action_planning"]]
    if res.get("gallup_access_url"):
        links.insert(0, {"title": "Open Gallup Access (your organization)",
                         "url": res["gallup_access_url"], "kind": "Action planning",
                         "why": "Create this plan for the team in Gallup Access."})
    return links


def manager_learning_links() -> list[dict[str, str]]:
    return [{**r, "kind": "Manager learning"} for r in load_resources()["manager_learning"]]


def resources_for(item: str | None) -> list[dict[str, str]]:
    """Element guide for the item, then action planning, then manager learning."""
    element = element_resource(item) if item else None
    return ([element] if element else []) + action_planning_links() + manager_learning_links()


def links_markdown(resources: list[dict[str, str]], with_why: bool = True) -> str:
    """Bulleted markdown links, grouped by kind."""
    lines, kind = [], None
    for r in resources:
        if r["kind"] != kind:
            kind = r["kind"]
            lines += ([""] if lines else []) + [f"**{kind}**", ""]
        lines.append(f"- [{r['title']}]({r['url']})" + (f": {r['why']}" if with_why and r.get("why") else ""))
    return "\n".join(lines)


# PROCESS 4 – ACTION PLANS (GALLUP ACTION-PLANNING FIELDS)

def _element(item: str) -> str:
    return f"{item} {ITEM_LABELS.get(item, item)}"


def action_plan_rows(plans: pd.DataFrame, start: date | None = None,
                     owner: str = "Team manager") -> pd.DataFrame:
    """One row per step for every team with a starting question
    (recommendations.team_plans output)."""
    start = start or date.today()
    help_link = load_resources()["action_planning"][0]["url"]
    rows = []
    for _, p in plans.iterrows():
        item = p.get("start_with")
        if not isinstance(item, str) or not item:
            continue
        actions = ACTION_LIBRARY[item]["actions"]
        question = ACTION_LIBRARY[item]["questions"][0]
        target = float(p["company_score"])
        goal = (f"Raise {ITEM_LABELS.get(item, item)} from {float(p['team_score']):.2f} to the "
                f"company average of {target:.2f} by the next survey.")
        steps = [actions[0], actions[1], f'Ask the team: "{question}" and review progress.']
        element = element_resource(item)
        for (when, days), step in zip(PLAN_STEPS, steps):
            rows.append({
                "team_id": p["team_id"], "department": p.get("department", ""),
                "q12_item": item, "element": _element(item), "goal": goal, "step": when,
                "next_step": step, "owner": owner,
                "due_date": (start + timedelta(days=days)).isoformat(),
                "success_indicator": f"Team average on {item} at or above {target:.2f} in the "
                                     "next survey; the team says the step happened.",
                "follow_up": (start + timedelta(days=FOLLOW_UP_DAYS)).isoformat(),
                "learn_more": element["url"] if element else "",
                "action_planning_help": help_link,
            })
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def plan_resources_summary(items: list[str]) -> list[dict[str, str]]:
    """Compact list for the AI agent: element guides for these items, then the
    action-planning and manager-learning links (title + url only)."""
    out = [{"for": _element(i), **{k: element_resource(i)[k] for k in ("title", "url")}}
           for i in dict.fromkeys(items) if element_resource(i)]
    out += [{"for": r["kind"], "title": r["title"], "url": r["url"]}
            for r in action_planning_links() + manager_learning_links()]
    return out


# PROCESS 5 – LINK CHECK

def all_links() -> list[dict[str, Any]]:
    res = load_resources()
    links = res["action_planning"] + res["manager_learning"] + list(res["elements"].values())
    if res.get("gallup_access_url"):
        links.append({"title": "Gallup Access", "url": res["gallup_access_url"]})
    return links


def check_links(timeout: float = 15.0, fetch=None) -> pd.DataFrame:
    """HTTP status of every link. `fetch(url) -> status` can be swapped in tests."""
    if fetch is None:
        import urllib.request

        def fetch(url: str) -> int:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return response.status
            except Exception as exc:                       # noqa: BLE001
                return getattr(exc, "code", 0) or 0

    rows = []
    for link in all_links():
        status = fetch(link["url"])
        scripted = link.get("script_check", True)
        rows.append({"title": link["title"], "url": link["url"], "status": status,
                     "ok": status == 200 or (not scripted and status in (401, 403)),
                     "note": "" if scripted else "Blocks scripts: check in a browser."})
    return pd.DataFrame(rows)


if __name__ == "__main__" and "--check" in sys.argv:
    report = check_links()
    print(report[["status", "ok", "title", "note"]].to_string(index=False))
    sys.exit(0 if report["ok"].all() else 1)
