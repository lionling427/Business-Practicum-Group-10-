"""Strengths: use CliftonStrengths results to tailor engagement advice.

Input: each person's Top 5 themes, from a Gallup Access export or entered by
hand, either as columns in the survey upload or as a separate file:
  employee_id, top1, top2, top3, top4, top5   (or one 'top5' cell, comma-separated)
  coached (optional, 1 = took part in strengths coaching)

What it adds:
  team_profiles()   each team's mix of the four domains and its most common
                    themes (teams of 5+ people with strengths data only)
  strength_play()   an action for a team's focus question that uses the team's
                    strongest domain (e.g. low Recognition + Relationship Building
                    = peer recognition)
  pilot_effect()    did coached teams improve more than teams that were not
                    coached, between the last two surveys? (difference in
                    differences, with a likely range from resampling teams)

Privacy: individual themes never leave this module; only team and company
totals for groups of 5+ people. CliftonStrengths and the theme names are Gallup
trademarks; this app is independent and not endorsed by Gallup.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

import re
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from .data_ingestion import Q12_ITEMS
from .insights import ITEM_LABELS


# PROCESS 2 – THE 34 THEMES AND FOUR DOMAINS

DOMAINS = ["Executing", "Influencing", "Relationship Building", "Strategic Thinking"]
DOMAIN_COLORS = ["#7B2D8E", "#E07A1F", "#1F6FB2", "#2E9E3E"]
THEMES = {
    **dict.fromkeys(["Achiever", "Arranger", "Belief", "Consistency", "Deliberative",
                     "Discipline", "Focus", "Responsibility", "Restorative"], "Executing"),
    **dict.fromkeys(["Activator", "Command", "Communication", "Competition", "Maximizer",
                     "Self-Assurance", "Significance", "Woo"], "Influencing"),
    **dict.fromkeys(["Adaptability", "Connectedness", "Developer", "Empathy", "Harmony",
                     "Includer", "Individualization", "Positivity", "Relator"],
                    "Relationship Building"),
    **dict.fromkeys(["Analytical", "Context", "Futuristic", "Ideation", "Input",
                     "Intellection", "Learner", "Strategic"], "Strategic Thinking"),
}
# One plain line per theme (the app's own wording, not Gallup's descriptions).
THEME_MEANING = {
    # Executing
    "Achiever": "Works hard and needs to get something done every day.",
    "Arranger": "Organises people and resources to find the best way to get things done.",
    "Belief": "Driven by strong core values that give the work meaning.",
    "Consistency": "Treats people fairly and the same; values clear rules.",
    "Deliberative": "Careful and thorough; spots risks before deciding.",
    "Discipline": "Likes routine, structure and order.",
    "Focus": "Sets a direction and stays on track to the goal.",
    "Responsibility": "Owns commitments and always follows through.",
    "Restorative": "Loves finding and fixing problems.",
    # Influencing
    "Activator": "Turns ideas into action; impatient to start.",
    "Command": "Takes charge, speaks up and makes decisions.",
    "Communication": "Puts thoughts into words easily; a natural presenter and storyteller.",
    "Competition": "Measures progress against others and plays to win.",
    "Maximizer": "Turns something good into something excellent.",
    "Self-Assurance": "Confident in own judgement and decisions.",
    "Significance": "Wants to make a big, visible impact and be recognised for it.",
    "Woo": "Short for 'Winning Others Over': loves meeting new people and winning them over.",
    # Relationship Building
    "Adaptability": "Goes with the flow and responds well to change in the moment.",
    "Connectedness": "Sees how people and events link together; believes things happen for a reason.",
    "Developer": "Sees potential in others and enjoys helping them grow.",
    "Empathy": "Senses how others feel and sees things from their point of view.",
    "Harmony": "Looks for agreement and avoids needless conflict.",
    "Includer": "Makes sure everyone feels part of the group.",
    "Individualization": "Notices what makes each person unique and works with that.",
    "Positivity": "Brings energy and enthusiasm that lifts others.",
    "Relator": "Builds deep, trusted relationships with a close circle.",
    # Strategic Thinking
    "Analytical": "Looks for evidence and searches for the real cause.",
    "Context": "Looks to the past to understand the present.",
    "Futuristic": "Inspired by what could be; paints a picture of the future.",
    "Ideation": "Loves new ideas and finding links between them.",
    "Input": "Collects information, ideas and resources.",
    "Intellection": "Enjoys thinking deeply and reflecting.",
    "Learner": "Loves learning and the process of getting better at something.",
    "Strategic": "Quickly sees the best route through a problem.",
}

DOMAIN_MEANING = {
    "Executing": "gets things done: plans, follows through, keeps standards",
    "Influencing": "takes charge, speaks up and brings others along",
    "Relationship Building": "builds trust, cares for people and holds the team together",
    "Strategic Thinking": "analyses, learns and imagines what could be",
}
MIN_GROUP = 5
TOP_COLUMNS = [f"top{i}" for i in range(1, 6)]
_KEY = {re.sub(r"[^a-z]", "", t.lower()): t for t in THEMES}


# PROCESS 3 – STRENGTH-AWARE ACTIONS (FOCUS QUESTION × STRONGEST DOMAIN)

PLAYS = {
    "Q01": {"Executing": "Turn this quarter's goals into a shared checklist with owners and dates.",
            "Influencing": "Have the team present back, in their own words, what success looks like this month.",
            "Relationship Building": "Agree expectations one-to-one and ask each person what they need to hit them.",
            "Strategic Thinking": "Map how each role's work connects to the team's goals, and fix the gaps found."},
    "Q02": {"Executing": "Run a 30-day 'fix the blockers' sprint with a visible list of tools and resources.",
            "Influencing": "Ask the team to make the case to leadership for the one resource that matters most.",
            "Relationship Building": "Pair people so those who found workarounds share them with the rest.",
            "Strategic Thinking": "Have the team analyse where time is lost and propose the top two fixes."},
    "Q03": {"Executing": "Let people choose the tasks they finish best and rebalance one task each.",
            "Influencing": "Give each person one area to lead where they can shine in front of others.",
            "Relationship Building": "Hold short strengths conversations: what energises you, what drains you?",
            "Strategic Thinking": "Match the next project's problems to the people who love solving them."},
    "Q04": {"Executing": "Make recognition a weekly routine: three specific thank-yous before Friday.",
            "Influencing": "Celebrate wins publicly in team meetings and with senior leaders.",
            "Relationship Building": "Start peer-to-peer recognition: colleagues name each other's contributions.",
            "Strategic Thinking": "Recognise the quality of thinking, not only results: name the idea and its impact."},
    "Q05": {"Executing": "Put a regular 1:1 in the calendar that is about the person, not the tasks.",
            "Influencing": "Model it: managers share something personal first and invite others to.",
            "Relationship Building": "Let the team's natural connectors organise check-ins and buddy pairs.",
            "Strategic Thinking": "Ask what would make each person's week better and act on one idea each."},
    "Q06": {"Executing": "Agree one development goal per person with monthly check-points.",
            "Influencing": "Give people stretch moments: presenting, leading a meeting, owning a client.",
            "Relationship Building": "Set up mentoring pairs inside the team.",
            "Strategic Thinking": "Build learning paths: what skill next, why, and how it will be used."},
    "Q07": {"Executing": "Close the loop on every idea within two weeks: done, planned, or why not.",
            "Influencing": "Hold a short 'challenge the plan' slot in meetings where anyone can push back.",
            "Relationship Building": "Ask quieter members for input first, before decisions are made.",
            "Strategic Thinking": "Invite the team to co-design the next change before it is announced."},
    "Q08": {"Executing": "Show how this week's work moved a customer or company goal.",
            "Influencing": "Bring in a customer or leader to tell the team why their work matters.",
            "Relationship Building": "Share stories of people the team's work has helped.",
            "Strategic Thinking": "Connect the team's work to the company's long-term direction in one picture."},
    "Q09": {"Executing": "Agree a short team quality standard and review it together each month.",
            "Influencing": "Let respected team members set the bar and coach others to it.",
            "Relationship Building": "Talk openly as a team about what 'good work' means to each person.",
            "Strategic Thinking": "Review one piece of work together each month and learn from it."},
    "Q10": {"Executing": "Create small working pairs on shared goals so people rely on each other.",
            "Influencing": "Run team challenges that bring people together around a common win.",
            "Relationship Building": "Make space for informal time: team lunches, welcome rituals for new people.",
            "Strategic Thinking": "Set up cross-team problem-solving groups around shared interests."},
    "Q11": {"Executing": "Hold a progress check every six months for everyone, with notes and next steps.",
            "Influencing": "Make progress talks forward-looking: where could you go next?",
            "Relationship Building": "Start each progress talk with what the person is proud of.",
            "Strategic Thinking": "Use the talk to plan the next six months of growth together."},
    "Q12": {"Executing": "Give each person one new learning task this quarter and track it.",
            "Influencing": "Let people teach others what they have learned in short team sessions.",
            "Relationship Building": "Pair people to learn from each other across roles.",
            "Strategic Thinking": "Give the team a real problem to research and present solutions on."},
}


def explain_themes(themes: str) -> list[str]:
    """'Woo, Learner' -> ['**Woo:** Short for ...', '**Learner:** Loves learning ...']"""
    return [f"**{t}:** {THEME_MEANING[t]}" for t in (x.strip() for x in themes.split(","))
            if t in THEME_MEANING]


def strength_play(item: str, domain: str) -> str:
    return PLAYS.get(item, {}).get(domain, "")


# PROCESS 4 – READ AND CHECK STRENGTHS DATA

def has_strengths(df: pd.DataFrame) -> bool:
    return "employee_id" in df.columns and (all(c in df.columns for c in TOP_COLUMNS)
                                            or "top5" in df.columns)


def normalize_theme(value: Any) -> str | None:
    return _KEY.get(re.sub(r"[^a-z]", "", str(value).lower()))


def read_strengths(df: pd.DataFrame) -> dict[str, Any]:
    """Long table (employee_id, rank, theme, domain) plus what could not be read."""
    if not has_strengths(df):
        raise ValueError("Needs employee_id and top1–top5 (or a comma-separated 'top5' column).")
    data = df.drop_duplicates("employee_id", keep="last").copy()
    if not all(c in data.columns for c in TOP_COLUMNS):
        split = data["top5"].astype(str).str.split(r"[;,|]", expand=True)
        for i, column in enumerate(TOP_COLUMNS):
            data[column] = split[i] if i in split.columns else np.nan
    long = data.melt(id_vars=["employee_id"], value_vars=TOP_COLUMNS,
                     var_name="rank", value_name="raw").dropna(subset=["raw"])
    long["raw"] = long["raw"].astype(str).str.strip()
    long = long[long["raw"].ne("") & long["raw"].str.lower().ne("nan")]
    long["rank"] = long["rank"].str[-1].astype(int)
    long["theme"] = long["raw"].map(normalize_theme)
    unknown = sorted(set(long.loc[long["theme"].isna(), "raw"].astype(str)))
    long = long.dropna(subset=["theme"])
    long["domain"] = long["theme"].map(THEMES)
    coached = (data.set_index("employee_id")["coached"].fillna(0).astype(int)
               if "coached" in data.columns else None)
    return {"long": long[["employee_id", "rank", "theme", "domain"]].reset_index(drop=True),
            "people": int(long["employee_id"].nunique()), "unknown": unknown,
            "coached": coached}


# PROCESS 5 – TEAM AND COMPANY PROFILES

def _latest_roster(survey: pd.DataFrame) -> pd.DataFrame:
    rows = survey
    if "wave" in rows.columns and rows["wave"].notna().any():
        rows = rows[rows["wave"] == rows["wave"].max()]
    keep = [c for c in ["employee_id", "team_id", "department"] if c in rows.columns]
    return rows[keep].drop_duplicates("employee_id")


def domain_mix(long: pd.DataFrame) -> pd.Series:
    """Share of all Top-5 themes falling in each domain (sums to 1)."""
    return long["domain"].value_counts(normalize=True).reindex(DOMAINS, fill_value=0.0)


def team_profiles(strengths: dict, survey: pd.DataFrame,
                  scorecard: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team with 5+ people who have strengths data."""
    roster = _latest_roster(survey)
    long = strengths["long"].merge(roster, on="employee_id", how="inner")
    if long.empty or "team_id" not in long.columns:
        return pd.DataFrame()
    team_size = roster.groupby("team_id")["employee_id"].nunique()
    rows = []
    for team, group in long.groupby("team_id"):
        people = group["employee_id"].nunique()
        if people < MIN_GROUP:
            continue
        mix = domain_mix(group)
        themes = group["theme"].value_counts()
        rows.append({"team_id": team,
                     "department": group["department"].mode().iat[0] if "department" in group else "",
                     "people_with_strengths": people,
                     "coverage": people / team_size.get(team, people),
                     **{d: float(mix[d]) for d in DOMAINS},
                     "lead_domain": mix.idxmax(),
                     "top_themes": ", ".join(themes.index[:3])})
    teams = pd.DataFrame(rows)
    if teams.empty:
        return teams
    if scorecard is not None and not scorecard.empty:
        teams = teams.merge(scorecard[["team_id", "focus_item", "focus_label", "q12_average"]],
                            on="team_id", how="left")
        teams["play"] = [strength_play(i, d) if isinstance(i, str) else ""
                         for i, d in zip(teams["focus_item"], teams["lead_domain"])]
    return teams.sort_values("team_id").reset_index(drop=True)


def company_profile(strengths: dict) -> dict[str, Any]:
    long = strengths["long"]
    themes = long["theme"].value_counts()
    return {"mix": domain_mix(long), "people": strengths["people"],
            "top_themes": (themes / strengths["people"]).head(10)}


# PROCESS 6 – PILOT: DID COACHED TEAMS IMPROVE MORE?

def pilot_effect(survey: pd.DataFrame, strengths: dict, measure: str = "Q03",
                 reps: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Change in a team measure between the last two surveys, coached vs not coached."""
    coached = strengths.get("coached")
    if coached is None or coached.sum() == 0:
        return {"status": "no_pilot",
                "text": "No coaching recorded: add a 'coached' column (1 = took part) to compare."}
    if "wave" not in survey.columns or survey["wave"].nunique() < 2:
        return {"status": "no_pilot", "text": "Needs two surveys: one before and one after coaching."}
    rows = survey[survey["responded"] == 1] if "responded" in survey.columns else survey
    before, after = sorted(rows["wave"].dropna().unique())[-2:]
    if measure == "engagement":
        rows = rows.assign(engagement=rows[[i for i in Q12_ITEMS if i in rows]].mean(axis=1))
    team_share = (_latest_roster(survey).assign(coached=lambda d: d["employee_id"].map(coached).fillna(0))
                  .groupby("team_id")["coached"].mean())
    means = rows.groupby(["team_id", "wave"])[measure].agg(["mean", "size"]).reset_index()
    wide = means.pivot(index="team_id", columns="wave", values="mean")
    sizes = means.pivot(index="team_id", columns="wave", values="size")
    ok = (sizes[[before, after]] >= MIN_GROUP).all(axis=1)
    change = (wide.loc[ok, after] - wide.loc[ok, before]).rename("change").to_frame()
    change["coached"] = change.index.map(team_share).fillna(0) >= 0.5
    yes, no = change.loc[change["coached"], "change"], change.loc[~change["coached"], "change"]
    if len(yes) < MIN_GROUP or len(no) < MIN_GROUP:
        return {"status": "no_pilot",
                "text": f"Needs {MIN_GROUP}+ coached and {MIN_GROUP}+ comparison teams "
                        f"(has {len(yes)} and {len(no)})."}
    effect = float(yes.mean() - no.mean())
    rng = np.random.default_rng(seed)
    draws = [rng.choice(yes.to_numpy(), len(yes)).mean() - rng.choice(no.to_numpy(), len(no)).mean()
             for _ in range(reps)]
    low, high = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    clear = low > 0 or high < 0
    name = "engagement" if measure == "engagement" else f"{ITEM_LABELS[measure]} ({measure})"
    text = (f"Coached teams changed by {yes.mean():+.2f} on {name} vs {no.mean():+.2f} for other "
            f"teams: a difference of {effect:+.2f} points (likely range {low:+.2f} to {high:+.2f}). "
            + ("That is a clear difference." if clear else
               "That is not a clear difference yet: it could be chance."))
    return {"status": "clear" if clear else "unclear", "effect": effect, "range": (low, high),
            "coached_teams": len(yes), "other_teams": len(no), "measure": measure,
            "text": text, "changes": change.reset_index()}


# PROCESS 7 – CHARTS

def domain_chart(mix: pd.Series, title: str = "") -> alt.LayerChart:
    """One labelled bar per domain: share of the group's Top 5 themes."""
    data = pd.DataFrame({"domain": mix.index, "share": mix.to_numpy()})
    data["label"] = data["share"].map("{:.0%}".format)
    base = alt.Chart(data).encode(y=alt.Y("domain:N", sort=DOMAINS, title=None,
                                          axis=alt.Axis(labelLimit=200)))
    bars = base.mark_bar(cornerRadiusEnd=3, height=20).encode(
        x=alt.X("share:Q", title="Share of Top 5 themes", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, max(0.5, float(data["share"].max()) * 1.25)])),
        color=alt.Color("domain:N", legend=None,
                        scale=alt.Scale(domain=DOMAINS, range=DOMAIN_COLORS)),
        tooltip=[alt.Tooltip("domain:N", title="Domain"),
                 alt.Tooltip("share:Q", title="Share of Top 5 themes", format=".0%")])
    text = base.mark_text(align="left", dx=5, fontSize=12).encode(x="share:Q", text="label:N")
    return (bars + text).properties(height=alt.Step(32), width="container", title=title)


def themes_chart(top_themes: pd.Series) -> alt.Chart:
    data = pd.DataFrame({"theme": top_themes.index, "share": top_themes.to_numpy()})
    data["domain"] = data["theme"].map(THEMES)
    data["meaning"] = data["theme"].map(THEME_MEANING)
    return alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=16).encode(
        y=alt.Y("theme:N", sort="-x", title=None),
        x=alt.X("share:Q", title="People with this theme in their Top 5", axis=alt.Axis(format="%")),
        color=alt.Color("domain:N", title=None, scale=alt.Scale(domain=DOMAINS, range=DOMAIN_COLORS),
                        legend=None),
        tooltip=[alt.Tooltip("theme:N", title="Theme"),
                 alt.Tooltip("meaning:N", title="What it means"),
                 alt.Tooltip("domain:N", title="Domain"),
                 alt.Tooltip("share:Q", format=".0%", title="Share of people")],
    ).properties(height=alt.Step(24), width="container")


# PROCESS 8 – DEMO STRENGTHS (FOR TRYING THE PAGE)

def demo_strengths(survey: pd.DataFrame, coached_share: float = 0.3, seed: int = 3) -> pd.DataFrame:
    """Random Top 5 per employee, with each team leaning towards one domain, and a
    'coached' flag for about 30% of teams. Random: it carries no real effect."""
    rng = np.random.default_rng(seed)
    roster = _latest_roster(survey)
    teams = roster["team_id"].unique()
    lean = dict(zip(teams, rng.choice(DOMAINS, len(teams))))
    coached_teams = set(rng.choice(teams, int(len(teams) * coached_share), replace=False))
    names = np.array(list(THEMES))
    rows = []
    for _, person in roster.iterrows():
        weights = np.array([3.0 if THEMES[t] == lean[person["team_id"]] else 1.0 for t in names])
        top = rng.choice(names, 5, replace=False, p=weights / weights.sum())
        rows.append({"employee_id": person["employee_id"], **dict(zip(TOP_COLUMNS, top)),
                     "coached": int(person["team_id"] in coached_teams)})
    return pd.DataFrame(rows)
