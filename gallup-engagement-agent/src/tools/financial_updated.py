"""Financial impact: translate verified engagement findings into money.

UPDATED VERSION (branch Will). Changes from the original financial.py follow the
"Financial Impact Review: Issues and Fixes" memo:

  Fix 1  costs today count ALL employees in the outcome file, not only survey
         respondents, and the default headcount is all employees (outcome_frame,
         default_assumptions).
  Fix 2  one scenario engine: engagement_value() and priorities.factor_value() call
         scenario.simulate() (each person's own risk), so Financials and Scenarios
         give the same number for the same scenario. dollar_impact() is no longer used.
  Fix 3  the headline is ONE named program (the "Fix first" factors, or the top
         factor) simulated jointly across turnover and absence, not a sum of two
         separate top drivers.
  Fix 4  every dollar figure carries a 95% range (low / high) from the models'
         confidence intervals, plus a break-even cost per person.
  Fix 5  money_warning(): the survey-response (X7) note for pages that show money.
  Fix 6  team_excess_cost(): all employees, small teams smoothed toward the company
         rate (k = 20), a department view, and a year-over-year check
         (team_cost_persistence) that decides whether team rankings are shown.

The data has no revenue or profit, so every figure here comes from the user's
own cost assumptions applied to verified results:
- What it costs today: voluntary turnover and unplanned absence, per year.
- What engagement could be worth (scenarios, from verified findings only):
    program     -> scenario.simulate(): everyone who answers 1-3 on the program's
                   questions moves up one point; value of fewer leavers and fewer
                   absence days, with a 95% range
    productivity-> recommendations.team_plans(): each team closes the gap on its
                   first lever, valued at the user's 'value of one rating point'
- Where the money is: departments and teams whose turnover and absence cost more
  than the company rate would (descriptive, no model needed).

All results scale from the data to the company headcount the user enters. They are
illustrative scenarios built on associations, not forecasts. Because leavers answer
the survey less often, the savings are likely understated (see money_warning).

If team-level financial results (e.g. profit per team) are added to the data
later, they can be modeled directly as an outcome instead of estimated here.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from .insights import analysis_frame
from .visualization import HIGH_COLOR, MUTED_COLOR
from .waves import model_wave


# PROCESS 2 – ASSUMPTIONS

TURNOVER = "left_voluntarily_12m"
ABSENCE = "unplanned_absence_days_6m"
WORKDAYS_PER_YEAR = 260
MIN_TEAM_SIZE = 5
SMOOTHING_K = 20            # small teams are pulled toward the company rate (fix 6)
PERSISTENCE_BAR = 0.75      # flagged teams must keep at least 75% of their extra cost next year
PERSISTENCE_SHARE = 0.01    # the flagged teams are the costliest 1% (at least 10)


@dataclass
class Assumptions:
    """Cost assumptions a leader enters. Defaults are placeholders, not facts."""

    avg_salary: float = 60_000.0
    replacement_cost_pct: float = 0.5          # cost to replace one leaver, share of salary
    absence_day_cost: float | None = None      # None -> salary / 260 workdays
    value_per_rating_point: float = 0.0        # yearly value of +1 performance point per person
    headcount: int | None = None               # None -> people in the data
    currency: str = "$"
    # EV additions (branch Will, from Sunny's EV brief and the Gallup utility-analysis notes)
    program_cost: float = 0.0                  # yearly cost of the program being evaluated (0 = not entered)
    effect_realization: float = 1.0            # share of the modeled effect the program really delivers
    sd_productivity_pct: float = 0.0           # SD of the yearly value of productivity, share of salary (0 = off)
    sources: dict = field(default_factory=dict)  # assumption name -> where the number came from

    @property
    def cost_per_leaver(self) -> float:
        return self.avg_salary * self.replacement_cost_pct

    @property
    def day_cost(self) -> float:
        return self.absence_day_cost if self.absence_day_cost else self.avg_salary / WORKDAYS_PER_YEAR

    def describe(self, sample: int) -> list[str]:
        c = self.currency
        lines = [
            f"Average salary {c}{self.avg_salary:,.0f}.",
            f"Replacing one leaver costs {self.replacement_cost_pct:.0%} of salary "
            f"({c}{self.cost_per_leaver:,.0f}).",
            f"One unplanned absence day costs {c}{self.day_cost:,.0f}"
            + (" (salary ÷ 260 workdays)." if not self.absence_day_cost else "."),
            f"Headcount {self._headcount(sample):,} (people in the data: {sample:,}); results "
            "scale by that ratio.",
        ]
        if self.value_per_rating_point > 0:
            lines.append(f"One extra performance-rating point is worth "
                         f"{c}{self.value_per_rating_point:,.0f} per person per year.")
        else:
            lines.append("Productivity value is not counted (value of a rating point = 0).")
        return lines

    def _headcount(self, sample: int) -> int:
        return int(self.headcount) if self.headcount else sample

    def scale(self, sample: int) -> float:
        return self._headcount(sample) / sample if sample else 1.0


# PROCESS 3 – WHO IS COUNTED (fix 1)

def outcome_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every employee in the model wave, whether or not they answered the survey.

    Costs need everyone's outcome: leavers answer the survey less often, so counting
    respondents only understates turnover and absence. Models still use
    insights.analysis_frame (respondents), because they need the survey answers."""
    wave = model_wave(df)
    return df if wave is None or "wave" not in df.columns else df[df["wave"] == wave]


def default_assumptions(df: pd.DataFrame, **changes: Any) -> Assumptions:
    """Placeholder assumptions with the headcount set to all employees in the data."""
    return Assumptions(headcount=len(outcome_frame(df)), **changes)


def unit_assumptions(a: Assumptions) -> Assumptions:
    """The same cost inputs with no headcount scaling (scale = 1)."""
    return replace(a, headcount=None)


SOURCE_DEFAULT = "Application default (placeholder)"
SOURCE_USER = "User-entered (source not stated)"
SOURCE_DATA = "Taken from your data"
SOURCE_OPTIONS = ["Company-provided", "External benchmark", "My own scenario assumption"]

REGISTER_ROWS = [
    ("avg_salary", "Average salary per year"),
    ("replacement_cost_pct", "Cost to replace one leaver (share of salary)"),
    ("absence_day_cost", "Cost of one unplanned absence day"),
    ("value_per_rating_point", "Value of +1 performance point, per person per year"),
    ("sd_productivity_pct", "SD of productivity value (share of salary)"),
    ("headcount", "Company headcount"),
    ("program_cost", "Yearly program cost"),
    ("effect_realization", "Share of the modeled effect the program delivers"),
]


def resolve_point_value(df: pd.DataFrame, a: Assumptions) -> Assumptions:
    """Price a performance point from the utility-analysis formula when the user gave an
    SD of productivity value but no value per rating point.

    Hunter-Schmidt / Brogden-Cronbach-Gleser utility analysis (the method Gallup cites for
    its productivity estimates): gain = people × effect × SDy × change in standard units.
    The performance model here is z-scored on both sides, so its standardized beta is the
    effect and value of one rating point = SDy ÷ standard deviation of the rating,
    with SDy = salary × sd_productivity_pct. An explicit value per point always wins."""
    if a.value_per_rating_point > 0 or a.sd_productivity_pct <= 0:
        return a
    data = outcome_frame(df)
    if "performance_rating" not in data.columns:
        return a
    sd = float(pd.to_numeric(data["performance_rating"], errors="coerce").std())
    if not sd > 0:
        return a
    return replace(a, value_per_rating_point=a.avg_salary * a.sd_productivity_pct / sd)


def assumption_register(a: Assumptions, sample: int | None = None) -> pd.DataFrame:
    """Every economic input with its value and where it came from. A number is labelled
    'application default' only while it still equals the placeholder."""
    default = Assumptions()
    shown = {
        "avg_salary": f"{a.currency}{a.avg_salary:,.0f}",
        "replacement_cost_pct": f"{a.replacement_cost_pct:.0%} ({a.currency}{a.cost_per_leaver:,.0f} per leaver)",
        "absence_day_cost": (f"{a.currency}{a.day_cost:,.0f}"
                             + ("" if a.absence_day_cost else " (salary ÷ 260 workdays)")),
        "value_per_rating_point": (f"{a.currency}{a.value_per_rating_point:,.0f}"
                                   if a.value_per_rating_point > 0 else "not counted"),
        "sd_productivity_pct": f"{a.sd_productivity_pct:.0%}" if a.sd_productivity_pct > 0 else "off",
        "headcount": (f"{a.headcount:,}" if a.headcount else
                      (f"{sample:,} (all employees in the data)" if sample else "all employees in the data")),
        "program_cost": (f"{a.currency}{a.program_cost:,.0f}" if a.program_cost > 0 else "not entered"),
        "effect_realization": f"{a.effect_realization:.0%}",
    }
    rows = []
    for key, label in REGISTER_ROWS:
        value, base = getattr(a, key), getattr(default, key)
        if key in a.sources:
            source = a.sources[key]
        elif key == "headcount" and not a.headcount:
            source = SOURCE_DATA
        elif key == "headcount":
            source = SOURCE_USER if (sample is None or a.headcount != sample) else SOURCE_DATA
        elif key in ("program_cost",) and not value:
            source = "Not entered"
        elif value == base:
            source = SOURCE_DEFAULT
        else:
            source = SOURCE_USER
        rows.append({"assumption": label, "value": shown[key], "source": source})
    return pd.DataFrame(rows)


# PROCESS 4 – WHAT IT COSTS TODAY

def current_costs(df: pd.DataFrame, a: Assumptions) -> dict[str, Any]:
    """Yearly cost of voluntary turnover and unplanned absence, company-wide,
    counted on all employees (not only survey respondents)."""
    data = outcome_frame(df)
    n = len(data)
    scale = a.scale(n)
    out: dict[str, Any] = {"sample": n, "headcount": a._headcount(n)}
    if TURNOVER in data.columns:
        leavers = float(data[TURNOVER].mean()) * n * scale
        out.update(leavers=leavers, turnover_cost=leavers * a.cost_per_leaver)
    if ABSENCE in data.columns:
        days = float(data[ABSENCE].mean()) * 2 * n * scale      # 6-month window -> year
        out.update(absence_days=days, absence_cost=days * a.day_cost)
    out["total_cost"] = out.get("turnover_cost", 0.0) + out.get("absence_cost", 0.0)
    return out


# PROCESS 5 – WHAT ENGAGEMENT COULD BE WORTH (fixes 2, 3, 4)

def _program_items(df, analysis, a, factors) -> list[str]:
    """Default program: the 'Fix first' factors, or the single top-valued factor."""
    from .priorities import factor_value          # local import: priorities imports this module

    table = factors if factors is not None else factor_value(df, analysis, a)
    valued = table[table["yearly_value"] > 0]
    fix = valued[valued["priority"] == "Fix first"]
    chosen = fix if not fix.empty else valued.head(1)
    return list(chosen["item"])


def engagement_value(df: pd.DataFrame, analysis: dict | None, a: Assumptions,
                     plans: pd.DataFrame | None = None,
                     tiers: list[str] | None = None,
                     factors: pd.DataFrame | None = None,
                     items: list[str] | None = None,
                     fits: dict | None = None) -> dict[str, Any]:
    """Yearly value of one named program plus the optional productivity plans.

    The program lifts every question in `items` by one point for the people who answer
    1-3, simulated jointly (scenario.simulate), so a person helped by two questions is
    counted once. `items` defaults to the 'Fix first' factors. Each row carries a 95%
    range (low, high)."""
    from .scenario import simulate                # local import: scenario imports this module

    scenarios: list[dict[str, Any]] = []
    warnings: list[str] = []
    program: dict[str, Any] | None = None
    low_total = high_total = 0.0
    a = resolve_point_value(df, a)

    if analysis is not None:
        items = items if items is not None else _program_items(df, analysis, a, factors)
        if items:
            from .audit import audit_sample
            from .insights import ITEM_LABELS

            sim = simulate(df, analysis, {i: 1.0 for i in items}, a, who="low", ramp_months=0,
                           fits=fits)
            t = sim["totals"]
            sample = audit_sample(df)
            scale = a.scale(len(sample))
            helped = float((sample[[i for i in items if i in sample.columns]] <= 3)
                           .any(axis=1).sum()) * scale
            names = " and ".join(ITEM_LABELS.get(i, i) for i in items)

            def money_of(bound: str, part: str) -> float:
                return (t[bound]["leavers"] * a.cost_per_leaver if part == "turnover"
                        else t[bound]["absence_days"] * a.day_cost)

            for source, part, units, unit_label in (
                    ("Turnover", "turnover", t["estimate"]["leavers"], "fewer leavers a year"),
                    ("Absence", "absence", t["estimate"]["absence_days"],
                     "fewer absence days a year")):
                if units <= 0:
                    continue
                lo, hi = sorted([money_of("cautious", part), money_of("optimistic", part)])
                scenarios.append({
                    "source": source,
                    "lever": names,
                    "what": (f"The {helped:,.0f} employees who answer 1–3 on {names} move up "
                             "1 point" if source == "Turnover" else
                             f"The same program, counted for unplanned absence"),
                    "units": units, "unit_label": unit_label,
                    "value": money_of("estimate", part), "low": lo, "high": hi,
                })
                low_total += lo
                high_total += hi
            if scenarios:
                program = {"items": list(items), "names": names, "people": helped,
                           "value": sum(s["value"] for s in scenarios),
                           "low": low_total, "high": high_total,
                           "physical": {b: {"leavers": t[b]["leavers"],
                                            "absence_days": t[b]["absence_days"]}
                                        for b in ("estimate", "cautious", "optimistic")}}
                if len(items) > 1:
                    warnings.append("The questions in the program overlap (the same people answer "
                                    "them), so the program is simulated jointly and counted once; "
                                    "do not add the single-question values.")

    if plans is not None and a.value_per_rating_point > 0:
        data = outcome_frame(df)
        scale = a.scale(len(data))
        chosen = plans if not tiers else plans[plans["tier"].isin(tiers)]
        chosen = chosen[chosen["estimated_gain"] > 0]
        if not chosen.empty:
            point_years = float((chosen["estimated_gain"] * chosen["people"]).sum()) * scale
            value = point_years * a.value_per_rating_point
            scenarios.append({
                "source": "Productivity",
                "lever": "Each team's first recommended step",
                "what": f"{len(chosen)} teams close the gap on their starting question",
                "units": point_years,
                "unit_label": "rating points × people",
                "value": value, "low": float("nan"), "high": float("nan"),
            })
            low_total += value
            high_total += value

    if program and any(s["source"] == "Productivity" for s in scenarios):
        warnings.append("The productivity row and the turnover and absence program describe "
                        "different pathways, but they can rest on the same engagement lifts and "
                        "the same people, so treat their sum as an upper bound.")
    table = pd.DataFrame(scenarios, columns=["source", "lever", "what", "units",
                                             "unit_label", "value", "low", "high"])
    total = float(table["value"].sum()) if len(table) else 0.0
    return {"scenarios": table, "total": total, "program": program, "warnings": warnings,
            "range": (low_total, high_total) if len(table) else (0.0, 0.0)}


def break_even_per_person(value: float, people: float) -> float:
    """Most a program can cost per person per year and still pay for itself."""
    return value / people if people else 0.0


def exposure_vs_opportunity(costs: dict[str, Any], value: dict[str, Any]) -> dict[str, float]:
    """Keep the two ideas apart: exposure is what turnover and absence cost today; opportunity
    is the smaller part a modeled engagement program is associated with. An opportunity is
    not a realized return."""
    exposure = float(costs.get("total_cost", 0.0))
    opportunity = float(value.get("total", 0.0))
    return {"exposure": exposure, "opportunity": opportunity,
            "opportunity_share": opportunity / exposure if exposure else 0.0}


def program_economics(value: dict[str, Any], a: Assumptions) -> dict[str, Any] | None:
    """ROI, payback and break-even for the named program, given the yearly program cost the
    user entered. Benefit = modeled value × the share of the effect the program delivers.
    None until a program cost is entered. Nothing here invents an effect size: the break-even
    outputs say what would have to be true."""
    program = value.get("program")
    if not program or a.program_cost <= 0:
        return None
    cost, r = a.program_cost, a.effect_realization
    cases = {}
    for name, modeled in (("downside", program["low"]), ("base", program["value"]),
                          ("upside", program["high"])):
        benefit = modeled * r
        cases[name] = {
            "benefit": benefit, "net": benefit - cost, "roi": (benefit - cost) / cost,
            "payback_months": 12 * cost / benefit if benefit > 0 else float("inf"),
            # share of the modeled effect that must be delivered for benefit to equal cost
            "required_realization": cost / modeled if modeled > 0 else float("inf"),
        }
    return {"cost": cost, "realization": r, "cases": cases,
            "break_even_cost": program["value"] * r,
            "people": program["people"],
            "break_even_cost_per_person": break_even_per_person(program["value"] * r,
                                                                program["people"])}


def break_even_lift(df: pd.DataFrame, analysis: dict, a: Assumptions, items: list[str],
                    max_lift: float = 3.0, fits: dict | None = None) -> dict[str, Any]:
    """Smallest average rise (in points) for the people who answer 1-3 on the program's
    questions for the modeled benefit to cover the program cost. Uses the same simulation as
    the Scenarios page at the estimate, so it answers 'what would have to be true' without
    assuming any lift. Benefit is turnover plus absence only, times the effect-realization share."""
    from .scenario import simulate

    cost, r = a.program_cost, a.effect_realization
    if cost <= 0 or not items:
        return {"status": "no_cost"}

    def benefit(lift: float) -> float:
        t = simulate(df, analysis, {i: lift for i in items}, a, who="low", ramp_months=0,
                     fits=fits)["totals"]["estimate"]
        return (t["leavers"] * a.cost_per_leaver + t["absence_days"] * a.day_cost) * r

    one = benefit(1.0)
    if one <= 0:
        return {"status": "no_effect"}
    lift = cost / one                      # linear first guess, refined below
    for _ in range(4):
        if lift > max_lift:
            break
        got = benefit(lift)
        if got <= 0 or abs(got - cost) / cost < 0.005:
            break
        lift *= cost / got
    if lift > max_lift:
        return {"status": "unreachable", "max_lift": max_lift, "benefit_at_max": benefit(max_lift)}
    return {"status": "ok", "lift": float(lift), "benefit_at_one_point": one}


def assumption_sensitivity(value: dict[str, Any], a: Assumptions) -> pd.DataFrame:
    """How the program's base value moves when the economic assumptions change, holding the
    statistical result fixed. Separate from the 95% range, which is statistical uncertainty only."""
    program = value.get("program")
    if not program:
        return pd.DataFrame(columns=["assumption", "setting", "value", "change"])
    est = program["physical"]["estimate"]
    base = est["leavers"] * a.cost_per_leaver + est["absence_days"] * a.day_cost
    rows = []

    def add(name: str, setting: str, v: float) -> None:
        rows.append({"assumption": name, "setting": setting, "value": v,
                     "change": v / base - 1 if base else 0.0})

    if est["leavers"] > 0:
        for pct in (0.25, 0.5, 1.0, 1.5, 2.0):
            add("Cost to replace a leaver", f"{pct:.0%} of salary",
                est["leavers"] * a.avg_salary * pct + est["absence_days"] * a.day_cost)
    if est["absence_days"] > 0:
        for mult in (0.5, 1.0, 1.5):
            add("Cost of an absence day", f"{mult:g} × {money(a.day_cost, a.currency)}",
                est["leavers"] * a.cost_per_leaver + est["absence_days"] * a.day_cost * mult)
    for share in (0.25, 0.5, 0.75, 1.0):
        add("Share of the modeled effect delivered", f"{share:.0%}", base * share)
    return pd.DataFrame(rows)


# PROCESS 6 – WHERE THE MONEY IS (fix 6)

def _team_table(data: pd.DataFrame, k: float) -> pd.DataFrame:
    """Per team: people, raw rates, and rates smoothed toward the company (and
    department) rate. Teams under MIN_TEAM_SIZE are dropped."""
    agg: dict[str, tuple] = {"people": ("team_id", "size")}
    if TURNOVER in data.columns:
        agg["leavers"] = (TURNOVER, "sum")
    if ABSENCE in data.columns:
        agg["absence_sum"] = (ABSENCE, "sum")
    if "department" in data.columns:
        agg["department"] = ("department", lambda s: s.mode().iat[0])
    teams = data.groupby("team_id").agg(**agg).reset_index()
    return teams[teams["people"] >= MIN_TEAM_SIZE].copy()


def team_excess_cost(df: pd.DataFrame, a: Assumptions, k: float = SMOOTHING_K,
                     frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per team: yearly turnover + absence cost above what the company rate would
    give for a team of that size. Counted on all employees. Rates of small teams are
    pulled toward the company rate first: smoothed = (events + k × company) ÷ (size + k),
    because one leaver more or less moves a 10-person team by a third of its rate.

    Columns ending in _dept compare each team with its own department instead of the
    company, so one high-turnover department does not fill the whole list.
    Descriptive only (no model)."""
    data = outcome_frame(df) if frame is None else frame
    if "team_id" not in data.columns:
        return pd.DataFrame()
    scale = a.scale(len(data))
    teams = _team_table(data, k)
    if teams.empty:
        return teams

    teams["excess_turnover_cost"] = 0.0
    teams["excess_absence_cost"] = 0.0
    teams["excess_turnover_cost_dept"] = 0.0
    teams["excess_absence_cost_dept"] = 0.0
    has_dept = "department" in teams.columns

    if "leavers" in teams:
        company = float(data[TURNOVER].mean())
        teams["turnover_rate"] = teams["leavers"] / teams["people"]
        smooth = (teams["leavers"] + k * company) / (teams["people"] + k)
        teams["excess_turnover_cost"] = (smooth - company) * teams["people"] * a.cost_per_leaver * scale
        if has_dept:
            dept = data.groupby("department")[TURNOVER].mean()
            base = teams["department"].map(dept)
            smooth_d = (teams["leavers"] + k * base) / (teams["people"] + k)
            teams["excess_turnover_cost_dept"] = ((smooth_d - base) * teams["people"]
                                                  * a.cost_per_leaver * scale)
    if "absence_sum" in teams:
        company = float(data[ABSENCE].mean())
        teams["absence_days"] = teams["absence_sum"] / teams["people"]
        smooth = (teams["absence_sum"] + k * company) / (teams["people"] + k)
        teams["excess_absence_cost"] = (smooth - company) * 2 * teams["people"] * a.day_cost * scale
        if has_dept:
            dept = data.groupby("department")[ABSENCE].mean()
            base = teams["department"].map(dept)
            smooth_d = (teams["absence_sum"] + k * base) / (teams["people"] + k)
            teams["excess_absence_cost_dept"] = ((smooth_d - base) * 2 * teams["people"]
                                                 * a.day_cost * scale)

    teams["excess_cost"] = teams["excess_turnover_cost"] + teams["excess_absence_cost"]
    teams["excess_cost_dept"] = (teams["excess_turnover_cost_dept"]
                                 + teams["excess_absence_cost_dept"])
    teams = teams.drop(columns=[c for c in ("leavers", "absence_sum") if c in teams])
    return teams.sort_values("excess_cost", ascending=False).reset_index(drop=True)


def department_excess_cost(df: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Per department: yearly turnover + absence cost above the company rate, on all
    employees. Departments are large, so no smoothing is needed."""
    data = outcome_frame(df)
    if "department" not in data.columns:
        return pd.DataFrame()
    scale = a.scale(len(data))
    agg: dict[str, tuple] = {"people": ("department", "size")}
    if TURNOVER in data.columns:
        agg["turnover_rate"] = (TURNOVER, "mean")
    if ABSENCE in data.columns:
        agg["absence_days"] = (ABSENCE, "mean")
    out = data.groupby("department").agg(**agg).reset_index()
    out["excess_turnover_cost"] = 0.0
    out["excess_absence_cost"] = 0.0
    if "turnover_rate" in out:
        out["excess_turnover_cost"] = ((out["turnover_rate"] - data[TURNOVER].mean())
                                       * out["people"] * a.cost_per_leaver * scale)
    if "absence_days" in out:
        out["excess_absence_cost"] = ((out["absence_days"] - data[ABSENCE].mean()) * 2
                                      * out["people"] * a.day_cost * scale)
    out["excess_cost"] = out["excess_turnover_cost"] + out["excess_absence_cost"]
    out["people"] = out["people"] * scale
    return out.sort_values("excess_cost", ascending=False).reset_index(drop=True)


def _outcome_waves(df: pd.DataFrame) -> list:
    if "wave" not in df.columns:
        return []
    cols = [c for c in (TURNOVER, ABSENCE) if c in df.columns]
    if not cols:
        return []
    return sorted(df.loc[df[cols].notna().any(axis=1), "wave"].dropna().unique())


def team_cost_persistence(df: pd.DataFrame, a: Assumptions, bar: float = PERSISTENCE_BAR,
                          share: float = PERSISTENCE_SHARE, min_flagged: int = 10,
                          k: float = SMOOTHING_K) -> dict[str, Any]:
    """Did the teams flagged in the earlier outcome year still cost extra in the next?

    Flags the costliest teams (smoothed, earlier wave) and compares what the page said
    they cost with what they really cost (unsmoothed) in the later wave. Passes when they
    keep at least `bar` of it. 'pending' with fewer than two outcome waves."""
    waves = _outcome_waves(df)
    if len(waves) < 2 or "team_id" not in df.columns:
        return {"status": "pending", "ratio": None, "said": None, "got": None, "flagged": 0,
                "text": "Only one year of outcomes: whether the costliest teams stay costly "
                        "has not been tested yet."}
    earlier, later = waves[-2], waves[-1]
    unit = unit_assumptions(a)
    before = team_excess_cost(df, unit, k, frame=df[df["wave"] == earlier])
    after = team_excess_cost(df, unit, 0, frame=df[df["wave"] == later])
    if before.empty or after.empty:
        return {"status": "pending", "ratio": None, "said": None, "got": None, "flagged": 0,
                "text": "Not enough teams in both years to test."}
    n_flag = max(min_flagged, int(math.ceil(share * len(before))))
    flagged = before.head(n_flag)
    said = float(flagged["excess_cost"].mean())
    got = float(after.set_index("team_id")["excess_cost"].reindex(flagged["team_id"]).mean())
    ratio = got / said if said > 0 else float("nan")
    status = "passed" if ratio >= bar else "failed"
    return {"status": status, "ratio": ratio, "said": said, "got": got, "flagged": n_flag,
            "earlier": earlier, "later": later,
            "text": (f"The {n_flag} costliest teams in wave {earlier:g} averaged "
                     f"{money(said, a.currency)} of extra cost; the same teams averaged "
                     f"{money(got, a.currency)} in wave {later:g} "
                     f"({ratio:.1%} kept; the bar is {bar:.0%}).")}


# PROCESS 7 – SURVEY-RESPONSE NOTE (fix 5)

def money_warning(df: pd.DataFrame, health_report: dict | None = None) -> str | None:
    """Plain-language note for every page that shows money (trap X7): leavers answer the
    survey less often, so the savings are likely understated. None when no such gap."""
    from .prediction_checks import survey_response_warning    # same rates as the prediction pages

    warning = survey_response_warning(outcome_frame(df), TURNOVER, health_report)
    if not warning:
        return None
    return (f"These savings are likely understated. People who later left were less likely "
            f"to answer the survey ({warning['leaver_response_rate']:.0%} vs "
            f"{warning['stayer_response_rate']:.0%} of those who stayed), so the links "
            "between engagement and turnover look weaker than they are, and the figures scale "
            "up from respondents, who leave less often than people who did not answer. The "
            "ranges shown reflect sampling error only, not this bias.")


# PROCESS 7B – CALCULATION RECEIPTS AND THE GALLUP BENCHMARK CHECK

# Published observed business-unit correlations between engagement and outcomes
# (Gallup Q12 meta-analysis, 11th edition, 2024, Table 3: mean observed r and observed SD
# across studies). Engagement here is the equally weighted mean of the Q12 items.
GALLUP_OBSERVED = {
    "turnover": {"label": "Voluntary turnover", "r": -0.10, "sd": 0.06},
    "absence": {"label": "Unplanned absence", "r": -0.26, "sd": 0.13},
    "performance": {"label": "Performance (Gallup: productivity)", "r": 0.14, "sd": 0.08},
}
GALLUP_TOP_VS_BOTTOM_QUARTILE = {
    "turnover": "21% lower in high-turnover organizations, 51% lower in low-turnover ones",
    "absence": "78% lower",
}


def gallup_benchmark_check(df: pd.DataFrame) -> pd.DataFrame:
    """Does the link between team engagement and team outcomes in this data look like what
    Gallup finds across organizations? Compares the team-level correlation of the Q12 average
    with each outcome to Gallup's published observed correlations (mean ± 1.28 SDs, roughly the
    middle 80% of studies). A plausibility check only: it does not feed any dollar figure,
    because Gallup's figures are business-unit correlations, not the individual-level effects
    the money pages use. Outcomes count everyone; the Q12 average counts respondents."""
    from .data_ingestion import Q12_ITEMS

    people = outcome_frame(df)
    survey = analysis_frame(df)
    items = [i for i in Q12_ITEMS if i in survey.columns]
    if "team_id" not in people.columns or not items:
        return pd.DataFrame()
    team = survey.assign(q12=survey[items].mean(axis=1)).groupby("team_id").agg(
        q12=("q12", "mean"), answered=("q12", "size"))
    out_cols = {"turnover": TURNOVER, "absence": ABSENCE, "performance": "performance_rating"}
    rows = []
    for key, column in out_cols.items():
        if column not in people.columns:
            continue
        size = people.groupby("team_id").size()
        rate = people.groupby("team_id")[column].mean()
        joined = pd.concat([team, rate.rename("outcome"), size.rename("people")], axis=1).dropna()
        joined = joined[(joined["people"] >= MIN_TEAM_SIZE) & (joined["answered"] >= MIN_TEAM_SIZE)]
        if len(joined) < 30 or joined["q12"].std() == 0 or joined["outcome"].std() == 0:
            continue
        ours = float(joined["q12"].corr(joined["outcome"]))
        g = GALLUP_OBSERVED[key]
        low, high = sorted([g["r"] - 1.28 * g["sd"], g["r"] + 1.28 * g["sd"]])
        if g["r"] < 0:
            verdict = ("weaker than Gallup usually finds" if ours > high else
                       "stronger than Gallup usually finds" if ours < low else
                       "within the range Gallup finds")
        else:
            verdict = ("weaker than Gallup usually finds" if ours < low else
                       "stronger than Gallup usually finds" if ours > high else
                       "within the range Gallup finds")
        rows.append({"outcome": g["label"], "teams": len(joined), "this_data_r": ours,
                     "gallup_r": g["r"], "gallup_low": low, "gallup_high": high,
                     "verdict": verdict})
    return pd.DataFrame(rows)


def _source_lines(a: Assumptions, sample: int) -> dict[str, tuple[str, str]]:
    register = assumption_register(a, sample)
    return {r["assumption"]: (r["value"], r["source"]) for _, r in register.iterrows()}


def calculation_receipts(df: pd.DataFrame, analysis: dict | None, a: Assumptions,
                         costs: dict[str, Any] | None = None, value: dict[str, Any] | None = None,
                         economics: dict[str, Any] | None = None,
                         health_report: dict | None = None) -> list[dict[str, Any]]:
    """One receipt per material dollar figure: population, model result, scenario, financial
    inputs, formula, result, source and limitations, so 'show me exactly how you calculated
    that' can be answered from the same numbers the page shows."""
    c = a.currency
    people = outcome_frame(df)
    n = len(people)
    wave = model_wave(df)
    costs = costs or current_costs(df, a)
    reg = _source_lines(a, n)

    def inputs(*labels: str) -> list[str]:
        return [f"{label}: {reg[label][0]}; source: {reg[label][1]}" for label in labels if label in reg]

    note = money_warning(df, health_report)
    common = ["These are associations in the data, not proven cause and effect."]
    if note:
        common.append(note)

    receipts: list[dict[str, Any]] = [{
        "figure": "What turnover and absence cost today",
        "result": money(costs["total_cost"], c),
        "population": (f"All {n:,} employees in survey wave {wave}, whether or not they answered "
                       f"the survey; scaled to a headcount of {costs['headcount']:,}."),
        "model": "None: counted directly from the outcome file.",
        "scenario": "Today, nothing changes.",
        "inputs": inputs("Average salary per year", "Cost to replace one leaver (share of salary)",
                         "Cost of one unplanned absence day", "Company headcount"),
        "formula": ("Turnover cost = leavers × cost per leaver. Absence cost = absence days × cost "
                    "per day. Leavers = turnover rate × employees × (headcount ÷ employees). "
                    "Absence days = average 6-month absence days × 2 × employees × (headcount ÷ employees)."),
        "calculation": [
            f"Turnover: {costs.get('leavers', 0):,.0f} leavers × {money(a.cost_per_leaver, c)} = "
            f"{money(costs.get('turnover_cost', 0), c)}",
            f"Absence: {costs.get('absence_days', 0):,.0f} days × {money(a.day_cost, c)} = "
            f"{money(costs.get('absence_cost', 0), c)}",
        ],
        "source": "individual_outcomes file: left_voluntarily_12m and unplanned_absence_days_6m.",
        "limits": ["The 6-month absence window is doubled to a year, which assumes absence is "
                   "spread evenly across the year.",
                   "Counts all voluntary turnover, not only the part linked to engagement."],
    }]

    program = (value or {}).get("program")
    if program and analysis is not None:
        findings = analysis["findings"].set_index(["item", "outcome"])
        model_lines = []
        for item in program["items"]:
            for outcome, unit in (("turnover", "odds ratio"), ("absence", "rate ratio")):
                if (item, outcome) in findings.index:
                    f = findings.loc[(item, outcome)]
                    model_lines.append(f"{item} and {outcome}: {unit} {float(f['effect']):.2f} per "
                                       f"point (95% interval {float(f['ci_low']):.2f} to "
                                       f"{float(f['ci_high']):.2f}), passed all checks")
        phys = program["physical"]
        lo, hi = value["range"]
        receipts.append({
            "figure": f"Value of lifting {program['names']}",
            "result": f"{money(program['value'], c)} a year (likely {money(program['low'], c)} to "
                      f"{money(program['high'], c)})",
            "population": (f"{program['people']:,.0f} employees (scaled to headcount) who answer "
                           f"1–3 on {program['names']}; fitted on survey respondents in wave {wave}."),
            "model": model_lines or ["No verified turnover or absence link."],
            "scenario": (f"Everyone who answers 1–3 on {program['names']} moves up one point; "
                         "everything else stays as it is today."),
            "inputs": inputs("Cost to replace one leaver (share of salary)",
                             "Cost of one unplanned absence day", "Company headcount"),
            "formula": ("For each person the verified model predicts the chance of leaving (and "
                        "expected absence days) today and again after the one-point rise. The "
                        "difference, added across people and scaled to headcount, is the leavers "
                        "and days avoided. Value = leavers avoided × cost per leaver + absence "
                        "days avoided × cost per day. The range repeats this with each model "
                        "link at the edges of its 95% interval."),
            "calculation": [
                f"Leavers avoided: {phys['estimate']['leavers']:,.0f} "
                f"(range {phys['cautious']['leavers']:,.0f} to {phys['optimistic']['leavers']:,.0f})",
                f"Absence days avoided: {phys['estimate']['absence_days']:,.0f}",
                f"Value: {phys['estimate']['leavers']:,.0f} × {money(a.cost_per_leaver, c)} + "
                f"{phys['estimate']['absence_days']:,.0f} × {money(a.day_cost, c)} = "
                f"{money(program['value'], c)}",
            ],
            "source": "Verified findings from the Outcomes analysis; same engine as the Scenarios page.",
            "limits": common + ["The range reflects sampling error only, not the economic "
                                "assumptions (see the sensitivity table) or the survey-response bias.",
                                "A modeled opportunity is not a realized return: it assumes a program "
                                "can actually move these scores."],
        })

    if economics:
        base = economics["cases"]["base"]
        receipts.append({
            "figure": "Return on the program cost you entered",
            "result": f"net {money(base['net'], c)}, ROI {base['roi']:.0%}",
            "population": "The same people as the program above.",
            "model": ["Uses the program value above; no new statistics."],
            "scenario": (f"The program costs {money(economics['cost'], c)} a year and delivers "
                         f"{economics['realization']:.0%} of the modeled effect."),
            "inputs": inputs("Yearly program cost", "Share of the modeled effect the program delivers"),
            "formula": ("Benefit = modeled value × share delivered. Net = benefit − cost. ROI = net ÷ cost. "
                        "Payback (months) = 12 × cost ÷ benefit. Break-even cost = benefit. "
                        "Share of the modeled effect needed to break even = cost ÷ modeled value."),
            "calculation": [
                f"Benefit: {money(base['benefit'], c)}; net: {money(base['net'], c)}; "
                f"payback: {base['payback_months']:.1f} months",
                f"Break-even: the program can cost up to {money(economics['break_even_cost'], c)} a year "
                f"({money(economics['break_even_cost_per_person'], c)} per person), or must deliver "
                f"at least {base['required_realization']:.0%} of the modeled effect.",
            ],
            "source": "Program cost and share delivered are your assumptions, not evidence.",
            "limits": ["The effect of any real program on these scores is not known from the data; "
                       "this shows what would have to be true, not what will happen."],
        })
    return receipts


def receipt_markdown(receipt: dict[str, Any]) -> str:
    """One receipt as markdown ('$' escaped for Streamlit)."""
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {x}" for x in items)

    text = (f"**{receipt['figure']}: {receipt['result']}**\n\n"
            f"**Who is counted:** {receipt['population']}\n\n"
            f"**Model result:** " + ("\n" + bullets(receipt["model"]) if isinstance(receipt["model"], list)
                                      else receipt["model"]) + "\n\n"
            f"**Scenario:** {receipt['scenario']}\n\n"
            f"**Inputs:**\n{bullets(receipt['inputs'])}\n\n"
            f"**Formula:** {receipt['formula']}\n\n"
            f"**Calculation:**\n{bullets(receipt['calculation'])}\n\n"
            f"**Source:** {receipt['source']}\n\n"
            f"**Limits:**\n{bullets(receipt['limits'])}")
    return md(text)


# PROCESS 8 – PLAIN LANGUAGE

def money(value: float, currency: str = "$") -> str:
    """1_234_567 -> '$1.2M', 45_300 -> '$45K', 900 -> '$900'."""
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e9:
        return f"{sign}{currency}{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{sign}{currency}{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{sign}{currency}{v / 1e3:.0f}K"
    return f"{sign}{currency}{v:,.0f}"


def share_text(part: float, whole: float) -> str:
    """'43%', or '<1%' / '0%' for tiny shares, so a small share never reads as nothing."""
    if not whole:
        return "0%"
    s = part / whole
    return "<1%" if 0 < s < 0.01 else f"{s:.0%}"


def range_text(low: float, high: float, currency: str = "$") -> str:
    return f"likely {money(low, currency)} to {money(high, currency)}"


def explain_value(costs: dict[str, Any], value: dict[str, Any], teams: pd.DataFrame,
                  a: Assumptions) -> dict[str, Any]:
    c = a.currency
    points = []
    if costs.get("total_cost"):
        points.append(f"Turnover and unplanned absence cost about "
                      f"**{money(costs['total_cost'], c)}** a year today.")
    for _, s in value["scenarios"].iterrows():
        rng = "" if pd.isna(s.get("low")) else f" ({range_text(s['low'], s['high'], c)})"
        points.append(f"**{s['source']}:** {s['what']} → about {s['units']:,.0f} "
                      f"{s['unit_label']}, worth **{money(s['value'], c)}** a year{rng}.")
    if not teams.empty:
        above = teams[teams["excess_cost"] > 0]
        if not above.empty:
            top = above.head(10)
            points.append(f"{len(above):,} teams cost more than the company rate would; the "
                          f"top 10 account for {share_text(top['excess_cost'].sum(), above['excess_cost'].sum())} "
                          "of that extra cost.")
    program = value.get("program")
    if value["total"] > 0 and program:
        low, high = value["range"]
        headline = (f"Lifting {program['names']} for the people who answer 1–3 could be worth "
                    f"about {money(value['total'], c)} a year ({range_text(low, high, c)}).")
    elif value["total"] > 0:
        headline = (f"Engagement improvements backed by your data could be worth about "
                    f"{money(value['total'], c)} a year.")
    else:
        headline = "Add turnover, absence or performance data to see what engagement could be worth."
    return {"headline": headline, "points": points}


# PROCESS 9 – CHARTS

def value_chart(value: dict[str, Any], currency: str = "$") -> alt.Chart:
    data = value["scenarios"].assign(label=lambda d: d["value"].map(lambda v: money(v, currency)))
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=22, color=HIGH_COLOR).encode(
        y=alt.Y("source:N", sort="-x", title=None),
        x=alt.X("value:Q", title=f"Estimated yearly value ({currency})", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("source:N", title="Scenario"),
                 alt.Tooltip("lever:N", title="Lever"),
                 alt.Tooltip("what:N", title="Assumes"),
                 alt.Tooltip("label:N", title="Value")])
    text = bars.mark_text(align="left", dx=6, color="#333").encode(text="label:N")
    return (bars + text).properties(height=60 * max(1, len(data)), width=560)


def excess_cost_chart(teams: pd.DataFrame, currency: str = "$", n: int = 15,
                      basis: str = "company") -> alt.Chart:
    """Top teams by extra yearly cost. basis='company' compares with the company rate,
    basis='department' with each team's own department."""
    suffix = "_dept" if basis == "department" else ""
    total = teams["excess_cost" + suffix]
    top = teams.assign(_total=total).sort_values("_total", ascending=False)
    top = top[top["_total"] > 0].head(n)
    long = top.melt(id_vars=["team_id"],
                    value_vars=["excess_turnover_cost" + suffix, "excess_absence_cost" + suffix],
                    var_name="kind", value_name="cost")
    long["kind"] = long["kind"].map({"excess_turnover_cost" + suffix: "Turnover",
                                     "excess_absence_cost" + suffix: "Absence"})
    long = long[long["cost"] > 0]
    where = "own department" if basis == "department" else "company rate"
    return alt.Chart(long).mark_bar(height=14).encode(
        y=alt.Y("team_id:N", sort=list(top["team_id"]), title=None,
                axis=alt.Axis(labelOverlap=False)),
        x=alt.X("sum(cost):Q", title=f"Extra yearly cost vs {where} ({currency})",
                axis=alt.Axis(format="~s")),
        color=alt.Color("kind:N", title=None,
                        scale=alt.Scale(domain=["Turnover", "Absence"],
                                        range=["#B5473F", MUTED_COLOR]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("team_id:N", title="Team"), alt.Tooltip("kind:N"),
                 alt.Tooltip("cost:Q", title="Extra cost", format=",.0f")],
    ).properties(height=alt.Step(26), width="container")


# PROCESS 10 – PAGE HELPERS: ASSUMPTION SUMMARY AND WHERE-THE-MONEY STORY

def md(text: str) -> str:
    """Escape '$' so Streamlit markdown does not read '$...$' as a math formula."""
    return text.replace("$", "\\$")


def assumptions_line(a: Assumptions, sample: int) -> str:
    """One compact line with the assumptions behind every figure."""
    c = a.currency
    return md(f"{c}{a.avg_salary:,.0f} salary · {a.replacement_cost_pct:.0%} of salary to replace "
            f"a leaver · {c}{a.day_cost:,.0f} per absence day · {a._headcount(sample):,} people"
            + (f" · {c}{a.value_per_rating_point:,.0f} per performance point"
               if a.value_per_rating_point > 0 else ""))


def explain_departments(departments: pd.DataFrame, a: Assumptions) -> dict[str, Any]:
    above = departments[departments["excess_cost"] > 0]
    if above.empty:
        return {"headline": "No department costs more than the company rate would.", "points": []}
    c = a.currency
    top = above.iloc[0]
    points = [md(f"**{r['department']}:** {money(r['excess_cost'], c)} a year more than the "
                 f"company rate ({r['turnover_rate']:.1%} turnover)")
              for _, r in above.head(3).iterrows()]
    return {"headline": md(f"**{top['department']}** costs {money(top['excess_cost'], c)} a year "
                           "more than the company rate would for its size."),
            "points": points}


def explain_where_money(teams: pd.DataFrame, a: Assumptions) -> dict[str, Any]:
    above = teams[teams["excess_cost"] > 0]
    if above.empty:
        return {"headline": "No team costs more than the company rate would.", "points": []}
    c = a.currency
    total = above["excess_cost"].sum()
    top10 = above.head(10)["excess_cost"].sum()
    points = [f"The 10 costliest teams account for {share_text(top10, total)} of that extra cost "
              f"({money(top10, c)})."]
    if "department" in above.columns:
        by_dept = above.groupby("department")["excess_cost"].sum().sort_values(ascending=False)
        points.append("Most of it sits in " + " and ".join(
            f"**{d}** ({money(v, c)})" for d, v in by_dept.head(2).items()) + ".")
    turnover = above["excess_turnover_cost"].clip(lower=0).sum()
    absence = above["excess_absence_cost"].clip(lower=0).sum()
    points.append(f"About {turnover / max(turnover + absence, 1):.0%} of the extra cost comes "
                  "from turnover, the rest from absence.")
    return {"headline": md(f"{len(above):,} teams cost {money(total, c)} a year more than "
                            "the company rate would for teams their size."),
            "points": [md(p) for p in points]}
