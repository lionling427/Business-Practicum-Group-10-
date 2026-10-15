# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.insights import analysis_frame, team_scorecard
from src.tools.visualization import (
    VISUAL_CONFIG,
    _auc,
    add_engagement_status,
    build_engagement_trend,
    build_sorting_groups,
    engagement_trend_chart,
    rank_sorting_power,
    segment_engagement_chart,
    sorting_effect_chart,
    sorting_power_chart,
    team_heatmap_chart,
)


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def data():

    path = DEFAULT_DATASET / "agent_data"

    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")

    keys = [
        column
        for column in survey.columns
        if column in individual.columns
    ]

    return survey.merge(individual, on=keys)


# PROCESS 3 – SORTING POWER MATH

def test_auc_perfect_none_and_reversed():

    high = pd.Series([4, 5, 5])
    low = pd.Series([1, 2, 2])

    assert _auc(high, low) == 1.0
    assert _auc(low, high) == 0.0
    assert _auc(high, high) == 0.5


def test_auc_empty_group_is_nan():

    assert pd.isna(_auc(pd.Series([4.0]), pd.Series([], dtype=float)))


# PROCESS 4 – ENGAGEMENT STATUS AND TREND

def test_engaged_flag_uses_threshold(data):

    status = add_engagement_status(data)

    engaged = status["q12_mean"] >= VISUAL_CONFIG["engaged_threshold"]

    assert (status["engaged"] == engaged.astype(int)).all()


def test_trend_has_one_row_per_wave(data):

    trend = build_engagement_trend(data)

    assert trend["has_trend"]
    assert trend["time_column"] == "wave"
    assert len(trend["table"]) == data["wave"].nunique()
    assert trend["table"]["pct_engaged"].between(0, 100).all()


def test_group_trend_adds_all_employees_and_hides_small_groups(data):

    trend = build_engagement_trend(data, "department")
    table = trend["table"]

    assert "All employees" in set(table["department"])
    assert (table["employees"] >= VISUAL_CONFIG["min_group_size"]).all()


def test_single_period_has_no_trend(data):

    trend = build_engagement_trend(data[data["wave"] == 1])

    assert not trend["has_trend"]


# PROCESS 5 – SORTING GROUPS

@pytest.mark.parametrize("outcome", ["performance", "turnover", "absence"])
@pytest.mark.parametrize("level", ["team", "employee"])
def test_sorting_groups_have_both_sides(data, outcome, level):

    units = build_sorting_groups(data, outcome, level)

    assert set(units["group"]) == {"High performers", "Low performers"}


def test_team_sorting_respects_min_team_size(data):

    units = build_sorting_groups(data, "performance", "team")

    assert (units["size"] >= VISUAL_CONFIG["min_team_size"]).all()


def test_high_performer_teams_have_better_outcomes(data):

    units = build_sorting_groups(data, "turnover", "team")

    means = units.groupby("group")["outcome"].mean()

    assert means["High performers"] < means["Low performers"]


def test_rank_sorting_power_covers_all_items(data):

    ranking = rank_sorting_power(data)

    assert len(ranking) == 12
    assert ranking["sorting_power"].is_monotonic_decreasing
    assert ranking["sorting_power"].between(0, 1).all()


def test_missing_outcome_column_raises(data):

    with pytest.raises(ValueError):
        build_sorting_groups(
            data.drop(columns=["performance_rating"]),
            "performance",
        )


# PROCESS 6 – CHARTS COMPILE

def test_all_charts_compile(data):

    ranking = rank_sorting_power(data)
    frame = analysis_frame(data)
    company_means = frame[[f"Q{i:02d}" for i in range(1, 13)]].mean()

    charts = [
        engagement_trend_chart(build_engagement_trend(data)),
        engagement_trend_chart(build_engagement_trend(data, "department")),
        engagement_trend_chart(
            build_engagement_trend(data[data["wave"] == 1], "department")
        ),
        sorting_power_chart(ranking),
        sorting_effect_chart(data, list(ranking["item"].head(2))),
        sorting_effect_chart(data, ["Q04"], "turnover", "employee"),
        segment_engagement_chart(data, "department"),
        team_heatmap_chart(team_scorecard(data), company_means),
    ]

    for chart in charts:
        assert chart.to_dict()


def test_sorting_uses_first_wave_only(data):

    units = build_sorting_groups(data, "performance", "employee")

    assert units["employee_id"].is_unique
    assert len(units) <= len(analysis_frame(data))


# PROCESS 7 – PLAIN-LANGUAGE EXPLANATIONS

from src.tools.visualization import (  # noqa: E402
    explain_engagement_trend,
    explain_sorting_items,
    explain_sorting_power,
    explain_team_heatmap,
)


@pytest.mark.parametrize("group", [None, "department"])
def test_trend_explanation_matches_table(data, group):

    trend = build_engagement_trend(data, group)
    story = explain_engagement_trend(trend)

    table = trend["table"]
    overall = table[table[group] == "All employees"] if group else table
    latest_pct = overall.sort_values("wave")["pct_engaged"].iloc[-1]

    assert story["headline"].startswith(f"{latest_pct:.0f}%")
    assert "wave 1 " in story["points"][0]
    assert story["how_to_read"]


def test_single_period_trend_explanation(data):

    story = explain_engagement_trend(
        build_engagement_trend(data[data["wave"] == 1], "department")
    )

    assert any("Highest department" in point for point in story["points"])


@pytest.mark.parametrize("outcome", ["performance", "turnover", "absence"])
@pytest.mark.parametrize("level", ["team", "employee"])
def test_sorting_explanation_names_top_item(data, outcome, level):

    ranking = rank_sorting_power(data, outcome, level)
    story = explain_sorting_power(ranking, outcome, level)

    assert ranking["item"].iloc[0] in story["headline"]

    pct = round(ranking["sorting_power"].iloc[0] * 100)
    assert f"{pct} times out of 100" in story["points"][0]

    reversed_items = ranking.loc[ranking["verdict"] == "Reversed", "item"]
    for item in reversed_items:
        assert any(item in point for point in story["points"])


def test_sorting_items_explanation_uses_group_means(data):

    ranking = rank_sorting_power(data)
    story = explain_sorting_items(ranking, ["Q03", "Q04"])

    q03 = ranking.set_index("item").loc["Q03"]

    assert len(story["points"]) == 2
    assert f"{q03['high_mean']:.2f}" in story["points"][0]


def test_heatmap_explanation_names_a_shown_team(data):

    teams = team_scorecard(data)
    frame = analysis_frame(data)
    company = frame[[f"Q{i:02d}" for i in range(1, 13)]].mean()

    story = explain_team_heatmap(teams, company)

    shown = set(teams["team_id"].head(25))

    assert any(team in story["points"][0] for team in shown)


# PROCESS 8 – OVERVIEW CARD

from src.tools.visualization import (  # noqa: E402
    build_overview,
    explain_overview,
    percentile_meter_chart,
    score_gauge_chart,
    trend_sparkline_chart,
)


def test_company_overview_matches_latest_wave(data):

    overview = build_overview(data)
    status = add_engagement_status(data)
    latest = status[status["wave"] == status["wave"].max()]

    assert overview["current"] == round(latest["q12_mean"].mean(), 2)
    assert overview["percentile"] is None
    assert len(overview["trend"]) == data["wave"].nunique()
    assert abs(overview["delta"] - (overview["current"] - overview["previous"])) < 0.011


def test_team_overview_ranks_against_other_teams(data):

    team = data["team_id"].value_counts().index[0]
    overview = build_overview(data, "team_id", team)

    assert 0 <= overview["percentile"] <= 100
    assert overview["n_peers"] > 1
    assert overview["percentile_label"].endswith(("st", "nd", "rd", "th"))
    assert "percentile" in explain_overview(overview)


def test_overview_refuses_tiny_groups(data):

    tiny = data.assign(team_id="BIG")
    tiny.loc[tiny.index[:3], "team_id"] = "TINY"

    with pytest.raises(ValueError):
        build_overview(tiny, "team_id", "TINY")


def test_overview_charts_compile(data):

    overview = build_overview(data, "department", data["department"].iloc[0])

    for chart in [
        score_gauge_chart(overview["current"], overview["company_current"]),
        score_gauge_chart(overview["current"]),
        percentile_meter_chart(overview["percentile"], "label"),
        trend_sparkline_chart(overview["trend"], overview["period_name"]),
    ]:
        assert chart.to_dict()


# PROCESS 9 – ENGAGEMENT PAGE: QUESTIONS AND GROUPS

from src.tools.visualization import (  # noqa: E402
    explain_item_results,
    explain_segment_engagement,
    item_results,
    item_results_chart,
    segment_engagement,
)


def test_item_results_latest_wave_ranked(data):

    results = item_results(data)
    table = results["table"]
    latest = add_engagement_status(data)
    latest = latest[latest["wave"] == latest["wave"].max()]

    assert len(table) == 12
    assert table["agree"].is_monotonic_decreasing
    assert results["responses"] == len(latest)
    assert table.loc[table["item"] == "Q01", "agree"].iloc[0] == pytest.approx(
        (latest["Q01"] >= 4).mean() * 100)
    assert "change" in table
    assert list(table["group"].head(3)) == ["Top 3"] * 3
    assert explain_item_results(results)["headline"]
    assert item_results_chart(results).to_dict()


def test_segment_engagement_uses_latest_wave_and_matches_company(data):

    result = segment_engagement(data, "department")
    latest = add_engagement_status(data)
    latest = latest[latest["wave"] == latest["wave"].max()]

    assert result["company"] == pytest.approx(latest["engaged"].mean() * 100)
    assert result["table"]["pct_engaged"].is_monotonic_decreasing
    assert "below the company" in explain_segment_engagement(result)["points"][0]


def test_trend_story_mentions_last_survey_with_three_waves(data):

    frames = [data.assign(wave=w) for w in (1, 2, 3)]
    story = explain_engagement_trend(build_engagement_trend(pd.concat(frames)))

    assert any("Since the last survey" in p for p in story["points"])
    assert any("Lowest point" in p for p in story["points"])


# PROCESS 10 – AGREE / NEUTRAL / DISAGREE SPLIT

from src.tools.visualization import (  # noqa: E402
    explain_sorting_distribution,
    sorting_distribution,
    sorting_distribution_chart,
)


@pytest.mark.parametrize("level", ["employee", "team"])
def test_sorting_distribution_shares_sum_to_100(data, level):

    dist = sorting_distribution(data, "Q03", "performance", level)
    totals = dist["table"].groupby("who")["share"].sum()

    assert totals.round(6).eq(100).all()
    assert set(dist["table"]["band"]) <= {"Disagree", "Neutral", "Agree"}
    assert "%" in explain_sorting_distribution(dist)
    assert sorting_distribution_chart(dist).to_dict()


def test_sorting_distribution_counts_everyone(data):

    dist = sorting_distribution(data, "Q03", "performance", "employee")
    units = build_sorting_groups(data, "performance", "employee")

    assert dist["n_better"] + dist["n_worse"] == len(units)
