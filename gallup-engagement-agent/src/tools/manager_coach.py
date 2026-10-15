"""AI manager coach: rewrites the verified manager brief in a warm, practical voice.

The AI only receives the facts in the brief. If the AI service is not configured,
or the answer contains a number that is not in the brief, the app falls back to
the deterministic brief so a manager never sees an unverified number.
"""


# PROCESS 1 – IMPORTS

import json
import re

from src.tools.insights import (
    brief_markdown,
)


# PROCESS 2 – CONFIGURATION

# Time frames named in COACH_PROMPT; passed to the AI so it may repeat them.
ACTION_WINDOW_DAYS = 30

CONVERSATION_MINUTES = 15

COACH_PROMPT = """
You are a supportive coach for a people manager. Audience: a busy, nontechnical manager.

Use ONLY the facts in TEAM_BRIEF. Do not invent or change any number.
Never claim cause and effect: say "linked to", never "causes" or "drives".
Do not mention statistics terms (odds ratio, regression, p-value).

Write in markdown, under 250 words. Use a short bold heading for each section
(no numbered lists for headings), with these sections:
1. One-sentence summary of where the team stands.
2. What is going well (use the strengths, under the wording in strengths_heading;
   if it says they are still below the company, say so honestly).
3. This month's focus (the focus item) and why it matters.
4. Three actions for the next 30 days (adapt the suggested actions to the team).
5. Two questions to open a 15-minute team conversation.
End with one encouraging sentence.

If TEAM_BRIEF has a "risk" section: open section 1 with its urgency and headline,
make at least one of the three actions come from its steps, and describe the risk
as a likelihood, never a certainty. Never suggest telling the team its risk label.
"""


# PROCESS 3 – NUMBER VERIFICATION

# A number not attached to a letter, so "Q04" and "T000123" are ignored.
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z\d.])\d+(?:,\d{3})*(?:\.\d+)?"
)

# Markdown list or heading numbering such as "1." or "## 2)".
LIST_MARKER_PATTERN = re.compile(
    r"^\s*(?:#+\s*)?\d+[.)]\s",
    re.MULTILINE,
)


def extract_numbers(text):

    text = LIST_MARKER_PATTERN.sub("", text)

    return {
        round(float(match.replace(",", "")), 2)
        for match in NUMBER_PATTERN.findall(text)
    }


def coach_facts(brief):
    """The brief plus the time frames the coach prompt asks the AI to use."""

    return {
        **brief,
        "action_window_days": ACTION_WINDOW_DAYS,
        "conversation_minutes": CONVERSATION_MINUTES,
    }


def find_unverified_numbers(answer, brief):

    allowed = extract_numbers(
        json.dumps(coach_facts(brief), default=str)
        + "\n"
        + brief_markdown(brief)
    )

    return sorted(
        extract_numbers(answer) - allowed
    )


# PROCESS 4 – AI COACH

def coach_brief(
    brief,
    client=None,
):
    """
    Rewrite the brief in a coaching voice.

    Returns a dict with:
    - text: markdown to show the manager
    - source: "ai" or "deterministic"
    - reason: why the fallback was used, or None
    """

    fallback = brief_markdown(brief)

    def use_fallback(reason):

        return {
            "text": fallback,
            "source": "deterministic",
            "reason": reason,
        }

    try:

        if client is None:

            # Imported here so the brief works without the AI package set up.
            from src.agent import get_azure_client

            client = get_azure_client()

        from src.agent import AZURE_DEPLOYMENT

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,

            messages=[
                {
                    "role": "system",
                    "content": COACH_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "TEAM_BRIEF:\n"
                        + json.dumps(
                            coach_facts(brief),
                            indent=2,
                            default=str,
                        )
                    ),
                },
            ],

            max_completion_tokens=4000,
        )

        answer = response.choices[0].message.content or ""

    except Exception as exc:

        return use_fallback(
            f"AI coach unavailable: {exc}"
        )

    if not answer.strip():

        return use_fallback(
            "AI coach returned an empty answer."
        )

    unverified = find_unverified_numbers(answer, brief)

    if unverified:

        return use_fallback(
            "AI answer contained numbers not in the brief: "
            + ", ".join(f"{number:g}" for number in unverified)
        )

    return {
        "text": answer,
        "source": "ai",
        "reason": None,
    }
