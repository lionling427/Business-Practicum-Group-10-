"""Browser check of the whole app with Playwright.

Starts the app (or uses one already running), uploads a data file in the
sidebar, opens every page from the navigation, clicks the analysis buttons,
saves a full-page screenshot per page and reports any error shown on a page.

    python scripts/screenshot_pages.py                       # dev data, own server
    python scripts/screenshot_pages.py --file ~/Downloads/gallup_q12_survey_5yr.csv
    python scripts/screenshot_pages.py --url http://localhost:8501 --pages Outcomes Financials

Screenshots go to docs/screenshots/ (or --out). Exit code 1 if any page shows
an error. One-time setup:  pip install playwright && python -m playwright install chromium
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGES = ["Overview", "CEO brief", "Engagement", "Trends", "Outcomes", "Teams", "Action story", "Predictions",
         "Recommendations", "Financials", "Scenarios", "Ask AI", "Data", "Audit"]
RUN_BUTTONS = ["Run statistical analysis", "Run change-over-time analysis",
               "Build recommendations", "Run statistical audit"]
IDLE_TIMEOUT_MS = 240_000


# PROCESS 2 – SERVER AND DATA

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def app_server(url: str | None):
    """Yield the app URL: the given one, or a fresh headless server."""
    if url:
        yield url.rstrip("/")
        return
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    address = f"http://localhost:{port}"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{address}/_stcore/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        yield address
    finally:
        process.terminate()
        process.wait(timeout=20)


def dev_csv() -> Path:
    """The dev dataset as one uploaded-style CSV (survey + outcomes)."""
    import pandas as pd

    from src.agent import DEFAULT_DATASET

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")
    data = survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])
    out = Path(tempfile.mkdtemp()) / "dev_upload.csv"
    data.to_csv(out, index=False)
    return out


# PROCESS 3 – BROWSER STEPS

def wait_idle(page) -> None:
    """Wait until Streamlit has finished running the script."""
    page.wait_for_timeout(800)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=IDLE_TIMEOUT_MS)
    page.wait_for_timeout(500)


def page_errors(page) -> list[str]:
    return [e.inner_text()[:300] for e in page.locator('[data-testid="stException"]').all()]


def check_app(url: str, data_file: Path, pages: list[str], out: Path,
              width: int = 1440) -> dict[str, list[str]]:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    problems: dict[str, list[str]] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 1000})
        page.goto(url)
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=60_000)
        wait_idle(page)

        page.locator('[data-testid="stSidebar"] input[type="file"]').set_input_files(str(data_file))
        wait_idle(page)

        nav = page.locator('[data-testid="stSidebarNav"]')
        for name in pages:
            # The link holds an icon name and the page title: match the title exactly.
            nav.locator('[data-testid="stSidebarNavLink"]').filter(
                has=page.get_by_text(name, exact=True)).first.click()
            wait_idle(page)
            for label in RUN_BUTTONS:
                button = page.get_by_role("button", name=label)
                if button.count() and button.first.is_visible():
                    button.first.click()
                    wait_idle(page)
            errors = page_errors(page)
            if errors:
                problems[name] = errors
            shot = out / f"{name.lower().replace(' ', '_')}.png"
            # Streamlit scrolls inside its main panel: grow the window to fit it.
            height = page.evaluate(
                "() => (document.querySelector('[data-testid=\"stMain\"]')"
                " || document.scrollingElement).scrollHeight")
            page.set_viewport_size({"width": width, "height": min(int(height) + 40, 12_000)})
            page.wait_for_timeout(800)
            page.screenshot(path=str(shot))
            page.set_viewport_size({"width": width, "height": 1000})
            print(f"{'ERROR' if errors else 'ok':5}  {name:16} {shot}")
        browser.close()
    return problems


# PROCESS 4 – COMMAND LINE

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", help="Use a running app instead of starting one.")
    parser.add_argument("--file", type=Path, help="Data file to upload (default: dev data).")
    parser.add_argument("--pages", nargs="+", default=PAGES, choices=PAGES)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    args = parser.parse_args(argv)

    data_file = args.file.expanduser() if args.file else dev_csv()
    with app_server(args.url) as url:
        problems = check_app(url, data_file, args.pages, args.out)
    for name, errors in problems.items():
        print(f"\n{name}:\n  " + "\n  ".join(errors))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
