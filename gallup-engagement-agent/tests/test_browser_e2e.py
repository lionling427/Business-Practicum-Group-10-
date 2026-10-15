"""Real-browser check of every page (Playwright). Opt-in because it starts a
server and takes about a minute:

    RUN_E2E=1 python -m pytest tests/test_browser_e2e.py
"""

# PROCESS 1 – IMPORTS

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(os.environ.get("RUN_E2E") != "1",
                                reason="browser test: set RUN_E2E=1 to run")


# PROCESS 2 – EVERY PAGE OPENS WITHOUT AN ERROR IN A REAL BROWSER

def test_every_page_renders_in_a_browser(tmp_path):

    pytest.importorskip("playwright")
    from screenshot_pages import PAGES, app_server, check_app, dev_csv

    with app_server(None) as url:
        problems = check_app(url, dev_csv(), PAGES, tmp_path)

    assert problems == {}
    assert len(list(tmp_path.glob("*.png"))) == len(PAGES)
