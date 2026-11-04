# PAGE BUILDER – turns src/layout.py into Streamlit pages.
#
# A page with several views shows a switcher and runs only the chosen page
# file, so each view behaves exactly as it does as a stand-alone page.

import re
import runpy

import streamlit as st

from src.layout import NAVIGATION, VIEW_HIDE_KEY
from src.ui import page_header

EMBEDDED_KEY = "_embedded_view"


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _view(view):
    """('Name', 'file.py') or ('Name', 'file.py', {'hide': [...]}) -> (name, path, hide)."""
    name, path = view[0], view[1]
    options = view[2] if len(view) > 2 else {}
    return name, path, tuple(options.get("hide", ()))


def reset_view_state():
    """Clear the view flags. Called at the start of every page run, so a flag left
    over from a run that was interrupted (a click reruns the app mid-page) can
    never hide the next page's title."""
    st.session_state[EMBEDDED_KEY] = False
    st.session_state[VIEW_HIDE_KEY] = ()


def run_view(path, hide=(), embedded=True):
    """Run a page file under the current page, hiding the given sections."""

    st.session_state[EMBEDDED_KEY] = embedded
    st.session_state[VIEW_HIDE_KEY] = hide
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        st.session_state[EMBEDDED_KEY] = False
        st.session_state[VIEW_HIDE_KEY] = ()


def _stack(entry):
    """A page whose views all show, one under the other, under the page's own title."""
    views = [_view(v) for v in entry["views"]]

    def render():
        reset_view_state()
        page_header(entry["title"], entry["icon"], entry.get("description", ""))
        for _, path, hide in views:
            run_view(path, hide)

    render.__name__ = f"page_{_slug(entry['title']).replace('-', '_')}"
    return render


def _hub(entry):
    views = [_view(v) for v in entry["views"]]
    names = [name for name, _, _ in views]
    by_name = {name: (path, hide) for name, path, hide in views}
    key = f"view_{_slug(entry['title'])}"

    def render():
        reset_view_state()
        page_header(entry["title"], entry["icon"], entry.get("description", ""))
        choice = st.segmented_control(
            "View", options=names, default=names[0], key=key,
            label_visibility="collapsed",
        ) or names[0]
        path, hide = by_name[choice]
        run_view(path, hide)

    render.__name__ = f"hub_{_slug(entry['title']).replace('-', '_')}"
    return render


def build_pages():
    """Return ({section: [st.Page]}, {page file: first st.Page that shows it})."""

    pages, by_file = {}, {}

    for section, entries in NAVIGATION.items():
        pages[section] = []

        for entry in entries:
            common = dict(title=entry["title"], icon=entry["icon"],
                          default=entry.get("default", False))
            views = [_view(v) for v in entry["views"]]

            # One view, or "stack": the views show one under the other with the
            # page's own title. Several views: a switcher at the top.
            if len(views) == 1 or entry.get("stack"):
                page = st.Page(_stack(entry), url_path=_slug(entry["title"]), **common)
            else:
                page = st.Page(_hub(entry), url_path=_slug(entry["title"]), **common)

            for _, path, _ in views:
                by_file.setdefault(path, page)
            pages[section].append(page)

    return pages, by_file
