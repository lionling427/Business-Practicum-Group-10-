# PROCESS 1 – IMPORTS

import json
from datetime import date

import streamlit as st

from src.tools.audit import (
    AREAS,
    STATUS_ORDER,
    audit_markdown,
    audit_record,
    run_audit,
)
from src.tools.financial import Assumptions, md
from src.tools.insights import analysis_frame
from src.ui import (
    OUTCOME_LEVEL,
    analysis_runner,
    page_header,
    require_data,
)


# PROCESS 2 – CONFIGURATION

BADGES = {
    "pass": ":green-badge[:material/check: Pass]",
    "warn": ":orange-badge[:material/warning: Warning]",
    "fail": ":red-badge[:material/error: Fail]",
    "skip": ":gray-badge[:material/remove: Not checked]",
}
MEANING = {
    "pass": "the check found no problem",
    "warn": "fine to share, but mention it",
    "fail": "fix before sharing",
    "skip": "not enough data to run this check",
}


def friendly(table):
    """Readable column names for the numbers behind a check."""
    return table.rename(columns=lambda c: str(c).replace("_", " ").capitalize())


# PROCESS 3 – HEADER AND DATA

page_header(
    "Statistical audit",
    ":material/verified:",
    "An independent check of the statistics behind every page: is the math right, are the "
    "results stable, how sure are the money figures, and can the run be reproduced.",
)

df, validation = require_data()

if validation["capabilities"]["level"] != OUTCOME_LEVEL:
    st.info("The audit checks the outcome models, so it needs turnover, absence or "
            "performance in the uploaded data.", icon=":material/info:")
    st.stop()

analysis = analysis_runner(df, key="run_analysis_audit")
if analysis is None:
    st.stop()

assumptions = (st.session_state.get("assumptions")
               or Assumptions(headcount=len(analysis_frame(df))))
raw = st.session_state.get("uploaded_raw", df)
audit_key = (st.session_state.get("upload_signature"), id(analysis), repr(assumptions))
stored = st.session_state.get("audit_result")
audit = stored["audit"] if stored and stored["key"] == audit_key else None


# PROCESS 4 – RUN BUTTON (THE AUDIT TAKES A FEW SECONDS)

if audit is None:

    with st.container(border=True):
        st.markdown("**:material/verified: Run the statistical audit**")
        st.caption("About 15 seconds. It refits every model with an independent statistics "
                   "library, repeats findings in other surveys and without each department, "
                   "and puts ranges on the money figures.")
        if st.button("Run statistical audit", type="primary", icon=":material/play_arrow:",
                     key="run_audit"):
            with st.spinner("Auditing the statistics..."):
                audit = run_audit(raw, analysis, st.session_state.get("health_report"),
                                  assumptions)
            st.session_state["audit_result"] = {"key": audit_key, "audit": audit}
            st.rerun()          # redraw without the run card

    st.stop()


# PROCESS 5 – VERDICT AND HOW TO READ IT

status, summary, detail = audit["verdict"]
show = {"pass": st.success, "warn": st.warning, "fail": st.error}[status]
show(f"**{summary}.** {detail}",
     icon={"pass": ":material/check_circle:", "warn": ":material/warning:",
           "fail": ":material/error:"}[status])

st.caption("  ·  ".join(f"{BADGES[s]} {MEANING[s]}" for s in STATUS_ORDER
                        if audit["counts"][s] or s != "skip"))


# PROCESS 6 – ONE TILE PER QUESTION

checks = audit["checks"]

for (area, (question, icon)), column in zip(AREAS.items(), st.columns(len(AREAS))):

    area_checks = [c for c in checks if c["area"] == area]
    worst = next((s for s in STATUS_ORDER if any(c["status"] == s for c in area_checks)), "skip")
    passed = sum(c["status"] == "pass" for c in area_checks)

    with column.container(border=True, height="stretch"):
        st.markdown(f"**{icon} {question}**")
        st.markdown(BADGES[worst])
        st.caption(f"{passed} of {len(area_checks)} checks passed")


# PROCESS 7 – EVERY CHECK, IN PLAIN WORDS

for area, (question, icon) in AREAS.items():

    area_checks = [c for c in checks if c["area"] == area]
    if not area_checks:
        continue

    with st.container(border=True):

        st.subheader(f"{icon} {question}", anchor=False)

        for position, check in enumerate(area_checks):

            if position:
                st.divider()

            st.markdown(f"{BADGES[check['status']]} **{check['check']}**")
            st.markdown(md(check["result"]))
            st.caption(f"Why it matters: {check['why']}")

            if not check["details"].empty:
                with st.expander("Show the numbers", icon=":material/table_chart:"):
                    st.dataframe(friendly(check["details"]), hide_index=True,
                                 width="stretch")


# PROCESS 8 – REVIEWER SIGN-OFF AND DOWNLOADS

with st.container(border=True):

    st.subheader(":material/approval: Reviewer sign-off", anchor=False)
    st.caption("A person who checked this audit records their decision. It is saved in the "
               "downloaded audit pack, not on a server.")

    with st.form("sign_off", border=False):
        name_col, role_col = st.columns(2)
        name = name_col.text_input("Reviewer name")
        role = role_col.text_input("Role", placeholder="e.g. People Analytics Lead")
        decision = st.radio(
            "Decision",
            ["Approved to share", "Approved with the caveats listed", "Not approved"],
            index=0 if status == "pass" else 1 if status == "warn" else 2,
            horizontal=True,
        )
        note = st.text_area("Note (optional)", height=80)
        if st.form_submit_button("Record sign-off", icon=":material/how_to_reg:"):
            if not name.strip():
                st.warning("Add the reviewer's name to record a sign-off.")
            else:
                st.session_state["audit_sign_off"] = {
                    "name": name.strip(), "role": role.strip(), "decision": decision,
                    "note": note.strip(), "date": date.today().isoformat()}

    sign_off = st.session_state.get("audit_sign_off")
    if sign_off:
        st.success(f"Signed off by **{sign_off['name']}** ({sign_off['role'] or 'no role'}) on "
                   f"{sign_off['date']}: {sign_off['decision']}.", icon=":material/how_to_reg:")

    record = audit_record(raw, analysis, checks, audit["verdict"], sign_off)

    with st.container(horizontal=True):
        st.download_button(
            "Download audit report (Markdown)",
            data=audit_markdown(audit, sign_off).encode(),
            file_name="statistical_audit.md",
            mime="text/markdown",
            icon=":material/description:",
        )
        st.download_button(
            "Download audit pack (JSON)",
            data=json.dumps(record, indent=2, default=str).encode(),
            file_name="statistical_audit_pack.json",
            mime="application/json",
            icon=":material/download:",
            help="Every check with its numbers, fingerprints, versions and the sign-off: "
            "everything needed to reproduce and review this run.",
        )
