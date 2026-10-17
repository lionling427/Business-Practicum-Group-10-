# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.financial import Assumptions
from src.tools.insights import analysis_frame
from src.tools.report_pack import (
    SECTIONS,
    Branding,
    build_pack,
    slide_previews,
    to_pdf,
    to_pptx,
)
from src.ui import OUTCOME_LEVEL, analysis_runner, page_header, require_data, trap_checks


# PROCESS 2 – HEADER AND DATA

page_header(
    "Report pack",
    ":material/slideshow:",
    "A branded slide deck and PDF for leaders to take away, built only from the app's "
    "verified results.",
)

df, validation = require_data()

if validation["capabilities"]["level"] != OUTCOME_LEVEL:
    st.info("The report pack needs business outcomes (turnover, absence or performance) in "
            "the uploaded data.", icon=":material/info:")
    st.stop()

analysis = analysis_runner(df, key="run_analysis_pack")
if analysis is None:
    st.stop()


# PROCESS 3 – BRANDING AND CONTENTS

with st.container(border=True):

    st.markdown("**:material/palette: Branding and contents**")
    name_col, by_col, color_col = st.columns([2, 2, 1])
    company = name_col.text_input("Organization name", value="Your organization",
                                  key="pack_company")
    prepared_by = by_col.text_input("Prepared by", value="People Analytics", key="pack_by")
    color = color_col.color_picker("Brand color", value="#1E7B34", key="pack_color")
    logo_file = st.file_uploader("Logo (optional, PNG or JPG)", type=["png", "jpg", "jpeg"],
                                 key="pack_logo")
    sections = st.pills("Slides to include", options=SECTIONS, selection_mode="multi",
                        default=SECTIONS, key="pack_sections")

branding = Branding(company=company.strip() or "Your organization", color=color,
                    logo=logo_file.getvalue() if logo_file else None,
                    prepared_by=prepared_by.strip() or "People Analytics")
assumptions = (st.session_state.get("assumptions")
               or Assumptions(headcount=len(analysis_frame(df))))
audit = (st.session_state.get("audit_result") or {}).get("audit")

slides = build_pack(df, analysis, assumptions, trap_checks(df),
                    st.session_state.get("change_analysis"), audit, sections or [])


# PROCESS 4 – PREVIEW

st.subheader(f"Preview · {len(slides)} slides", anchor=False)
st.caption("Every number comes from the app's verified results. Money uses the cost "
           "assumptions on the Financials page"
           + ("; the audit verdict is included." if audit else
              "; run the Audit page to add its verdict to 'How sure we are'."))

with st.spinner("Drawing the slides..."):
    previews = slide_previews(slides, branding, dpi=50)

for row in range(0, len(previews), 3):
    for image, column in zip(previews[row:row + 3], st.columns(3)):
        column.image(image, width="stretch")


# PROCESS 5 – DOWNLOAD

file_name = "".join(ch if ch.isalnum() else "_" for ch in branding.company).strip("_").lower()

with st.container(horizontal=True):
    st.download_button("Download PowerPoint", data=to_pptx(slides, branding),
                       file_name=f"{file_name}_engagement_report.pptx",
                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                       icon=":material/slideshow:", type="primary",
                       help="Editable slides: text and tables can be changed in PowerPoint, "
                       "Keynote or Google Slides.")
    st.download_button("Download PDF", data=to_pdf(slides, branding),
                       file_name=f"{file_name}_engagement_report.pdf", mime="application/pdf",
                       icon=":material/picture_as_pdf:",
                       help="The same slides as a PDF, ready to email.")
