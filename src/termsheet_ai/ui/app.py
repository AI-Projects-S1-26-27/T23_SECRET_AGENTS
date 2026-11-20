"""TermSheet AI Streamlit entry point.  Run:  streamlit run app.py  (repo root)."""

import streamlit as st

from termsheet_ai.ui import views
from termsheet_ai.ui.data_source import (
    build_pipeline,
    load_mock_payload,
    resolve_gemini_api_key,
    run_pipeline_on_upload,
    uses_plaintext_ingestor,
)
from termsheet_ai.ui.theme import CSS, SEVERITY_ORDER, sev_label
from termsheet_ai.ui.viewmodel import build_clause_rows

DEMO = "Demo result (mock data)"
UPLOAD = "Upload a term sheet"


def _sidebar():
    st.sidebar.header("Term sheet")
    mode = st.sidebar.radio("Source", [DEMO, UPLOAD], label_visibility="collapsed")
    payload = None
    if mode == DEMO:
        payload = load_mock_payload()
    else:
        use_cloud = st.session_state.get("use_cloud_llm", False)
        if use_cloud:
            key = resolve_gemini_api_key()
            if key:
                st.sidebar.info("Engine: Live Google Gemini 1.5 Flash Connected")
            else:
                st.sidebar.warning(
                    "Engine: Cloud LLM enabled, but GEMINI_API_KEY is not found in Secrets."
                )
        else:
            st.sidebar.caption("Engine: Deterministic Offline Mode")

        upload = st.sidebar.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
        if (
            upload is not None
            and uses_plaintext_ingestor(build_pipeline(use_cloud_llm=use_cloud))
            and not upload.name.lower().endswith(".txt")
        ):
            st.sidebar.warning(
                "The current ingestor reads plain text only. PDF/DOCX parsing arrives with the real S1 stage."
            )
        if upload is not None and st.sidebar.button("Analyse", type="primary"):
            spinner_msg = (
                "Running pipeline with Gemini Cloud LLM..."
                if use_cloud
                else "Running deterministic offline pipeline..."
            )
            with st.spinner(spinner_msg):
                try:
                    st.session_state["payload"] = run_pipeline_on_upload(
                        upload.name, upload.getvalue(), use_cloud_llm=use_cloud
                    )
                except Exception as exc:  # surface any pipeline failure instead of a stack trace
                    st.session_state.pop("payload", None)
                    st.sidebar.error(f"Pipeline failed: {exc}")
        payload = st.session_state.get("payload")

    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    severities = st.sidebar.multiselect(
        "Severities to show", SEVERITY_ORDER, default=list(SEVERITY_ORDER), format_func=sev_label
    )
    only_flagged = st.sidebar.checkbox("Only clauses with flags", value=False)
    return mode, payload, severities, only_flagged


def main() -> None:
    st.set_page_config(page_title="TermSheet AI", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    header_col, toggle_col = st.columns([3, 1])
    with header_col:
        st.title("TermSheet AI")
        st.caption("Plain-English term sheet review with an auditable rule trace.")
    with toggle_col:
        st.write("")  # vertical spacing alignment
        use_cloud_llm = st.toggle(
            "Cloud LLM (Gemini)",
            value=st.session_state.get("use_cloud_llm", False),
            key="use_cloud_llm",
            help="When toggled ON, uses Google Gemini Cloud LLM for Stage S3 fact extraction and Stage S5 explanations. When OFF, runs deterministic local mode.",
        )
        if use_cloud_llm:
            key = resolve_gemini_api_key()
            if key:
                st.caption("Status: Gemini 1.5 Flash Active")
            else:
                st.caption("Status: Missing GEMINI_API_KEY in Secrets")
        else:
            st.caption("Status: Offline Deterministic")

    mode, payload, severities, only_flagged = _sidebar()
    if payload is None:
        st.info(
            "Upload a term sheet in the sidebar and press Analyse, or switch to the demo result."
        )
        return
    if mode == DEMO:
        st.info("You are viewing bundled mock data. Nothing here came from the live pipeline.")

    views.render_header(payload)
    clauses_tab, flags_tab, facts_tab, investor_tab, debug_tab = st.tabs(
        ["Clauses", "Flags and audit trace", "Extracted facts", "Investor lookup", "Debug"]
    )
    with clauses_tab:
        views.render_clauses_tab(build_clause_rows(payload), severities, only_flagged)
    with flags_tab:
        views.render_flags_tab(payload)
    with facts_tab:
        views.render_facts_tab(payload)
    with investor_tab:
        views.render_investor_tab()
    with debug_tab:
        views.render_debug_tab(payload)


if __name__ == "__main__":
    main()
