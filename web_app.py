"""
Streamlit web interface for the Research Paper Insight Generator.

This file is only a frontend layer. All extraction / NLP logic is imported
from app/main.py (process_pdf), so the standalone pipeline stays unchanged.

Run locally:   streamlit run app/web_app.py
Run in Docker: docker compose up --build
"""

import os
import tempfile

import pandas as pd
import streamlit as st

# Reuse the existing pipeline (importing main.py does not run it,
# because main() is guarded by `if __name__ == "__main__"`).
from main import OUTPUT_DIR, process_pdf

# CSV column order is taken from the existing pipeline output.
CSV_NAME = "research_paper_insights.csv"
WEB_CSV_PATH = os.path.join(OUTPUT_DIR, "web_" + CSV_NAME)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def run_pipeline(uploaded_files):
    """Save uploads to a temp folder and run the existing process_pdf()."""
    records, errors = [], []

    # Same ordering rule as main.py: sorted by filename, IDs RP1, RP2, ...
    uploaded_files = sorted(uploaded_files, key=lambda f: f.name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for index, uploaded in enumerate(uploaded_files, start=1):
            paper_id = f"RP{index}"
            pdf_path = os.path.join(tmp_dir, os.path.basename(uploaded.name))

            with open(pdf_path, "wb") as f:
                f.write(uploaded.getbuffer())

            try:
                records.append(process_pdf(pdf_path, paper_id))
            except Exception as error:
                errors.append(f"{uploaded.name}: {error}")

    return (pd.DataFrame(records) if records else None), errors


def quality_checks(df):
    """Same checks as the quality-check block in main.py."""
    return {
        "Papers processed": len(df),
        "Number of columns": len(df.columns),
        "Duplicate Paper IDs": int(df["Paper_ID"].duplicated().sum()),
        "Empty abstracts": int(
            (df["Abstract"].fillna("").str.strip() == "").sum()
        ),
        "Empty summaries": int(
            (df["Summary"].fillna("").str.strip() == "").sum()
        ),
    }


def show_tags(keyword_string):
    """Show TF-IDF keywords as small tags."""
    words = [w.strip() for w in str(keyword_string).split(",") if w.strip()]
    if not words:
        st.write("Not available")
        return
    st.markdown(" ".join(f"`{w}`" for w in words))


def show_paper(row):
    """Display one paper's insights."""
    with st.container(border=True):
        st.subheader(f"{row['Paper_ID']} — {row['Title']}")
        st.caption(f"File: {row['File_Name']}")

        st.markdown(f"**Authors:** {row['Authors']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preprint Date", str(row["Preprint_Date"]))
        pub_year = row["Publication_Year"]
        c2.metric(
            "Publication Year",
            "Not available" if pd.isna(pub_year) or pub_year is None
            else str(pub_year),
        )
        c3.metric("Pages", int(row["Pages"]))
        c4.metric("Word Count", f"{int(row['Word_Count']):,}")

        with st.expander("Abstract", expanded=False):
            st.write(row["Abstract"])

        st.markdown("**Keywords**")
        st.write(row["Keywords"])

        st.markdown("**Extractive Summary**")
        st.write(row["Summary"])

        st.markdown("**TF-IDF Keywords**")
        show_tags(row["TFIDF_Keywords"])


# ------------------------------------------------------------
# Page
# ------------------------------------------------------------

st.set_page_config(
    page_title="Research Paper Insight Generator",
    page_icon="📄",
    layout="wide",
)

# 1. Header
st.title("Research Paper Insight Generator")
st.subheader("Upload a research paper and generate structured insights automatically.")
st.caption(
    "Built with open-source tools: PyMuPDF for PDF processing, and "
    "regex + TF-IDF + cosine similarity (scikit-learn) for NLP."
)
st.divider()

# 2. File upload
st.header("1. Upload PDFs")
uploaded_files = st.file_uploader(
    "Choose one or more research-paper PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write("**Uploaded files:**")
    for f in uploaded_files:
        st.write(f"- {f.name}")

# 3. Process button
st.header("2. Process")
if st.button("Generate Insights", type="primary", disabled=not uploaded_files):
    with st.spinner("Processing PDFs..."):
        df, errors = run_pipeline(uploaded_files)

    if df is None:
        st.session_state.pop("results", None)
        st.error("No papers could be processed.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(WEB_CSV_PATH, index=False)
        st.session_state["results"] = df
        st.success(f"Processing complete: {len(df)} paper(s) processed.")

    for message in errors:
        st.warning(f"Failed — {message}")

if not uploaded_files:
    st.info("Upload at least one PDF to enable the button.")

# Results stay on screen after the download button is clicked
df = st.session_state.get("results")

if df is not None:
    # 4. Paper insights
    st.header("3. Paper Insights")
    for _, row in df.iterrows():
        show_paper(row)

    # 5. Output dataset
    st.header("4. Output Dataset")
    st.dataframe(df)

    # 6. Quality checks
    st.header("5. Data Quality Check")
    checks = quality_checks(df)
    cols = st.columns(len(checks))
    for col, (label, value) in zip(cols, checks.items()):
        col.metric(label, value)

    # 7. Download
    st.header("6. Download")
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=CSV_NAME,
        mime="text/csv",
    )
    st.caption(f"A copy is also saved on the server at: {WEB_CSV_PATH}")
