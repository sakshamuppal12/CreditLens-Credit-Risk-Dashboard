from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# ---------- CONFIGURATION ----------
st.set_page_config(
    page_title="CreditLens | Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

# The app checks the project folder and common data subfolders.
WORKBOOK_CANDIDATES = [
    ROOT / "AssessmentDataScientist.xlsx",
    ROOT / "data" / "AssessmentDataScientist.xlsx",
    ROOT / "AssessmentDataScientist.xlsm",
    ROOT / "data" / "AssessmentDataScientist.xlsm",
]

CHARTS = [
    (
        "01 / Discrimination",
        "roc.png",
        "ROC curve from the existing model run. Confirm the evaluation split "
        "and prediction provenance before interpreting its AUC.",
    ),
    (
        "02 / Risk ranking",
        "quint.png",
        "Observed Bad-label rates by score quintile. Confirm that the scores "
        "are out-of-sample predictions.",
    ),
    (
        "03 / Model interpretation",
        "coef.png",
        "Existing standardized coefficients. Associations do not establish "
        "causation or justify lending decisions.",
    ),
]


# ---------- WORKBOOK LOADING ----------
def locate_workbook():
    for candidate in WORKBOOK_CANDIDATES:
        if candidate.exists():
            return candidate

    # If the workbook was renamed, accept one Excel file in the project root.
    possible = [
        *ROOT.glob("*.xlsx"),
        *ROOT.glob("*.xlsm"),
    ]
    if len(possible) == 1:
        return possible[0]

    return None


@st.cache_data(show_spinner="Reading workbook...")
def load_workbook(workbook_path, modified_time):
    # modified_time is included in the cache key so edits to the workbook
    # trigger a fresh read on the next rerun.
    applications = pd.read_excel(
        workbook_path,
        sheet_name="Application Data",
    )
    loans = pd.read_excel(
        workbook_path,
        sheet_name="Loan Performance",
    )
    return applications, loans


def load_data():
    workbook_path = locate_workbook()

    if workbook_path is None:
        return None, None, None, (
            "Workbook not found. Place AssessmentDataScientist.xlsx next "
            "to app.py or inside a data folder. If you renamed it, ensure "
            "there is only one Excel workbook in the project folder."
        )

    try:
        applications, loans = load_workbook(
            str(workbook_path),
            workbook_path.stat().st_mtime,
        )
        required_app_sheets = {"customer_id"}
        required_loan_columns = {"idLoan", "flgGood"}

        missing_app = required_app_sheets - set(applications.columns)
        missing_loan = required_loan_columns - set(loans.columns)

        if missing_app or missing_loan:
            problems = []
            if missing_app:
                problems.append(
                    "Application Data is missing: "
                    + ", ".join(sorted(missing_app))
                )
            if missing_loan:
                problems.append(
                    "Loan Performance is missing: "
                    + ", ".join(sorted(missing_loan))
                )
            return workbook_path, None, None, ". ".join(problems)

        return workbook_path, applications, loans, None

    except ValueError as exc:
        return workbook_path, None, None, (
            "Could not find the expected worksheet names. Expected "
            "'Application Data' and 'Loan Performance'. Details: "
            + str(exc)
        )
    except Exception as exc:
        return workbook_path, None, None, (
            f"Could not read workbook ({type(exc).__name__}): {exc}"
        )


workbook_path, applications, loans, load_error = load_data()


# ---------- DESIGN SYSTEM ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b1120;
        --panel: #111b2e;
        --border: #25344d;
        --muted: #91a3bd;
        --teal: #48d6bd;
    }

    .stApp {
        background: var(--bg);
        color: #edf3fc;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: rgba(11,17,32,0.95); }

    [data-testid="stSidebar"] {
        background: #0e1728;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

    .block-container {
        max-width: 1450px;
        padding-top: 2.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.6px;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: #f5f8ff;
        letter-spacing: -0.8px;
    }

    .brand span { color: var(--teal); }

    .eyebrow {
        color: #48d6bd;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .hero {
        padding: 28px 30px;
        background: linear-gradient(120deg, #14243d 0%, #101a2c 65%, #102a35 100%);
        border: 1px solid #2b405c;
        border-radius: 18px;
        margin: 10px 0 24px 0;
    }

    .hero h1 {
        font-size: clamp(27px, 3vw, 40px);
        line-height: 1.18;
        margin: 0 0 10px 0;
        color: #f5f8ff;
    }

    .hero p {
        color: #aabbd2;
        font-size: 14px;
        line-height: 1.7;
        margin: 0;
        max-width: 760px;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 20px;
        background: #332b1b;
        border: 1px solid #62502c;
        color: #ffd78b;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .kpi {
        background: linear-gradient(145deg, #142039, #111a2b);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 19px;
        min-height: 128px;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 13px;
    }

    .kpi-value {
        font-family: 'Manrope', sans-serif;
        font-size: 30px;
        font-weight: 800;
        color: #f4f7ff;
        line-height: 1.1;
    }

    .kpi-note {
        color: #7f95b2;
        font-size: 11px;
        margin-top: 10px;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 19px;
        font-weight: 800;
        margin: 27px 0 5px 0;
        color: #f0f5ff;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 17px;
    }

    .panel {
        background: #111b2e;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .panel-title {
        font-family: 'Manrope', sans-serif;
        font-size: 15px;
        font-weight: 700;
        color: #eaf1ff;
        margin-bottom: 8px;
    }

    .muted {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
    }

    .insight {
        border-left: 3px solid var(--teal);
        background: #11272d;
        padding: 14px 16px;
        border-radius: 0 10px 10px 0;
        color: #d9f5ef;
        font-size: 13px;
        line-height: 1.7;
        margin: 12px 0;
    }

    .warning-box {
        border: 1px solid #6d5630;
        background: #282319;
        border-radius: 12px;
        padding: 15px 17px;
        color: #f5dca5;
        font-size: 12px;
        line-height: 1.7;
        margin: 18px 0;
    }

    .tag {
        display: inline-block;
        padding: 4px 8px;
        background: #1c304a;
        border: 1px solid #304a69;
        border-radius: 6px;
        color: #a9caff;
        font-size: 10px;
        font-weight: 700;
    }

    hr { border-color: var(--border) !important; }

    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid var(--border);
    }

    #MainMenu, footer { visibility: hidden; }

    @media (max-width: 700px) {
        .hero { padding: 20px; }
        .kpi { padding: 14px; }
        .kpi-value { font-size: 24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- REUSABLE COMPONENTS ----------
def kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        '<div class="brand">Credit<span>Lens</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted">RISK INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**WORKSPACE**")
    page = st.radio(
        "Navigation",
        ["Overview", "Model Diagnostics", "Data & Limitations"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        '<span class="tag">ASSESSMENT PROJECT</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Exploratory analytics based on the supplied '
        'loan-performance workbook.</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("CREDITLENS · DEVELOPMENT BUILD")


# ---------- PAGE HEADER ----------
st.markdown(
    '<div class="eyebrow">DATA SCIENCE / CREDIT ANALYTICS</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="status-pill">● EXPLORATORY BUILD · NOT PRODUCTION VALIDATED</div>
        <h1>Credit Risk Intelligence</h1>
        <p>Explore loan-performance labels, inspect model diagnostics, and
        understand the data-quality issues that must be resolved before
        reliable credit-risk predictions can be made.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-box">
        <b>Model status:</b> Predictive results and hypothetical profit estimates
        have not been validated for lending decisions. Existing charts are
        displayed for diagnostic review only.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- HANDLE DATA ERRORS ----------
if load_error:
    st.error(load_error)
    st.info(
        "Check the workbook filename, worksheet names, and required columns. "
        "The app has not modified the workbook."
    )
    with st.expander("Expected workbook structure"):
        st.code(
            "AssessmentDataScientist.xlsx\n"
            "├── Application Data (must contain customer_id)\n"
            "└── Loan Performance (must contain idLoan and flgGood)"
        )
    st.stop()


# ---------- OVERVIEW ----------
if page == "Overview":
    section(
        "Portfolio at a glance",
        f"Metrics calculated from {workbook_path.name}.",
    )

    total_applications = len(applications)
    total_loans = len(loans)

    labels = loans["flgGood"].astype("string").str.strip().str.casefold()
    good_count = int((labels == "good").sum())
    bad_count = int((labels == "bad").sum())
    other_count = total_loans - good_count - bad_count

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        kpi("APPLICATION RECORDS", f"{total_applications:,}", "Rows in Application Data")
    with c2:
        kpi("LOAN RECORDS", f"{total_loans:,}", "Rows in Loan Performance")
    with c3:
        good_rate = good_count / total_loans if total_loans else 0
        kpi("GOOD LOAN LABELS", f"{good_count:,}", f"{good_rate:.1%} of loan records")

    section(
        "Observed loan labels",
        "Historical labels from the original loan sheet, not model predictions.",
    )

    label_data = pd.DataFrame(
        {
            "Outcome": ["Good", "Bad"],
            "Records": [good_count, bad_count],
        }
    )

    chart_col, summary_col = st.columns([1.0, 0.72], gap="large")

    with chart_col:
        fig = px.bar(
            label_data,
            x="Outcome",
            y="Records",
            color="Outcome",
            color_discrete_map={
                "Good": "#48d6bd",
                "Bad": "#f07178",
            },
            text="Records",
            template="plotly_dark",
        )
        fig.update_layout(
            height=330,
            showlegend=False,
            plot_bgcolor="#111b2e",
            paper_bgcolor="#111b2e",
            font_color="#edf3fc",
            margin=dict(l=12, r=12, t=28, b=12),
            xaxis_title=None,
            yaxis_title="Loan records",
            yaxis=dict(gridcolor="#25344d", zerolinecolor="#25344d"),
            xaxis=dict(showgrid=False),
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )

    with summary_col:
        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">Label summary</div>'
            '<div class="muted">Observed labels in all loan records</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        s1, s2 = st.columns(2)
        with s1:
            st.metric("Good", f"{good_count:,}")
            st.caption(
                f"{good_count / total_loans:.1%} of all loans"
                if total_loans else "No loan records"
            )
        with s2:
            st.metric("Bad", f"{bad_count:,}")
            st.caption(
                f"{bad_count / total_loans:.1%} of all loans"
                if total_loans else "No loan records"
            )

        if other_count:
            st.warning(
                f"{other_count:,} labels are blank or unrecognized."
            )

    # Conservative exact-link count: only count loans whose customer ID
    # prefix exactly matches an application customer_id after case-folding.
    # Do not infer or repair malformed IDs.
    app_ids = (
        applications["customer_id"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    loan_ids = (
        loans["idLoan"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(r"-\\d{2}$", "", regex=True)
    )
    valid_app_ids = set(app_ids.dropna())
    exact_link_mask = loan_ids.isin(valid_app_ids)
    linked_loan_count = int(exact_link_mask.sum())
    unmatched_loan_count = int(total_loans - linked_loan_count)

    section(
        "Linked-data coverage",
        "Exact identifier matches only; IDs are not inferred or silently corrected.",
    )
    l1, l2, l3 = st.columns(3, gap="medium")
    with l1:
        kpi(
            "EXACT-LINKED LOANS",
            f"{linked_loan_count:,}",
            "Loan IDs with an exact application ID match",
        )
    with l2:
        kpi(
            "UNMATCHED LOANS",
            f"{unmatched_loan_count:,}",
            "Excluded from application-linked analysis",
        )
    with l3:
        link_rate = linked_loan_count / total_loans if total_loans else 0
        kpi(
            "LINKAGE RATE",
            f"{link_rate:.1%}",
            "Of all loan records",
        )

    st.caption(
        "This linkage check strips a trailing loan sequence such as '-01' "
        "or '-02' from idLoan before matching. Review this rule against the "
        "official identifier specification; it does not repair IDs altered by Excel."
    )

    section(
        "Data quality snapshot",
        "Missing values and duplicate rows in each source sheet.",
    )

    q1, q2 = st.columns(2, gap="medium")
    for container, frame, title in [
        (q1, applications, "Application Data"),
        (q2, loans, "Loan Performance"),
    ]:
        with container:
            st.markdown(f"#### {title}")
            missing = frame.isna().sum()
            missing = missing[missing > 0]

            if missing.empty:
                st.success("No missing cells detected.")
            else:
                missing_df = pd.DataFrame(
                    {
                        "Column": missing.index.astype(str),
                        "Missing cells": missing.values.astype(int),
                        "Missing rate (%)": (
                            missing.values / len(frame) * 100
                            if len(frame)
                            else 0
                        ),
                    }
                )
                missing_df["Missing rate (%)"] = missing_df[
                    "Missing rate (%)"
                ].round(2)
                st.dataframe(
                    missing_df,
                    hide_index=True,
                    use_container_width=True,
                )

            st.metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")

    st.markdown(
        """
        <div class="insight">
            <b>Interpretation note</b><br>
            Source-level counts and missingness are descriptive checks. They
            do not establish predictive validity or justify lending decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- MODEL DIAGNOSTICS ----------
elif page == "Model Diagnostics":
    section(
        "Model diagnostics",
        "Existing charts from the current modeling script.",
    )

    st.markdown(
        """
        <div class="muted">These figures are existing outputs. The model's
        evaluation design has not been independently validated.</div>
        """,
        unsafe_allow_html=True,
    )

    for title, filename, description in CHARTS:
        st.markdown(
            f'<div class="section-title">{title}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="section-subtitle">{description}</div>',
            unsafe_allow_html=True,
        )

        image_path = ROOT / filename
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(
                f"Chart not found: {filename}. Place it beside app.py "
                "or update the path in CHARTS."
            )
        st.divider()

    st.markdown(
        """
        <div class="warning-box">
            Do not treat the displayed AUC, coefficients, risk quintiles, or
            profit estimates as validated lending evidence until the target,
            feature timing, and customer-grouped evaluation have been reviewed.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- DATA & LIMITATIONS ----------
elif page == "Data & Limitations":
    section(
        "Data quality & limitations",
        "Known issues from the prior workbook audit; verify them against your current source.",
    )

    issues = [
        (
            "01",
            "Target definition",
            "The authoritative rule defining Good and Bad is not documented.",
        ),
        (
            "02",
            "Outcome horizon",
            "The period used to observe loan performance is not documented.",
        ),
        (
            "03",
            "Feature timing",
            "Availability of candidate predictors at application time has not been verified.",
        ),
        (
            "04",
            "Unmatched IDs",
            "Seven application IDs were reported as unmatched in the prior audit. "
            "Do not correct them without an authoritative crosswalk.",
        ),
        (
            "05",
            "Validation design",
            "The existing modeling script should be checked for customer overlap "
            "between training and evaluation data.",
        ),
        (
            "06",
            "Potential leakage",
            "Fields such as amount_approved may only be known after a lending decision.",
        ),
        (
            "07",
            "Profit assumptions",
            "Profit scenarios rely on assumptions that require verification against "
            "actual recovery and cost data.",
        ),
    ]

    for number, title, description in issues:
        st.markdown(
            f"""
            <div class="panel">
                <div style="color:#48d6bd;font-size:11px;font-weight:800;
                            letter-spacing:1px">{number} / REVIEW REQUIRED</div>
                <div class="panel-title" style="margin-top:8px">{title}</div>
                <div class="muted">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section(
        "Recommended next steps",
        "Resolve these issues before considering predictive use.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="muted">
            <b>1.</b> Obtain the authoritative Good/Bad definition and observation period.<br><br>
            <b>2.</b> Verify feature availability at the intended prediction time.<br><br>
            <b>3.</b> Preserve unresolved IDs until a reliable crosswalk is found.<br><br>
            <b>4.</b> Use customer-grouped validation and report genuine out-of-sample metrics.<br><br>
            <b>5.</b> Document privacy and fairness risks before any proposed lending use.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- FOOTER ----------
st.markdown(
    """
    <hr>
    <div style="display:flex;justify-content:space-between;gap:12px;
    flex-wrap:wrap;color:#8192aa;font-size:11px;padding:8px 0">
        <span>CREDITLENS · CREDIT RISK INTELLIGENCE</span>
        <span>EXPLORATORY ANALYTICS · NOT FOR LENDING DECISIONS</span>
    </div>
    """,
    unsafe_allow_html=True,
)
